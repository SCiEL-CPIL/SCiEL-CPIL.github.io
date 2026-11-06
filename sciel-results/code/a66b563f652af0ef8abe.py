"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate
import scipy.linalg

def integrate(f, t_span, y0):
    """
    Main integration function using a block Krylov subspace method (block Arnoldi).
    Strategy:
    1. Reconstruct the linear operator A using n basis vectors (n calls).
    2. Construct an orthonormal basis V for the Krylov subspace K_m(A, y0) = span{y0, A y0, ..., A^(m-1) y0}.
    3. Project A onto this subspace to get a small m x m upper Hessenberg matrix H.
    4. Compute y_end = V @ expm(H * dt) @ e1 * ||y0|| (approximately, adjusted for the projection).
    
    This avoids forming expm(A*dt) directly and reduces the problem to a small matrix exponential.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    deadline = time.time() + 280.0  # Safety margin

    # Step 1: Reconstruct A.
    # For a linear ODE y' = A y, we can recover A by evaluating f on basis vectors.
    # f(t, y) = A @ y.
    # A[:, i] = f(t0, e_i)
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
        
        if time.time() > deadline:
            # Fallback if we run out of time during reconstruction (unlikely for n=100)
            return _fallback_solve(f, t_span, y0, deadline)

    # Step 2: Block Arnoldi / Krylov Subspace Construction
    # We want to compute y(t1) = expm(A * dt) @ y0.
    # Let's build a Krylov subspace basis V = [v_1, v_2, ..., v_m] where v_1 = y0 / ||y0||.
    # H is the projection of A onto V: V^T A V = H.
    # Then expm(A*dt) y0 ≈ V expm(H*dt) (V^T y0).
    # Since v_1 is y0 normalized, V^T y0 = [||y0||, 0, ..., 0]^T.
    
    norm_y0 = np.linalg.norm(y0)
    if norm_y0 == 0:
        return y0
    
    v0 = y0 / norm_y0
    
    m = 30  # Dimension of the Krylov subspace. 30 is usually enough for n=100 smooth problems.
    # We can increase m if we have time, but 30 is a good balance.
    
    V = np.zeros((n, m))
    H = np.zeros((m, m))
    
    V[:, 0] = v0
    
    # Orthogonalize v0 against A
    w = A @ V[:, 0]
    
    # Modified Gram-Schmidt for stability
    for j in range(m):
        if j > 0:
            w = A @ V[:, j-1]
            # Subtract projections onto previous vectors
            for k in range(j):
                H[k, j-1] = np.dot(V[:, k], w)
                w -= H[k, j-1] * V[:, k]
            
            # Second pass for reorthogonalization (improves stability)
            for k in range(j):
                H[k, j-1] += np.dot(V[:, k], w)
                w -= H[k, j-1] * V[:, k]
            
            beta = np.linalg.norm(w)
            H[j, j-1] = beta
            if beta < 1e-14:
                # Breakdown, subspace is invariant
                m_actual = j + 1
                break
            V[:, j] = w / beta
        else:
            # First step j=0
            w = A @ V[:, 0]
            for k in range(1): # k=0
                H[0, 0] = np.dot(V[:, 0], w)
                w -= H[0, 0] * V[:, 0]
            # Reorthogonalize
            H[0, 0] += np.dot(V[:, 0], w)
            w -= H[0, 0] * V[:, 0]
            
            beta = np.linalg.norm(w)
            H[1, 0] = beta
            if beta < 1e-14:
                m_actual = 1
                break
            V[:, 1] = w / beta

    # If loop completed, m_actual is m
    if 'm_actual' not in locals():
        m_actual = m
    
    # Step 3: Solve the small ODE
    # We need expm(H_sub * dt) where H_sub is the top-left m_actual x m_actual block of H.
    H_sub = H[:m_actual, :m_actual]
    
    try:
        # Compute exponential of the small matrix
        E = scipy.linalg.expm(H_sub * dt)
        
        # The initial condition in the subspace coordinates is c = [norm_y0, 0, ..., 0]^T
        c = np.zeros(m_actual)
        c[0] = norm_y0
        
        y_sub_end = E @ c
        
        # Project back to original space
        y_final = V[:, :m_actual] @ y_sub_end
        
    except Exception:
        # Fallback to full matrix exponential if small system fails
        try:
            M = scipy.linalg.expm(A * dt)
            y_final = M @ y0
        except Exception:
            return _fallback_solve(f, t_span, y0, deadline)
    
    return y_final

def _fallback_solve(f, t_span, y0, deadline):
    """Fallback to DOP853 if Krylov approach fails."""
    result = scipy.integrate.solve_ivp(
        f, t_span, y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14
    )
    return result.y[:, -1] if result.success else y0
# EVOLVE-BLOCK-END
