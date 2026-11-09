import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    """
    Solves the linear autonomous ODE y' = A y for a 10x10 2D heat equation.
    Attempts to infer the matrix A structure (Dirichlet, Neumann, or Periodic Laplacian)
    with minimal function calls (2-3). Falls back to full reconstruction (100 calls) 
    if structure inference fails.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    y0 = np.asarray(y0, dtype=float).flatten()
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0

    if N == 0:
        return y0
    
    if dt == 0:
        return y0

    # If N is not 100, we can't assume 10x10 structure. 
    # But the task says "fixed 10x10". 
    # If N != 100, fallback to general method.
    if N != 100:
        return _general_solve(f, t0, t1, y0, deadline)

    # 1. Infer Structure
    # Pick an interior node (5,5) -> index 55
    # Pick a corner node (0,0) -> index 0
    
    idx_int = 55
    idx_cor = 0
    
    e_int = np.zeros(N)
    e_int[idx_int] = 1.0
    
    e_cor = np.zeros(N)
    e_cor[idx_cor] = 1.0
    
    try:
        v_int = np.asarray(f(0.0, e_int), dtype=float).flatten()
        v_cor = np.asarray(f(0.0, e_cor), dtype=float).flatten()
    except Exception:
        # If f fails on these, fallback
        return _general_solve(f, t0, t1, y0, deadline)

    if time.time() > deadline:
        return _fast_rk4(f, t0, t1, y0, 1000)

    # Extract weights from interior node
    # Neighbors of 55 (5,5) are 54 (5,4), 56 (5,6), 45 (4,5), 65 (6,5)
    w_c = v_int[55]
    w_n = v_int[54]
    
    # Check if weights are valid (non-zero)
    if abs(w_n) < 1e-12:
        return _general_solve(f, t0, t1, y0, deadline)

    # Construct candidate matrices
    A_dir = _build_laplacian_dirichlet(10, w_c, w_n)
    A_per = _build_laplacian_periodic(10, w_c, w_n)
    A_neu = _build_laplacian_neumann(10, w_c, w_n)

    # Evaluate which candidate fits the observations best
    # We have observations for e_int and e_cor
    
    err_dir = np.linalg.norm(A_dir @ e_int - v_int) + np.linalg.norm(A_dir @ e_cor - v_cor)
    err_per = np.linalg.norm(A_per @ e_int - v_int) + np.linalg.norm(A_per @ e_cor - v_cor)
    err_neu = np.linalg.norm(A_neu @ e_int - v_int) + np.linalg.norm(A_neu @ e_cor - v_cor)

    errors = [err_dir, err_per, err_neu]
    best_idx = np.argmin(errors)
    best_err = errors[best_idx]

    # Threshold for accepting inferred structure
    # If the error is very small, we trust the structure
    # Scale relative to vector norm
    scale = np.linalg.norm(v_int) + np.linalg.norm(v_cor)
    if scale > 0 and best_err / scale < 1e-10:
        A = [A_dir, A_per, A_neu][best_idx]
        # Use 2 calls
    else:
        # Fallback to full reconstruction
        # We already used 2 calls. We need 98 more? 
        # Or just do 100 total? 
        # Let's just do the full loop, it's fast.
        A = _reconstruct_A(f, N, deadline)
        if A is None:
             # Timeout during reconstruction
             return _fast_rk4(f, t0, t1, y0, 1000)

    if time.time() > deadline:
        return _fast_rk4(f, t0, t1, y0, 1000)

    # Compute exact solution
    try:
        expA = expm(dt * A)
        y_end = expA @ y0
    except Exception:
        y_end = _fast_rk4(f, t0, t1, y0, 1000)

    return y_end

def _build_laplacian_dirichlet(n, w_c, w_n):
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_c
            
            # Left
            if j > 0:
                A[idx, i * n + (j - 1)] = w_n
            # Right
            if j < n - 1:
                A[idx, i * n + (j + 1)] = w_n
            # Up
            if i > 0:
                A[idx, (i - 1) * n + j] = w_n
            # Down
            if i < n - 1:
                A[idx, (i + 1) * n + j] = w_n
    return A

def _build_laplacian_periodic(n, w_c, w_n):
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_c
            
            # Left (wrap)
            j_left = (j - 1) % n
            A[idx, i * n + j_left] = w_n
            # Right (wrap)
            j_right = (j + 1) % n
            A[idx, i * n + j_right] = w_n
            # Up (wrap)
            i_up = (i - 1) % n
            A[idx, i_up * n + j] = w_n
            # Down (wrap)
            i_down = (i + 1) % n
            A[idx, i_down * n + j] = w_n
    return A

def _build_laplacian_neumann(n, w_c, w_n):
    # Neumann BC: du/dx = 0 at boundaries.
    # Discretization: ghost nodes u[-1] = u[1], u[N] = u[N-2].
    # For a corner (0,0):
    # u[1,0] + u[-1,0] + u[0,1] + u[0,-1] - 4u[0,0]
    # = u[1,0] + u[1,0] + u[0,1] + u[0,1] - 4u[0,0]
    # = 2u[1,0] + 2u[0,1] - 4u[0,0]
    # So A[0,0] = w_c (which is -4c)
    # A[0,1] = 2*w_n
    # A[1,0] = 2*w_n
    # For edge (0,j) where 0<j<n-1:
    # u[j+1,0] + u[j-1,0] + u[j,1] + u[j,-1] - 4u[j,0]
    # = u[j+1,0] + u[j-1,0] + u[j,1] + u[j,1] - 4u[j,0]
    # = u[j+1,0] + u[j-1,0] + 2u[j,1] - 4u[j,0]
    # A[0,j] = w_c
    # A[0,j+1] = w_n
    # A[0,j-1] = w_n
    # A[1,j] = 2*w_n
    
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_c
            
            # Left
            if j > 0:
                A[idx, i * n + (j - 1)] = w_n
            else:
                # Corner or Edge with left boundary
                if i > 0 and i < n - 1:
                    # Edge (not corner)
                    A[idx, i * n + 1] = 2.0 * w_n
                elif i == 0 or i == n - 1:
                    # Corner
                    A[idx, i * n + 1] = 2.0 * w_n
                    # Also need to handle the other boundary neighbor?
                    # For corner (0,0): neighbors are (1,0) and (0,1).
                    # We are setting A[0,0 -> 0,1] here? No, j=0.
                    # Left neighbor is (0, -1) -> ghost (0,1). So A[0,0 -> 0,1] gets w_n from left ghost.
                    # Up neighbor is (-1, 0) -> ghost (1,0). So A[0,0 -> 1,0] gets w_n from up ghost.
                    # Total A[0,0 -> 0,1] = w_n (from left ghost) + w_n (from actual right? No, actual right is (0,1)).
                    # Wait.
                    # Corner (0,0):
                    # Left ghost (0,-1) -> (0,1). Weight w_n.
                    # Right actual (0,1). Weight w_n.
                    # Sum for A[0,0 -> 0,1] is 2*w_n.
                    # Up ghost (-1,0) -> (1,0). Weight w_n.
                    # Down actual (1,0). Weight w_n.
                    # Sum for A[0,0 -> 1,0] is 2*w_n.
                    
                    # So for corner, both existing neighbors get 2*w_n.
                    # My loop logic:
                    # If j==0: Left ghost. Add w_n to (i, 1).
                    # If i==0: Up ghost. Add w_n to (1, j).
                    # If j==n-1: Right ghost. Add w_n to (i, n-2).
                    # If i==n-1: Down ghost. Add w_n to (n-2, j).
                    
                    # Let's restart Neumann construction logic to be precise.
                    pass
    # Simpler Neumann Construction:
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_c
            
            # Left
            if j > 0:
                A[idx, i * n + (j - 1)] += w_n
            else:
                # Ghost left is (i, 1)
                A[idx, i * n + 1] += w_n
            
            # Right
            if j < n - 1:
                A[idx, i * n + (j + 1)] += w_n
            else:
                # Ghost right is (i, n-2)
                A[idx, i * n + (n - 2)] += w_n
            
            # Up
            if i > 0:
                A[idx, (i - 1) * n + j] += w_n
            else:
                # Ghost up is (1, j)
                A[idx, 1 * n + j] += w_n
            
            # Down
            if i < n - 1:
                A[idx, (i + 1) * n + j] += w_n
            else:
                # Ghost down is (n-2, j)
                A[idx, (n - 2) * n + j] += w_n
    return A

def _reconstruct_A(f, N, deadline):
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        if time.time() > deadline:
            return None
        e_i = np.zeros(N, dtype=float)
        e_i[i] = 1.0
        col = f(0.0, e_i)
        A[:, i] = np.asarray(col, dtype=float).flatten()
    return A

def _general_solve(f, t0, t1, y0, deadline):
    # Fallback for N != 100 or structure failure
    # Use RK4 with reasonable steps
    # Or if N is small, reconstruct
    N = len(y0)
    if N <= 200 and time.time() < deadline - 10:
        A = _reconstruct_A(f, N, deadline)
        if A is not None:
            try:
                expA = expm((t1 - t0) * A)
                return expA @ y0
            except:
                pass
    return _fast_rk4(f, t0, t1, y0, 1000)

def _fast_rk4(f, t0, t1, y0, steps):
    """
    Fast RK4 fallback.
    """
    h = (t1 - t0) / steps
    t, y = t0, np.asarray(y0, dtype=float).flatten()
    
    for _ in range(steps):
        k1 = np.asarray(f(t, y), dtype=float).flatten()
        k2 = np.asarray(f(t + h/2, y + h/2 * k1), dtype=float).flatten()
        k3 = np.asarray(f(t + h/2, y + h/2 * k2), dtype=float).flatten()
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float).flatten()
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
    return y
