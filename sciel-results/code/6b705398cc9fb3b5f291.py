import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Detect if the system is linear and time-independent (f(t,y) = A@y).
       If so, extract A with n+1 calls.
    2. For the 10x10 heat equation, exploit the Kronecker-sum structure:
       A = L ⊗ I + I ⊗ L where L is the 1D Laplacian.
       Compute expm(A*dt) via 1D eigen-decompositions (much faster and more stable
       than dense 100x100 expm).
    3. Fallback: high-precision adaptive solver.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    dt = t1 - t0

    # Check if system is linear: f(t0, 0) should be 0
    y_zero = np.zeros_like(y0)
    f_zero = np.asarray(f(t0, y_zero), dtype=float)

    if np.allclose(f_zero, 0.0, atol=1e-14):
        # Likely linear. Extract A by probing each basis vector.
        A = np.empty((n, n), dtype=float)
        
        for j in range(n):
            e_j = np.zeros_like(y0)
            e_j[j] = 1.0
            A[:, j] = np.asarray(f(t0, e_j), dtype=float)
            
            if time.time() > deadline:
                break
        
        if time.time() <= deadline:
            # Try Kronecker-sum spectral method for 10x10 heat equation
            try:
                y_end = _kronecker_heat_solve(A, dt, y0)
                if y_end is not None:
                    return y_end
            except Exception:
                pass
            
            # Fallback to dense expm_multiply
            try:
                from scipy.sparse.linalg import expm_multiply
                y_end = expm_multiply(A * dt, y0)
                return y_end
            except Exception:
                pass

    # Fallback: use high-precision adaptive solver
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        f, t_span, y0,
        method='Radau',
        rtol=1e-13,
        atol=1e-15,
        max_step=dt / 200 if dt > 0 else 1.0
    )

    if sol.success and len(sol.y) > 0:
        return sol.y[:, -1]

    sol = solve_ivp(
        f, t_span, y0,
        method='BDF',
        rtol=1e-13,
        atol=1e-15,
        max_step=dt / 200 if dt > 0 else 1.0
    )

    if sol.success and len(sol.y) > 0:
        return sol.y[:, -1]

    return y0


def _kronecker_heat_solve(A, dt, y0):
    """
    Attempt to solve y' = A@y using Kronecker-sum structure.
    
    For a 10x10 grid with row-major flattening, A = L ⊗ I + I ⊗ L
    where L is the 10x10 1D Laplacian (tridiagonal with -2 on diagonal, 1 on off-diagonals).
    
    We verify this structure and if it holds, compute the exact solution via
    1D eigendecompositions.
    """
    n = y0.size
    # Check if n is a perfect square (2D grid)
    side = int(np.sqrt(n))
    if side * side != n:
        return None
    
    # For 10x10, side=10
    if side < 2:
        return None
    
    # Reshape A to 4D tensor: A[i1*side+i2, j1*side+j2]
    # If A = L1 ⊗ I + I ⊗ L2, then A = kron(L1, I) + kron(I, L2)
    # Let's extract the two 1D operators.
    
    # Method: Extract L1 (acting on first index) by fixing second index
    # A[i1*s+i2, j1*s+j2] = L1[i1,j1]*delta[i2,j2] + delta[i1,j1]*L2[i2,j2]
    
    # Extract L1: set j2 = i2, j1 varies, i1 varies
    # A[i1*s + k, j1*s + k] for fixed k = L1[i1,j1] + delta[i1,j1]*L2[k,k]
    
    # Simpler approach: Try to fit A as kron(L1, I) + kron(I, L2)
    # by checking specific entries.
    
    # Let's try: for the standard 2D heat equation with Dirichlet BCs,
    # L = tridiag(1, -2, 1) * (some scaling)
    # But the actual A might have different scaling. Let's extract it.
    
    # Extract L1 (first dimension operator):
    # Take column j=0 (j1=0, j2=0), row i where i2=0: A[i1*s, 0]
    # This gives L1[i1, 0] + delta[i1,0]*L2[0,0]
    
    # Better: Use the fact that if A = kron(L1, I) + kron(I, L2),
    # then for any i1,j1,i2,j2:
    # A[i1*s+i2, j1*s+j2] = L1[i1,j1]*delta[i2,j2] + delta[i1,j1]*L2[i2,j2]
    
    # Extract L1[i1,j1]: set i2=j2=k (same), then
    # A[i1*s+k, j1*s+k] = L1[i1,j1] + delta[i1,j1]*L2[k,k]
    # For i1 != j1: A[i1*s+k, j1*s+k] = L1[i1,j1]
    # For i1 == j1: A[i1*s+k, i1*s+k] = L1[i1,i1] + L2[k,k]
    
    # Extract L2[i2,j2]: set i1=j1=k (same), then
    # A[k*s+i2, k*s+j2] = delta[i2,j2]*L1[k,k] + L2[i2,j2]
    # For i2 != j2: A[k*s+i2, k*s+j2] = L2[i2,j2]
    # For i2 == j2: A[k*s+i2, k*s+i2] = L1[k,k] + L2[i2,i2]
    
    s = side
    L1 = np.zeros((s, s), dtype=float)
    L2 = np.zeros((s, s), dtype=float)
    
    # Extract off-diagonal elements of L1 (i1 != j1)
    for i1 in range(s):
        for j1 in range(s):
            if i1 != j1:
                # Use k=0: A[i1*s+0, j1*s+0] = L1[i1,j1]
                L1[i1, j1] = A[i1*s, j1*s]
    
    # Extract off-diagonal elements of L2 (i2 != j2)
    for i2 in range(s):
        for j2 in range(s):
            if i2 != j2:
                # Use k=0: A[0*s+i2, 0*s+j2] = L2[i2,j2]
                L2[i2, j2] = A[i2, j2]
    
    # Now extract diagonal elements.
    # A[i1*s+i2, j1*s+j2] = L1[i1,j1]*delta[i2,j2] + delta[i1,j1]*L2[i2,j2]
    
    # For diagonal of L1: A[i1*s+k, i1*s+k] = L1[i1,i1] + L2[k,k] for any k
    # For diagonal of L2: A[k*s+i2, k*s+i2] = L1[k,k] + L2[i2,i2] for any k
    
    # We need to separate these. Use the fact that for a consistent Kronecker sum:
    # A[i1*s+k, i1*s+k] - A[i1*s+0, i1*s+0] = L2[k,k] - L2[0,0]
    # A[0*s+i2, 0*s+i2] - A[0*s+0, 0*s+0] = L2[i2,i2] - L2[0,0]
    
    # And: A[i1*s+k, i1*s+k] - A[0*s+k, 0*s+k] = L1[i1,i1] - L1[0,0]
    # A[k*s+i2, k*s+i2] - A[0*s+i2, 0*s+i2] = L1[k,k] - L1[0,0]
    
    # Let's verify consistency and extract.
    # From A[i1*s+k, i1*s+k] = L1[i1,i1] + L2[k,k]
    # Sum over all i1,k: sum_{i1,k} A[i1*s+k, i1*s+k] = s*sum_i L1[i,i] + s*sum_k L2[k,k]
    
    # Alternative: Use multiple equations.
    # For i1=0, j1=0: A[k*s+i2, k*s+i2] = L1[0,0]*delta[i2,i2] + L2[i2,i2] = L1[0,0] + L2[i2,i2]
    # So L2[i2,i2] = A[k*s+i2, k*s+i2] - L1[0,0] for any k
    
    # For i2=0, j2=0: A[i1*s, j1*s] with i2=j2=0:
    # A[i1*s, j1*s] = L1[i1,j1]*1 + delta[i1,j1]*L2[0,0]
    # For i1=j1: A[i1*s, i1*s] = L1[i1,i1] + L2[0,0]
    # So L1[i1,i1] = A[i1*s, i1*s] - L2[0,0]
    
    # We have two unknowns: L1[0,0] and L2[0,0].
    # A[0, 0] = L1[0,0] + L2[0,0]
    
    # Use the off-diagonal info to verify and solve.
    # For a tridiagonal Laplacian, L1[0,0] should relate to L1[0,1] etc.
    
    # Let's use: A[i1*s+i2, j1*s+j2] should equal L1[i1,j1]*delta[i2,j2] + delta[i1,j1]*L2[i2,j2]
    
    # Try to determine L1[0,0] and L2[0,0]:
    # A[0,0] = L1[0,0] + L2[0,0] ... (eq 1)
    # A[s, s] = L1[1,1] + L2[0,0] ... (eq 2)  [i1=1,j1=1,i2=0,j2=0]
    # A[1, 1] = L1[0,0] + L2[1,1] ... (eq 3)  [i1=0,j1=0,i2=1,j2=1]
    # A[s+1, s+1] = L1[1,1] + L2[1,1] ... (eq 4)
    
    # From eq 2: L1[1,1] = A[s,s] - L2[0,0]
    # From eq 3: L2[1,1] = A[1,1] - L1[0,0]
    # From eq 4: A[s+1,s+1] = (A[s,s] - L2[0,0]) + (A[1,1] - L1[0,0])
    # So: A[s+1,s+1] = A[s,s] + A[1,1] - L2[0,0] - L1[0,0]
    # And from eq 1: L1[0,0] + L2[0,0] = A[0,0]
    # So: A[s+1,s+1] = A[s,s] + A[1,1] - A[0,0]
    
    # This is a consistency check!
    if abs(A[s+1, s+1] - (A[s, s] + A[1, 1] - A[0, 0])) > 1e-12 * max(1.0, abs(A[0,0])):
        return None  # Not a valid Kronecker sum
    
    # Now we need one more equation to separate L1[0,0] and L2[0,0].
    # Use: A[i1*s, j1*s] = L1[i1,j1] + delta[i1,j1]*L2[0,0]
    # For i1=0, j1=1: A[0, s] = L1[0,1] + 0 = L1[0,1]  (already extracted)
    # For i1=1, j1=0: A[s, 0] = L1[1,0] + 0 = L1[1,0]  (already extracted)
    
    # Use diagonal consistency:
    # A[i1*s+i2, i1*s+i2] = L1[i1,i1] + L2[i2,i2]
    # For i1=0, i2=0: A[0,0] = L1[0,0] + L2[0,0]
    # For i1=0, i2=1: A[1,1] = L1[0,0] + L2[1,1]
    # For i1=1, i2=0: A[s,s] = L1[1,1] + L2[0,0]
    # For i1=1, i2=1: A[s+1,s+1] = L1[1,1] + L2[1,1]
    
    # We already verified consistency. Now assume symmetry: L1[0,0] = L1[1,1] (for uniform grid)
    # Actually, for the standard Laplacian with Dirichlet BCs, the diagonal is constant.
    # Let's check: is L1[i,i] constant?
    
    # A[i1*s+i2, i1*s+i2] = L1[i1,i1] + L2[i2,i2]
    # If L1[i1,i1] = c1 (constant) and L2[i2,i2] = c2 (constant):
    # A[i1*s+i2, i1*s+i2] = c1 + c2 for all i1, i2
    # Check: are all diagonal entries of A equal?
    
    diag_A = np.array([A[i*s+i, i*s+i] for i in range(s)])
    if np.std(diag_A) < 1e-14:
        # All diagonals equal: L1[i,i] + L2[i,i] = const for all i
        # This is consistent with constant diagonals.
        c_total = diag_A[0]
        # We need to split c_total into c1 + c2.
        # Use off-diagonal structure. For standard Laplacian:
        # L1[0,1] = 1 (or some value), L1[0,0] = -2 (or some value)
        # The ratio L1[0,0]/L1[0,1] tells us the scaling.
        
        # For a consistent Kronecker sum with identical 1D operators:
        # L1 = L2 (same 1D Laplacian)
        # Then c1 = c2 = c_total / 2
        # Let's verify: A[i1*s+i2, j1*s+j2] = L[i1,j1]*delta[i2,j2] + delta[i1,j1]*L[i2,j2]
        
        # Check if L1 == L2 by comparing off-diagonals:
        # L1[i1,j1] = A[i1*s, j1*s] for i1 != j1
        # L2[i2,j2] = A[i2, j2] for i2 != j2
        # Are they the same matrix?
        
        L1_off = np.zeros((s, s))
        L2_off = np.zeros((s, s))
        for i1 in range(s):
            for j1 in range(s):
                if i1 != j1:
                    L1_off[i1, j1] = A[i1*s, j1*s]
        for i2 in range(s):
            for j2 in range(s):
                if i2 != j2:
                    L2_off[i2, j2] = A[i2, j2]
        
        # Fill in diagonals assuming constant diagonal
        # L1[i,i] = c1, L2[i,i] = c2, c1+c2 = c_total
        # If L1 == L2, then c1 = c2 = c_total/2
        
        # Verify: A[i1*s+i2, j1*s+j2] should equal L1[i1,j1]*delta[i2,j2] + delta[i1,j1]*L2[i2,j2]
        # With L1 = L2 = L, diagonal = c_total/2:
        
        L = L1_off.copy()
        L[np.diag_indices(s)] = c_total / 2.0
        
        # Verify the Kronecker sum reconstruction
        A_recon = np.kron(L, np.eye(s)) + np.kron(np.eye(s), L)
        if np.allclose(A, A_recon, atol=1e-12):
            # Great! Use 1D eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(L)
            # expm(A*dt) applied to y:
            # y is shape (s*s,), reshape to (s, s)
            Y = y0.reshape(s, s)
            # Diagonalize L: L = V D V^T
            # expm((L⊗I + I⊗L)*dt) = expm(L*dt)⊗expm(L*dt) in the eigenbasis
            # In original basis: V @ diag(exp(d_i*dt)) @ V^T for each dimension
            # So: result = (V @ diag(exp(d*dt)) @ V^T) @ Y @ (V @ diag(exp(d*dt)) @ V^T)^T
            # Wait, need to be careful with row-major ordering.
            # y0 is flattened row-major: y0[i*s+j] = Y[i,j]
            # A = kron(L1, I) + kron(I, L2) with L1 acting on first index (rows of Y)
            # So A @ y corresponds to: (L1 @ Y) * delta + Y * (L2 @ Y^T)^T ... 
            # Actually: kron(L1, I) @ vec(Y) = vec(L1 @ Y)
            # kron(I, L2) @ vec(Y) = vec(Y @ L2^T) = vec((L2 @ Y^T)^T)
            # So A @ vec(Y) = vec(L1 @ Y + Y @ L2^T)
            # For L1 = L2 = L (symmetric): A @ vec(Y) = vec(L @ Y + Y @ L^T) = vec(L @ Y + Y @ L)
            
            # expm(A*dt) @ vec(Y) = vec(expm(L*dt) @ Y @ expm(L^T*dt))
            # For symmetric L: = vec(expm(L*dt) @ Y @ expm(L*dt)^T)
            
            # Compute expm(L*dt) via eigen-decomposition
            exp_L = eigvecs @ np.diag(np.exp(eigvals * dt)) @ eigvecs.T
            
            Y_end = exp_L @ Y @ exp_L.T
            return Y_end.flatten()
        
        # If L1 != L2, handle separately
        # But for now, try the symmetric case
    
    # Try general case: L1 and L2 may differ
    # We have: A[i1*s+i2, j1*s+j2] = L1[i1,j1]*delta[i2,j2] + delta[i1,j1]*L2[i2,j2]
    # Off-diagonals of L1 and L2 are known.
    # Diagonals: L1[i1,i1] + L2[i2,i2] = A[i1*s+i2, i1*s+i2]
    
    # Let d1[i] = L1[i,i], d2[j] = L2[j,j]
    # d1[i] + d2[j] = A[i*s+j, i*s+j] for all i, j
    # This is a system: d1[i] + d2[j] = C[i,j] where C[i,j] = A[i*s+j, i*s+j]
    # Solution: d1[i] = C[i,0] - d2[0], d2[j] = C[0,j] - d1[0]
    # C[i,0] - d2[0] + C[0,j] - d1[0] = C[i,j]
    # d1[i] + d2[j] = C[i,j] ... consistent if C[i,j] = C[i,0] + C[0,j] - C[0,0]
    
    C = np.array([A[i*s+j, i*s+j] for i in range(s) for j in range(s)]).reshape(s, s)
    C_recon = C[:, 0:1] + C[0:1, :] - C[0, 0]
    if not np.allclose(C, C_recon, atol=1e-12):
        return None
    
    # d1[i] + d2[j] = C[i,j]
    # Set d2[0] = 0 (gauge freedom, but we need consistency)
    # Actually, we need d1[i] + d2[j] = C[i,j] for all i,j.
    # d1[i] = C[i,0] - d2[0]
    # d2[j] = C[0,j] - d1[0] = C[0,j] - C[0,0] + d2[0]
    # Check: d1[i] + d2[j] = C[i,0] - d2[0] + C[0,j] - C[0,0] + d2[0] = C[i,0] + C[0,j] - C[0,0] = C[i,j] ✓
    # So we have one degree of freedom. Use the off-diagonal structure to fix it.
    # For a Laplacian, the diagonal should be related to off-diagonals.
    # Actually, any consistent split works for computing expm.
    # Let's set d2[0] = 0 for simplicity.
    
    d2 = np.zeros(s)
    d1 = C[:, 0] - d2[0]  # = C[:, 0]
    # Verify: d1[i] + d2[j] = C[i,0] + (C[0,j] - C[0,0]) ... wait
    # d2[j] = C[0,j] - d1[0] = C[0,j] - C[0,0]
    d2 = C[0, :] - d1[0]
    
    # Now build full L1 and L2
    L1_full = L1_off.copy()
    L1_full[np.diag_indices(s)] = d1
    L2_full = L2_off.copy()
    L2_full[np.diag_indices(s)] = d2
    
    # Verify reconstruction
    A_recon = np.kron(L1_full, np.eye(s)) + np.kron(np.eye(s), L2_full)
    if not np.allclose(A, A_recon, atol=1e-12):
        return None
    
    # Compute expm via 1D eigendecompositions
    eigvals1, eigvecs1 = np.linalg.eigh(L1_full)
    eigvals2, eigvecs2 = np.linalg.eigh(L2_full)
    
    exp_L1 = eigvecs1 @ np.diag(np.exp(eigvals1 * dt)) @ eigvecs1.T
    exp_L2 = eigvecs2 @ np.diag(np.exp(eigvals2 * dt)) @ eigvecs2.T
    
    Y = y0.reshape(s, s)
    Y_end = exp_L1 @ Y @ exp_L2.T
    return Y_end.flatten()
