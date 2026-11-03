import time
import numpy as np

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    trivial_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_rank = trivial_rank
    
    # Initialize with trivial decomposition
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
                
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    def snap_to_half_integer(mat):
        return np.round(mat * 2) / 2

    def _is_exact(U, V, W, n, m, p, tol=1e-8):
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Check all terms
        # Target: sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] == delta(j, j')
        # Note: The indices in the prompt's W are k*n+i.
        # The target tensor T_{(ij), (j'k), (ki)} = delta(j, j')
        
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        target = 1.0 if j == j2 else 0.0
                        val = 0.0
                        # Sum over r
                        # U[i*m+j, :] dot (V[j2*p+k, :] * W[k*n+i, :])
                        val = np.dot(U[i * m + j, :], V[j2 * p + k, :] * W[k * n + i, :])
                        if abs(val - target) > tol:
                            return False
        return True

    def _compute_error(U, V, W, n, m, p):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        target = 1.0 if j == j2 else 0.0
                        val = np.dot(U[i * m + j, :], V[j2 * p + k, :] * W[k * n + i, :])
                        err += (val - target) ** 2
        return err

    # Fast ALS Step
    def _als_step(U, V, W, n, m, p):
        nm = n * m
        mp = m * p
        pn = p * n
        R = U.shape[1]
        
        # Update U
        for i in range(n):
            for j in range(m):
                # Build matrix A for lstsq
                # Rows correspond to (j2, k)
                # Columns correspond to r
                A = np.zeros((m * p, R))
                b = np.zeros(m * p)
                idx = 0
                for j2 in range(m):
                    for k in range(p):
                        # Coefficients for U[i*m+j, r]
                        # Term is V[j2*p+k, r] * W[k*n+i, r]
                        A[idx, :] = V[j2 * p + k, :] * W[k * n + i, :]
                        b[idx] = 1.0 if j == j2 else 0.0
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = x
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n * m, R)) # Rows correspond to (i, j2) ?? No, (i, j2) is not right.
                # Target: sum_r U[i*m+j2, r] V[j*p+k, r] W[k*n+i, r] == delta(j2, j) ??
                # Wait, the target is delta(j, j') where j is from U and j' is from V.
                # In the update for V[j*p+k], the variable is V[j*p+k, r].
                # The term is U[i*m+j2, r] * V[j*p+k, r] * W[k*n+i, r].
                # The target is delta(j2, j) ?
                # Let's re-verify the index mapping.
                # T_{(ij), (j'k), (ki)} = delta(j, j').
                # Here j is the index in U, j' is the index in V.
                # So for fixed j (in V update), we sum over i, j2 (index in U).
                # The target is delta(j2, j).
                A = np.zeros((n * m, R))
                b = np.zeros(n * m)
                idx = 0
                for i in range(n):
                    for j2 in range(m):
                        A[idx, :] = U[i * m + j2, :] * W[k * n + i, :]
                        b[idx] = 1.0 if j2 == j else 0.0
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = x
                except:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                # Variable W[k*n+i, r]
                # Term: U[i*m+j2, r] * V[j*p+k, r] * W[k*n+i, r]
                # Target: delta(j2, j) ?
                # The sum is over j2 (from U) and j (from V).
                # Wait, the target delta(j, j') depends on the specific j and j'.
                # In the full sum, we sum over r.
                # For a fixed (i, k), the output is a matrix in j, j'.
                # The target matrix is I (identity).
                # So we need sum_r (U_i(j) V_k(j') W_ki) = delta(j, j').
                # This must hold for all j, j'.
                # So we have m*m equations for each (i, k).
                A = np.zeros((m * m, R))
                b = np.zeros(m * m)
                idx = 0
                for j2 in range(m):
                    for j in range(m):
                        A[idx, :] = U[i * m + j2, :] * V[j * p + k, :]
                        b[idx] = 1.0 if j2 == j else 0.0
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = x
                except:
                    pass

        return U, V, W

    # Search for lower rank
    for R in range(trivial_rank - 1, 20, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + R * 10 + trial)
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                U, V, W = _als_step(U, V, W, n, m, p)
                
                err = _compute_error(U, V, W, n, m, p)
                if err < 1e-10:
                    # Snap and verify
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    
                    if _is_exact(U_snap, V_snap, W_snap, n, m, p):
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        best_rank = R
                        found = True
                        break
                elif err > 1e6:
                    break
            
            if found:
                break
        
        if found:
            continue # Try lower rank
        else:
            if time.time() > deadline - 5:
                break

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
