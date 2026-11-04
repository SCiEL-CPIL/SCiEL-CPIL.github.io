"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(result, atol=1e-8):
        U = np.array(result["U"], dtype=np.float64)
        V = np.array(result["V"], dtype=np.float64)
        W = np.array(result["W"], dtype=np.float64)
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        return np.allclose(T, target, atol=atol)

    # Build target tensor once
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    def build_system_for_W(U, V, R):
        """Build linear system A * W_flat = b for solving W given U, V.
        Variables: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,R)
        Equations: for each (i, j, jp, k): sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+i, r] = delta(j,jp)
        """
        num_eqs = n * m * m * p
        num_vars = P * R  # P = p*n
        
        A = np.zeros((num_eqs, num_vars))
        b = np.zeros(num_eqs)
        
        # Build b: 1 when j == jp
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * m * p + j * p + k
                    b[eq_idx] = 1.0
        
        # Build A column by column for each (k, i, r) variable
        for r in range(R):
            u_col = U[:, r]  # length N = n*m
            v_col = V[:, r]  # length M = m*p
            
            for k in range(p):
                for i in range(n):
                    var_idx = (k * n + i) * R + r
                    
                    # Coefficient for equation (i, j, jp, k) is U[i*m+j, r] * V[jp*p+k, r]
                    u_vals = u_col[i*m : (i+1)*m]  # length m, index j
                    v_vals = v_col[k::p]  # length m, index jp: v_col[jp*p+k]
                    
                    for j in range(m):
                        for jp in range(m):
                            eq_idx = (i * m + j) * m * p + jp * p + k
                            A[eq_idx, var_idx] = u_vals[j] * v_vals[jp]
        
        return A, b

    def solve_for_W(U, V, R, method='lstsq'):
        """Solve for W given U and V."""
        A, b = build_system_for_W(U, V, R)
        try:
            if method == 'lstsq':
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            else:
                # Try using pseudo-inverse for potentially better conditioning
                W_flat = np.linalg.pinv(A) @ b
            W = W_flat.reshape(P, R)
            return W
        except:
            return None

    def als_decompose(target, rank, max_iter=30, tol=1e-10):
        """ALS decomposition with structured W-solve and polish."""
        # Initialize with small random values, but use schoolbook-like structure
        U = np.random.randn(N, rank) * 0.3
        V = np.random.randn(M, rank) * 0.3
        W = np.random.randn(P, rank) * 0.3
        
        prev_err = np.inf
        for iteration in range(max_iter):
            if time_remaining() < 20:
                break
            
            # Update W: fix U, V - use structured linear solve
            W_new = solve_for_W(U, V, rank)
            if W_new is not None:
                W = W_new
            
            # Update U: fix V, W
            H_U = np.einsum('br,cr->bcr', V, W).reshape(M * P, rank)
            target_flat = target.reshape(N, M * P)
            try:
                U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Update V: fix U, W
            H_V = np.einsum('ar,cr->acr', U, W).reshape(N * P, rank)
            target_v = target.transpose(1, 0, 2).reshape(M, N * P)
            try:
                V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Check convergence
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - target)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        # Polish: alternate structured W-solve with lstsq for U, V
        if time_remaining() > 25:
            for _ in range(15):
                if time_remaining() < 20:
                    break
                W_new = solve_for_W(U, V, rank)
                if W_new is not None:
                    W = W_new
                
                H_U = np.einsum('br,cr->bcr', V, W).reshape(M * P, rank)
                target_flat = target.reshape(N, M * P)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(H_U, target_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                H_V = np.einsum('ar,cr->acr', U, W).reshape(N * P, rank)
                target_v = target.transpose(1, 0, 2).reshape(M, N * P)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(H_V, target_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
        
        return U, V, W

    def try_rank(r, num_trials=4):
        """Try to find a decomposition of rank r."""
        if time_remaining() < 30:
            return None
        for trial in range(num_trials):
            if time_remaining() < 25:
                break
            U, V, W = als_decompose(target, r, max_iter=35)
            result = {"U": U, "V": V, "W": W}
            if verify(result):
                return result
        return None

    # Try ranks below schoolbook, starting from 119 down
    # Focus on a narrower range first for speed
    for r in range(best_rank - 1, max(best_rank - 20, 100), -1):
        if time_remaining() < 40:
            break
        result = try_rank(r, num_trials=3)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            # Continue searching for even lower ranks
            continue

    # If we have time, try a few more lower ranks
    if time_remaining() > 50:
        for r in range(max(best_rank - 1, 100), max(best_rank - 15, 95), -1):
            if time_remaining() < 35:
                break
            result = try_rank(r, num_trials=2)
            if result is not None:
                best = result
                best_rank = r
                print(f"Found decomposition with rank {r}")

    # Final verification
    if not verify(best, atol=1e-7):
        best = schoolbook()
        best_rank = n * m * p

    return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}
# EVOLVE-BLOCK-END
