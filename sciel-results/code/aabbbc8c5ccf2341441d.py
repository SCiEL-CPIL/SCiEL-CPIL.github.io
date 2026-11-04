import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1

    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    best_rank_found = schoolbook_rank

    def check_exact(U, V, W):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_approx, atol=1e-10)

    def try_integer_rounding(U, V, W):
        # Try rounding to nearest integer
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if check_exact(U_int, V_int, W_int):
            return U_int, V_int, W_int

        # Try rounding to nearest half-integer
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        if check_exact(U_half, V_half, W_half):
            return U_half, V_half, W_half

        # If not exact, check error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T - T_approx)
        
        if error < 0.1:
            # Local search around the rounded solution
            U_cur = U_int.copy()
            V_cur = V_int.copy()
            W_cur = W_int.copy()
            
            # Check current
            if check_exact(U_cur, V_cur, W_cur):
                return U_cur, V_cur, W_cur

            # Simple coordinate descent on integer variables
            # Limit steps to avoid timeout
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                # Pick a random matrix and element
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = U_cur
                elif mat_idx == 1:
                    mat = V_cur
                else:
                    mat = W_cur
                    
                idx = np.random.randint(mat.size)
                row, col = np.unravel_index(idx, mat.shape)
                
                orig_val = mat[row, col]
                # Try changing by -1, 1
                for delta in [-1, 1]:
                    mat[row, col] = orig_val + delta
                    if check_exact(U_cur, V_cur, W_cur):
                        return U_cur, V_cur, W_cur
                    mat[row, col] = orig_val

        return None

    # Lower bound is max(n*m, m*p, p*n) = 15
    lower_bound = max(n*m, m*p, p*n)

    for target_rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 15:
            break

        max_restarts = 30 if target_rank < 30 else 15

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5

            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 8:
                    break
                
                # Update U
                for a in range(n*m):
                    # A_a is (m*p*p*n) x target_rank
                    # A_a[b*p*n + c, r] = V[b, r] * W[c, r]
                    A_a = (V[:, :, None] * W[None, :, :]).reshape(m*p*p*n, target_rank)
                    t_a = T[a].flatten()
                    u_sol, residuals, rank, sv = np.linalg.lstsq(A_a, t_a, rcond=None)
                    U[a] = u_sol

                # Update V
                for b in range(m*p):
                    # A_b is (n*m*p*n) x target_rank
                    # A_b[a*p*n + c, r] = U[a, r] * W[c, r]
                    A_b = (U[:, :, None] * W[None, :, :]).reshape(n*m*p*n, target_rank)
                    t_b = T[:, b, :].flatten()
                    v_sol, residuals, rank, sv = np.linalg.lstsq(A_b, t_b, rcond=None)
                    V[b] = v_sol

                # Update W
                for c in range(p*n):
                    # A_c is (n*m*m*p) x target_rank
                    # A_c[a*m*p + b, r] = U[a, r] * V[b, r]
                    A_c = (U[:, :, None] * V[None, :, :]).reshape(n*m*m*p, target_rank)
                    t_c = T[:, :, c].flatten()
                    w_sol, residuals, rank, sv = np.linalg.lstsq(A_c, t_c, rcond=None)
                    W[c] = w_sol

            # Check convergence / rounding
            rounded = try_integer_rounding(U, V, W)
            if rounded is not None:
                return {"U": rounded[0].tolist(), "V": rounded[1].tolist(), "W": rounded[2].tolist()}

    return best_result
