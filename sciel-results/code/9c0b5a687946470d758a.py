import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    best_result = None
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute the target tensor unfoldings
    T1 = np.zeros((n*m, m*p * p*n))
    T2 = np.zeros((m*p, n*m * p*n))
    T3 = np.zeros((p*n, n*m * m*p))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                T1[a, b * (p*n) + c] = 1.0
                T2[b, a * (p*n) + c] = 1.0
                T3[c, a * (m*p) + b] = 1.0
                
    T1_T = T1.T
    T2_T = T2.T
    T3_T = T3.T

    def khatri_rao(A, B):
        dim1, R = A.shape
        dim2, _ = B.shape
        prod = A.T[:, :, np.newaxis] * B.T[:, np.newaxis, :]
        return prod.reshape(R, -1).T

    def check_exact(U, V, W):
        MWV = khatri_rao(W, V)
        T1_recon = U @ MWV.T
        if not np.allclose(T1_recon, T1, atol=1e-9):
            return False
        MUW = khatri_rao(U, W)
        T2_recon = V @ MUW.T
        if not np.allclose(T2_recon, T2, atol=1e-9):
            return False
        MVU = khatri_rao(V, U)
        T3_recon = W @ MVU.T
        if not np.allclose(T3_recon, T3, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if check_exact(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    def local_search(U, V, W, max_steps=100):
        # Try to improve the snapped solution by local moves
        # This is a heuristic
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        def get_error(U, V, W):
            MWV = khatri_rao(W, V)
            T1_recon = U @ MWV.T
            err1 = np.linalg.norm(T1_recon - T1)
            MUW = khatri_rao(U, W)
            T2_recon = V @ MUW.T
            err2 = np.linalg.norm(T2_recon - T2)
            MVU = khatri_rao(V, U)
            T3_recon = W @ MVU.T
            err3 = np.linalg.norm(T3_recon - T3)
            return err1 + err2 + err3

        curr_err = get_error(U_s, V_s, W_s)
        if curr_err < 1e-9:
            return U_s, V_s, W_s, True

        for _ in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            # Try perturbing one element in U, V, or W
            for factor_name, factor in [('U', U_s), ('V', V_s), ('W', W_s)]:
                # Randomly pick an element
                idx = np.random.randint(0, factor.size)
                r, c = np.unravel_index(idx, factor.shape)
                orig_val = factor[r, c]
                # Try neighbors: orig_val +/- 0.5
                for delta in [-0.5, 0.5]:
                    new_val = orig_val + delta
                    # Ensure it's a half-integer (it will be if orig_val is)
                    if factor_name == 'U':
                        U_new = U_s.copy()
                        U_new[r, c] = new_val
                        err = get_error(U_new, V_s, W_s)
                    elif factor_name == 'V':
                        V_new = V_s.copy()
                        V_new[r, c] = new_val
                        err = get_error(U_s, V_new, W_s)
                    else:
                        W_new = W_s.copy()
                        W_new[r, c] = new_val
                        err = get_error(U_s, V_s, W_new)
                    
                    if err < curr_err - 1e-12:
                        if factor_name == 'U':
                            U_s = U_new
                        elif factor_name == 'V':
                            V_s = V_new
                        else:
                            W_s = W_new
                        curr_err = err
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
                
        if curr_err < 1e-9:
            return U_s, V_s, W_s, True
        return None

    def solve_als(U, V, W, max_iter=100):
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            MWV = khatri_rao(W, V)
            U = np.linalg.lstsq(MWV, T1_T, rcond=None)[0].T
            
            MUW = khatri_rao(U, W)
            V = np.linalg.lstsq(MUW, T2_T, rcond=None)[0].T
            
            MVU = khatri_rao(V, U)
            W = np.linalg.lstsq(MVU, T3_T, rcond=None)[0].T
            
            MWV = khatri_rao(W, V)
            T1_recon = U @ MWV.T
            err = np.linalg.norm(T1_recon - T1)
            if err < 1e-4:
                res = snap_and_verify(U, V, W)
                if res:
                    return res, True
                # Try local search if snap failed but error is small
                ls_res = local_search(U, V, W)
                if ls_res:
                    return ls_res[:3], True
        return (U, V, W), False

    # Search for lower rank
    # Try ranks from 59 down to 50
    # If not found, fall back to schoolbook
    found_solution = None
    
    for R in range(59, 49, -1):
        if time.time() > deadline:
            break
            
        restarts = 50
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            sol, success = solve_als(U, V, W)
            if success:
                found_solution = sol
                found_rank = R
                break
        if found_solution:
            break

    if found_solution:
        U, V, W = found_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
