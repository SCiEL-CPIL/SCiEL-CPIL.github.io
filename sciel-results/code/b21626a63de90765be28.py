import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 140.0

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return U, V, W, rank

    # Rank-44 construction:
    # Split 3x3x5 into:
    #   3x3x3 part (k=0,1,2): Use Strassen on (i,j) in {0,1}x{0,1} for k in {0,1} (rank 7),
    #     plus schoolbook for remaining (i,j) pairs for k in {0,1,2}
    #   3x3x2 part (k=3,4): Schoolbook (rank 18)
    #
    # For 3x3x3:
    #   Strassen on i in {0,1}, j in {0,1}, k in {0,1}: 7 terms
    #   Schoolbook for (i=0 or 1, j=2) x k in {0,1}: 4 terms
    #   Schoolbook for i in {0,1}, j in {0,1}, k=2: 4 terms
    #   Schoolbook for i=2, j in {0,1,2}, k in {0,1,2}: 9 terms
    #   Total 3x3x3: 7 + 4 + 4 + 9 = 24
    #   Wait, that's 24 not 23. Let me recount.
    #
    # Actually Strassen for 2x2x2 uses rank 7. The remaining terms:
    #   (i=0,1) x (j=2) x (k=0,1): 4 terms
    #   (i=0,1) x (j=0,1) x (k=2): 4 terms
    #   (i=2) x (j=0,1,2) x (k=0,1,2): 9 terms
    #   Total: 7 + 4 + 4 + 9 = 24
    #
    # Plus 3x3x2 schoolbook: 18
    # Total: 24 + 18 = 42
    #
    # Hmm, that gives 42 not 44. Let me be more careful.
    #
    # Strassen 2x2x2: computes for i in {0,1}, j in {0,1}, k in {0,1}
    # That covers 4 of the 9 (i,j) pairs in the 3x3x3 block for k in {0,1}.
    #
    # Remaining for 3x3x3:
    #   j=2, i in {0,1}, k in {0,1}: 4 terms (schoolbook)
    #   j=2, i=2, k in {0,1}: 2 terms (schoolbook)
    #   j in {0,1,2}, i=2, k=2: 3 terms (schoolbook)
    #   j in {0,1}, i in {0,1}, k=2: 4 terms (schoolbook)
    #   j=2, i=2, k=2: 1 term (schoolbook)
    #   Total remaining: 4 + 2 + 3 + 4 + 1 = 14
    #
    # Total 3x3x3: 7 + 14 = 21? No that's too low.
    #
    # Let me just enumerate:
    # All (i,j,k) for 3x3x3: 27 terms
    # Strassen covers: i in {0,1}, j in {0,1}, k in {0,1}: 4 output entries, rank 7
    # The other 27 - 4 = 23 output entries need individual terms? No, Strassen handles those 4 with 7 terms.
    # Remaining 23 output entries: schoolbook = 23 terms
    # Total 3x3x3: 7 + 23 = 30. That's worse than 27.
    #
    # OK so Strassen on 2x2x2 within 3x3x3 doesn't help because the other 23 terms still need rank 23.
    #
    # Let me try a different approach. Use the fact that for the 3x3x2 part (k=3,4),
    # we can share structure with the 3x3x3 part.
    #
    # Actually, let me just use a cleaner construction:
    # Total rank 44 = 27 (3x3x3 schoolbook) + 17 (3x3x2 with 1 shared term)
    #
    # For 3x3x2: schoolbook is 18. Can we save 1?
    # C[i,3] = sum_j A[i,j]*B[j,3], C[i,4] = sum_j A[i,j]*B[j,4]
    # If we make B[j,3] and B[j,4] related... but they're arbitrary.
    #
    # Let me just go with 44 = some verified construction.
    #
    # Simplest: Use rank 44 by starting with schoolbook 45 and merging one pair.
    #
    # Actually, let me try the numerical search approach with a better initialization.
    
    from scipy.optimize import least_squares
    
    def residuals(params, R):
        U_flat = params[:9 * R].reshape(9, R)
        V_flat = params[9 * R:9 * R + 15 * R].reshape(15, R)
        W_flat = params[9 * R + 15 * R:].reshape(15, R)
        
        res = np.zeros(45)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    u = U_flat[i * 3 + j]
                    v = V_flat[j * 5 + k]
                    w = W_flat[k * 3 + i]
                    res[idx] = np.dot(u, v * w) - 1.0
                    idx += 1
        return res

    best_U, best_V, best_W, best_rank = None, None, None, 45

    # Try rank 44 with multiple restarts
    for target_rank in [44]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for restart in range(10):
            if time.time() > deadline - 30:
                break
            
            rng = np.random.default_rng(42 + restart * 7)
            U_init = rng.normal(size=(9, target_rank)) * 0.5
            V_init = rng.normal(size=(15, target_rank)) * 0.5
            W_init = rng.normal(size=(15, target_rank)) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(
                    lambda p: residuals(p, target_rank),
                    x0,
                    method='lm',
                    x_scale='jac',
                    max_nfev=20000
                )
                
                if res.cost < 1e-12:
                    U_sol = res.x[:9 * target_rank].reshape(9, target_rank)
                    V_sol = res.x[9 * target_rank:9 * target_rank + 15 * target_rank].reshape(15, target_rank)
                    W_sol = res.x[9 * target_rank + 15 * target_rank:].reshape(15, target_rank)
                    
                    # Try rounding to integers first
                    U_r = np.round(U_sol).astype(float)
                    V_r = np.round(V_sol).astype(float)
                    W_r = np.round(W_sol).astype(float)
                    
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    if verify_decomposition(U_list, V_list, W_list, 3, 3, 5):
                        best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                        found = True
                        break
                    
                    # Try rounding to half-integers
                    U_r = np.round(U_sol * 2) / 2.0
                    V_r = np.round(V_sol * 2) / 2.0
                    W_r = np.round(W_sol * 2) / 2.0
                    
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    if verify_decomposition(U_list, V_list, W_list, 3, 3, 5):
                        best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            break

    if best_U is None:
        # Fallback: try a structured rank-44 construction
        # Use schoolbook for all except try to merge one term
        # Actually just use schoolbook 45 as fallback
        best_U, best_V, best_W, best_rank = make_schoolbook(3, 3, 5)
    
    return {"U": best_U, "V": best_V, "W": best_W}
