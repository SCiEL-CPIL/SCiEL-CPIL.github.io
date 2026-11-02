# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 138  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    N, M, P = n * m, m * p, p * n  # 9, 12, 12
    
    # Build the target tensor T[N, M, P]
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Precompute flattened targets for vectorized ALS
    # For solving W: RHS is T reshaped to (N*M, P)
    T_flat_W = T.reshape(N * M, P)
    # For solving U: RHS is T reshaped to (M*P, N)
    # T[a, b, c] -> index in (M*P, N) is (b*P+c, a)
    T_flat_U = T.transpose(1, 2, 0).reshape(M * P, N)
    # For solving V: RHS is T reshaped to (N*P, M)
    # T[a, b, c] -> index in (N*P, M) is (a*P+c, b)
    T_flat_V = T.transpose(0, 2, 1).reshape(N * P, M)

    def verify(U, V, W, tol=1e-9):
        """Verify that U@V@W reconstructs T exactly."""
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)
    
    def error(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sqrt(np.sum((T_rec - T) ** 2))

    def solve_W(U, V, r):
        """Vectorized solve for W."""
        # A shape (N*M, r), A[(a,b), r] = U[a,r] * V[b,r]
        # We can construct A efficiently using outer products or broadcasting
        # A = (U[:, :, None] * V[None, :, :]).reshape(N*M, r)
        A = (U[:, :, None] * V[None, :, :]).reshape(N * M, r)
        # Solve A @ X = T_flat_W, where X is (r, P)
        X, _, _, _ = np.linalg.lstsq(A, T_flat_W, rcond=1e-12)
        return X.T  # W is (P, r)
    
    def solve_U(V, W, r):
        """Vectorized solve for U."""
        # A shape (M*P, r), A[(b,c), r] = V[b,r] * W[c,r]
        A = (V[:, :, None] * W[None, :, :]).reshape(M * P, r)
        # Solve A @ X = T_flat_U, where X is (r, N)
        X, _, _, _ = np.linalg.lstsq(A, T_flat_U, rcond=1e-12)
        return X.T  # U is (N, r)
    
    def solve_V(U, W, r):
        """Vectorized solve for V."""
        # A shape (N*P, r), A[(a,c), r] = U[a,r] * W[c,r]
        A = (U[:, :, None] * W[None, :, :]).reshape(N * P, r)
        # Solve A @ X = T_flat_V, where X is (r, M)
        X, _, _, _ = np.linalg.lstsq(A, T_flat_V, rcond=1e-12)
        return X.T  # V is (M, r)

    # Fallback: schoolbook rank-36
    rank_school = n * m * p
    U_sch = np.zeros((N, rank_school))
    V_sch = np.zeros((M, rank_school))
    W_sch = np.zeros((P, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j, col] = 1
                V_sch[j * p + k, col] = 1
                W_sch[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    
    # Try ranks from 18 upward
    # We can afford more trials now due to speedup
    for target_rank in range(18, 36):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Increase trials for lower ranks
        max_trials = 20 if target_rank <= 24 else 10
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 7919 + target_rank * 31)
            
            # Random initialization
            U = np.random.randn(N, target_rank) * 0.5
            V = np.random.randn(M, target_rank) * 0.5
            W = np.random.randn(P, target_rank) * 0.5
            
            prev_err = float('inf')
            for it in range(100):
                if time.time() > deadline - 2:
                    break
                
                W = solve_W(U, V, target_rank)
                U = solve_U(V, W, target_rank)
                V = solve_V(U, W, target_rank)
                
                if it % 5 == 0:
                    err = error(U, V, W)
                    if err < 1e-10:
                        break
                    if abs(prev_err - err) < 1e-12:
                        break
                    prev_err = err
            
            err = error(U, V, W)
            if err < 1e-8:
                # Try to snap to half-integers
                def snap(arr, step=0.5, tol=1e-6):
                    s = np.round(arr / step) * step
                    if np.allclose(arr, s, atol=tol):
                        return s
                    return None
                
                Us, Vs, Ws = snap(U), snap(V), snap(W)
                if Us is not None and Vs is not None and Ws is not None:
                    if verify(Us, Vs, Ws, tol=1e-8):
                        best_result = {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                        best_rank = target_rank
                        found = True
                        break
                else:
                    # If snap fails, check if the float solution is good enough
                    if verify(U, V, W, tol=1e-7):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            break
    
    # Final check
    U_final = np.array(best_result["U"], dtype=float)
    V_final = np.array(best_result["V"], dtype=float)
    W_final = np.array(best_result["W"], dtype=float)
    
    # Ensure correct shapes
    assert U_final.shape == (N, best_rank), f"U shape {U_final.shape} != ({N}, {best_rank})"
    assert V_final.shape == (M, best_rank), f"V shape {V_final.shape} != ({M}, {best_rank})"
    assert W_final.shape == (P, best_rank), f"W shape {W_final.shape} != ({P}, {best_rank})"
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
