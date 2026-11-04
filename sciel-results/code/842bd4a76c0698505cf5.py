import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    timeout = 120.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        # Check if U, V, W form a valid decomposition
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # for all i, j, k
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for rr in range(r):
                        val += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank_optimized(r, time_budget=10.0):
        """Try to find rank-r decomposition using ALS"""
        start = time.time()
        
        for seed in range(20):
            if time.time() - start > time_budget:
                break
            np.random.seed(1000 + seed)
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize to prevent divergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # ALS iterations
            for step in range(100):
                # Solve W: for each (k,i), minimize ||A @ W_col - 1||^2
                for k in range(p):
                    for i in range(n):
                        A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        try:
                            W_col, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = W_col
                        except:
                            return None
                            
                # Solve U: for each (i,j), minimize ||A @ U_col - 1||^2
                for i in range(n):
                    for j in range(m):
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        try:
                            U_col, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = U_col
                        except:
                            return None
                            
                # Solve V: for each (j,k), minimize ||A @ V_col - 1||^2
                for j in range(m):
                    for k in range(p):
                        A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        try:
                            V_col, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = V_col
                        except:
                            return None
                            
                # Check loss
                loss = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            loss += (val - 1.0)**2
                            
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if loss < 1e-10:
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                    # Also try integer rounding
                    U_r2 = np.round(U)
                    V_r2 = np.round(V)
                    W_r2 = np.round(W)
                    if verify(U_r2.tolist(), V_r2.tolist(), W_r2.tolist()):
                        return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}
                    
                    # Try scaling
                    for scale in [0.5, 1.0, 2.0]:
                        U_r3 = np.round(U * scale * 2) / 2
                        V_r3 = np.round(V * scale * 2) / 2
                        W_r3 = np.round(W * (1/scale) * 2) / 2 if scale != 0 else W
                        if verify(U_r3.tolist(), V_r3.tolist(), W_r3.tolist()):
                            return {"U": U_r3.tolist(), "V": V_r3.tolist(), "W": W_r3.tolist()}

            # After all seeds, check if best solution is close enough
            if best_loss < 1e-8:
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return None
    
    # Strategy: Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But actual rank is likely higher
    
    # First, try to exploit symmetry: map 3x3x8 to equivalent problems
    # M(n,m,p) tensor T_{ijk} where i in [0,n), j in [0,m), k in [0,p)
    # Permutation: swap i and k -> M(p,m,n) with T'_{kji} = T_{ijk}
    # This means rank(M(n,m,p)) = rank(M(p,m,n))
    
    # For 3x3x8, we can try M(8,3,3) which is 8x3x3
    # The rank of 8x3x3 should be related to rank of 3x8x3
    
    # Let's try ranks starting from theoretical minimum
    # Lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    
    # Try a range of ranks
    ranks_to_try = [24, 25, 26, 27, 28, 29, 30, 32, 36, 40, 48, 54, 60, 66, 72]
    
    for r in ranks_to_try:
        remaining_time = timeout - (time.time() - start_time) - 5.0  # 5s safety margin
        if remaining_time < 2.0:
            break
        
        # Allocate time based on rank (lower ranks get more attempts)
        time_budget = min(15.0, remaining_time / max(1, len(ranks_to_try) - ranks_to_try.index(r)))
        
        if r == 24 or r == 25 or r == 26:
            time_budget = min(20.0, remaining_time)
        
        res = try_rank_optimized(r, time_budget)
        if res:
            return res
    
    # Fall back to schoolbook
    return schoolbook()
