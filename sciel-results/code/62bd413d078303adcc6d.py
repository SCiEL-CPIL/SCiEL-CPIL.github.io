"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 140.0

def _check_time():
    return (time.time() - _START_TIME) < TIME_LIMIT

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build target tensor T[a, b, c] where:
    # a = i*m + j (A index), b = j'*p + k (B index), c = k'*n + i' (output index)
    # T[a, b, c] = 1 iff a=(i,j), b=(j,k), c=(k,i) for same i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank using iterative optimization
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try various target ranks - focus on ranks that might be achievable
    # For 3x3x3x7, the lower bound is at least max(n*p, m*p, n*m) = max(21,21,9) = 21
    # but realistically, for these dimensions, rank is likely close to schoolbook
    for target_rank in [45, 50, 55, 60]:
        if not _check_time():
            break
        
        found = False
        for attempt in range(2):
            if not _check_time():
                break
            
            np.random.seed(123 + target_rank * 10 + attempt)
            U = np.random.randn(nm, target_rank) * 0.3
            V = np.random.randn(mp, target_rank) * 0.3
            W = np.random.randn(pn, target_rank) * 0.3
            
            # ALS-style optimization with efficient updates
            for it in range(30):
                if not _check_time():
                    break
                
                # Update U: for each row a, solve linear system
                for a in range(nm):
                    # M_a[bc, r] = V[b, r] * W[c, r]
                    M_a = np.zeros((mp * pn, target_rank))
                    T_a = np.zeros(mp * pn)
                    idx = 0
                    for b in range(mp):
                        for c in range(pn):
                            M_a[idx, :] = V[b, :] * W[c, :]
                            T_a[idx] = T[a, b, c]
                            idx += 1
                    U[a, :], _, _, _ = np.linalg.lstsq(M_a, T_a, rcond=None)
                
                # Update V
                for b in range(mp):
                    M_b = np.zeros((nm * pn, target_rank))
                    T_b = np.zeros(nm * pn)
                    idx = 0
                    for a in range(nm):
                        for c in range(pn):
                            M_b[idx, :] = U[a, :] * W[c, :]
                            T_b[idx] = T[a, b, c]
                            idx += 1
                    V[b, :], _, _, _ = np.linalg.lstsq(M_b, T_b, rcond=None)
                
                # Update W
                for c in range(pn):
                    M_c = np.zeros((nm * mp, target_rank))
                    T_c = np.zeros(nm * mp)
                    idx = 0
                    for a in range(nm):
                        for b in range(mp):
                            M_c[idx, :] = U[a, :] * V[b, :]
                            T_c[idx] = T[a, b, c]
                            idx += 1
                    W[c, :], _, _, _ = np.linalg.lstsq(M_c, T_c, rcond=None)
            
            # Check if solution is good
            error = 0.0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        error += (val - T[a, b, c]) ** 2
            
            if error < 1e-8:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded solution
                err_r = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                            err_r += (val - T[a, b, c]) ** 2
                
                if err_r < 1e-8:
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
