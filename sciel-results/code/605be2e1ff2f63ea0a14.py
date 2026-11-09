"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 8]

def construct():
    """Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Index helpers
    def idx_u(i, j):
        return i * m + j
    
    def idx_v(j, k):
        return j * p + k
    
    def idx_w(k, i):
        return k * n + i
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p  # 72
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[idx_u(i, j), col] = 1.0
                V_sb[idx_v(j, k), col] = 1.0
                W_sb[idx_w(k, i), col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = schoolbook_rank
    
    # All (i,j,k) triples
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 24
    lower_bound = max(n * m, m * p, p * n)
    
    # Try candidate ranks
    for candidate_rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts for this rank
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + candidate_rank * 100 + restart)
            
            U = np.zeros((n * m, candidate_rank))
            V = np.zeros((m * p, candidate_rank))
            W = np.zeros((p * n, candidate_rank))
            
            residual = np.ones((n, m, p))
            
            for step in range(candidate_rank):
                if time.time() > deadline - 2:
                    break
                
                # Find best rank-1 component: maximize number of satisfied equations
                best_score = -1
                best_u, best_v, best_w = None, None, None
                
                # Try structured candidates: for each (i,j) or (j,k) or (k,i) pattern
                # Try simple binary/half-integer patterns
                
                # Generate candidate rank-1 components
                candidates = []
                
                # Pattern 1: fix i, vary j,k
                for i_fixed in range(n):
                    for j_fixed in range(m):
                        u = np.zeros(n * m)
                        v = np.zeros(m * p)
                        w = np.zeros(p * n)
                        u[idx_u(i_fixed, j_fixed)] = 1.0
                        for k in range(p):
                            v[idx_v(j_fixed, k)] = 1.0
                            w[idx_w(k, i_fixed)] = 1.0
                        candidates.append((u, v, w))
                
                # Pattern 2: fix j, vary i,k
                for j_fixed in range(m):
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    for i in range(n):
                        u[idx_u(i, j_fixed)] = 1.0
                    for k in range(p):
                        v[idx_v(j_fixed, k)] = 1.0
                        w[idx_w(k, i)] = 1.0 if i < n else 0.0
                    # This pattern doesn't work well, skip
                    # Instead, try: for fixed j, u[i,j]=1, v[j,k]=1, w[k,i]=1
                    w2 = np.zeros(p * n)
                    for i in range(n):
                        for k in range(p):
                            w2[idx_w(k, i)] = 1.0
                    candidates.append((u, v, w2))
                
                # Pattern 3: fix k, vary i,j
                for k_fixed in range(p):
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    for i in range(n):
                        for j in range(m):
                            u[idx_u(i, j)] = 1.0
                        v[idx_v(j, k_fixed)] = 1.0
                        w[idx_w(k_fixed, i)] = 1.0
                    candidates.append((u, v, w))
                
                # Pattern 4: single (i,j,k)
                for i, j, k in all_triples:
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    u[idx_u(i, j)] = 1.0
                    v[idx_v(j, k)] = 1.0
                    w[idx_w(k, i)] = 1.0
                    candidates.append((u, v, w))
                
                # Score each candidate
                for u, v, w in candidates:
                    # Compute how many equations this would satisfy
                    score = 0
                    for i, j, k in all_triples:
                        val = u[idx_u(i, j)] * v[idx_v(j, k)] * w[idx_w(k, i)]
                        if abs(val - residual[i, j, k]) < 1e-9:
                            score += 1
                        elif abs(val) > 1e-9 and abs(residual[i, j, k]) > 1e-9:
                            # This would make it worse
                            score -= 100
                    if score > best_score:
                        best_score = score
                        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                
                if best_u is None or best_score <= 0:
                    break
                
                U[:, step] = best_u
                V[:, step] = best_v
                W[:, step] = best_w
                
                # Update residual
                for i, j, k in all_triples:
                    residual[i, j, k] -= best_u[idx_u(i, j)] * best_v[idx_v(j, k)] * best_w[idx_w(k, i)]
                
                # Check if all satisfied
                if np.allclose(residual, 0, atol=1e-9):
                    if step + 1 < best_rank:
                        best_rank = step + 1
                        best_U = U[:, :step+1].copy()
                        best_V = V[:, :step+1].copy()
                        best_W = W[:, :step+1].copy()
                    break
        
        # If we found a valid decomposition at this rank, we can stop
        if best_rank == candidate_rank:
            break
    
    # Verify the best decomposition
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(best_U[idx_u(i, j), :] * best_V[idx_v(j, k), :] * best_W[idx_w(k, i), :])
                if abs(val - 1.0) > 1e-6:
                    verified = False
                    break
            if not verified:
                break
    
    if not verified:
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = schoolbook_rank
    
    # Convert to lists with integer/half-integer entries
    U_list = [[0] * best_rank for _ in range(n * m)]
    V_list = [[0] * best_rank for _ in range(m * p)]
    W_list = [[0] * best_rank for _ in range(p * n)]
    
    for r in range(best_rank):
        for idx in range(n * m):
            U_list[idx][r] = float(best_U[idx, r])
        for idx in range(m * p):
            V_list[idx][r] = float(best_V[idx, r])
        for idx in range(p * n):
            W_list[idx][r] = float(best_W[idx, r])
    
    return {"U": U_list, "V": V_list, "W": W_list}
