"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook decomposition as fallback
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
        return U, V, W, rank
    
    # Greedy rank-1 decomposition
    def greedy_decompose(max_rank=96):
        T_rem = T.copy().astype(float)
        U_cols = []
        V_cols = []
        W_cols = []
        
        for s in range(max_rank):
            if time.time() > deadline - 5:
                break
            
            # Find the best rank-1 tensor to subtract
            # A rank-1 tensor is u ⊗ v ⊗ w
            # We want to maximize the overlap with T_rem
            # This is equivalent to finding the best u, v, w
            
            # Simplified: try all possible "support patterns"
            # For efficiency, we'll try a subset
            
            best_score = -1
            best_u, best_v, best_w = None, None, None
            
            # Try random support patterns
            for _ in range(1000):
                if time.time() > deadline - 5:
                    break
                
                # Random support for u
                u_support = random.sample(range(n * m), random.randint(1, 4))
                # Random support for v
                v_support = random.sample(range(m * p), random.randint(1, 4))
                # Random support for w
                w_support = random.sample(range(p * n), random.randint(1, 4))
                
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                
                for idx in u_support:
                    u[idx] = random.choice([-1, 1])
                for idx in v_support:
                    v[idx] = random.choice([-1, 1])
                for idx in w_support:
                    w[idx] = random.choice([-1, 1])
                
                # Compute the tensor u ⊗ v ⊗ w
                # T_candidate[a,b,c] = u[a] * v[b] * w[c]
                # Score = sum of T_rem[a,b,c] * T_candidate[a,b,c]
                # = sum over a,b,c of T_rem[a,b,c] * u[a] * v[b] * w[c]
                
                # Efficient computation:
                # For each s, the contribution is u[a] * v[b] * w[c] * T_rem[a,b,c]
                # = sum_a u[a] * (sum_b v[b] * (sum_c w[c] * T_rem[a,b,c]))
                
                # Precompute for efficiency
                # This is still O(n*m * m*p * p*n) = O(12*32*24) = O(9216) per candidate
                # With 1000 candidates, that's ~9M operations, which is fast
                
                score = 0
                for a in u_support:
                    if u[a] == 0:
                        continue
                    for b in v_support:
                        if v[b] == 0:
                            continue
                        for c in w_support:
                            if w[c] == 0:
                                continue
                            score += u[a] * v[b] * w[c] * T_rem[a, b, c]
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_u is None or best_score <= 0:
                break
            
            # Subtract the rank-1 tensor
            for a in range(n * m):
                if best_u[a] == 0:
                    continue
                for b in range(m * p):
                    if best_v[b] == 0:
                        continue
                    for c in range(p * n):
                        if best_w[c] == 0:
                            continue
                        T_rem[a, b, c] -= best_u[a] * best_v[b] * best_w[c]
            
            U_cols.append(best_u.tolist())
            V_cols.append(best_v.tolist())
            W_cols.append(best_w.tolist())
            
            # Check if T_rem is zero
            if np.allclose(T_rem, 0, atol=1e-6):
                break
        
        if not U_cols:
            return None
        
        # Transpose to get U, V, W
        rank = len(U_cols)
        U = [[U_cols[s][a] for s in range(rank)] for a in range(n * m)]
        V = [[V_cols[s][b] for s in range(rank)] for b in range(m * p)]
        W = [[W_cols[s][c] for s in range(rank)] for c in range(p * n)]
        
        return U, V, W, rank
    
    # Try greedy decomposition
    result = greedy_decompose(max_rank=96)
    if result is not None:
        U, V, W, rank = result
        # Verify
        T_check = np.zeros((n * m, m * p, p * n), dtype=float)
        for s in range(rank):
            for a in range(n * m):
                if U[a][s] == 0:
                    continue
                for b in range(m * p):
                    if V[b][s] == 0:
                        continue
                    for c in range(p * n):
                        if W[c][s] == 0:
                            continue
                        T_check[a, b, c] += U[a][s] * V[b][s] * W[c][s]
        
        if np.allclose(T_check, T, atol=1e-6):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, _ = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
