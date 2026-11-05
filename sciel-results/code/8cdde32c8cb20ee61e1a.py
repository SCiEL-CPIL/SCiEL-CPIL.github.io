import numpy as np
import time
from itertools import product

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Pre-compute indices for the 60 tensor entries
    idx_u = np.zeros(n * m * p, dtype=np.int32)
    idx_v = np.zeros(n * m * p, dtype=np.int32)
    idx_w = np.zeros(n * m * p, dtype=np.int32)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u[idx] = i * m + j
                idx_v[idx] = j * p + k
                idx_w[idx] = k * n + i
                idx += 1
    
    def check_decomposition_fast(U, V, W):
        vals = U[idx_u] * V[idx_v] * W[idx_w]
        sums = vals.sum(axis=1)
        return np.allclose(sums, 1.0, atol=1e-10)
    
    def get_optimized_strassen():
        """
        Optimized Strassen-based decomposition for 2x5x6.
        
        Strategy:
        - Split j into {0,1}, {2,3}, {4}
        - Split k into {0,1}, {2,3}, {4,5}
        - Use Strassen for 2x2 blocks where possible
        """
        rank = 54
        U = np.zeros((nm, rank), dtype=np.int8)
        V = np.zeros((mp, rank), dtype=np.int8)
        W = np.zeros((pn, rank), dtype=np.int8)
        
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            # M0: (A00+A11)(B00+B11)
            U[i_base * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M1: (A10+A11)B00
            U[(i_base + 1) * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] -= 1
            col += 1
            
            # M2: A00(B01-B11)
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M3: A11(B10-B00)
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[j_base * p + k_base, col] -= 1
            W[k_base * n + i_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # M4: (A00+A01)B11
            U[i_base * m + j_base, col] += 1
            U[i_base * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base + 1) * m + j_base, col] += 1
            U[i_base * m + j_base, col] -= 1
            V[j_base * p + k_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M6: (A01-A11)(B10+B11)
            U[i_base * m + (j_base + 1), col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] -= 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1
        
        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            # 1: A00 B00 -> C00
            U[i_base * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1
            
            # 2: A00 B01 -> C01
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # 3: A10 B00 -> C10
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # 4: A10 B01 -> C11
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
        
        # Apply Strassen blocks for j pairs (0,1) and (2,3)
        for j_base in [0, 2]:
            for k_base in [0, 2, 4]:
                add_strassen_block(0, j_base, k_base)
        
        # Apply naive blocks for j=4
        for k_base in [0, 2, 4]:
            add_naive_2112_block(0, 4, k_base)
        
        if col != rank:
            rank_fb = n * m * p
            U_fb = np.zeros((nm, rank_fb), dtype=np.int8)
            V_fb = np.zeros((mp, rank_fb), dtype=np.int8)
            W_fb = np.zeros((pn, rank_fb), dtype=np.int8)
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, c] = 1
                        V_fb[j * p + k, c] = 1
                        W_fb[k * n + i, c] = 1
                        c += 1
            return U_fb, V_fb, W_fb, rank_fb
             
        return U, V, W, rank
    
    # Get the deterministic rank-54 solution first
    best_U, best_V, best_W, best_rank = get_optimized_strassen()
    
    # Try to find lower rank using structured random search
    for R in range(best_rank - 1, 35, -1):
        if time.time() > deadline - 2:
            break
            
        start_rank_time = time.time()
        time_per_rank = 1.0
        
        attempts = 0
        while time.time() - start_rank_time < time_per_rank and attempts < 100:
            attempts += 1
            
            U = np.zeros((nm, R), dtype=np.int8)
            V = np.zeros((mp, R), dtype=np.int8)
            W = np.zeros((pn, R), dtype=np.int8)
            
            for r in range(R):
                i, j = np.random.randint(0, n), np.random.randint(0, m)
                k = np.random.randint(0, p)
                
                if np.random.random() < 0.3:
                    i2, j2 = np.random.randint(0, n), np.random.randint(0, m)
                    k2 = np.random.randint(0, p)
                    U[i*m+j, r] = np.random.choice([-1, 1])
                    U[i2*m+j2, r] = np.random.choice([-1, 1])
                    V[j*p+k, r] = np.random.choice([-1, 1])
                    V[j2*p+k2, r] = np.random.choice([-1, 1])
                    W[k*n+i, r] = np.random.choice([-1, 1])
                    W[k2*n+i2, r] = np.random.choice([-1, 1])
                else:
                    U[i*m+j, r] = np.random.choice([-1, 1])
                    V[j*p+k, r] = np.random.choice([-1, 1])
                    W[k*n+i, r] = np.random.choice([-1, 1])
            
            if check_decomposition_fast(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
        
        if time.time() > deadline - 5:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
