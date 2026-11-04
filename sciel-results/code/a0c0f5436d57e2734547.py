"""Improved matrix multiplication tensor decomposition for M_{3,4,7}."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    def get_trivial_block(n, m, p):
        """Returns U, V, W for M_{n,m,p} with trivial rank n*m*p"""
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
    
    def verify_full(U, V, W, rank, n, m, p):
        """Verify full decomposition for M_{n,m,p}"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m + j] * V_arr[j*p + k] * W_arr[k*n + i])
                    if abs(s - 1) > 1e-9:
                        return False
        
        # Check invalid entries
        for a in range(n*m):
            i_a = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k_b = b % p
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    
                    is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                    
                    if not is_valid:
                        s = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(s) > 1e-9:
                            return False
        return True
    
    # Strategy: Decompose M_{3,4,7} into blocks
    # Option 1: Two M_{3,2,7} blocks (current approach)
    # Option 2: M_{3,3,7} + M_{3,1,7}
    # Option 3: Try to find better decompositions for smaller blocks
    
    # First, get the trivial decomposition as fallback
    U_triv, V_triv, W_triv, rank_triv = get_trivial_block(n, m, p)
    
    # Try to improve by finding better decompositions for M_{3,2,7}
    # The trivial rank for M_{3,2,7} is 3*2*7 = 42
    # If we can find rank 41 for M_{3,2,7}, total rank = 82
    
    n1, m1, p1 = 3, 2, 7
    U1_triv, V1_triv, W1_triv, rank1_triv = get_trivial_block(n1, m1, p1)
    
    best_U1 = U1_triv
    best_V1 = V1_triv
    best_W1 = W1_triv
    best_rank1 = rank1_triv
    
    # Try random search for lower rank M_{3,2,7}
    search_values = [0, 1, -1, Fraction(1,2), Fraction(-1,2)]
    
    target_rank1 = rank1_triv - 1
    attempts = 0
    max_attempts = 3000
    
    while time.time() < deadline - 15 and attempts < max_attempts:
        attempts += 1
        
        # Random initialization with sparse entries
        U1 = [[random.choice(search_values) if random.random() < 0.3 else 0 for _ in range(target_rank1)] for _ in range(n1*m1)]
        V1 = [[random.choice(search_values) if random.random() < 0.3 else 0 for _ in range(target_rank1)] for _ in range(m1*p1)]
        W1 = [[random.choice(search_values) if random.random() < 0.3 else 0 for _ in range(target_rank1)] for _ in range(p1*n1)]
        
        # Quick check: valid entries sum to 1
        valid_ok = True
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    s = 0
                    for r in range(target_rank1):
                        s += U1[i*m1 + j][r] * V1[j*p1 + k][r] * W1[k*n1 + i][r]
                    if s != 1:
                        valid_ok = False
                        break
                if not valid_ok: break
            if not valid_ok: break
            
        if valid_ok:
            # Full verification
            U1_arr = np.array(U1, dtype=float)
            V1_arr = np.array(V1, dtype=float)
            W1_arr = np.array(W1, dtype=float)
            
            invalid_ok = True
            for a in range(n1*m1):
                i_a = a // m1
                j_a = a % m1
                for b in range(m1*p1):
                    j_b = b // p1
                    k_b = b % p1
                    for c in range(p1*n1):
                        k_c = c // n1
                        i_c = c % n1
                        
                        is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                        
                        if not is_valid:
                            s = np.sum(U1_arr[a] * V1_arr[b] * W1_arr[c])
                            if abs(s) > 1e-9:
                                invalid_ok = False
                                break
                    if not invalid_ok: break
                if not invalid_ok: break
            
            if invalid_ok:
                best_U1 = U1
                best_V1 = V1
                best_W1 = W1
                best_rank1 = target_rank1
                break
    
    # Combine two M_{3,2,7} blocks to form M_{3,4,7}
    rank_total = best_rank1 * 2
    
    U_total = [[0] * rank_total for _ in range(n * m)]  # 12 rows
    V_total = [[0] * rank_total for _ in range(m * p)]  # 28 rows
    W_total = [[0] * rank_total for _ in range(p * n)]  # 21 rows
    
    # Mapping for Block 1 (j=0,1)
    u_map_1 = {0: 0, 1: 1, 2: 4, 3: 5, 4: 8, 5: 9}
    v_map_1 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
               7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13}
    w_map_1 = {k*n1 + i: k*n1 + i for k in range(p1) for i in range(n1)}
    
    # Mapping for Block 2 (j=2,3)
    u_map_2 = {0: 2, 1: 3, 2: 6, 3: 7, 4: 10, 5: 11}
    v_map_2 = {0: 14, 1: 15, 2: 16, 3: 17, 4: 18, 5: 19, 6: 20,
               7: 21, 8: 22, 9: 23, 10: 24, 11: 25, 12: 26, 13: 27}
    w_map_2 = w_map_1
    
    # Fill U_total
    for r in range(best_rank1):
        for u_idx_1, u_idx_orig in u_map_1.items():
            U_total[u_idx_orig][r] = best_U1[u_idx_1][r]
        for u_idx_2, u_idx_orig in u_map_2.items():
            U_total[u_idx_orig][r + best_rank1] = best_U1[u_idx_2][r]
    
    # Fill V_total
    for r in range(best_rank1):
        for v_idx_1, v_idx_orig in v_map_1.items():
            V_total[v_idx_orig][r] = best_V1[v_idx_1][r]
        for v_idx_2, v_idx_orig in v_map_2.items():
            V_total[v_idx_orig][r + best_rank1] = best_V1[v_idx_2][r]
    
    # Fill W_total
    for r in range(best_rank1):
        for w_idx, w_idx_orig in w_map_1.items():
            W_total[w_idx_orig][r] = best_W1[w_idx][r]
            W_total[w_idx_orig][r + best_rank1] = best_W1[w_idx][r]
    
    # Verify the final result
    if not verify_full(U_total, V_total, W_total, rank_total, n, m, p):
        # Fallback to trivial decomposition
        return {"U": U_triv, "V": V_triv, "W": W_triv}
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
