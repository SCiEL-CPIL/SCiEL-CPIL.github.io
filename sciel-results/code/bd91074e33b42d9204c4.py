"""Improved matrix multiplication tensor decomposition for M_{3,4,7}."""
import time
import numpy as np
from fractions import Fraction
import random
from itertools import product

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
    
    def try_reduce_rank(U, V, W, rank, n, m, p, max_rounds=5):
        """Try to reduce rank by finding linear dependencies among terms"""
        for round_num in range(max_rounds):
            if time.time() > deadline - 10:
                break
            
            # Try to find pairs of terms that can be merged
            found_reduction = False
            for r1 in range(rank):
                if time.time() > deadline - 10:
                    break
                for r2 in range(r1 + 1, rank):
                    # Check if terms r1 and r2 can be merged
                    # We need to find alpha such that we can eliminate one term
                    # This is a complex constraint; try simple cases
                    
                    # Simple case: if U[:,r1] = U[:,r2], V[:,r1] = V[:,r2], W[:,r1] = W[:,r2]
                    # then we can combine them
                    if all(U[i][r1] == U[i][r2] for i in range(n*m)) and \
                       all(V[i][r1] == V[i][r2] for i in range(m*p)) and \
                       all(W[i][r1] == W[i][r2] for i in range(p*n)):
                        # Merge: add W values and remove one term
                        for i in range(p*n):
                            W[i][r1] += W[i][r2]
                        # Remove term r2
                        for i in range(n*m):
                            U[i].pop(r2)
                        for i in range(m*p):
                            V[i].pop(r2)
                        for i in range(p*n):
                            W[i].pop(r2)
                        rank -= 1
                        found_reduction = True
                        break
                if found_reduction:
                    break
            
            if not found_reduction:
                break
        
        return U, V, W, rank
    
    # First, get the trivial decomposition as fallback
    U_triv, V_triv, W_triv, rank_triv = get_trivial_block(n, m, p)
    
    # Try to reduce the trivial decomposition
    U_reduced, V_reduced, W_reduced, rank_reduced = try_reduce_rank(
        [row[:] for row in U_triv], 
        [row[:] for row in V_triv], 
        [row[:] for row in W_triv], 
        rank_triv, n, m, p
    )
    
    # Also try block decomposition approach
    n1, m1, p1 = 3, 2, 7
    U1_triv, V1_triv, W1_triv, rank1_triv = get_trivial_block(n1, m1, p1)
    
    # Try to reduce M_{3,2,7}
    U1_reduced, V1_reduced, W1_reduced, rank1_reduced = try_reduce_rank(
        [row[:] for row in U1_triv], 
        [row[:] for row in V1_triv], 
        [row[:] for row in W1_triv], 
        rank1_triv, n1, m1, p1
    )
    
    # Combine two M_{3,2,7} blocks
    rank_total_block = rank1_reduced * 2
    
    U_total = [[0] * rank_total_block for _ in range(n * m)]
    V_total = [[0] * rank_total_block for _ in range(m * p)]
    W_total = [[0] * rank_total_block for _ in range(p * n)]
    
    u_map_1 = {0: 0, 1: 1, 2: 4, 3: 5, 4: 8, 5: 9}
    v_map_1 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
               7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13}
    w_map_1 = {k*n1 + i: k*n1 + i for k in range(p1) for i in range(n1)}
    
    u_map_2 = {0: 2, 1: 3, 2: 6, 3: 7, 4: 10, 5: 11}
    v_map_2 = {0: 14, 1: 15, 2: 16, 3: 17, 4: 18, 5: 19, 6: 20,
               7: 21, 8: 22, 9: 23, 10: 24, 11: 25, 12: 26, 13: 27}
    w_map_2 = w_map_1
    
    for r in range(rank1_reduced):
        for u_idx_1, u_idx_orig in u_map_1.items():
            U_total[u_idx_orig][r] = U1_reduced[u_idx_1][r]
        for u_idx_2, u_idx_orig in u_map_2.items():
            U_total[u_idx_orig][r + rank1_reduced] = U1_reduced[u_idx_2][r]
    
    for r in range(rank1_reduced):
        for v_idx_1, v_idx_orig in v_map_1.items():
            V_total[v_idx_orig][r] = V1_reduced[v_idx_1][r]
        for v_idx_2, v_idx_orig in v_map_2.items():
            V_total[v_idx_orig][r + rank1_reduced] = V1_reduced[v_idx_2][r]
    
    for r in range(rank1_reduced):
        for w_idx, w_idx_orig in w_map_1.items():
            W_total[w_idx_orig][r] = W1_reduced[w_idx][r]
            W_total[w_idx_orig][r + rank1_reduced] = W1_reduced[w_idx][r]
    
    # Choose the best result
    best_rank = min(rank_reduced, rank_total_block)
    
    if rank_reduced <= rank_total_block:
        if verify_full(U_reduced, V_reduced, W_reduced, rank_reduced, n, m, p):
            return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
        else:
            return {"U": U_triv, "V": V_triv, "W": W_triv}
    else:
        if verify_full(U_total, V_total, W_total, rank_total_block, n, m, p):
            return {"U": U_total, "V": V_total, "W": W_total}
        else:
            return {"U": U_triv, "V": V_triv, "W": W_triv}
# EVOLVE-BLOCK-END
