"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    
    # Structural Change: Decompose M_{3,4,7} into two M_{3,2,7} blocks.
    # m=4 is split into m1=2, m2=2.
    # C = A1 B1 + A2 B2
    # Rank(M_{3,4,7}) <= Rank(M_{3,2,7}) + Rank(M_{3,2,7}) = 2 * Rank(M_{3,2,7})
    
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

    def verify_block(U, V, W, rank, n, m, p):
        """Verify decomposition for a block"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(np.dot(U_arr[i*m + j], V_arr[j*p + k]), W_arr[k*n + i])
                    # Note: U, V, W are (rows, rank). 
                    # sum_r U[a,r] V[b,r] W[c,r]
                    # This is element-wise product sum.
                    s = np.sum(U_arr[i*m + j] * V_arr[j*p + k] * W_arr[k*n + i])
                    if abs(s - 1) > 1e-9:
                        return False
        
        # Check invalid entries (sample a few to save time, or check all if small)
        # For 3,2,7: 6 * 14 * 21 = 1764 entries. Total 3*2*7=42 valid.
        # 1722 invalid entries.
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                    
                    if not is_valid:
                        s = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(s) > 1e-9:
                            return False
        return True

    # Step 1: Get trivial blocks for M_{3,2,7}
    n1, m1, p1 = 3, 2, 7
    U1_triv, V1_triv, W1_triv, rank1_triv = get_trivial_block(n1, m1, p1)
    
    # Step 2: Try to find a lower rank for M_{3,2,7}
    # Search for rank 41, 40, etc.
    best_U1 = U1_triv
    best_V1 = V1_triv
    best_W1 = W1_triv
    best_rank1 = rank1_triv
    
    search_values = [0, 1, -1, Fraction(1,2), Fraction(-1,2), Fraction(1,4), Fraction(-1,4)]
    
    # We have about 140 seconds. 
    # Verification of 3,2,7 is fast.
    # Let's try a limited random search for rank 41.
    
    target_rank1 = rank1_triv - 1
    attempts = 0
    max_attempts = 5000 # Adjust based on time
    
    while time.time() < deadline - 10 and attempts < max_attempts:
        attempts += 1
        
        # Random initialization
        U1 = [[random.choice(search_values) for _ in range(target_rank1)] for _ in range(n1*m1)]
        V1 = [[random.choice(search_values) for _ in range(target_rank1)] for _ in range(m1*p1)]
        W1 = [[random.choice(search_values) for _ in range(target_rank1)] for _ in range(p1*n1)]
        
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
            if verify_block(U1, V1, W1, target_rank1, n1, m1, p1):
                best_U1 = U1
                best_V1 = V1
                best_W1 = W1
                best_rank1 = target_rank1
                break
        
    # Step 3: Combine two blocks to form M_{3,4,7}
    # Block 1: A1 (3x2) * B1 (2x7) -> C1 (3x7)
    # Block 2: A2 (3x2) * B2 (2x7) -> C2 (3x7)
    # Total C = C1 + C2
    
    # The trivial M_{3,4,7} decomposition corresponds to:
    # U_total has 84 cols.
    # Cols 0-41 correspond to Block 1 (j=0,1)
    # Cols 42-83 correspond to Block 2 (j=2,3)
    
    # If we found a reduced rank for Block 1 (best_rank1 < 42),
    # We can use that for Block 1 and trivial for Block 2.
    # Total rank = best_rank1 + 42.
    
    # Alternatively, use best_rank1 for both?
    # The blocks are independent in the trivial decomposition.
    # We can use the reduced decomposition for both blocks if it is valid for M_{3,2,7}.
    # Yes, M_{3,2,7} is the same structure for both halves.
    
    rank_total = best_rank1 * 2
    
    U_total = [[0] * rank_total for _ in range(n * m)] # 12 rows
    V_total = [[0] * rank_total for _ in range(m * p)] # 28 rows
    W_total = [[0] * rank_total for _ in range(p * n)] # 21 rows
    
    # Fill Block 1 (j=0,1 in original m=4)
    # Original indices:
    # U row i*m + j. For j=0,1: rows 0..5 (since n=3, m=4, i*m+j -> 0,1,2, 4,5,6? No.
    # U index is i*m + j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # i=2, j=0 -> 8
    # i=2, j=1 -> 9
    
    # V index is j*p + k.
    # j=0, k=0..6 -> 0..6
    # j=1, k=0..6 -> 7..13
    
    # W index is k*n + i.
    # k=0..6, i=0..2.
    
    # Block 1 uses j=0,1.
    # We map the reduced U1 (6 rows) to U_total rows corresponding to j=0,1.
    # U1 row index: i*m1 + j1 where m1=2, j1 in 0,1.
    # i=0, j1=0 -> 0. Original U row 0 (i=0, j=0).
    # i=0, j1=1 -> 1. Original U row 1 (i=0, j=1).
    # i=1, j1=0 -> 2. Original U row 4 (i=1, j=0).
    # i=1, j1=1 -> 3. Original U row 5 (i=1, j=1).
    # i=2, j1=0 -> 4. Original U row 8 (i=2, j=0).
    # i=2, j1=1 -> 5. Original U row 9 (i=2, j=1).
    
    # V1 row index: j1*p1 + k where p1=7.
    # j1=0, k=0..6 -> 0..6. Original V rows 0..6 (j=0).
    # j1=1, k=0..6 -> 7..13. Original V rows 7..13 (j=1).
    
    # W1 row index: k*n1 + i. Same as original W rows.
    
    # Mapping U:
    u_map_1 = {0: 0, 1: 1, 2: 4, 3: 5, 4: 8, 5: 9}
    # Mapping V:
    v_map_1 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
               7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13}
    # Mapping W:
    w_map_1 = {k*n1 + i: k*n1 + i for k in range(p1) for i in range(n1)}
    
    # Fill Block 2 (j=2,3 in original m=4)
    # U1 row index: i*m1 + j1.
    # i=0, j1=0 -> 0. Original U row 2 (i=0, j=2).
    # i=0, j1=1 -> 1. Original U row 3 (i=0, j=3).
    # i=1, j1=0 -> 2. Original U row 6 (i=1, j=2).
    # i=1, j1=1 -> 3. Original U row 7 (i=1, j=3).
    # i=2, j1=0 -> 4. Original U row 10 (i=2, j=2).
    # i=2, j1=1 -> 5. Original U row 11 (i=2, j=3).
    
    u_map_2 = {0: 2, 1: 3, 2: 6, 3: 7, 4: 10, 5: 11}
    
    # V1 row index: j1*p1 + k.
    # j1=0, k=0..6 -> 0..6. Original V rows 14..20 (j=2).
    # j1=1, k=0..6 -> 7..13. Original V rows 21..27 (j=3).
    
    v_map_2 = {0: 14, 1: 15, 2: 16, 3: 17, 4: 18, 5: 19, 6: 20,
               7: 21, 8: 22, 9: 23, 10: 24, 11: 25, 12: 26, 13: 27}
    
    # W1 row index: k*n1 + i. Same as original W rows.
    w_map_2 = w_map_1 # W is shared in index space, but values are separate columns in total rank.
    # Wait, W_total has columns 0..rank_total-1.
    # Block 1 uses columns 0..best_rank1-1.
    # Block 2 uses columns best_rank1..2*best_rank1-1.
    
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
            
            # Block 2 uses the same W values but in different columns
            W_total[w_idx_orig][r + best_rank1] = best_W1[w_idx][r]
            
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
