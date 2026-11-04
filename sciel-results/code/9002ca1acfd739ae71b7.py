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
        """Verify full decomposition for M_{n,m,p} using numpy for speed"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m + j], V_arr[j*p + k] * W_arr[k*n + i])
                    if abs(s - 1) > 1e-9:
                        return False
        
        # Check invalid entries using vectorized approach
        for a in range(n*m):
            i_a = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    
                    is_valid = (i_a == i_c) and (k_b == k_c)
                    
                    if not is_valid:
                        s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                        if abs(s) > 1e-9:
                            return False
        return True
    
    def verify_block(U, V, W, rank, n, m, p):
        """Verify block decomposition - same as verify_full but for sub-blocks"""
        return verify_full(U, V, W, rank, n, m, p)
    
    def count_errors(U, V, W, rank, n, m, p):
        """Count number of constraint violations (for local search)"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        errors = 0
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m + j], V_arr[j*p + k] * W_arr[k*n + i])
                    if abs(s - 1) > 1e-9:
                        errors += 1
        
        # Check invalid entries
        for a in range(n*m):
            i_a = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    
                    is_valid = (i_a == i_c) and (k_b == k_c)
                    
                    if not is_valid:
                        s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                        if abs(s) > 1e-9:
                            errors += 1
        return errors
    
    def combine_blocks(U1, V1, W1, rank1, n, m1, p1, m2, m_offset):
        """Combine two M_{n,m1,p} and M_{n,m2,p} blocks into M_{n,m1+m2,p}"""
        m_total = m1 + m2
        rank_total = rank1 * 2
        
        U_total = [[0] * rank_total for _ in range(n * m_total)]
        V_total = [[0] * rank_total for _ in range(m_total * p1)]
        W_total = [[0] * rank_total for _ in range(p1 * n)]
        
        # Block 1: j = 0..m1-1
        for r in range(rank1):
            for i in range(n):
                for j in range(m1):
                    U_total[i * m_total + j][r] = U1[i * m1 + j][r]
            for j in range(m1):
                for k in range(p1):
                    V_total[j * p1 + k][r] = V1[j * p1 + k][r]
            for k in range(p1):
                for i in range(n):
                    W_total[k * n + i][r] = W1[k * n + i][r]
        
        # Block 2: j = m1..m1+m2-1
        for r in range(rank1):
            for i in range(n):
                for j in range(m2):
                    U_total[i * m_total + m1 + j][r + rank1] = U1[i * m1 + j][r]
            for j in range(m2):
                for k in range(p1):
                    V_total[(m1 + j) * p1 + k][r + rank1] = V1[j * p1 + k][r]
            for k in range(p1):
                for i in range(n):
                    W_total[k * n + i][r + rank1] = W1[k * n + i][r]
        
        return U_total, V_total, W_total, rank_total
    
    # Get trivial decomposition as fallback
    U_triv, V_triv, W_triv, rank_triv = get_trivial_block(n, m, p)
    
    best_total_rank = rank_triv
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    
    # Try different block decompositions
    block_configs = [
        (2, 2, 7),   # M_{3,2,7} + M_{3,2,7}
        (1, 3, 7),   # M_{3,1,7} + M_{3,3,7}
        (3, 1, 7),   # M_{3,3,7} + M_{3,1,7}
    ]
    
    for m1, m2, p1 in block_configs:
        if time.time() > deadline - 20:
            break
        
        # Get trivial decompositions for blocks
        U1_triv, V1_triv, W1_triv, rank1_triv = get_trivial_block(n, m1, p1)
        U2_triv, V2_triv, W2_triv, rank2_triv = get_trivial_block(n, m2, p1)
        
        # Start with trivial ranks
        best_U1, best_V1, best_W1, best_rank1 = U1_triv, V1_triv, W1_triv, rank1_triv
        best_U2, best_V2, best_W2, best_rank2 = U2_triv, V2_triv, W2_triv, rank2_triv
        
        # Try to improve M_{n,m1,p}
        if m1 >= 2:
            target_rank = rank1_triv - 1
            search_values = [0, 1, -1, 0.5, -0.5]
            attempts = 0
            max_attempts = 500
            
            while time.time() < deadline - 25 and attempts < max_attempts:
                attempts += 1
                
                # Structured initialization: start from trivial and remove one column
                U1 = [row[:] for row in U1_triv]
                V1 = [row[:] for row in V1_triv]
                W1 = [row[:] for row in W1_triv]
                
                # Remove a random column
                col_to_remove = random.randint(0, rank1_triv - 1)
                for i in range(len(U1)):
                    U1[i].pop(col_to_remove)
                for i in range(len(V1)):
                    V1[i].pop(col_to_remove)
                for i in range(len(W1)):
                    W1[i].pop(col_to_remove)
                
                # Try to fix by adjusting remaining entries
                if random.random() < 0.5:
                    # Scale approach: divide by (rank1_triv - 1)
                    scale = 1.0 / (rank1_triv - 1)
                    for i in range(len(U1)):
                        for j in range(len(U1[i])):
                            U1[i][j] = U1[i][j] * scale
                    for i in range(len(V1)):
                        for j in range(len(V1[i])):
                            V1[i][j] = V1[i][j] * scale
                    for i in range(len(W1)):
                        for j in range(len(W1[i])):
                            W1[i][j] = W1[i][j] * scale
                else:
                    # Random perturbation
                    for i in range(len(U1)):
                        for j in range(len(U1[i])):
                            if random.random() < 0.1:
                                U1[i][j] = random.choice(search_values)
                    for i in range(len(V1)):
                        for j in range(len(V1[i])):
                            if random.random() < 0.1:
                                V1[i][j] = random.choice(search_values)
                    for i in range(len(W1)):
                        for j in range(len(W1[i])):
                            if random.random() < 0.1:
                                W1[i][j] = random.choice(search_values)
                
                # Quick verification
                if verify_block(U1, V1, W1, target_rank, n, m1, p1):
                    best_U1, best_V1, best_W1, best_rank1 = U1, V1, W1, target_rank
                    break
        
        # Try to improve M_{n,m2,p}
        if m2 >= 2:
            target_rank = rank2_triv - 1
            search_values = [0, 1, -1, 0.5, -0.5]
            attempts = 0
            max_attempts = 500
            
            while time.time() < deadline - 25 and attempts < max_attempts:
                attempts += 1
                
                U2 = [row[:] for row in U2_triv]
                V2 = [row[:] for row in V2_triv]
                W2 = [row[:] for row in W2_triv]
                
                col_to_remove = random.randint(0, rank2_triv - 1)
                for i in range(len(U2)):
                    U2[i].pop(col_to_remove)
                for i in range(len(V2)):
                    V2[i].pop(col_to_remove)
                for i in range(len(W2)):
                    W2[i].pop(col_to_remove)
                
                if random.random() < 0.5:
                    scale = 1.0 / (rank2_triv - 1)
                    for i in range(len(U2)):
                        for j in range(len(U2[i])):
                            U2[i][j] = U2[i][j] * scale
                    for i in range(len(V2)):
                        for j in range(len(V2[i])):
                            V2[i][j] = V2[i][j] * scale
                    for i in range(len(W2)):
                        for j in range(len(W2[i])):
                            W2[i][j] = W2[i][j] * scale
                else:
                    for i in range(len(U2)):
                        for j in range(len(U2[i])):
                            if random.random() < 0.1:
                                U2[i][j] = random.choice(search_values)
                    for i in range(len(V2)):
                        for j in range(len(V2[i])):
                            if random.random() < 0.1:
                                V2[i][j] = random.choice(search_values)
                    for i in range(len(W2)):
                        for j in range(len(W2[i])):
                            if random.random() < 0.1:
                                W2[i][j] = random.choice(search_values)
                
                if verify_block(U2, V2, W2, target_rank, n, m2, p1):
                    best_U2, best_V2, best_W2, best_rank2 = U2, V2, W2, target_rank
                    break
        
        # Combine blocks
        total_rank = best_rank1 + best_rank2
        
        if total_rank < best_total_rank:
            U_total, V_total, W_total, _ = combine_blocks(
                best_U1, best_V1, best_W1, best_rank1, 
                n, m1, p1, m2, m1
            )
            
            if verify_full(U_total, V_total, W_total, total_rank, n, m1 + m2, p1):
                best_total_rank = total_rank
                best_result = {"U": U_total, "V": V_total, "W": W_total}
    
    # Additional: try direct search on full M_{3,4,7} with structured patterns
    if time.time() < deadline - 30:
        # Try rank 84 (trivial is 84)
        # Use a pattern-based approach
        search_values = [0, 1, -1, 0.5, -0.5]
        attempts = 0
        max_attempts = 200
        
        while time.time() < deadline - 10 and attempts < max_attempts:
            attempts += 1
            
            # Start from trivial and try removing columns
            U = [row[:] for row in U_triv]
            V = [row[:] for row in V_triv]
            W = [row[:] for row in W_triv]
            
            # Try to remove 1-3 columns
            cols_to_remove = random.sample(range(rank_triv), min(3, rank_triv))
            for col in sorted(cols_to_remove, reverse=True):
                for i in range(len(U)):
                    U[i].pop(col)
                for i in range(len(V)):
                    V[i].pop(col)
                for i in range(len(W)):
                    W[i].pop(col)
            
            new_rank = rank_triv - len(cols_to_remove)
            
            # Try to fix with scaling
            if random.random() < 0.7:
                scale = rank_triv / new_rank
                for i in range(len(U)):
                    for j in range(len(U[i])):
                        U[i][j] = U[i][j] * scale
                for i in range(len(V)):
                    for j in range(len(V[i])):
                        V[i][j] = V[i][j] * scale
                for i in range(len(W)):
                    for j in range(len(W[i])):
                        W[i][j] = W[i][j] * scale
            
            if verify_full(U, V, W, new_rank, n, m, p):
                if new_rank < best_total_rank:
                    best_total_rank = new_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break
    
    return best_result
# EVOLVE-BLOCK-END
