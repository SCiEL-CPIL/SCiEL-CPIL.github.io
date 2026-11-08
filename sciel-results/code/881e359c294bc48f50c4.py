import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = rank_schoolbook
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
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    # Winograd-like block decomposition strategy
    # Partition A (2x3) into row blocks: A_0 (1x3), A_1 (1x3)
    # Partition B (3x5) into column blocks: B_0 (3x1), B_1 (3x1), B_2 (3x1), B_3 (3x1), B_4 (3x1)
    # Use bilinear identities to reduce rank
    
    # Strategy 1: Try to use a 2x2 fast multiplication identity on pairs of rows
    # For 2x2 matrices, we can use 7 multiplications instead of 4
    # Apply this to the row dimension
    
    # Let's try a systematic search with structured initializations
    # First, try rank 20-29 with various structured approaches
    
    for target_rank in range(20, 30):
        if time.time() > deadline:
            break
        
        # Try different structured initializations
        for init_type in range(5):
            if time.time() > deadline:
                break
            
            max_trials = 50 if init_type < 3 else 20
            
            for trial in range(max_trials):
                if time.time() > deadline:
                    break
                
                # Different initialization strategies
                if init_type == 0:
                    # Random initialization
                    U = np.random.randn(n * m, target_rank)
                    V = np.random.randn(m * p, target_rank)
                    W = np.random.randn(p * n, target_rank)
                elif init_type == 1:
                    # Sparse initialization - mostly zeros with some ones
                    U = np.zeros((n * m, target_rank))
                    V = np.zeros((m * p, target_rank))
                    W = np.zeros((p * n, target_rank))
                    for c in range(target_rank):
                        # Randomly select a few non-zero entries
                        for _ in range(2):
                            U[np.random.randint(n * m), c] = 1
                        for _ in range(2):
                            V[np.random.randint(m * p), c] = 1
                        for _ in range(2):
                            W[np.random.randint(p * n), c] = 1
                elif init_type == 2:
                    # Block-structured initialization
                    U = np.zeros((n * m, target_rank))
                    V = np.zeros((m * p, target_rank))
                    W = np.zeros((p * n, target_rank))
                    
                    # Divide target_rank into blocks
                    block_size = target_rank // 3
                    for b in range(3):
                        start = b * block_size
                        end = start + block_size if b < 2 else target_rank
                        
                        # Each block handles a subset of the multiplication
                        for c in range(start, end):
                            # Randomly assign to rows/cols
                            row_i = np.random.randint(n)
                            col_j = np.random.randint(m)
                            U[row_i * m + col_j, c] = 1
                            
                            for _ in range(3):
                                V[np.random.randint(m * p), c] = 1
                            for _ in range(3):
                                W[np.random.randint(p * n), c] = 1
                elif init_type == 3:
                    # Winograd-inspired: use differences
                    U = np.zeros((n * m, target_rank))
                    V = np.zeros((m * p, target_rank))
                    W = np.zeros((p * n, target_rank))
                    
                    # Use the 2x2 fast multiplication pattern
                    # M1 = (A00 + A11)(B00 + B11)
                    # M2 = (A10 + A11)B00
                    # M3 = A00(B11 - B00)
                    # M4 = A11(B00 - B10)
                    # M5 = (A00 + A10)B11
                    # M6 = (A11 - A00)B10
                    # M7 = (A01 - A10)B01
                    
                    # Adapt for our 2x3 x 3x5 case
                    for c in range(min(7, target_rank)):
                        if c == 0:
                            # M1-like
                            U[0, c] = 1
                            U[4, c] = 1
                            V[0, c] = 1
                            V[4, c] = 1
                            W[0, c] = 1
                            W[6, c] = 1
                        elif c == 1:
                            # M2-like
                            U[2, c] = 1
                            U[4, c] = 1
                            V[0, c] = 1
                            W[2, c] = 1
                            W[8, c] = 1
                        elif c == 2:
                            # M3-like
                            U[0, c] = 1
                            V[4, c] = 1
                            V[0, c] = -1
                            W[0, c] = 1
                        elif c == 3:
                            # M4-like
                            U[4, c] = 1
                            V[0, c] = 1
                            V[2, c] = -1
                            W[4, c] = 1
                        elif c == 4:
                            # M5-like
                            U[0, c] = 1
                            U[2, c] = 1
                            V[4, c] = 1
                            W[2, c] = 1
                        elif c == 5:
                            # M6-like
                            U[4, c] = 1
                            U[0, c] = -1
                            V[2, c] = 1
                            W[4, c] = 1
                        elif c == 6:
                            # M7-like
                            U[1, c] = 1
                            U[2, c] = -1
                            V[1, c] = 1
                            W[6, c] = 1
                    
                    # Fill remaining ranks with random
                    for c in range(7, target_rank):
                        U[:, c] = np.random.randn(n * m) * 0.1
                        V[:, c] = np.random.randn(m * p) * 0.1
                        W[:, c] = np.random.randn(p * n) * 0.1
                else:
                    # Hybrid: start from schoolbook and perturb
                    U = np.zeros((n * m, target_rank))
                    V = np.zeros((m * p, target_rank))
                    W = np.zeros((p * n, target_rank))
                    
                    # Initialize first min(target_rank, 30) as schoolbook
                    for c in range(min(target_rank, 30)):
                        i = c // 15
                        j = (c // 5) % 3
                        k = c % 5
                        if i < n and j < m and k < p:
                            U[i * m + j, c] = 1
                            V[j * p + k, c] = 1
                            W[k * n + i, c] = 1
                    
                    # Perturb
                    for c in range(target_rank):
                        U[:, c] += np.random.randn(n * m) * 0.05
                        V[:, c] += np.random.randn(m * p) * 0.05
                        W[:, c] += np.random.randn(p * n) * 0.05
                
                # Check if the decomposition is valid
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(target_rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(val - 1) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Convert to list format
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                
                if time.time() > deadline:
                    break
    
    return best_result
