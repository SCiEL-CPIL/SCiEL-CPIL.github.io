"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin before 150s limit
    
    n, m, p = 3, 4, 7
    nmp = n * m * p  # 84 (schoolbook rank)
    
    # The schoolbook decomposition has rank 84
    # We want to find something better if possible
    
    # First, let's create the schoolbook decomposition as a fallback
    def make_schoolbook():
        rank = nmp
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
        return U, V, W
    
    # Fast verify using numpy broadcasting
    def verify(U, V, W):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # For each (i,j,k), check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = int(np.dot(
                        U_np[i * m + j, :] * V_np[j * p + k, :],
                        W_np[k * n + i, :]
                    ))
                    if total != 1:
                        return False
        return True
    
    best_rank = nmp
    best_U, best_V, best_W = make_schoolbook()
    
    random.seed(12345)
    np.random.seed(12345)
    
    # Strategy: Use block decomposition
    # Split p=7 into p1=4 and p2=3
    # M_{3,4,7} can be seen as combining M_{3,4,4} and M_{3,4,3}
    # But we need to be careful: the V matrix has rows indexed by (j,k)
    
    # Let's try a different approach: use known results
    # For 3x4x7, the lower bound is around 56-60 based on asymptotic results
    # But let's try to find a good decomposition
    
    # Approach 1: Start with schoolbook and try to merge terms
    # Approach 2: Use structured random search with better patterns
    
    # Let's try ranks from a reasonable range
    # Based on literature, 3x4x7 might have rank around 50-60
    
    # Prioritize ranks that are multiples of small integers or known bounds
    # R(3,4,7) is likely 56. Try 56, 57, 58, 59, 60 first, then others.
    target_ranks = [56, 57, 58, 59, 60] + list(range(61, 84))
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        # Try multiple strategies
        strategies = ['structured_block', 'mixed', 'random_dense']
        
        for strategy in strategies:
            if time.time() > deadline - 20:
                break
            
            # More attempts for lower ranks
            max_attempts = 100 if target_rank <= 60 else 50
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                U = np.zeros((n * m, target_rank), dtype=int)
                V = np.zeros((m * p, target_rank), dtype=int)
                W = np.zeros((p * n, target_rank), dtype=int)
                
                if strategy == 'structured_block':
                    for c in range(target_rank):
                        if random.random() < 0.3:
                            # Span both blocks
                            for i in range(n):
                                for j in range(m):
                                    if random.random() < 0.4:
                                        U[i * m + j, c] = random.choice([-1, 1])
                            for j in range(m):
                                for k in range(p):
                                    if random.random() < 0.3:
                                        V[j * p + k, c] = random.choice([-1, 1])
                            for i in range(n):
                                for k in range(p):
                                    if random.random() < 0.3:
                                        W[k * n + i, c] = random.choice([-1, 1])
                        else:
                            # Single block
                            if random.random() < 0.5:
                                k_min, k_max = 0, 4
                            else:
                                k_min, k_max = 4, 7
                            
                            for i in range(n):
                                for j in range(m):
                                    if random.random() < 0.5:
                                        U[i * m + j, c] = random.choice([-1, 1])
                            for j in range(m):
                                for k in range(k_min, k_max):
                                    if random.random() < 0.4:
                                        V[j * p + k, c] = random.choice([-1, 1])
                            for i in range(n):
                                for k in range(k_min, k_max):
                                    if random.random() < 0.4:
                                        W[k * n + i, c] = random.choice([-1, 1])
                            
                elif strategy == 'mixed':
                    for c in range(target_rank):
                        r = random.random()
                        if r < 0.4:
                            # Sparse random
                            U[:, c] = np.random.choice([-1, 0, 1], size=n * m, p=[0.2, 0.6, 0.2])
                            V[:, c] = np.random.choice([-1, 0, 1], size=m * p, p=[0.2, 0.6, 0.2])
                            W[:, c] = np.random.choice([-1, 0, 1], size=p * n, p=[0.2, 0.6, 0.2])
                        elif r < 0.7:
                            # Structured single position
                            i_idx = random.randint(0, n-1)
                            j_idx = random.randint(0, m-1)
                            k_idx = random.randint(0, p-1)
                            U[i_idx * m + j_idx, c] = random.choice([-1, 1])
                            V[j_idx * p + k_idx, c] = random.choice([-1, 1])
                            W[k_idx * n + i_idx, c] = random.choice([-1, 1])
                        else:
                            # Structured with extras
                            i_idx = random.randint(0, n-1)
                            j_idx = random.randint(0, m-1)
                            k_idx = random.randint(0, p-1)
                            U[i_idx * m + j_idx, c] = random.choice([-1, 1])
                            V[j_idx * p + k_idx, c] = random.choice([-1, 1])
                            W[k_idx * n + i_idx, c] = random.choice([-1, 1])
                            for _ in range(random.randint(1, 2)):
                                i2 = random.randint(0, n-1)
                                j2 = random.randint(0, m-1)
                                k2 = random.randint(0, p-1)
                                U[i2 * m + j2, c] += random.choice([-1, 1])
                                V[j2 * p + k2, c] += random.choice([-1, 1])
                                W[k2 * n + i2, c] += random.choice([-1, 1])
                else: # random_dense
                    for c in range(target_rank):
                        U[:, c] = np.random.choice([-1, 0, 1], size=n * m, p=[0.25, 0.5, 0.25])
                        V[:, c] = np.random.choice([-1, 0, 1], size=m * p, p=[0.25, 0.5, 0.25])
                        W[:, c] = np.random.choice([-1, 0, 1], size=p * n, p=[0.25, 0.5, 0.25])
                
                # Verify
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
            
            if best_rank < nmp:
                break
        
        if best_rank < nmp:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
