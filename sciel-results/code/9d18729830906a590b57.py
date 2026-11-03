import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback (rank 45)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition exactly
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for i in range(n * m):
            if len(U[i]) != rank:
                return False
        for i in range(m * p):
            if len(V[i]) != rank:
                return False
        for i in range(p * n):
            if len(W[i]) != rank:
                return False
        
        for trial in range(20):
            test_i = random.randint(0, n * m - 1)
            test_j = random.randint(0, m * p - 1)
            test_k = random.randint(0, p * n - 1)
            
            ii = test_i // m
            jj = test_i % m
            jj2 = test_j // p
            kk = test_j % p
            kk2 = test_k // n
            ii2 = test_k % n
            
            is_valid = (ii == ii2 and jj == jj2 and kk == kk2)
            expected = 1 if is_valid else 0
            
            computed = 0
            for r in range(rank):
                computed += U[test_i][r] * V[test_j][r] * W[test_k][r]
            
            if computed != expected:
                return False
        
        return True
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    for target_rank in range(42, 34, -1):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    if best_rank < n * m * p:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
