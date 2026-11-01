import time
import random
import numpy as np

def verify_decomposition(U, V, W, n=2, m=2, p=7):
    """
    Verifies if U, V, W form a valid decomposition of the matrix multiplication tensor.
    """
    n, m, p = 2, 2, 7
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for the 150s limit
    
    n, m, p = 2, 2, 7
    schoolbook_rank = n * m * p  # 28
    
    # 1. Construct Schoolbook (Rank 28) as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # 2. Try to find a rank 27 decomposition
    target_rank = 27
    found = False
    
    # Use a structured random search
    # We initialize with the schoolbook and perturb it, or just random search.
    # Given the small dimension, random search in a bounded box might work.
    
    search_space_sizes = [3, 5, 7, 9]
    
    for size in search_space_sizes:
        if found:
            break
        if time.time() > deadline - 20:
            break
        
        low = -size // 2
        high = (size - 1) // 2
        
        max_attempts = 20000
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            U_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.randint(low, high) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = target_rank
                found = True
                break
                
    return {"U": best_U, "V": best_V, "W": best_W}
