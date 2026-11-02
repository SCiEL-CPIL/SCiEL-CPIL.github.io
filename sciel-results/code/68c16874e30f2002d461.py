import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per seed, or numpy? 
    # Seed used lists. I will use lists for compatibility if evaluator expects lists, 
    # but numpy is faster. The prompt says "factor matrices ... with integer/half-integer entries".
    # The seed returns lists. I should probably return lists to be safe, 
    # or numpy arrays if allowed. 
    # "Python standard library, NumPy and SciPy are available."
    # The seed returns {"U": U, "V": V, "W": W} where U,V,W are lists of lists.
    # I will stick to lists to match the interface exactly, or convert if needed.
    # Actually, numpy arrays are often accepted. But let's stick to the seed's type: lists.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # c maps to (i_c, k_c)
    for c in range(rank):
        i_c = c // p
        k_c = c % p
        
        # U[i*m + j, c] = 1 if i == i_c
        for j in range(m):
            U[i_c * m + j][c] = 1
            
        # V[j*p + k, c] = 1 if k == k_c
        for j in range(m):
            V[j * p + k_c][c] = 1
            
        # W[k*n + i, c] = 1 if k == k_c and i == i_c
        W[k_c * n + i_c][c] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
