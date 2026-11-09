"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

import time
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Schoolbook decomposition as fallback
    def get_schoolbook():
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
        
        return {"U": U, "V": V, "W": W}
    
    best_result = get_schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W):
        rank = len(U[0])
        # Check that U[i,j], V[j,k], W[k,i] satisfy the tensor equation
        # For each (i,j,k): sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = delta
        # Actually, the tensor is T[i*m+j, j*p+k, k*n+i] = 1 if (i,j,k) valid, else 0
        # The decomposition requires: T[a,b,c] = sum_r U[a][r] * V[b][r] * W[c][r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 1:
                        return False
        
        # Check that T is zero elsewhere
        # T[a,b,c] should be 0 if (a,b,c) doesn't correspond to a valid (i,j,k)
        # a = i*m+j, b = j'*p+k', c = k''*n+i''
        # Valid only if j=j'=j'' and i=i'' and k=k'
        # Actually, the tensor is only non-zero at (i*m+j, j*p+k, k*n+i) for valid i,j,k
        # Let's check all possible (a,b,c) combinations
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Determine if (a,b,c) corresponds to a valid tensor entry
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    if j == j2 and i == i2 and k == k2:
                        expected = 1
                    else:
                        expected = 0
                    
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != expected:
                        return False
        
        return True
    
    # Check if all entries are integer or half-integer
    def check_half_integer(mat):
        for row in mat:
            for val in row:
                # val should be integer or half-integer
                if val * 2 != int(val * 2):
                    return False
        return True
    
    # Try to find a lower rank decomposition
    # Strategy: For each output position (i,k), we need to compute sum_j A[i,j]*B[j,k]
    # We can try to share the j-dimension computations
    
    # For 2x3 * 3x10, the key is that for each (i,k), we sum over j=0,1,2
    # If we could use a rank-1 method for the 3-term sum, we'd save
    
    # Let's try a systematic search for small ranks
    # For rank r, we need U (6 x r), V (30 x r), W (20 x r)
    
    # Use a greedy approach: start with schoolbook and try to merge terms
    
    # Alternative: try specific structured decompositions
    
    # For the 3-term sum in j, we can use:
    # sum_{j=0}^{2} x_j * y_j = (x_0+x_1+x_2)(y_0+y_1+y_2) - (x_0+x_1)(y_0+y_1) - (x_0+x_2)(y_0+y_2) - (x_1+x_2)(y_1+y_2) + 2*(x_1*y_1 + x_2*y_2)
    # This is getting complex. Let's try a simpler approach.
    
    # For each (i,k) pair, we have 3 terms. If we process all (i,k) pairs together,
    # we have 2*10 = 20 outputs, each requiring 3 multiplications = 60 total.
    
    # Can we do better? The tensor has rank at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # So rank >= 30.
    
    # Let's try ranks from 30 to 60 and see if we can find a valid decomposition
    
    # For efficiency, let's use a randomized local search
    import random
    
    def random_decomposition(rank, seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Initialize with random small values
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
        
        return U, V, W
    
    # Try a few random initializations and local search
    max_search_time = 100  # seconds
    search_start = time.time()
    
    for rank in range(30, min(60, best_rank)):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        for attempt in range(10):
            if time.time() > deadline:
                break
            if time.time() - search_start > max_search_time:
                break
            
            U, V, W = random_decomposition(rank, seed=rank*100+attempt)
            
            # Local search: try to fix errors
            # This is complex, so let's just check if it works
            if verify(U, V, W):
                if check_half_integer(U) and check_half_integer(V) and check_half_integer(W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
        
        if best_rank == rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
