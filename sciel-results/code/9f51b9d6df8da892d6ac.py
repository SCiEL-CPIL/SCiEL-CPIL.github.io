"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    # Structural approach: exploit the block-diagonal structure of the 2x3x9 tensor.
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k can be viewed as
    # 9 independent 2x3x1 multiplications (one per k).
    # For each k, we need to decompose the 2x3x1 tensor of rank 6.
    # 
    # Key insight: The 2x3x1 tensor has rank 6 (schoolbook is optimal for this small case).
    # However, we can try to find a lower-rank decomposition by exploiting
    # the specific structure. For 2x3x1, the tensor is:
    #   T[i,j,0] = 1 for i in {0,1}, j in {0,1,2}
    # This is an outer product of a 2-vector and 3-vector, so it's rank 1 in the
    # tensor sense? No, matrix multiplication tensor is different.
    #
    # Actually, for multiplying a 2x3 matrix by a 3x1 matrix, the result is 2x1.
    # The tensor has dimensions (2,3,1) = 6 entries. The schoolbook rank is 6.
    # But can we do better? The rank of the matrix multiplication tensor M_{n,m,p}
    # is known to be n*m*p for small cases, but there might be savings.
    #
    # Let's try a direct approach: attempt to find a rank-5 or lower decomposition
    # for the full 2x3x9 tensor using random search with exact rational arithmetic.
    
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return U, V, W, rank
    
    # Verify a decomposition
    def _verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    # Start with schoolbook
    U_best, V_best, W_best, rank_best = _schoolbook()
    
    # Try to find lower-rank decompositions
    # Lower bound: max(n*m, m*p, n*p) = max(6, 27, 18) = 27
    # But the tensor rank might be higher. Let's try ranks from 27 upward.
    
    rng = random.Random(42)
    
    for rank in range(27, rank_best):
        if time.time() > deadline:
            break
        
        attempts = 0
        max_attempts = 2000
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            
            attempts += 1
            
            # Generate random rational matrices
            def rand_frac():
                # Use small integers to keep fractions manageable
                return Fraction(rng.randint(-2, 2), 1)
            
            U = [[rand_frac() for _ in range(rank)] for _ in range(n * m)]
            V = [[rand_frac() for _ in range(rank)] for _ in range(m * p)]
            W = [[rand_frac() for _ in range(rank)] for _ in range(p * n)]
            
            if _verify(U, V, W, rank):
                U_best, V_best, W_best, rank_best = U, V, W, rank
                break
    
    # Convert to float for the interface
    U_out = [[float(x) for x in row] for row in U_best]
    V_out = [[float(x) for x in row] for row in V_best]
    W_out = [[float(x) for x in row] for row in W_best]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
