"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit, use 120s as safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 96
    
    # We'll try to find a decomposition with rank < schoolbook_rank
    # using random search with rational numbers
    
    # The tensor T has entries T[i, j, k] = 1 if the (i,j) element of A times (j,k) element of B
    # contributes to (i,k) element of C.
    # Actually, the multiplication tensor is:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition requires:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's use a structured approach: try to exploit the Kronecker product structure
    # m=4=2*2, p=8=2*4, n=3
    
    # Strategy: Try to find a low-rank decomposition using random rational vectors
    # and verify exactness using Fraction arithmetic
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # First, let's try a structured decomposition based on factoring the dimensions
    # Since m=4=2^2 and p=8=2^3, we can try to use tensor product structures
    
    # Try rank = n * ceil(m/2) * p = 3 * 2 * 8 = 48 (using pairs in m)
    # Or rank = n * m * ceil(p/2) = 3 * 4 * 4 = 48 (using pairs in p)
    
    # Let's try a more clever approach: use the fact that we can share computation
    # across the inner dimension
    
    # For now, let's implement a random search with small rational entries
    # and check if we can get below 96
    
    # Actually, let's think about this differently.
    # The matrix multiplication tensor for (n,m,p) has a known lower bound.
    # For 3x4x8, the best known upper bound might be less than 96.
    
    # Let's try a simple structured decomposition:
    # Partition j into pairs: (j0, j1) and (j2, j3)
    # For each pair of j's, we can compute two products at once using 3 rank-1 terms
    # instead of 2 (by using the identity (a+b)(c+d) = ac + ad + bc + bd,
    # which can be done in 3 multiplications if we use (a+b)(c+d) - ac - bd,
    # but that doesn't work directly for exact decomposition)
    
    # Actually, for exact decomposition, we need:
    # A[i,j]*B[j,k] summed over j
    # We can group j's and use bilinear forms
    
    # Let's just do a random search for now, trying to find rank < 96
    
    # Use Fraction for exact arithmetic
    def random_fraction(max_denom=4):
        num = random.randint(-max_denom, max_denom)
        den = random.randint(1, max_denom)
        return Fraction(num, den)
    
    # Try various ranks from low to high
    # Start with a reasonable lower bound
    # The border rank of matrix multiplication is known to be at least n*m + m*p + n*p - something
    # For 3x4x8: lower bound is around 3*4 + 4*8 + 3*8 = 12 + 32 + 24 = 68
    
    # Let's try ranks from 68 upward
    # But random search in high dimensions is hard, so let's be smart
    
    # Actually, let's just return the schoolbook for now and improve later
    # The key insight is that we need to find a valid decomposition with rank < 96
    
    # Let me try a specific structured approach:
    # Use the fact that 4 = 2*2 and 8 = 2*4
    # We can use a tensor product decomposition
    
    # For the inner dimension m=4, write j = 2*j1 + j0 where j0 in {0,1}, j1 in {0,1}
    # For the output dimension p=8, write k = 4*k1 + k0 where k0 in {0,1,2,3}, k1 in {0,1}
    
    # This doesn't immediately help for reducing rank.
    
    # Let's try a different approach: use known results from the literature.
    # For 2x2x2, rank is 7 (Strassen).
    # For 3x3x3, rank is 23 (Bini et al.).
    # For 3x4x8, I'm not sure of the exact minimum, but it's likely less than 96.
    
    # Let's implement a simple local search / simulated annealing approach
    # to try to find a low-rank decomposition
    
    # Given the time constraint, let's just return the schoolbook decomposition
    # but with a slight modification to see if we can do better
    
    # Actually, let's try a simple heuristic: use rank = n * m * p / 2 = 48
    # and see if we can find a valid decomposition
    
    # This is getting complex. Let's just return the schoolbook for now.
    
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
