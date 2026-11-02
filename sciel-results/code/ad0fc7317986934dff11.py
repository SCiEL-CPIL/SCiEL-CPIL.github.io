"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    # Schoolbook decomposition of rank n*m*p
    def schoolbook():
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

    best = schoolbook()
    best_rank = n * m * p

    # Nonzero entries of the tensor
    nz = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]

    # Verify a decomposition: check that sum_t U[a,t]*V[b,t]*W[c,t] = 1 for (a,b,c) in nz, 0 otherwise
    def verify(U, V, W):
        r = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    s = sum(U[a][t] * V[b][t] * W[c][t] for t in range(r))
                    expected = 1 if (a, b, c) in nz else 0
                    if s != expected:
                        return False
        return True

    # Try to find a decomposition of rank r by fixing W to a specific structure
    # and solving for U and V.
    # 
    # Key insight: If we fix W to have a particular structure, the equations become
    # bilinear in U and V. We can try several structured W patterns.
    #
    # Strategy: For each candidate rank r from lower bound to schoolbook:
    #   Try multiple structured W patterns
    #   For each pattern, solve for U and V using iterative refinement
    #   Verify with exact arithmetic

    # Lower bound
    lb = max(n*m, m*p, p*n)

    # Candidate ranks to try (from low to high)
    candidate_ranks = list(range(lb, n*m*p + 1))

    # Try to find a decomposition with rank r
    def try_rank(r):
        if time.time() > deadline - 5:
            return None

        # Try several structured W patterns
        # Pattern 1: W is a subset of identity-like columns
        # For each subset of size r from p*n columns, set W[c,t] = 1 if c == t_index, 0 otherwise
        # Then we need: for each t, U[:,t] and V[:,t] such that
        #   U[a,t]*V[b,t] = 1 if (a,b,c_t) in nz else 0, where c_t is the t-th selected column
        
        # More practically: try random sparse W patterns and solve

        # Method: Use a greedy approach
        # 1. Initialize W with random small rational entries
        # 2. Iterate: given W, solve for U and V (linear system)
        # 3. Check if solution is valid

        # Actually, let's use a more direct approach:
        # Fix W to be a random matrix with entries in {0, 1, -1, 1/2}
        # Then the equations sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c] are bilinear
        # We can linearize by fixing V and solving for U, or vice versa

        # Alternative: Use the fact that for small ranks, we can try all patterns
        # of W with entries in a small set

        # For now, try a simple approach:
        # Set W[c,t] = 1 if (c,t) is in some pattern, else 0
        # Try all patterns where W has exactly r ones arranged in a structured way

        # Actually, let's try a different approach:
        # Use the "alternating least squares" style but with exact arithmetic
        # and a structured initialization

        # Initialize W as identity-like: W[c,t] = 1 if c == t and t < min(r, p*n), else 0
        W = [[0]*r for _ in range(p*n)]
        for t in range(min(r, p*n)):
            W[t][t] = 1

        # Now solve for U and V
        # For each t where W has a 1, we need:
        #   U[a,t]*V[b,t] = T[a,b,c] for c = t (if t < p*n)
        # But T[a,b,c] = 1 only for specific (a,b) pairs
        # This is the same problem as before

        # Let's try a completely different approach:
        # Use a known construction for rectangular matrix multiplication
        # 
        # For multiplying an n×m matrix by an m×p matrix, one approach is:
        # Split the m dimension into blocks and use Strassen-like tricks
        # 
        # For m=4, we can split into two 2×2 blocks
        # A = [A11 A12; A21 A22] where each Aij is n×2
        # B = [B11 B12; B21 B22] where each Bij is 2×p
        # 
        # Then AB = [A11*B11 + A12*B21, A11*B12 + A12*B22;
        #            A21*B11 + A22*B21, A21*B12 + A22*B22]
        # 
        # This is 8 multiplications of n×2 by 2×p matrices
        # Each such multiplication has rank n*p (schoolbook) or less
        # So total rank is at most 8*n*p = 8*3*5 = 120, which is worse than 60
        # 
        # Not helpful. Let's try another approach.

        # Try to use a "bilinear" structure where we can share computations
        # 
        # For the tensor T[i*m+j, j*p+k, k*n+i] = 1:
        # We can think of this as: for each (i,k), the slice over j is:
        #   T[i*m+j, j*p+k, k*n+i] = 1 for j = 0..m-1
        # 
        # This is like a "diagonal" in the (a,b) plane for fixed (i,k)
        # 
        # One idea: use m separate rank-1 tensors for each j, but that's schoolbook
        # 
        # Another idea: use a "convolution" structure
        # 
        # Let's try to find a decomposition where U and V have a banded structure
        # and W is sparse

        # For now, return None if we can't find something better
        return None

    # Try each candidate rank
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            if verify(result['U'], result['V'], result['W']):
                if r < best_rank:
                    best = result
                    best_rank = r
                break

    return best
# EVOLVE-BLOCK-END
