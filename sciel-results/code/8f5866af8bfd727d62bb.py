"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 3*4*8 = 96

    # The matrix multiplication tensor T[i,j,k] = e_{i,j} * e_{j,k} * e_{k,i}
    # We want to find rank r such that T = sum_{c=1}^r U_c * V_c * W_c
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # and T_{(i,j),(j,k),(k,i)} = sum_c U[(i,j),c] * V[(j,k),c] * W[(k,i),c]

    # Known lower bound: for n x m x p matrix multiplication, rank >= max(n*m, m*p, p*n, (n*m*p)^(1/2))
    # For 3x4x8: max(12, 32, 24, sqrt(96)) = 32
    # But we can do better with specific constructions.

    # Let's try to find a decomposition with rank < 96.
    # One approach: use the fact that we can combine some terms.

    # Strategy: Try random integer/half-integer matrices and check if they work.
    # But this is hard. Let's use a more structured approach.

    # Known result: For 3x4 matrix multiplication, the minimum rank is 11 (over reals).
    # For 4x8, it's more complex.

    # Let's try a different approach: use the "substitution" method.
    # If we have a decomposition for smaller matrices, we can combine them.

    # Actually, let's try to use a known construction for rectangular matrix multiplication.
    # For 3x4x8, we can think of it as (3x4) * (4x8).

    # Let's try to find a rank-minimal decomposition using a greedy approach.
    # We'll start with the schoolbook and try to merge terms.

    # Actually, let's use a different strategy: 
    # The tensor T has entries T[i,j,k] = 1 if i, j, k are compatible, else 0.
    # In our case, T is the standard matrix multiplication tensor.

    # Let's try to use a known result: the rank of the 3x4x8 matrix multiplication tensor.
    # This is not well-known, so let's try to find a good upper bound.

    # Approach: Use a random search with structured initialization.
    # We'll try ranks from a lower bound up to the schoolbook rank.

    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But this is not tight. A better lower bound is (n*m*p)^(1/2) = sqrt(96) ≈ 9.8, so >= 10.
    # Actually, the rank must be at least max(n, m, p) = 8, and also at least (n*m*p)^(1/2).

    # Let's try to find a decomposition with rank around 40-50.

    # Method: We'll use a randomized approach.
    # Initialize U, V, W with random small integers or half-integers.
    # Then check if U * V * W reconstructs T.
    # If not, adjust.

    # But this is hard to do efficiently. Let's use a more structured approach.

    # Actually, let's use the following construction:
    # For 3x4x8, we can use the fact that 4 = 2*2 and 8 = 2*4.
    # So we can decompose the 4x8 matrix into blocks.

    # Let's try a specific construction:
    # A = 3x4, B = 4x8.
    # Write A as [A1, A2] where A1 is 3x2, A2 is 3x2.
    # Write B as [B1; B2] where B1 is 2x8, B2 is 2x8.
    # Then A*B = A1*B1 + A2*B2.
    # Each term is 3x2 times 2x8, which has rank at most 3*2*8 = 48.
    # But we can do better.

    # Actually, let's just use a simple construction that we know works.
    # The schoolbook construction has rank 96.
    # Let's see if we can do better by sharing some computations.

    # For now, let's return the schoolbook construction, but let's try to improve it.

    # Actually, let's try a random search for a better rank.
    # We'll try ranks from 40 to 96 and see if we can find a valid decomposition.

    # But this is computationally expensive. Let's use a smarter approach.

    # Let's use the following idea: 
    # The matrix multiplication tensor for 3x4x8 can be decomposed using the 
    # "border rank" or "approximate rank" approach, but we need exact decomposition.

    # For now, let's return the schoolbook construction.
    # This is a safe fallback.

    rank = rank_schoolbook
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

    # Check if we have time to try to improve
    if time.time() < deadline - 10:
        # Try to find a better decomposition
        # We'll use a simple greedy approach
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
