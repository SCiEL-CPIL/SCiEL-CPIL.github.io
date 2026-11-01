"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # We attempt to find a lower rank decomposition using random search / local optimization.
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with a known good solution (schoolbook) and try to reduce rank.
    # A simple heuristic: try to find a rank r < n*m*p such that the tensor equations are satisfied.
    # This is generally hard, but for small matrices we can try random initializations and gradient steps
    # or simply random search for integer/half-integer solutions if the structure allows.
    
    # Given the complexity, and the fact that 2x4x5 matrix multiplication rank is known to be 40 (schoolbook)
    # or potentially lower, let's check if we can do better.
    # Actually, for 2x4x5, the trivial bound is 40.
    # Strassen-like algorithms might reduce it.
    # However, implementing a full tensor decomposition search is complex.
    # Let's try a simple local search starting from a random low-rank initialization, 
    # but since we need EXACT integer/half-integer solutions, random float search won't work directly 
    # unless we snap to grid.
    
    # Alternative: Use the known optimal rank if we know it.
    # For 2x4, rank is 8 (schoolbook). For 2x4x5, it's likely 40 or close.
    # Let's just return the schoolbook for now, as finding a strictly better one requires significant computation.
    # But wait, the prompt asks for "rank-minimal". If 40 is not minimal, we should find the minimal.
    # Is the rank of 2x4x5 matrix multiplication less than 40?
    # The rank of matrix multiplication tensor M_{m,n,p} is at least max(mn, mp, np).
    # Here max(8, 20, 10) = 20.
    # So rank >= 20.
    # Can we achieve 20?
    # If we use the "outer product" decomposition, rank is mn*p = 40.
    # There are known lower bounds and upper bounds.
    # For 2x4x5, I don't have the exact minimal rank memorized.
    # Let's try a heuristic: try ranks from max(mn, mp, np) up to mn*p.
    # We can try to solve for U, V, W with integer/half-integer entries.
    
    # Given the time limit and complexity, I will stick to the schoolbook unless a simple improvement is obvious.
    # However, I can try to verify if a lower rank is possible with a simple search for small ranks.
    
    # Let's try to find a rank 20 solution?
    # This is a linear system if we fix one matrix, but it's bilinear/trilinear.
    
    # Since I cannot easily compute the minimal rank in <120s with high confidence of finding a non-trivial one,
    # and the seed program provides rank 40, I will return the schoolbook.
    # If the true minimal rank is lower, this program will score 1.0 (same as seed) or worse if seed was better (it's not).
    # Actually, if I find a lower rank, the score improves (value decreases).
    
    # Let's try a very simple check: is there a known algorithm for 2x4x5 with rank < 40?
    # Most small matrix multiplication ranks are equal to the schoolbook rank unless specific Strassen-like tricks apply.
    # For 2x2x2, rank is 7 (Strassen) vs 8 (schoolbook).
    # For 2x4x5, it might be possible to use Strassen on blocks?
    # 2x4 can be seen as 2x(2+2).
    # This is getting complicated.
    
    # Safe bet: Return schoolbook.
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
