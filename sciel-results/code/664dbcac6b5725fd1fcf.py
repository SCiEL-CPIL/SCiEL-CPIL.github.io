"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Returns U (16 x rank), V (32 x rank), W (32 x rank) with integer/half-integer entries.
    Score is the rank.
    """
    n, m, p = DIMENSIONS
    
    # Known best rank for 4x4 * 4x4 is 25 (Stothers).
    # For 4x4 * 4x8, we can potentially use a decomposition that respects the structure.
    # However, 4x4 * 4x8 is equivalent to two 4x4 * 4x4 multiplications if we view it as [A] * [B1 B2].
    # The rank of a direct sum is the sum of ranks, so 25+25=50.
    # But maybe there's a better way.
    
    # Let's try to find a small rank decomposition using a local search or known algorithms.
    # Since we have a time limit, and finding the exact minimum rank for arbitrary dimensions is hard,
    # we will start with a known good upper bound and try to improve it if time permits.
    # 
    # For 4x4 * 4x4, rank 25 is known.
    # For 4x4 * 4x8, if we consider the tensor as T in V* x V* x V** where dim(V*)=16, dim(V**)=16, dim(V***)=32? No.
    # U is 16 x r, V is 32 x r, W is 32 x r.
    # The tensor is in (R^16)* x (R^32)* x (R^32)*.
    # Actually, the multiplication tensor T_{ijk} where i in 0..15, j in 0..31, k in 0..31.
    # T_{(i1,i2), (j1,j2,k), ...}
    
    # Let's use the schoolbook decomposition as a baseline, which is rank 4*4*8 = 128.
    # Rank 128 is safe.
    
    # Can we do better?
    # Strassen's algorithm for 2x2 * 2x2 has rank 7.
    # For 4x4 * 4x4, using Strassen recursively gives 7^3 = 343? No, 2^3=8, so 7^3=343 is wrong.
    # 4x4 * 4x4 using Strassen: divide into 2x2 blocks. 7 multiplications of 2x2 * 2x2.
    # Each 2x2 * 2x2 has rank 7. So total rank 7*7 = 49? No, the operations are linear combinations.
    # The rank of the tensor for 4x4 * 4x4 using Strassen is 7 * (rank of 2x2 * 2x2) = 7*7=49? 
    # Wait, the rank of the 4x4 * 4x4 tensor is at most 7 * 7 = 49? No.
    # The rank of a block tensor is not simply the product.
    # Actually, Stothers found rank 25 for 4x4 * 4x4.
    
    # Let's try to construct a rank 25 decomposition for 4x4 * 4x4 and then adapt it.
    # But we don't have the explicit 25-rank decomposition in memory.
    
    # Alternative: Use a local search to minimize the rank.
    # Given the time constraint (120s), and the complexity of finding a minimal rank decomposition,
    # it might be safer to return a known valid decomposition, even if not minimal.
    # 
    # However, the task asks for "rank-minimal". If we return 128, and the minimum is 25, we get a low score.
    # But if we can find a better one, we get a higher score.
    
    # Let's try a simple local search approach.
    # We start with a random initialization of U, V, W with rank R.
    # We try to decrease R by finding dependencies.
    
    # For now, let's return the schoolbook decomposition of rank 128.
    # This is a valid decomposition.
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
# EVOLVE-BLOCK-END
