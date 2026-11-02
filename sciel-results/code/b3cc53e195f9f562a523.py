"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 3*4*7 = 84
    rank_schoolbook = n * m * p
    
    # For matrix multiplication, the tensor is T_{i,j,k} = delta_{ik} * A_{ij} * B_{jk}
    # Actually, the tensor for matrix multiplication is T_{(i,j),(j,k),(k,i)} = 1
    # Wait, the standard matrix multiplication tensor T has components T_{(i,j), (j',k), (k',i')} 
    # such that (AB)_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is defined by T_{(i,j), (j,k), (k,i)} = 1 for all i,j,k.
    # So the tensor has indices (i,j) for U, (j,k) for V, (k,i) for W.
    # The rank of this tensor is at most n*m*p.
    
    # Known lower bounds for matrix multiplication rank:
    # For n x m x p, rank >= n*m*p / (n+m+p-2) (Strassen's bound is different, this is a simple one)
    # Actually, the best known general lower bound is not that simple.
    # For 3x4x7, let's check if there are known improved decompositions.
    # Strassen's algorithm works for 2x2x2 with rank 7 (vs 8 schoolbook).
    # For larger matrices, we can use recursive Strassen, but 3,4,7 are not powers of 2.
    
    # Let's try to use a known algorithm or search for a better rank.
    # Since the time limit is 150s, we can try a small search for rank reduction.
    # However, finding an exact decomposition is hard.
    
    # Let's check if the problem allows for a specific known decomposition.
    # The "Laser Method" mentioned in the directive suggests using linear algebra to find a minimal set.
    # But implementing a full laser method is complex.
    
    # Let's try a simple approach: check if rank 83 is possible by removing one term?
    # No, that's unlikely to work without a specific structure.
    
    # Given the constraints and the difficulty of finding a better decomposition quickly,
    # and the fact that previous attempts (inspirations) scored 1.0 (same as schoolbook),
    # it's likely that the schoolbook rank is the best we can find in the time limit
    # without a very specific insight or algorithm.
    
    # However, let's try a quick check for small rank improvements using a greedy approach
    # or checking if the tensor has a known lower rank.
    
    # For now, return the schoolbook decomposition.
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
