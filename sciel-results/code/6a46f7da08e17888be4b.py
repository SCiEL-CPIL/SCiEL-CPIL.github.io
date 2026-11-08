"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with the schoolbook decomposition as a safe baseline
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using structured ansatz
    # The matrix multiplication tensor T_{ijk} = 1 for i in [0,n), j in [0,m), k in [0,p)
    # where the bilinear map is C_{ik} = sum_j A_{ij} B_{jk}
    # We want to express this as sum_r U_{i,r} V_{j,r} W_{k,r} ... wait, the tensor indices are different.
    
    # The tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
    # T_{i, j, k} where i is row of A (and C), j is col of A / row of B, k is col of B (and C)
    # Actually, the standard tensor is T_{i, j, k} = 1 if we consider the product C_{ik} = sum_j A_{ij} B_{jk}
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to it.
    # A rank-1 tensor is u_i * v_j * w_k.
    # So we need sum_r u_{i,r} v_{j,r} w_{k,r} = T_{i,j,k}
    
    # But the interface asks for U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # And the decomposition is sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} = ?
    # Wait, the trace convention W[k*n+i, r] suggests the output index is (k, i).
    # Let's re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # This usually means the tensor is T_{i, j, k} and the decomposition is:
    # T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}
    # Where U is flattened to (n*m, rank), V to (m*p, rank), W to (p*n, rank).
    
    # The schoolbook method uses rank n*m*p.
    # Can we do better?
    # For small matrices, the border rank is often lower than the rank.
    # However, we need an EXACT decomposition with integer/half-integer entries.
    
    # Let's try a few known low-rank decompositions if they exist for these sizes.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x4x7, we can potentially use block structures.
    
    # Strategy: Use a Kronecker product structure.
    # A (2x4) = A1 (2x2) @ [I 0; 0 I] ... no, that's not right.
    # We can view 4 = 2*2 and 7 as 4+3 or similar.
    
    # Let's try to construct a decomposition using Strassen-like blocks.
    # Split A into A_left (2x2) and A_right (2x2)
    # Split B into B_left (2x4) and B_right (2x3) ... this is getting complicated.
    
    # Given the time constraint and the difficulty of finding a provably optimal low-rank exact decomposition for 2x4x7 with integer/half-integer entries,
    # and the fact that the current score is 0.0 (invalid or timeout), the safest bet is to return a valid schoolbook decomposition quickly.
    # The schoolbook rank is 56.
    
    # Let's double check if we can do a simple improvement.
    # If we use Strassen for the 2x2 blocks:
    # We can treat the 2x4 matrix as two 2x2 blocks.
    # A = [A1 A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x4? No, B is 4x7.
    # Let B = [B_top; B_bottom] where B_top is 2x7 and B_bottom is 2x7.
    # Then C = A1 * B_top + A2 * B_bottom.
    # This is a sum of two 2x2 * 2x7 products.
    # The rank of a 2x2 * 2x7 product?
    # We can further decompose 2x7 into 2x4 + 2x3.
    # This leads to a recursive structure.
    
    # However, ensuring exactness with integer/half-integer entries is tricky.
    # Strassen's algorithm uses coefficients 1, -1, 0.5, etc.
    # Let's try to implement a Strassen-based decomposition for the 2x2 part and see if it generalizes.
    
    # For now, to ensure a non-zero score, we return the schoolbook decomposition.
    # It is valid and fast.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
