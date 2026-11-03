"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 48
    # We need to find if a lower rank decomposition exists.
    # For matrix multiplication, the rank is the tensor rank.
    # Known lower bounds and upper bounds for small cases:
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x3x3, rank is 16? No, schoolbook is 18. Actually, it's known to be 16.
    # For 2x4x6, let's try to find a decomposition.
    
    # Since we have a time limit of 120s, we can attempt a search for a lower rank.
    # However, finding a tensor decomposition is NP-hard in general.
    # For small dimensions, we can use linear algebra techniques.
    
    # The tensor T is in U^* \otimes V^* \otimes W^*, where U has dim n*m, V has dim m*p, W has dim p*n.
    # Wait, the indices are:
    # U[i*m + j, c] corresponds to input A[i, j]
    # V[j*p + k, c] corresponds to input B[j, k]
    # W[k*n + i, c] corresponds to output C[i, k]
    # The equation is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{i,i'} delta_{j,j'} delta_{k,k'} ?
    # No, the matrix multiplication is C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor is T[i, j, k] = 1 if it's a valid multiplication term?
    # The tensor decomposition is:
    # C[i, k] = sum_c (sum_j U[i*m+j, c] A[i, j]) (sum_j V[j*p+k, c] B[j, k]) * W[k*n+i, c]?
    # No, the standard form is:
    # C[i, k] = sum_c (sum_j U_c[i, j] A[i, j]) (sum_l V_c[j, l] B[j, l]) ...
    # Let's stick to the interface:
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    # The decomposition satisfies:
    # For all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if it's the identity map?
    # No, the tensor is the multiplication tensor.
    # T[i, j, k] = 1 for all i, j, k? No.
    # The multiplication tensor has entries T[i, j, k] where i is row of A, j is col of A / row of B, k is col of B.
    # Actually, the tensor is in (C^n x C^m)^* \otimes (C^m x C^p)^* \otimes (C^n x C^p)^*
    # The entries are T_{i, j, k} = 1 if the indices match for multiplication.
    # Specifically, C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # So the tensor is T = sum_{i, j, k} e_i \otimes e_j \otimes f_k, where f_k is the basis for the output.
    # Wait, the output index is (k, i) in the W matrix? No, W[k*n+i, c].
    # So the output index is (i, k) but stored as k*n+i.
    # The tensor is T = sum_{i, j, k} e_{i, j} \otimes e_{j, k} \otimes f_{k, i}
    # where e_{i, j} is the basis vector for A_{i, j}, e_{j, k} for B_{j, k}, and f_{k, i} for C_{i, k}.
    
    # We want to find rank r such that T = sum_{c=1}^r u_c \otimes v_c \otimes w_c
    # where u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}.
    # The entries are:
    # U[i*m+j, c] = u_c[i, j]
    # V[j*p+k, c] = v_c[j, k]
    # W[k*n+i, c] = w_c[k, i]
    
    # For n=2, m=4, p=6, the schoolbook rank is 2*4*6 = 48.
    # We can try to find a lower rank decomposition using known algorithms or search.
    # Given the time constraint, we will return the schoolbook decomposition, which is always valid.
    # If we have time, we could try to search for a lower rank, but it's risky.
    
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
