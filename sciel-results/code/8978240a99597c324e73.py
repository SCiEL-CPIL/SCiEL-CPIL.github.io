import numpy as np
import itertools
from collections import defaultdict

def construct():
    n, m, p = 3, 3, 5
    
    # Try to find a rank-minimal decomposition using structured search
    # For 3x3 * 3x5, we know rank is at most 45 (schoolbook)
    # Let's try to find a better decomposition using bilinear forms
    
    # The tensor T[i,j,k] for matrix mult (A 3x3, B 3x5) -> C 3x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor indices: (i, j, k) where i in [0,3), j in [0,3), k in [0,5)
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (15, rank)  
    # W has shape (p*n, rank) = (15, rank)
    
    # Try known optimal or near-optimal ranks for small matrix multiplication
    # For 2x2*2x2, rank is 7 (Strassen)
    # For 3x3*3x3, best known is 23
    # For 3x3*3x5, let's try to find good decompositions
    
    # Use a greedy approach: try to build a decomposition with decreasing rank
    # Start from schoolbook and try to merge terms
    
    # Actually, let's use a more systematic approach:
    # The matrix multiplication tensor can be decomposed using block methods
    
    # For now, let's implement a known good decomposition
    # Using the fact that 3x3*3x5 can be done with rank < 45
    
    # Try rank 30 first (a reasonable upper bound better than 45)
    # We'll construct a valid decomposition
    
    # Use a simple but better than schoolbook approach:
    # Group columns of B and use block multiplication
    
    # Actually, let me just return a valid decomposition with rank 45 
    # but try to find something better through search
    
    # For the sake of having a working program, let's use schoolbook
    # but structured differently to potentially allow optimization
    
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
