"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions for the matrices
    # A is n x m (4x4)
    # B is m x p (4x8)
    # C = A B is n x p (4x8)
    
    # We want to find rank r such that:
    # C_{i,k} = sum_{c=1}^r U_{i,j}^{(c)} V_{j,k}^{(c)}
    # Wait, the standard decomposition is:
    # C_{i,k} = sum_{j=1}^m A_{i,j} B_{j,k}
    # In tensor form, the multiplication tensor T has entries T[i,j,k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k].
    # Actually, the tensor for matrix multiplication C = AB is defined such that:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # This can be written as a sum of rank-1 tensors.
    # A rank-1 tensor is U_{i,j} * V_{j,k} * W_{i,k}.
    # The standard schoolbook algorithm has rank n*m*p.
    # Here n=4, m=4, p=8. Rank = 4*4*8 = 128.
    
    # The task asks for a rank-minimal exact tensor decomposition.
    # For 4x4 by 4x8, the best known bounds are likely around the schoolbook or slightly better if we exploit structure.
    # However, finding the *exact* minimal rank is a hard problem.
    # The prompt suggests using randomized tensor rank minimization or ALS.
    # But we need *exact* integer/half-integer entries.
    # Randomized methods usually produce floating point numbers.
    # We can try to find a low rank decomposition over rationals or half-integers.
    
    # Let's first check if we can do better than 128.
    # The multiplication of (4x4) by (4x8) is equivalent to 8 separate (4x4) by (4x1) multiplications.
    # Or 4 separate (4x4) by (4x8) ... no.
    # It's 8 columns of B, each is 4x1.
    # So we need to multiply a 4x4 matrix by 8 different 4x1 vectors.
    # The rank of the tensor for 4x4 * 4x1 is at most 16 (schoolbook).
    # Can we do 4x4 * 4x1 with rank < 16?
    # The minimal rank for 4x4 * 4x1 (matrix-vector multiplication) is known to be 16?
    # Actually, for matrix-vector multiplication A (n x m) * v (m x 1) = w (n x 1).
    # The tensor rank is at least max(n, m, n+m-1)? No.
    # For 2x2 * 2x1, rank is 4 (schoolbook). Can it be 3?
    # C_1 = a_11 v_1 + a_12 v_2
    # C_2 = a_21 v_1 + a_22 v_2
    # Rank 3? 
    # If we use Strassen-like ideas, maybe.
    # But for 4x4 * 4x1, the rank is likely 16.
    # If each column requires rank 16, and they are independent, the total rank might be 8*16 = 128.
    # However, the A matrix is shared across all 8 columns.
    # This sharing might allow for a lower rank decomposition of the whole 4x4 * 4x8 problem.
    
    # Let's try to construct a decomposition.
    # We can use the schoolbook algorithm as a baseline.
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (32, rank)
    # Note: W index is k*n + i.
    
    # Let's try to find a rank-128 solution first (schoolbook) to ensure we have a valid answer.
    # Then we can try to reduce it.
    
    # Schoolbook:
    # For each i in 0..3, j in 0..3, k in 0..7:
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    
    # This gives rank 128.
    
    # Can we do better?
    # Consider the structure.
    # C_{i,k} = sum_{j=0}^3 A_{i,j} B_{j,k}
    
    # Let's try a simple reduction.
    # If we group columns, maybe we can reuse computations?
    # But the tensor rank is a property of the whole bilinear map.
    
    # Let's assume the minimal rank is 128 for now, as proving a lower rank is difficult without a specific algorithm.
    # The prompt asks to "Find a rank-minimal exact tensor decomposition".
    # If we can't prove minimality, we should return the best we can find.
    # Schoolbook is a safe bet.
    
    # However, the prompt mentions "half-integer entries".
    # Schoolbook uses 0s and 1s, which are integers.
    
    # Let's implement the schoolbook decomposition.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists of lists as per the interface?
    # The interface says: 'U', 'V', 'W': factor matrices of shapes ...
    # The seed program returns lists of lists.
    # Let's stick to that to be safe, or numpy arrays?
    # The seed program:
    # U = [[0] * rank for _ in range(n * m)]
    # So it returns lists.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
