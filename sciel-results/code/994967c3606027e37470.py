"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # We'll use a block decomposition:
    # Split 5x5 into 3x3 + 2x2 blocks:
    # A = [[A11(3x3), A12(3x2)],
    #      [A21(2x3), A22(2x2)]]
    # B = [[B11(3x3), B12(3x2)],
    #      [B21(2x3), B22(2x2)]]
    # C = [[C11(3x3), C12(3x2)],
    #      [C21(2x3), C22(2x2)]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # For the 3x3*3x3 product, we'll use Laderman's rank-15 algorithm.
    # For the other products, we'll use schoolbook.
    #
    # Rank counts:
    # C11: 15 (Laderman) + 18 (3x2*2x3 schoolbook) = 33
    # C12: 18 (3x3*3x2 schoolbook) + 12 (3x2*2x2 schoolbook) = 30
    # C21: 18 (2x3*3x3 schoolbook) + 12 (2x2*2x3 schoolbook) = 30
    # C22: 12 (2x3*3x2 schoolbook) + 4 (2x2*2x2 schoolbook) = 16
    # Total: 33 + 30 + 30 + 16 = 109
    #
    # This is better than 125. Let's implement it.

    def laderman_3x3(A, B):
        """
        Laderman's 15-multiplication algorithm for 3x3 matrix multiplication.
        Returns C = A @ B.
        """
        # A and B are 3x3 numpy arrays
        # Flatten to 1D for easier indexing
        a = A.flatten()
        b = B.flatten()
        
        # 15 multiplications
        # Using the formulation from Laderman (1976)
        # M1 = (a0+a1+a2)*(b0+b1+b2)
        # M2 = (a0-a1+a2)*(b0-b1+b2)
        # M3 = (a0+a1-a2)*(b0+b1-b2)
        # M4 = (a0-a1-a2)*(b0-b1-b2)
        # M5 = (a0+a2)*(b0+b2)
        # M6 = (a1+a2)*(b1+b2)
        # M7 = (a0+a1)*(b0+b1)
        # M8 = a0*b0
        # M9 = a1*b1
        # M10 = a2*b2
        # M11 = (a0+a2-b1)*(b0+b2-b1)
        # M12 = (a1+a0-b2)*(b1+b0-b2)
        # M13 = (a2+a1-b0)*(b2+b1-b0)
        # M14 = (a0+a1+a2-b0-b1-b2)*(b0+b1+b2-a0-a1-a2)  # This doesn't seem right
        # 
        # Let me use a different, verified formulation.
        # Actually, let me just use schoolbook for now to ensure correctness.
        # The Laderman algorithm is complex and error-prone.
        
        # For now, use schoolbook
        C = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[i, j] += A[i, k] * B[k, j]
        return C

    # Build the decomposition
    # We'll create U, V, W matrices
    # U has shape (n*m, rank) = (25, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (25, rank)
    
    # First, let's count the total rank
    # C11: 15 + 18 = 33
    # C12: 18 + 12 = 30
    # C21: 18 + 12 = 30
    # C22: 12 + 4 = 16
    # Total: 109
    
    total_rank = 109
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper function to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1
    
    # Block indices:
    # A11: rows 0-2, cols 0-2 -> U indices 0-8
    # A12: rows 0-2, cols 3-4 -> U indices 9-14
    # A21: rows 3-4, cols 0-2 -> U indices 15-20
    # A22: rows 3-4, cols 3-4 -> U indices 21-24
    #
    # B11: rows 0-2, cols 0-2 -> V indices 0-8
    # B12: rows 0-2, cols 3-4 -> V indices 9-14
    # B21: rows 3-4, cols 0-2 -> V indices 15-20
    # B22: rows 3-4, cols 3-4 -> V indices 21-24
    #
    # C11: rows 0-2, cols 0-2 -> W indices 0-8
    # C12: rows 0-2, cols 3-4 -> W indices 9-14
    # C21: rows 3-4, cols 0-2 -> W indices 15-20
    # C22: rows 3-4, cols 3-4 -> W indices 21-24

    # C11 = A11*B11 (Laderman, rank 15) + A12*B21 (schoolbook, rank 18)
    # Laderman for A11*B11:
    # A11 indices: 0-8, B11 indices: 0-8, C11 indices: 0-8
    
    # For simplicity, let's just use schoolbook for everything for now
    # to ensure correctness, then optimize.
    
    # Actually, let me implement Laderman properly.
    # The 15 multiplications for 3x3:
    # Using a known formulation:
    
    # Let me use a different approach: just use schoolbook for all blocks
    # This gives rank 125, same as baseline.
    
    # For now, return schoolbook
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

# EVOLVE-BLOCK-END
