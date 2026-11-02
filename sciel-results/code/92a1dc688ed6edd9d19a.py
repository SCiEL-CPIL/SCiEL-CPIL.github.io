"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 110  # safety margin before 120s limit
    
    schoolbook_rank = n * m * p  # 80
    
    def build_schoolbook():
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, schoolbook_rank
    
    def verify_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def to_nice_list(mat):
        """Convert numpy array to list of integers/half-integers."""
        result = []
        for row in mat:
            nice_row = []
            for val in row:
                rounded = round(val * 2) / 2.0
                nice_row.append(rounded)
            result.append(nice_row)
        return result
    
    # Winograd/Hadamard approach:
    # Decompose the 4x4x5 matrix multiplication using algebraic structure
    # Use block decomposition with Hadamard transforms to reduce rank
    
    def hadamard_4x4():
        """Construct 4x4 Hadamard matrix."""
        H = np.array([
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1]
        ], dtype=float)
        return H / 2.0  # Normalize so H @ H.T = I
    
    def try_winograd_decomposition():
        """
        Try to find a decomposition using Winograd-style factorization.
        
        The idea: partition the 4x4x5 multiplication into blocks and use
        transform matrices to reduce the number of scalar multiplications.
        
        For a 4x4 matrix times 4x5 matrix:
        - Partition into 2x2 blocks where possible
        - Use Hadamard transforms on the block structure
        """
        H = hadamard_4x4()
        
        # Strategy: Use tensor product structure
        # C = A * B where A is 4x4, B is 4x5
        # Partition A into 2x2 blocks of 2x2 matrices
        # Partition B into 2x2 blocks of 2x5 matrices
        
        # For now, try a structured approach based on Kronecker products
        # of smaller Hadamard transforms
        
        # Base case: 2x2 matrix multiplication has rank 7 (Strassen)
        # But we need exact integer/half-integer entries
        
        # Try rank 48: use 6 "virtual" multiplications per 2x2 block
        # 4x4x5 = (2x2x4) * (2x2x5) with block structure
        
        # Construct using tensor products
        # U = I_2 ⊗ H_2 type structure
        
        H2 = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2)
        
        # Build factor matrices using Kronecker structure
        # For 4x4x5, we can use rank = 4*4*5 - savings
        
        # Let's try a specific construction:
        # Use the fact that we can share some computations
        
        # Approach: Build a decomposition with rank 64
        # by using 2x2 block structure with Strassen-like savings
        
        rank = 64
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Index mapping for 2x2 block structure
        # A is 4x4, partition into blocks A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
        # B is 4x5, partition into blocks B[0:2,0:5], B[2:4,0:5]
        # C is 4x5, partition into blocks C[0:2,0:5], C[2:4,0:5]
        
        # For each 2x2 block of A times 2x5 block of B, we need 2x5 outputs
        # Using Strassen on 2x2: 7 multiplications per 2x2x2 sub-block
        # But we have 2x5, so we need more
        
        # Simpler approach: just use schoolbook but try to find a better structured rank
        # For 4x4x5, known lower bounds suggest rank around 48-56 is possible
        
        # Let's try a direct construction with rank 48
        rank = 48
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Use a structured pattern based on modular arithmetic
        # This is a heuristic construction that may or may not work
        
        # For each (i,j,k) in the output tensor, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Try assigning each (i,j,k) to a unique rank index with some sharing
        # Total (i,j,k) combinations: 4*4*5 = 80
        # We want to reduce to 48 by sharing
        
        # Pattern: share when i and k are related
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
                    else:
                        # Need to share - this simple approach won't work
                        # Fall back to schoolbook
                        return None
        
        # This simple construction only works if rank >= 80
        return None
    
    # First try the Winograd approach
    result = try_winograd_decomposition()
    
    if result is not None:
        U, V, W, rank = result
        if verify_decomposition(U, V, W, rank):
            U_list = to_nice_list(U)
            V_list = to_nice_list(V)
            W_list = to_nice_list(W)
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    U, V, W, rank = build_schoolbook()
    
    U_list = to_nice_list(U)
    V_list = to_nice_list(V)
    W_list = to_nice_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
