"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    We use the block-diagonal decomposition:
      - Split the 3x10 matrix into a 2x10 block and a 1x10 block.
      - The 2x10 part is further split into five 2x2 blocks, each using Strassen (rank 7).
      - The 1x10 part uses schoolbook (rank 10).
    Total rank: 5*7 + 10 = 45.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    # We split m=3 into m1=2 and m2=1
    m1 = 2
    m2 = 1
    # Split p=10 into 5 blocks of size 2
    p_blocks = 5
    p_block_size = 2
    
    # Strassen for 2x2 matrices: rank 7
    # The 7 products are:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B22-B12)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each 2x2 block of the 2x10 part, we need 7 rank-1 terms.
    # For the 1x10 part, we need 10 rank-1 terms.
    # Total rank = 5*7 + 10 = 45.
    
    rank = 5 * 7 + 10
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Process the 2x10 part (rows 0,1 of A; rows 0,1 of B; cols 0-9 of C)
    for block in range(p_blocks):
        k_start = block * p_block_size
        # Strassen terms for this 2x2 block
        # M1 = (A11+A22)*(B11+B22)
        U[0, col] = 1; U[1, col] = 1  # A row 0 (i=0,j=0), A row 1 (i=0,j=1)
        V[0, col] = 1; V[2, col] = 1  # B row 0 (j=0,k=0), B row 2 (j=1,k=0)
        W[k_start * n + 0, col] = 1; W[k_start * n + 1, col] = 1  # C col k=0, i=0,1
        col += 1
        
        # M2 = (A21+A22)*B11
        U[1, col] = 1; U[2, col] = 1  # A row 1 (i=0,j=1), A row 2 (i=1,j=0)
        V[0, col] = 1  # B row 0 (j=0,k=0)
        W[k_start * n + 1, col] = 1  # C col k=0, i=1
        col += 1
        
        # M3 = A11*(B11-B21)
        U[0, col] = 1  # A row 0 (i=0,j=0)
        V[0, col] = 1; V[1, col] = -1  # B row 0 (j=0,k=0), B row 1 (j=0,k=1)
        W[k_start * n + 0, col] = 1; W[(k_start+1) * n + 0, col] = 1  # C col k=0,1, i=0
        col += 1
        
        # M4 = A22*(B22-B12)
        U[2, col] = 1  # A row 2 (i=1,j=0)
        V[2, col] = 1; V[1, col] = -1  # B row 2 (j=1,k=0), B row 1 (j=0,k=1)
        W[k_start * n + 1, col] = 1; W[(k_start+1) * n + 1, col] = 1  # C col k=0,1, i=1
        col += 1
        
        # M5 = (A11+A12)*B22
        U[0, col] = 1; U[1, col] = 1  # A row 0 (i=0,j=0), A row 1 (i=0,j=1)
        V[2, col] = 1  # B row 2 (j=1,k=0)
        W[k_start * n + 0, col] = -1; W[(k_start+1) * n + 0, col] = 1  # C col k=0,1, i=0
        col += 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[1, col] = 1; U[0, col] = -1  # A row 1 (i=0,j=1), A row 0 (i=0,j=0)
        V[0, col] = 1; V[1, col] = 1  # B row 0 (j=0,k=0), B row 1 (j=0,k=1)
        W[k_start * n + 1, col] = 1; W[(k_start+1) * n + 1, col] = 1  # C col k=0,1, i=1
        col += 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[1, col] = 1; U[2, col] = -1  # A row 1 (i=0,j=1), A row 2 (i=1,j=0)
        V[1, col] = 1; V[2, col] = 1  # B row 1 (j=0,k=1), B row 2 (j=1,k=0)
        W[k_start * n + 0, col] = 1; W[k_start * n + 1, col] = -1  # C col k=0, i=0,1
        col += 1
    
    # Process the 1x10 part (row 2 of A; row 2 of B; all cols of C)
    for k in range(p):
        U[2, col] = 1  # A row 2 (i=1,j=0)
        V[2 * p + k, col] = 1  # B row 2 (j=2,k)
        W[k * n + 0, col] = 1; W[k * n + 1, col] = 1  # C col k, i=0,1
        col += 1
    
    # Convert to list of lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
