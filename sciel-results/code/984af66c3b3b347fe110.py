"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 3x4 by 4x7.
    Uses Strassen's algorithm on two disjoint 2x2x2 sub-blocks to reduce rank from 84 to 82.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit

    n, m, p = DIMENSIONS  # 3, 4, 7

    # We'll build the decomposition manually.
    # Start with schoolbook (rank 84), then replace two 2x2x2 blocks with Strassen (rank 7 each, saving 1 each).
    
    # Total rank = 84 - 2 = 82
    rank = 82
    
    # U: (n*m, rank) = (12, 82)
    # V: (m*p, rank) = (28, 82)
    # W: (p*n, rank) = (21, 82)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # We need to carefully assign indices.
    # Let's define the two Strassen blocks:
    # Block A: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2
    # Block B: i in {0,1}, j in {2,3}, k in {0,1} -> 2x2x2
    
    # Wait, the problem says 3x4 * 4x7. So i in [0,2], j in [0,3], k in [0,6].
    # Let's pick two disjoint 2x2x2 blocks that don't overlap.
    
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    # Block 2: i={0,1}, j={2,3}, k={2,3}
    
    # These are disjoint in the j and k dimensions.
    
    # For each block, we'll use 7 terms (Strassen) instead of 8 terms (schoolbook).
    # The remaining 84 - 8 - 8 = 68 terms will be schoolbook.
    # Total rank = 7 + 7 + 68 = 82.
    
    # Let's build this step by step.
    
    # Helper function to add a schoolbook term
    def add_schoolbook(i, j, k, col):
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
    
    # Helper function to add Strassen terms for a 2x2x2 block
    # i0, i1: row indices (2 values)
    # j0, j1: column indices of A / row indices of B (2 values)
    # k0, k1: column indices of B / column indices of C (2 values)
    def add_strassen(i0, i1, j0, j1, k0, k1, base_col):
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        # M3 = A[i0,j0] * (B[j1,k0] - B[j1,k1])
        # M4 = A[i1,j1] * (B[j0,k1] - B[j0,k0])
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        
        # C[i0,k0] = M1 + M4 - M5 - M7
        # C[i0,k1] = M3 + M5
        # C[i1,k0] = M2 + M4
        # C[i1,k1] = M1 - M2 + M3 + M7
        
        # M1
        U[i0 * m + j0, base_col + 0] = 1
        U[i1 * m + j1, base_col + 0] = 1
        V[j0 * p + k0, base_col + 0] = 1
        V[j1 * p + k1, base_col + 0] = 1
        W[k0 * n + i0, base_col + 0] = 1
        W[k1 * n + i1, base_col + 0] = 1
        
        # M2
        U[i1 * m + j0, base_col + 1] = 1
        U[i1 * m + j1, base_col + 1] = 1
        V[j0 * p + k0, base_col + 1] = 1
        W[k0 * n + i1, base_col + 1] = 1
        
        # M3
        U[i0 * m + j0, base_col + 2] = 1
        V[j1 * p + k0, base_col + 2] = 1
        V[j1 * p + k1, base_col + 2] = -1
        W[k0 * n + i0, base_col + 2] = 1
        
        # M4
        U[i1 * m + j1, base_col + 3] = 1
        V[j0 * p + k1, base_col + 3] = 1
        V[j0 * p + k0, base_col + 3] = -1
        W[k0 * n + i1, base_col + 3] = 1
        W[k1 * n + i0, base_col + 3] = 1
        
        # M5
        U[i0 * m + j0, base_col + 4] = 1
        U[i0 * m + j1, base_col + 4] = 1
        V[j1 * p + k1, base_col + 4] = 1
        W[k1 * n + i0, base_col + 4] = -1
        W[k1 * n + i1, base_col + 4] = 1
        
        # M6
        U[i1 * m + j0, base_col + 5] = 1
        U[i0 * m + j0, base_col + 5] = -1
        V[j0 * p + k0, base_col + 5] = 1
        V[j0 * p + k1, base_col + 5] = 1
        W[k0 * n + i1, base_col + 5] = 1
        
        # M7
        U[i0 * m + j1, base_col + 6] = 1
        U[i1 * m + j1, base_col + 6] = -1
        V[j1 * p + k0, base_col + 6] = 1
        V[j1 * p + k1, base_col + 6] = 1
        W[k1 * n + i0, base_col + 6] = -1
        W[k0 * n + i1, base_col + 6] = 1
    
    # Now let's assign the columns.
    # Block 1: i={0,1}, j={0,1}, k={0,1} -> uses 7 columns (0-6)
    # Block 2: i={0,1}, j={2,3}, k={2,3} -> uses 7 columns (7-13)
    # Remaining schoolbook terms: 68 terms -> columns 14-81
    
    col = 0
    
    # Add Block 1 (Strassen)
    add_strassen(0, 1, 0, 1, 0, 1, col)
    col += 7
    
    # Add Block 2 (Strassen)
    add_strassen(0, 1, 2, 3, 2, 3, col)
    col += 7
    
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this term is in Block 1 or Block 2
                in_block1 = (i in (0, 1)) and (j in (0, 1)) and (k in (0, 1))
                in_block2 = (i in (0, 1)) and (j in (2, 3)) and (k in (2, 3))
                
                if not in_block1 and not in_block2:
                    add_schoolbook(i, j, k, col)
                    col += 1
    
    # Check if we used exactly 82 columns
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Convert to lists for the required format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
