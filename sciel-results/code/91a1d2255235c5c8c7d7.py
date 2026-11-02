"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    Uses a hybrid decomposition: Schoolbook for most terms, but Strassen for two disjoint
    2x2x2 sub-blocks to reduce the rank from 90 to 88.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Initialize with Schoolbook rank 90
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    # Store indices of schoolbook terms to potentially replace
    schoolbook_indices = {}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                schoolbook_indices[(i, j, k)] = column
                column += 1
                
    # Define Strassen blocks to replace
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {1,2}, j in {3,4}, k in {3,4}
    
    blocks = [
        {'i': [0, 1], 'j': [0, 1], 'k': [0, 1]},
        {'i': [1, 2], 'j': [3, 4], 'k': [3, 4]}
    ]
    
    # We will overwrite the specific columns corresponding to these blocks with Strassen terms.
    # Since Strassen has 7 terms and Schoolbook has 8 for a 2x2x2 block, we can reuse 7 columns
    # and leave the 8th column as zeros (or merge it, but reusing is easier).
    # Actually, we just need to ensure the total rank is minimized.
    # Current Rank 90.
    # We replace 8 terms with 7 terms in Block 1.
    # We replace 8 terms with 7 terms in Block 2.
    # Total Rank 90 - 8 + 7 - 8 + 7 = 88.
    
    # We need to identify the 8 columns for each block and replace them.
    
    for block in blocks:
        i1, i2 = block['i']
        j1, j2 = block['j']
        k1, k2 = block['k']
        
        # Get the 8 schoolbook column indices for this block
        cols_to_replace = []
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in [k1, k2]:
                    cols_to_replace.append(schoolbook_indices[(i, j, k)])
        
        # We will use the first 7 of these columns for Strassen terms.
        # The 8th column will be set to 0.
        strassen_cols = cols_to_replace[:7]
        zero_col = cols_to_replace[7]
        
        # Strassen factors
        # Local indices 0,1 map to global i1,i2 etc.
        
        # U factors (A)
        # M1: A[i1,j1] + A[i2,j2]
        U[i1 * m + j1, strassen_cols[0]] = 1
        U[i2 * m + j2, strassen_cols[0]] = 1
        
        # M2: A[i2,j1] + A[i2,j2]
        U[i2 * m + j1, strassen_cols[1]] = 1
        U[i2 * m + j2, strassen_cols[1]] = 1
        
        # M3: A[i1,j1]
        U[i1 * m + j1, strassen_cols[2]] = 1
        
        # M4: A[i2,j2]
        U[i2 * m + j2, strassen_cols[3]] = 1
        
        # M5: A[i1,j1] + A[i1,j2]
        U[i1 * m + j1, strassen_cols[4]] = 1
        U[i1 * m + j2, strassen_cols[4]] = 1
        
        # M6: A[i2,j1] - A[i1,j1]
        U[i2 * m + j1, strassen_cols[5]] = 1
        U[i1 * m + j1, strassen_cols[5]] = -1
        
        # M7: A[i1,j2] - A[i2,j2]
        U[i1 * m + j2, strassen_cols[6]] = 1
        U[i2 * m + j2, strassen_cols[6]] = -1
        
        # V factors (B)
        # M1: B[j1,k1] + B[j2,k2]
        V[j1 * p + k1, strassen_cols[0]] = 1
        V[j2 * p + k2, strassen_cols[0]] = 1
        
        # M2: B[j1,k1]
        V[j1 * p + k1, strassen_cols[1]] = 1
        
        # M3: B[j2,k1] - B[j2,k2]
        V[j2 * p + k1, strassen_cols[2]] = 1
        V[j2 * p + k2, strassen_cols[2]] = -1
        
        # M4: B[j1,k2] - B[j1,k1]
        V[j1 * p + k2, strassen_cols[3]] = 1
        V[j1 * p + k1, strassen_cols[3]] = -1
        
        # M5: B[j2,k2]
        V[j2 * p + k2, strassen_cols[4]] = 1
        
        # M6: B[j1,k1] + B[j1,k2]
        V[j1 * p + k1, strassen_cols[5]] = 1
        V[j1 * p + k2, strassen_cols[5]] = 1
        
        # M7: B[j2,k1] + B[j2,k2]
        V[j2 * p + k1, strassen_cols[6]] = 1
        V[j2 * p + k2, strassen_cols[6]] = 1
        
        # W factors (C)
        # C indices: k * n + i
        # M1: C[i1,k1] + C[i2,k2]
        W[k1 * n + i1, strassen_cols[0]] = 1
        W[k2 * n + i2, strassen_cols[0]] = 1
        
        # M2: C[i2,k1] - C[i2,k2]
        W[k1 * n + i2, strassen_cols[1]] = 1
        W[k2 * n + i2, strassen_cols[1]] = -1
        
        # M3: C[i1,k2] + C[i2,k2]
        W[k2 * n + i1, strassen_cols[2]] = 1
        W[k2 * n + i2, strassen_cols[2]] = 1
        
        # M4: C[i1,k1] + C[i2,k1]
        W[k1 * n + i1, strassen_cols[3]] = 1
        W[k1 * n + i2, strassen_cols[3]] = 1
        
        # M5: -C[i1,k1] + C[i1,k2]
        W[k1 * n + i1, strassen_cols[4]] = -1
        W[k2 * n + i1, strassen_cols[4]] = 1
        
        # M6: C[i2,k2]
        W[k2 * n + i2, strassen_cols[5]] = 1
        
        # M7: C[i1,k1]
        W[k1 * n + i1, strassen_cols[6]] = 1
        
        # Zero out the 8th column
        U[:, zero_col] = 0
        V[:, zero_col] = 0
        W[:, zero_col] = 0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
