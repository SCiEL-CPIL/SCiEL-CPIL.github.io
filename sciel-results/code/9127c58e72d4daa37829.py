import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialize Schoolbook Decomposition
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Indices:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    rank0 = n * m * p
    U = [[0] * rank0 for _ in range(n * m)]
    V = [[0] * rank0 for _ in range(m * p)]
    W = [[0] * rank0 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Convert to list of columns for easier manipulation
    # U_cols[c][row], V_cols[c][row], W_cols[c][row]
    U_cols = [[U[r][c] for r in range(n*m)] for c in range(rank0)]
    V_cols = [[V[r][c] for r in range(m*p)] for c in range(rank0)]
    W_cols = [[W[r][c] for r in range(p*n)] for c in range(rank0)]
    
    current_rank = rank0
    
    # 2. Try to reduce rank by finding linear dependencies
    # We look for a set of columns that are linearly dependent.
    # Specifically, if column c is a linear combination of others, we can remove it.
    # This is equivalent to reducing the rank of the tensor.
    
    # To be efficient and exact, we use a greedy approach:
    # Iterate through columns. For each column, check if it can be removed.
    # A column c can be removed if the tensor equation holds without it,
    # which implies that the "missing" term U_c \otimes V_c \otimes W_c
    # must be zero? No, that's not right.
    #
    # Actually, if we have a decomposition T = sum U_r \otimes V_r \otimes W_r,
    # and we want to reduce rank, we need to find a different set of vectors.
    #
    # A simpler valid improvement for small tensors:
    # Check for "trivial" reductions where a factor is zero (none in schoolbook).
    # 
    # Let's try a different valid reduction strategy:
    # If the tensor allows for a Strassen-like decomposition in any 2x2x2 sub-block,
    # we can replace 8 terms with 7.
    #
    # Given the complexity of general search, and the "Refinement" directive,
    # let's implement a strict check for 2x2x2 Strassen substitutions.
    # This is a "family" improvement: optimizing the structure.
    
    # Identify 2x2x2 blocks in the indices (i,j,k).
    # A block is defined by pairs of indices (i1, i2), (j1, j2), (k1, k2).
    # We look for 4 such pairs in i, 4 in j, 4 in k? No, Strassen is 2x2x2.
    # We need i1, i2; j1, j2; k1, k2.
    # The 8 terms are (i1,j1,k1), (i1,j1,k2), (i1,j2,k1), (i1,j2,k2),
    #                 (i2,j1,k1), (i2,j1,k2), (i2,j2,k1), (i2,j2,k2).
    
    # We will iterate through all possible 2x2x2 sub-tensors.
    # If we find one where the 8 corresponding columns in U, V, W form a "standard"
    # pattern (i.e., only these 8 columns are non-zero for these specific row/col indices),
    # we can replace them with 7 Strassen terms.
    
    # However, in the schoolbook, each column is unique.
    # To apply Strassen, we need to *merge* or *recombine*.
    # This is difficult to do in-place with the existing column structure without
    # rewriting the entire factorization.
    
    # Alternative Strategy: Return the Schoolbook but with a verification step.
    # If we cannot find a reduction quickly, return the best found.
    # For 3x5x5, is there a known rank < 75?
    # Research suggests small matrix multiplication ranks are often equal to schoolbook
    # except for specific cases like 2x2x2.
    # 3x3x3 is 23 (vs 27).
    # 3x4x4? 4x4x4?
    # 3x5x5: It is likely that 75 is minimal or very close.
    
    # Since I must provide a "stronger algorithm" and "distinct" program,
    # and the previous "Strassen" idea was rejected (possibly due to similarity or failure),
    # I will implement a **Bounded Randomized Search** for rank reduction.
    #
    # Algorithm:
    # 1. Start with Schoolbook (Rank 75).
    # 2. Generate random integer matrices U', V', W' of rank R < 75.
    # 3. Check if U' \otimes V' \otimes W' == T.
    #    This is expensive.
    #
    # Better: Use the property that if we can find 7 vectors that span the 8-dim space
    # of a 2x2x2 block, we reduce rank by 1.
    
    # Let's try to explicitly find one 2x2x2 block that can be Strassen-ized.
    # We need to pick i1,i2 in [0,2], j1,j2 in [0,4], k1,k2 in [0,4].
    # Let's pick i1=0, i2=1, j1=0, j2=1, k1=0, k2=1.
    # This involves rows of U: 0*m+0, 0*m+1, 1*m+0, 1*m+1.
    # Rows of V: 0*p+0, 0*p+1, 1*p+0, 1*p+1.
    # Rows of W: 0*n+0, 0*n+1, 1*n+0, 1*n+1.
    
    # The 8 columns in the schoolbook for this block are:
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1).
    
    # We will replace these 8 columns with 7 new columns defined by Strassen formulas.
    # This reduces rank by 1.
    
    # We can do this for multiple disjoint blocks.
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Block 3: i in {0,1}, j in {0,1}, k in {4, ?} -- need 2 k's. Only 0..4.
    # Let's find disjoint blocks.
    
    # Disjoint 2x2x2 blocks in 3x5x5:
    # i can be {0,1} (2 rows left: 2)
    # j can be {0,1}, {2,3} (1 row left: 4)
    # k can be {0,1}, {2,3}, {4} (1 row left)
    
    # Possible blocks:
    # B1: i{0,1}, j{0,1}, k{0,1}
    # B2: i{0,1}, j{2,3}, k{2,3}
    # B3: i{0,1}, j{0,1}, k{4, ?} -> No, k needs 2 distinct indices.
    # We can use k{4, 0}? But k0 is used in B1. They must be disjoint in terms of
    # the *columns* being replaced.
    
    # If we replace the 8 columns of B1, we are free to use other columns.
    # B2 uses j{2,3}, k{2,3}. Disjoint from B1?
    # B1 uses j{0,1}, k{0,1}.
    # B2 uses j{2,3}, k{2,3}.
    # They are disjoint.
    
    # Can we fit a third?
    # B3: i{0,1}, j{4, ?}, k{4, ?}.
    # j has 0,1,2,3,4. Used 0,1,2,3. Left 4. Need 2. Impossible.
    # k has 0,1,2,3,4. Used 0,1,2,3. Left 4. Need 2. Impossible.
    
    # So we can fit 2 disjoint Strassen blocks.
    # Rank reduction: 2 * (8-7) = 2.
    # New Rank: 75 - 2 = 73.
    
    # Let's implement this specific, deterministic optimization.
    
    # Identify the columns to replace for B1 and B2.
    # We will build new U, V, W matrices.
    
    # It is safer to construct the final matrices directly.
    
    U_new = [[0] * (rank0 - 2) for _ in range(n*m)]
    V_new = [[0] * (rank0 - 2) for _ in range(m*p)]
    W_new = [[0] * (rank0 - 2) for _ in range(p*n)]
    
    # Mapping from old column index to new column index
    # We skip the 16 columns involved in the two blocks.
    # Block 1 cols: (0,0,0)..(1,1,1)
    # Block 2 cols: (0,2,2)..(1,3,3)
    
    # Let's define the set of "Strassen-ized" columns.
    # We will create 7 new columns for each block.
    
    col_idx = 0
    
    # Helper to add a standard schoolbook column
    def add_schoolbook(i, j, k):
        nonlocal col_idx
        if col_idx >= len(U_new[0]):
            return
        U_new[i*m + j][col_idx] = 1
        V_new[j*p + k][col_idx] = 1
        W_new[k*n + i][col_idx] = 1
        col_idx += 1

    # Helper to add Strassen columns for a block (i1,i2, j1,j2, k1,k2)
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col_idx
        # 7 formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U entries for A (i,j)
        # V entries for B (j,k)
        # W entries for C (i,k) -- Note: Output is C = AB. C[i,k]
        
        # M1:
        # u = e_i1_j1 + e_i2_j2
        # v = e_j1_k1 + e_j2_k2
        # w = e_k1_i1 + e_k2_i2
        U_new[i1*m + j1][col_idx] = 1
        U_new[i2*m + j2][col_idx] = 1
        V_new[j1*p + k1][col_idx] = 1
        V_new[j2*p + k2][col_idx] = 1
        W_new[k1*n + i1][col_idx] = 1
        W_new[k2*n + i2][col_idx] = 1
        col_idx += 1
        
        # M2:
        # u = e_i2_j1 + e_i2_j2
        # v = e_j1_k1
        # w = e_k1_i2
        U_new[i2*m + j1][col_idx] = 1
        U_new[i2*m + j2][col_idx] = 1
        V_new[j1*p + k1][col_idx] = 1
        W_new[k1*n + i2][col_idx] = 1
        col_idx += 1
        
        # M3:
        # u = e_i1_j1
        # v = e_j2_k1 - e_j1_k1  (B21 - B11? No, indices are j,k. B[j,k])
        # Strassen: C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M4
        # C21 = M2 + M5
        # C22 = M1 - M3 + M6 - M7
        
        # Let's re-verify Strassen formulas with standard indexing A(i,j), B(j,k), C(i,k).
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # M3:
        # u = e_i1_j1
        # v = e_j2_k1 - e_j1_k1  (Term B21 is B[j2, k1], B11 is B[j1, k1])
        # w = e_k1_i1
        U_new[i1*m + j1][col_idx] = 1
        V_new[j2*p + k1][col_idx] = 1
        V_new[j1*p + k1][col_idx] = -1
        W_new[k1*n + i1][col_idx] = 1
        col_idx += 1
        
        # M4:
        # u = e_i2_j2
        # v = e_j1_k2 - e_j2_k2  (B12 is B[j1, k2], B22 is B[j2, k2])
        # w = e_k2_i2
        U_new[i2*m + j2][col_idx] = 1
        V_new[j1*p + k2][col_idx] = 1
        V_new[j2*p + k2][col_idx] = -1
        W_new[k2*n + i2][col_idx] = 1
        col_idx += 1
        
        # M5:
        # u = e_i1_j1 + e_i1_j2
        # v = e_j2_k2
        # w = e_k2_i1
        U_new[i1*m + j1][col_idx] = 1
        U_new[i1*m + j2][col_idx] = 1
        V_new[j2*p + k2][col_idx] = 1
        W_new[k2*n + i1][col_idx] = 1
        col_idx += 1
        
        # M6:
        # u = e_i2_j1 - e_i1_j1
        # v = e_j1_k1 + e_j1_k2
        # w = e_k1_i2
        U_new[i2*m + j1][col_idx] = 1
        U_new[i1*m + j1][col_idx] = -1
        V_new[j1*p + k1][col_idx] = 1
        V_new[j1*p + k2][col_idx] = 1
        W_new[k1*n + i2][col_idx] = 1
        col_idx += 1
        
        # M7:
        # u = e_i1_j2 - e_i2_j2
        # v = e_j2_k1 + e_j2_k2
        # w = e_k2_i2
        U_new[i1*m + j2][col_idx] = 1
        U_new[i2*m + j2][col_idx] = -1
        V_new[j2*p + k1][col_idx] = 1
        V_new[j2*p + k2][col_idx] = 1
        W_new[k2*n + i2][col_idx] = 1
        col_idx += 1

    # Define the blocks to replace
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    block1_cols = set()
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                # Find old column index for (i,j,k)
                # Old index logic:
                # for i... for j... for k... col++
                # idx = i*m*p + j*p + k
                old_idx = i*m*p + j*p + k
                block1_cols.add(old_idx)
                
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    block2_cols = set()
    for i in [0, 1]:
        for j in [2, 3]:
            for k in [2, 3]:
                old_idx = i*m*p + j*p + k
                block2_cols.add(old_idx)
                
    # Iterate through all old columns
    for old_idx in range(rank0):
        if old_idx in block1_cols:
            continue
        elif old_idx in block2_cols:
            continue
        else:
            # Recover i,j,k from old_idx
            k = old_idx % p
            j = (old_idx // p) % m
            i = (old_idx // (p*m)) % n
            add_schoolbook(i, j, k)
            
    # Add Strassen columns for Block 1
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Add Strassen columns for Block 2
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # Trim if we added too many (should be exact: 75 - 16 + 14 = 73)
    # Current col_idx should be 73.
    final_rank = col_idx
    
    U_final = [row[:final_rank] for row in U_new]
    V_final = [row[:final_rank] for row in V_new]
    W_final = [row[:final_rank] for row in W_new]
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
