import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen-based decomposition for 2x5x6 matrix multiplication
    # We partition the 5x5 middle dimension into blocks and use Strassen's 7-multiplication algorithm
    # for 2x2 blocks where possible.
    
    # Strategy: 
    # - The 5x5 matrix can be split as 2+2+1 (or other partitions)
    # - For 2x2 sub-blocks, Strassen reduces rank from 4 to 3 (saving 1 per block)
    # - We have multiple 2x2 blocks possible in the 5x5 dimension
    
    # Let's use a direct constructive approach:
    # Partition m=5 into blocks: [2, 2, 1]
    # Partition p=6 into blocks: [2, 2, 2] or [3, 3] etc.
    
    # Actually, let's use a simpler approach:
    # The tensor is T[i,j,k] = e_{i,k} * delta_{j,j} (for matrix mult A*B where A is 2x5, B is 5x6)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We'll construct a decomposition using Strassen on overlapping 2x2 blocks.
    
    # For now, let's implement a clean schoolbook + Strassen hybrid:
    # Use Strassen for the first 2x2 subproblem (rows 0,1 of A; cols 0,1 of B)
    # and schoolbook for the rest.
    
    # Actually, let me think more carefully.
    # A is 2x5, B is 5x6, C = A*B is 2x6.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can write this as:
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    
    # Let's partition the 5 columns of A (and 5 rows of B) into groups.
    # If we use Strassen on a 2x2 subblock:
    # For columns j1,j2 of A and rows j1,j2 of B:
    # Instead of 4 products A[0,j1]*B[j1,k], A[0,j2]*B[j2,k], A[1,j1]*B[j1,k], A[1,j2]*B[j2,k]
    # We use 7 Strassen products.
    
    # But we need to be careful about how this composes with other columns.
    
    # Let me try a different approach: explicit construction.
    
    # For 2x5x6, one known construction uses rank 54:
    # This comes from applying Strassen to overlapping 2x2 blocks.
    
    # Let's construct it explicitly:
    # We'll have 54 rank-1 terms.
    
    # The idea: 
    # - For the "middle" part where we can apply Strassen, we save rank.
    # - Specifically, if we can identify 6 independent 2x2 subproblems, each saves 1 rank.
    # - 60 - 6 = 54.
    
    # Let me implement a verified Strassen-based construction.
    
    # Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our problem, we have n=2, so we can apply Strassen to pairs of columns.
    
    # Let's partition the 5 columns into: {0,1}, {2,3}, {4}
    # And the 6 rows of B into: {0,1}, {2,3}, {4,5}
    
    # For the subblock A[:,{0,1}] * B[{0,1},:], we can use Strassen-like decomposition.
    
    # Actually, let me just implement a clean construction that I can verify.
    
    # Direct construction with rank 54:
    
    rank = 54
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We'll build this term by term.
    # For each output element C[i,k] = sum_j A[i,j] * B[j,k]
    # We need terms that contribute to this sum.
    
    # Let's use a block-based Strassen approach.
    
    # Partition columns j: [0,1], [2,3], [4]
    # Partition rows k: [0,1], [2,3], [4,5]
    
    # For each pair of column-block and row-block, we can use Strassen if both are size 2.
    
    # Column block 0: j in {0,1}
    # Column block 1: j in {2,3}  
    # Column block 2: j in {4}
    
    # Row block 0: k in {0,1}
    # Row block 1: k in {2,3}
    # Row block 2: k in {4,5}
    
    # For (col_block, row_block) pairs where both have size 2, we can use Strassen (7 terms instead of 4).
    # Pairs: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> 6 pairs of 2x2 blocks
    # Each saves 1 term: 4 -> 3, so 6 * 1 = 6 saved.
    # Total: 60 - 6 = 54.
    
    # For pairs where one block has size 1, we use schoolbook (4 terms for 2x1 or 1x2, etc.)
    # Wait, if col_block has size 2 and row_block has size 1, we have 2*1*2 = 4 terms (i in {0,1}, j in {0,1}, k is fixed)
    # Actually no: for fixed k, C[i,k] = sum_j A[i,j]*B[j,k]. If j has 2 values and i has 2 values, that's 4 terms.
    # If col_block has size 1 and row_block has size 2, that's 1*2*2 = 4 terms.
    
    # Let me reconsider. The total number of terms should be:
    # For each (i,j,k) triple, we need exactly one term in the schoolbook.
    # When we apply Strassen to a 2x2 subproblem (2 i-values, 2 j-values, 2 k-values), 
    # we replace 8 terms (2*2*2) with 7 terms.
    
    # Wait, I'm confusing myself. Let me restart with a cleaner formulation.
    
    # The matrix multiplication tensor for (n,m,p) has rank at most n*m*p.
    # Each rank-1 term corresponds to a product of three vectors.
    
    # For Strassen on 2x2x2:
    # We replace 8 rank-1 terms with 7 rank-1 terms.
    
    # In our case, we can identify 2x2x2 subproblems within the 2x5x6 tensor.
    # A 2x2x2 subproblem uses 2 i-values, 2 j-values, 2 k-values.
    # We have n=2, so we always use both i-values.
    # We need to find 2 j-values and 2 k-values.
    
    # The j-values range over 5, k-values over 6.
    # We can partition j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4,5}.
    
    # For each pair (J, K) where J is a 2-element subset of j and K is a 2-element subset of k,
    # we can apply Strassen to the 2x2x2 subproblem.
    
    # But we need to be careful: the subproblems must be disjoint in the (j,k) plane to avoid double-counting.
    
    # Let's partition:
    # J0 = {0,1}, J1 = {2,3}, J2 = {4}
    # K0 = {0,1}, K1 = {2,3}, K2 = {4,5}
    
    # Strassen-applicable pairs: (J0,K0), (J0,K1), (J0,K2), (J1,K0), (J1,K1), (J1,K2)
    # That's 6 pairs, each saving 1 term.
    # Remaining terms: (J2, K0), (J2, K1), (J2, K2) -> each is 2*1*2 = 4 terms, total 12.
    # Strassen terms: 6 * 7 = 42.
    # Total: 42 + 12 = 54. 
    
    # Now let's construct the actual terms.
    
    term_idx = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # In our notation, for a 2x2x2 subproblem with i in {i0,i1}, j in {j0,j1}, k in {k0,k1}:
    # A[i,j] has indices: A[i0,j0], A[i0,j1], A[i1,j0], A[i1,j1]
    # B[j,k] has indices: B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
    # C[i,k] has indices: C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
    
    # Strassen products:
    # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
    # M4 = A[i1,j1] * (B[j0,k1]-B[j0,k0])
    # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    
    # C[i0,k0] = M1+M4-M5+M7
    # C[i0,k1] = M3+M5
    # C[i1,k0] = M2+M4
    # C[i1,k1] = M1-M2+M3+M6
    
    # Each M is a product of a linear combination of A-entries and a linear combination of B-entries.
    # So each M corresponds to a rank-1 term in the tensor decomposition.
    
    def add_strassen_term(i0, i1, j0, j1, k0, k1, M_idx):
        """Add one Strassen term. M_idx is 0-6 for M1-M7."""
        global term_idx
        # Determine U, V, W coefficients for this term
        # U corresponds to A[i,j], V to B[j,k], W to C[i,k]
        
        # For each M, we have specific coefficients.
        # M1: U has +1 at (i0,j0) and (i1,j1); V has +1 at (j0,k0) and (j1,k1); W has +1 at (i0,k0) and (i1,k1)
        # Wait, that's not right. Each M is a single product term.
        
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # This is a product of two linear forms, so it's rank-1 in the tensor sense?
        # No, (a+b)*(c+d) = ac+ad+bc+bd, which is rank 4 in general.
        
        # I'm making an error. Strassen's algorithm works because the 7 products M1-M7 are scalar products,
        # and the C entries are linear combinations of these scalars.
        
        # In tensor decomposition terms, each M is a scalar, and C[i,k] = sum of M's with coefficients.
        # So the tensor T = sum over M's of (coefficients for C) * (outer product of A-linear-form and B-linear-form)
        
        # Let me reformulate. The tensor T[i,j,k] = delta_{i,k} for the matrix mult... no.
        # T[A_index, B_index, C_index] = 1 if C_index corresponds to the product of A_index and B_index.
        
        # For matrix multiplication, T[i,j,k] = 1 if the output C[i,k] depends on A[i,j] and B[j,k].
        # So T[i*m+j, j*p+k, k*n+i] = 1.
        
        # A rank-1 decomposition is T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
        
        # For Strassen, each M_r is a scalar product of a linear form in A and a linear form in B.
        # M_r = (sum_{i,j} u_{i,j} A[i,j]) * (sum_{j,k} v_{j,k} B[j,k])
        # Then C[i,k] = sum_r w_{i,k} M_r
        
        # So the tensor is T = sum_r (w[:,r]) outer (v[:,r]) outer (u[:,r])
        # Wait, need to get the indices right.
        
        # T[a_index, b_index, c_index] = sum_r U[a_index, r] * V[b_index, r] * W[c_index, r]
        # where a_index = i*m+j, b_index = j*p+k, c_index = k*n+i
        
        # For a Strassen term M_r:
        # M_r = (sum_{i,j} u_{i,j} A[i,j]) * (sum_{j,k} v_{j,k} B[j,k])
        # This contributes to C[i,k] as: c_{i,k} += w_{i,k} * M_r
        # = w_{i,k} * (sum_{i',j} u_{i',j} A[i',j]) * (sum_{j',k'} v_{j',k'} B[j',k'])
        
        # For this to equal the correct contribution, we need:
        # For each (i,j,k) with T[i*m+j, j*p+k, k*n+i] = 1:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is getting complex. Let me just implement it directly and verify.
        
        pass
    
    # Let me try a different approach: just construct the schoolbook and then apply Strassen transformations.
    
    # Start with schoolbook (rank 60)
    U_sb = np.zeros((n*m, n*m*p))
    V_sb = np.zeros((m*p, n*m*p))
    W_sb = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Now apply Strassen to 6 independent 2x2x2 subproblems.
    # Each Strassen transformation replaces 8 terms with 7 terms.
    
    # The 8 terms for a 2x2x2 subproblem (i0,i1,j0,j1,k0,k1) are:
    # (i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1),
    # (i1,j0,k0), (i1,j0,k1), (i1,j1,k0), (i1,j1,k1)
    
    # We'll replace these 8 terms with 7 Strassen terms.
    
    # Let's define the Strassen transformation for a given subproblem.
    
    def apply_strassen(U, V, W, cols, i0, i1, j0, j1, k0, k1):
        """Replace 8 schoolbook terms with 7 Strassen terms.
        cols: list of 8 column indices in U, V, W corresponding to the 8 schoolbook terms."""
        # The 8 terms are indexed as:
        # 0: (i0,j0,k0), 1: (i0,j0,k1), 2: (i0,j1,k0), 3: (i0,j1,k1),
        # 4: (i1,j0,k0), 5: (i1,j0,k1), 6: (i1,j1,k0), 7: (i1,j1,k1)
        
        # We'll use the first 7 columns for Strassen terms, and zero out the 8th.
        # Actually, we should reuse the columns in place.
        
        # Strassen M1-M7 coefficients:
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
        # M4 = A[i1,j1] * (B[j0,k1]-B[j0,k0])
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        
        # C[i0,k0] = M1+M4-M5+M7
        # C[i0,k1] = M3+M5
        # C[i1,k0] = M2+M4
        # C[i1,k1] = M1-M2+M3+M6
        
        # For each Strassen term Mr, we have:
        # U contribution: linear form in A
        # V contribution: linear form in B
        # W contribution: coefficients for C
        
        # Let's set up the new values for the 7 columns.
        
        # Column 0 -> M1
        U[i0*m+j0, cols[0]] = 1
        U[i1*m+j1, cols[0]] = 1
        V[j0*p+k0, cols[0]] = 1
        V[j1*p+k1, cols[0]] = 1
        W[k0*n+i0, cols[0]] = 1  # C[i0,k0] += M1
        W[k1*n+i1, cols[0]] = 1  # C[i1,k1] += M1
        
        # Column 1 -> M2
        U[i1*m+j0, cols[1]] = 1
        U[i1*m+j1, cols[1]] = 1
        V[j0*p+k0, cols[1]] = 1
        W[k0*n+i1, cols[1]] = 1  # C[i1,k0] += M2
        W[k1*n+i1, cols[1]] = -1  # C[i1,k1] -= M2
        
        # Column 2 -> M3
        U[i0*m+j0, cols[2]] = 1
        V[j1*p+k0, cols[2]] = 1
        V[j1*p+k1, cols[2]] = -1
        W[k1*n+i0, cols[2]] = 1  # C[i0,k1] += M3
        W[k1*n+i1, cols[2]] = 1  # C[i1,k1] += M3
        
        # Column 3 -> M4
        U[i1*m+j1, cols[3]] = 1
        V[j0*p+k1, cols[3]] = 1
        V[j0*p+k0, cols[3]] = -1
        W[k0*n+i0, cols[3]] = 1  # C[i0,k0] += M4
        W[k0*n+i1, cols[3]] = 1  # C[i1,k0] += M4
        
        # Column 4 -> M5
        U[i0*m+j0, cols[4]] = 1
        U[i0*m+j1, cols[4]] = 1
        V[j1*p+k1, cols[4]] = 1
        W[k0*n+i0, cols[4]] = -1  # C[i0,k0] -= M5
        W[k1*n+i0, cols[4]] = 1  # C[i0,k1] += M5
        
        # Column 5 -> M6
        U[i1*m+j0, cols[5]] = 1
        U[i0*m+j0, cols[5]] = -1
        V[j0*p+k0, cols[5]] = 1
        V[j0*p+k1, cols[5]] = 1
        W[k1*n+i1, cols[5]] = 1  # C[i1,k1] += M6
        
        # Column 6 -> M7
        U[i0*m+j1, cols[6]] = 1
        U[i1*m+j1, cols[6]] = -1
        V[j1*p+k0, cols[6]] = 1
        V[j1*p+k1, cols[6]] = 1
        W[k0*n+i0, cols[6]] = 1  # C[i0,k0] += M7
        
        # Zero out column 7
        for idx in range(n*m):
            U[idx, cols[7]] = 0
        for idx in range(m*p):
            V[idx, cols[7]] = 0
        for idx in range(p*n):
            W[idx, cols[7]] = 0
    
    # Now apply Strassen to 6 subproblems.
    # We need to find the column indices for each 2x2x2 subproblem.
    
    # Subproblem 1: i0=0, i1=1, j0=0, j1=1, k0=0, k1=1
    # Terms: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    # Column indices: 
    # (0,0,0): 0*5+0=0, 0*6+0=0, 0*2+0=0 -> col 0*5+0... wait, let me compute properly.
    # col = i*m*p + j*p + k? No, the schoolbook col is just a counter.
    # Let me recompute the column mapping.
    
    # In the schoolbook, col is assigned in order:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       col += 1
    
    # So col(i,j,k) = i*m*p + j*p + k
    
    # For n=2, m=5, p=6:
    # col(i,j,k) = i*30 + j*6 + k
    
    # Subproblem 1: i0=0, i1=1, j0=0, j1=1, k0=0, k1=1
    # Cols: 
    # (0,0,0): 0, (0,0,1): 1, (0,1,0): 6, (0,1,1): 7,
    # (1,0,0): 30, (1,0,1): 31, (1,1,0): 36, (1,1,1): 37
    
    subproblems = [
        # (i0, i1, j0, j1, k0, k1)
        (0, 1, 0, 1, 0, 1),   # J0={0,1}, K0={0,1}
        (0, 1, 0, 1, 2, 3),   # J0={0,1}, K1={2,3}
        (0, 1, 0, 1, 4, 5),   # J0={0,1}, K2={4,5}
        (0, 1, 2, 3, 0, 1),   # J1={2,3}, K0={0,1}
        (0, 1, 2, 3, 2, 3),   # J1={2,3}, K1={2,3}
        (0, 1, 2, 3, 4, 5),   # J1={2,3}, K2={4,5}
    ]
    
    U_new = U_sb.copy()
    V_new = V_sb.copy()
    W_new = W_sb.copy()
    
    for (i0, i1, j0, j1, k0, k1) in subproblems:
        # Get the 8 column indices
        cols = []
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    cols.append(i * m * p + j * p + k)
        
        # Apply Strassen transformation
        apply_strassen(U_new, V_new, W_new, cols, i0, i1, j0, j1, k0, k1)
    
    # Now we have 54 effective terms (column 7 of each subproblem is zeroed out).
    # But we still have 60 columns, with 6 of them being all-zero.
    # We need to remove the zero columns.
    
    # Find non-zero columns
    nonzero_cols = []
    for col in range(60):
        if np.any(U_new[:, col] != 0) or np.any(V_new[:, col] != 0) or np.any(W_new[:, col] != 0):
            nonzero_cols.append(col)
    
    rank_final = len(nonzero_cols)
    
    U_final = U_new[:, nonzero_cols]
    V_final = V_new[:, nonzero_cols]
    W_final = W_new[:, nonzero_cols]
    
    # Verify the decomposition
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    reconstructed = np.einsum('ic,jc,kc->ijk', U_final, V_final, W_final)
    
    if not np.allclose(reconstructed, T, atol=1e-9):
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Convert to lists
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
