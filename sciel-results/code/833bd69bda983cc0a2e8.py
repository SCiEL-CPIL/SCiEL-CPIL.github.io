"""Matrix multiplication 3x5x7 rank-minimal decomposition using Strassen blocks."""
import time

DIMENSIONS = [3, 5, 7]

# Deadline guard
_DEADLINE = time.time() + 140.0  # 140s out of 150s wall-clock limit

def _strassen_2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is (U_entry, V_entry, W_entry) where:
      U_entry: which of the 4 sub-blocks of A (indices 0-3)
      V_entry: which of the 4 sub-blocks of B (indices 0-3)
      W_entry: which of the 4 sub-blocks of C (indices 0-3)
      plus signs for the linear combinations.
    
    Actually, for the tensor decomposition, each rank-1 term corresponds to:
      U factor: linear combination of A sub-blocks
      V factor: linear combination of B sub-blocks  
      W factor: linear combination of C sub-blocks
    
    For 2x2 Strassen, the 7 products are:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B21-B22)
      M4 = A22(B12-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Each entry: list of (subblock_index, sign) for U, V, W
    # Subblock indices: 0=A11, 1=A12, 2=A21, 3=A22 (and same for B, C)
    terms = [
        # (U: [(idx, sign)], V: [(idx, sign)], W: [(idx, sign)])
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,-1),(2,-1),(1,1)]),  # M1
        ([(2,1),(3,1)], [(0,1)],       [(1,1),(2,1)]),                 # M2
        ([(0,1)],       [(2,1),(3,-1)],[(1,1),(3,1)]),                 # M3
        ([(3,1)],       [(1,1),(0,-1)],[(0,1),(2,1)]),                 # M4
        ([(0,1),(1,1)],[(3,1)],       [(0,-1),(1,1)]),                 # M5
        ([(2,1),(0,-1)],[(0,1),(1,1)],[(3,1)]),                        # M6
        ([(1,1),(3,-1)],[(2,1),(3,1)],[(0,1)]),                        # M7
    ]
    return terms

def _build_strassen_2x2_block(row_offset, col_offset, U, V, W, col_start):
    """
    Add Strassen 2x2 block terms starting at column col_start.
    row_offset: starting row in A (0-based), so covers rows [row_offset, row_offset+2)
    col_offset: starting col in A / row in B, covers [col_offset, col_offset+2)
    Returns new column index after adding 7 terms.
    """
    terms = _strassen_2x2()
    for (u_terms, v_terms, w_terms) in terms:
        col = col_start
        # U: for each (sub_idx, sign), add sign to U[row_offset + sub_row][col]
        # sub_idx 0 -> row 0,col 0; 1 -> row 0,col 1; 2 -> row 1,col 0; 3 -> row 1,col 1
        for (sub_idx, sign) in u_terms:
            r = sub_idx // 2
            c = sub_idx % 2
            U[row_offset + r][col] += sign
        # V: sub_idx 0 -> row 0,col 0; 1 -> row 0,col 1; 2 -> row 1,col 0; 3 -> row 1,col 1
        for (sub_idx, sign) in v_terms:
            r = sub_idx // 2
            c = sub_idx % 2
            V[col_offset + r][col] += sign
        # W: W[k*n+i, col], sub_idx 0 -> k=0,i=0; 1 -> k=0,i=1; 2 -> k=1,i=0; 3 -> k=1,i=1
        for (sub_idx, sign) in w_terms:
            k = sub_idx // 2
            i = sub_idx % 2
            W[k * 3 + i][col] += sign
        col_start += 1
    return col_start

def _add_schoolbook_term(i, j, k, U, V, W, col):
    """Add a single schoolbook rank-1 term for C[i,k] += A[i,j]*B[j,k]."""
    n, m, p = 3, 5, 7
    U[i * m + j][col] = 1
    V[j * p + k][col] = 1
    W[k * n + i][col] = 1

def construct():
    n, m, p = 3, 5, 7
    # We'll use Strassen for 2x2 blocks where possible, schoolbook for the rest.
    # 
    # Strategy: Partition the 3x5x7 multiplication into blocks.
    # A is 3x5, B is 5x7, C is 3x7.
    # 
    # We can decompose by rows of A (3 rows) and columns of B (7 cols), with 5 intermediate.
    # 
    # A better approach: use Strassen on 2x2 blocks in the "middle" dimension.
    # 
    # Let's think of it as: for each pair of rows of A and pair of cols of B,
    # we have a 2x1 * 1x2 product (not 2x2). That doesn't help directly.
    #
    # Actually, the standard approach for rectangular matrix multiplication:
    # We can group the 5 intermediate indices into blocks.
    # 
    # Alternative: Use the fact that 3x5x7 = sum over j of A[:,j] * B[j,:]
    # Each term is rank 1 (outer product of column j of A and row j of B).
    # Total schoolbook rank = 3*5*7 = 105.
    #
    # Strassen helps when we have 2x2x2 sub-problems. 
    # For a 2x2x2 block: A is 2x2, B is 2x2, C is 2x2.
    # 
    # In our 3x5x7:
    # - We can take rows 0,1 of A (2 rows), cols 0,1 of B (2 cols), and the 2 intermediate indices.
    #   But the intermediate dimension is 5, not 2.
    #
    # Let me reconsider. The tensor for matrix multiplication (n,m,p) has:
    #   U indexed by (i,j) where i in [0,n), j in [0,m)  -> shape (n*m, rank)
    #   V indexed by (j,k) where j in [0,m), k in [0,p)  -> shape (m*p, rank)
    #   W indexed by (k,i) where k in [0,p), i in [0,n)  -> shape (p*n, rank)
    #
    # A rank-1 term (u, v, w) satisfies: for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k} (the multiplication tensor)
    #
    # Actually the multiplication tensor T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]B[j,k]).
    # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    #
    # Schoolbook: one term per (i,j,k) with U[i*m+j]=V[j*p+k]=W[k*n+i]=1.
    #
    # Strassen 2x2: For a 2x2x2 sub-problem with indices (i1,i2) x (j1,j2) x (k1,k2):
    #   7 terms that compute all 8 products C[i,k] for i in {i1,i2}, k in {k1,k2}
    #   using the 4 intermediate products from j1,j2.
    #
    # So I need to identify 2x2x2 sub-blocks. In 3x5x7:
    # - i dimension: 3 (can do one 2x2 block on i=0,1)
    # - j dimension: 5 (can do two 2x2 blocks on j=0,1 and j=2,3, plus j=4)
    # - k dimension: 7 (can do three 2x2 blocks on k=0,1; k=2,3; k=4,5, plus k=6)
    #
    # A 2x2x2 block uses 2 values of i, 2 values of j, 2 values of k.
    # 
    # Let me enumerate all possible 2x2x2 blocks:
    # i-blocks: {0,1} (only one, since n=3)
    # j-blocks: {0,1}, {2,3} (two blocks, j=4 is leftover)
    # k-blocks: {0,1}, {2,3}, {4,5} (three blocks, k=6 is leftover)
    #
    # For each combination of (i-block, j-block, k-block), we can apply Strassen.
    # Number of full 2x2x2 blocks: 1 * 2 * 3 = 6 blocks, each using 7 rank terms = 42 terms.
    #
    # Remaining terms (schoolbook):
    # - i=2 with all j,k: 5*7 = 35 terms
    # - i in {0,1} with j=4, all k: 2*7 = 14 terms
    # - i in {0,1}, j in {0,1,2,3}, k=6: 2*4*1 = 8 terms
    # 
    # Wait, let me be more careful. The full decomposition covers all (i,j,k) triples.
    # A 2x2x2 block on (i1,i2) x (j1,j2) x (k1,k2) covers 8 (i,j,k) triples.
    # 
    # Total triples: 3*5*7 = 105
    # 6 Strassen blocks cover: 6*8 = 48 triples, using 6*7 = 42 rank terms.
    # Remaining: 105 - 48 = 57 triples, needing 57 schoolbook terms.
    # Total rank: 42 + 57 = 99.
    #
    # Let's verify which triples are covered:
    # i-block {0,1}, j-block {0,1}, k-block {0,1}: covers i in {0,1}, j in {0,1}, k in {0,1}
    # i-block {0,1}, j-block {0,1}, k-block {2,3}: covers i in {0,1}, j in {0,1}, k in {2,3}
    # i-block {0,1}, j-block {0,1}, k-block {4,5}: covers i in {0,1}, j in {0,1}, k in {4,5}
    # i-block {0,1}, j-block {2,3}, k-block {0,1}: covers i in {0,1}, j in {2,3}, k in {0,1}
    # i-block {0,1}, j-block {2,3}, k-block {2,3}: covers i in {0,1}, j in {2,3}, k in {2,3}
    # i-block {0,1}, j-block {2,3}, k-block {4,5}: covers i in {0,1}, j in {2,3}, k in {4,5}
    #
    # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Not covered:
    #   i=2, any j, any k: 5*7 = 35
    #   i in {0,1}, j=4, any k: 2*7 = 14
    #   i in {0,1}, j in {0,1,2,3}, k=6: 2*4*1 = 8
    # Total not covered: 35 + 14 + 8 = 57. ✓
    
    n, m, p = 3, 5, 7
    U = [[0] * 99 for _ in range(n * m)]
    V = [[0] * 99 for _ in range(m * p)]
    W = [[0] * 99 for _ in range(p * n)]
    
    col = 0
    
    # Define the Strassen blocks
    i_blocks = [(0, 1)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for (i1, i2) in i_blocks:
        for (j1, j2) in j_blocks:
            for (k1, k2) in k_blocks:
                # Add 7 Strassen terms for this 2x2x2 block
                terms = _strassen_2x2()
                for (u_terms, v_terms, w_terms) in terms:
                    # U: rows i1,i2; cols j1,j2
                    # sub_idx 0 -> (i1,j1), 1 -> (i1,j2), 2 -> (i2,j1), 3 -> (i2,j2)
                    for (sub_idx, sign) in u_terms:
                        r = sub_idx // 2  # 0 for i1, 1 for i2
                        c = sub_idx % 2   # 0 for j1, 1 for j2
                        i = i1 + r
                        j = j1 + c
                        U[i * m + j][col] += sign
                    # V: rows j1,j2; cols k1,k2
                    for (sub_idx, sign) in v_terms:
                        r = sub_idx // 2
                        c = sub_idx % 2
                        j = j1 + r
                        k = k1 + c
                        V[j * p + k][col] += sign
                    # W: W[k*n+i, col]
                    for (sub_idx, sign) in w_terms:
                        k = k1 + (sub_idx // 2)
                        i = i1 + (sub_idx % 2)
                        W[k * n + i][col] += sign
                    col += 1
    
    # Add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this triple is covered by a Strassen block
                covered = False
                if i in (0, 1) and j in (0, 1, 2, 3) and k in (0, 1, 2, 3, 4, 5):
                    covered = True
                if not covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Sanity check: col should be 99
    assert col == 99, f"Expected 99 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
