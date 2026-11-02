"""
Rewrite: hybrid Strassen/schoolbook decomposition for 2x3 * 3x7 matrix multiplication.

Strategy:
- Partition the 3x7 matrix into a 2x2 block (rows 0,1) and a 1x7 block (row 2).
- Use Strassen's 7-multiplication algorithm for the 2x2 * 2x2 part.
- Use schoolbook (4 multiplications) for the 2x1 * 1x7 part (i.e., 2x2 * 1x2).
- Use schoolbook (3 multiplications) for the 2x1 * 1x7 part? Wait, 2x3 * 3x7:
  Actually:
  A is 2x3: A = [A11 A12 A13] where A11, A12, A13 are 2x1 columns? No.
  
Let me re-think. 

A is 2x3, B is 3x7.

We can partition B's 3 rows into 2+1:
  B = [B1; B2] where B1 is 2x7 and B2 is 1x7.

Then C = A1*B1 + A2*B2 where:
  A1 is 2x2 (first 2 columns of A)
  A2 is 2x1 (last column of A)
  B1 is 2x7 (first 2 rows of B)
  B2 is 1x7 (last row of B)

For the 2x2 * 2x7 part:
  We can further partition B1 into two 2x3 and one 2x1? Or use Strassen for 2x2*2x2 blocks.
  
  Actually, 2x2 * 2x7: We can view B1 as three 2x2 blocks plus one 2x1 block? 
  7 = 2+2+2+1. So B1 = [B1a B1b B1c B1d] where B1a, B1b, B1c are 2x2 and B1d is 2x1.
  
  For each 2x2 * 2x2 block, use Strassen (7 mults).
  For the 2x2 * 2x1 block, use schoolbook (4 mults).
  
  So 2x2 * 2x7 costs 3*7 + 4 = 21 + 4 = 25 mults.

For the 2x1 * 1x7 part:
  This is just a row vector (length 7) times a column vector (2x1) producing 2x7.
  C2 = A2 * B2 where A2 is 2x1, B2 is 1x7.
  C2[i][j] = A2[i] * B2[j]
  This requires 2*7 = 14 mults in schoolbook.
  
  Can we do better? This is essentially outer product of a 2-vector and a 7-vector.
  Rank of the tensor for 2x1 * 1x7 -> 2x7 is 2*1*7 = 14? No, the tensor for outer product 
  of vectors of length m and n producing m x n has rank... 
  
  Actually, the matrix multiplication tensor for (2x1)*(1x7) -> 2x7 is just the outer product 
  tensor. Each output entry C[i][j] = A[i] * B[j]. These are 14 independent bilinear terms.
  The rank of this tensor is 14 (each term is a rank-1 tensor, and they are linearly independent).
  
  So total rank = 25 + 14 = 39.

This matches the known improvement from 42 to 39.

Let me implement this.
"""

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    # A is n x m = 2 x 3
    # B is m x p = 3 x 7
    # C is n x p = 2 x 7
    #
    # Indexing convention:
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (21, rank)
    # W has shape (p*n, rank) = (14, rank)
    #
    # C[k*n + i] = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] ... 
    # Wait, the standard decomposition is:
    # C = A * B
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # In the tensor form: T[i,j,k] = delta[i,j,k] meaning C[i,k] = sum_j A[i,j] B[j,k]
    #
    # The factorization: C[i,k] = sum_c U[i,m+j? no...
    #
    # The problem states:
    # U: (n*m, rank) -> indexed by (i, j) where i in [0,n), j in [0,m)
    # V: (m*p, rank) -> indexed by (j, k) where j in [0,m), k in [0,p)
    # W: (p*n, rank) -> indexed by (k, i) where k in [0,p), i in [0,n)
    #
    # And C[k*n + i] = sum_c U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
    # Wait, that's not right. Let me re-read.
    #
    # "output factor follows the official trace convention W[k*n+i,r]"
    #
    # So the decomposition is:
    # C[i][k] (where i is row, k is column of C, C is n x p)
    # C[i][k] = sum_c (sum_j U[i*m+j][c] * A[i][j]) * (sum_j V[j*p+k][c] * B[j][k]) * W[k*n+i][c]
    #
    # Hmm, that doesn't seem right either. Let me think about the standard form.
    #
    # Standard: C[i,k] = sum_c U[i][c] * V[k][c] * W[i,k][c] where U and V are linear maps.
    # But here U, V, W are matrices with specific shapes.
    #
    # I think the convention is:
    # C[i,k] = sum_c (linear combination of A using U[:,c]) * (linear combination of B using V[:,c]) * W[k*n+i][c]
    #
    # More precisely:
    # Let A be flattened to a vector of length n*m, indexed by (i,j) -> i*m+j
    # Let B be flattened to a vector of length m*p, indexed by (j,k) -> j*p+k
    # Then C[i,k] = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] ... no, this sums over j too?
    #
    # I think the correct interpretation is:
    # C[i,k] = sum_c (sum_j U[i*m+j][c] * A[i,j]) * (sum_j V[j*p+k][c] * B[j,k]) * W[k*n+i][c]
    #
    # But that would make the U and V terms depend on j in a summed way, which is unusual.
    #
    # Actually, I think it's:
    # C[i,k] = sum_c (sum_j U[i*m+j][c] * A[i,j]) * (sum_j V[j*p+k][c] * B[j,k]) * W[k*n+i][c]
    #
    # No wait, that would be a product of three things, not a sum. Let me look at the schoolbook example.
    #
    # In the schoolbook:
    # U[i*m+j][c] = 1, V[j*p+k][c] = 1, W[k*n+i][c] = 1 for each (i,j,k) tuple assigned to column c.
    # Then C[i,k] = sum_c (sum_j U[i*m+j][c] * A[i,j]) * (sum_j V[j*p+k][c] * B[j,k]) * W[k*n+i][c]
    #
    # For a given column c corresponding to (i',j',k'):
    # U[i*m+j][c] = 1 if i=i' and j=j', else 0. So sum_j U[i*m+j][c]*A[i,j] = A[i'][j'] if i=i' else 0.
    # V[j*p+k][c] = 1 if j=j' and k=k', else 0. So sum_j V[j*p+k][c]*B[j,k] = B[j'][k'] if k=k' else 0.
    # W[k*n+i][c] = 1 if k=k' and i=i', else 0.
    #
    # So the term is non-zero only if i=i', k=k', and then it's A[i'][j']*B[j'][k']*1.
    # Summing over c (which iterates over all (i',j',k') tuples with i'=i, k'=k, and any j'),
    # we get sum_j' A[i][j']*B[j'][k] = C[i][k].
    #
    # So the formula is:
    # C[i,k] = sum_c (sum_j U[i*m+j][c] * A[i,j]) * (sum_j V[j*p+k][c] * B[j,k]) * W[k*n+i][c]
    #
    # This is a product of three terms per column c, summed over c.
    #
    # Now let's implement the hybrid decomposition.
    
    # Partition:
    # A (2x3): A1 (2x2) uses columns 0,1; A2 (2x1) uses column 2
    # B (3x7): B1 (2x7) uses rows 0,1; B2 (1x7) uses row 2
    
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x7):
    # Partition B1 into B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1)
    # B1 columns 0,1 -> B1a; columns 2,3 -> B1b; columns 4,5 -> B1c; column 6 -> B1d
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal rank
        U.append([0] * rank + [u_val])
        V.append([0] * rank + [v_val])
        W.append([0] * rank + [w_val])
        rank += 1
    
    def strassen_2x2_2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """
        Add Strassen decomposition for a 2x2 * 2x2 block.
        A_rows: list of 2 row indices in A (for the 2x2 block)
        A_cols: list of 2 col indices in A
        B_rows: list of 2 row indices in B
        B_cols: list of 2 col indices in B
        C_rows: list of 2 row indices in C
        C_cols: list of 2 col indices in C
        """
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A indices: A[A_rows[0]][A_cols[0]] = A11, A[A_rows[0]][A_cols[1]] = A12,
        #             A[A_rows[1]][A_cols[0]] = A21, A[A_rows[1]][A_cols[1]] = A22
        # B indices: B[B_rows[0]][B_cols[0]] = B11, B[B_rows[0]][B_cols[1]] = B12,
        #             B[B_rows[1]][B_cols[0]] = B21, B[B_rows[1]][B_cols[1]] = B22
        # C indices: C[C_rows[0]][C_cols[0]] = C11, C[C_rows[0]][C_cols[1]] = C12,
        #             C[C_rows[1]][C_cols[0]] = C21, C[C_rows[1]][C_cols[1]] = C22
        
        a11, a12 = A_rows[0], A_cols[0], A_cols[1]  # (row, col)
        a21, a22 = A_rows[1], A_cols[0], A_cols[1]
        
        b11, b12 = B_rows[0], B_cols[0], B_cols[1]
        b21, b22 = B_rows[1], B_cols[0], B_cols[1]
        
        c11, c12 = C_rows[0], C_cols[0], C_cols[1]
        c21, c22 = C_rows[1], C_cols[0], C_cols[1]
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11+A22 -> U[a11*m+a11_col][c]=1, U[a21*m+a22_col][c]=1
        # V: B11+B22 -> V[b11*p+b11_col][c]=1, V[b21*p+b22_col][c]=1
        # W: 1
        add_term(
            [1, 1],  # U for A11 and A22
            [1, 1],  # V for B11 and B22
            [1]       # W
        )
        # But add_term only takes single values. I need to handle multiple U/V entries per column.
        
        # Let me restructure. I'll build U, V, W as lists of dicts or handle it differently.
        pass
    
    # Let me use a different approach: accumulate contributions to each (row, col) of U, V, W.
    
    U_entries = {}  # (i*m+j) -> list of (col, val)
    V_entries = {}  # (j*p+k) -> list of (col, val)
    W_entries = {}  # (k*n+i) -> list of (col, val)
    
    col = 0
    def add(u_indices, v_indices, w_indices, w_val):
        nonlocal col
        for idx in u_indices:
            U_entries.setdefault(idx, []).append((col, 1))
        for idx in v_indices:
            V_entries.setdefault(idx, []).append((col, 1))
        for idx in w_indices:
            W_entries.setdefault(idx, []).append((col, w_val))
        col += 1
    
    def strassen(a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        """
        a11, a12, a21, a22 are (i,j) indices for A
        b11, b12, b21, b22 are (j,k) indices for B
        c11, c12, c21, c22 are (k,i) indices for W
        """
        # M1 = (A11+A22)(B11+B22), W=1
        add([a11, a22], [b11, b22], [c11, c22], 1)
        # M2 = (A21+A22)B11, W=1
        add([a21, a22], [b11], [c21, c22], 1)
        # M3 = A11(B21-B22), W=1
        add([a11], [b21, b22], [c12, c22], 1)
        # But M3 needs B21 - B22, so V[b21]=1, V[b22]=-1
        # I need to support negative values in V.
        pass
    
    # I need to support signed entries. Let me redo with a list-based approach.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal rank
        for key in range(n * m):
            U_list.append([0] * rank + [u_dict.get(key, 0)])
        for key in range(m * p):
            V_list.append([0] * rank + [v_dict.get(key, 0)])
        for key in range(p * n):
            W_list.append([0] * rank + [w_dict.get(key, 0)])
        rank += 1
    
    # This is getting complex. Let me just directly build the matrices.
    
    # Start fresh
    U_mat = []
    V_mat = []
    W_mat = []
    r = 0
    
    def add(u_dict, v_dict, w_dict):
        nonlocal r
        for i in range(n * m):
            U_mat.append([0] * r + [u_dict.get(i, 0)])
        for j in range(m * p):
            V_mat.append([0] * r + [v_dict.get(j, 0)])
        for k in range(p * n):
            W_mat.append([0] * r + [w_dict.get(k, 0)])
        r += 1
    
    # Strassen for 2x2 * 2x2
    def strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        a_r: [r1, r2] rows in A
        a_c: [c1, c2] cols in A
        b_r: [r1, r2] rows in B
        b_c: [c1, c2] cols in B
        c_r: [r1, r2] rows in C
        c_c: [c1, c2] cols in C
        """
        A11 = a_r[0] * m + a_c[0]
        A12 = a_r[0] * m + a_c[1]
        A21 = a_r[1] * m + a_c[0]
        A22 = a_r[1] * m + a_c[1]
        
        B11 = b_r[0] * p + b_c[0]
        B12 = b_r[0] * p + b_c[1]
        B21 = b_r[1] * p + b_c[0]
        B22 = b_r[1] * p + b_c[1]
        
        C11 = c_c[0] * n + c_r[0]
        C12 = c_c[1] * n + c_r[0]
        C21 = c_c[0] * n + c_r[1]
        C22 = c_c[1] * n + c_r[1]
        
        # M1 = (A11+A22)(B11+B22)
        add({A11: 1, A22: 1}, {B11: 1, B22: 1}, {C11: 1, C22: 1})
        # M2 = (A21+A22)B11
        add({A21: 1, A22: 1}, {B11: 1}, {C21: 1, C22: 1})
        # M3 = A11(B21-B22)
        add({A11: 1}, {B21: 1, B22: -1}, {C12: 1, C22: 1})
        # M4 = A22(B12-B21)
        add({A22: 1}, {B12: 1, B21: -1}, {C11: 1, C22: 1})
        # M5 = (A11+A12)B22
        add({A11: 1, A12: 1}, {B22: 1}, {C11: -1, C22: 1})
        # M6 = (A21-A11)(B11+B12)
        add({A21: 1, A11: -1}, {B11: 1, B12: 1}, {C22: 1})
        # M7 = (A12-A22)(B21+B22)
        add({A12: 1, A22: -1}, {B21: 1, B22: 1}, {C11: 1, C12: 1})
    
    # Schoolbook for 2x2 * 2x1 (i.e., 2x2 * 1x2 transposed? No, 2x2 * 2x1)
    # Actually B1d is 2x1, so it's 2x2 * 2x1.
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # For 2x2 * 2x1, k has only 1 value.
    def schoolbook_2x2_2x1(a_r, a_c, b_r, b_k, c_r, c_k):
        """
        a_r: 2 rows, a_c: 2 cols
        b_r: 2 rows, b_k: 1 col
        c_r: 2 rows, c_k: 1 col
        """
        for i in range(2):
            for j in range(2):
                Aij = a_r[i] * m + a_c[j]
                Bjk = b_r[j] * p + b_k[0]
                Cki = c_k[0] * n + c_r[i]
                add({Aij: 1}, {Bjk: 1}, {Cki: 1})
    
    # Schoolbook for 2x1 * 1x7
    # C[i][k] = A[i] * B[k]
    def schoolbook_2x1_1x7(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        a_r: 2 rows, a_c: 1 col
        b_r: 1 row, b_c: 7 cols
        c_r: 2 rows, c_c: 7 cols
        """
        for i in range(2):
            for k in range(7):
                Aij = a_r[i] * m + a_c[0]
                Bjk = b_r[0] * p + b_c[k]
                Cki = c_c[k] * n + c_r[i]
                add({Aij: 1}, {Bjk: 1}, {Cki: 1})
    
    # A1 is 2x2: rows [0,1], cols [0,1]
    # A2 is 2x1: rows [0,1], col [2]
    # B1 is 2x7: rows [0,1], cols [0..6]
    # B2 is 1x7: row [2], cols [0..6]
    # C is 2x7: rows [0,1], cols [0..6]
    
    # Strassen for B1a: cols [0,1]
    strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    # Strassen for B1b: cols [2,3]
    strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    # Strassen for B1c: cols [4,5]
    strassen_block([0, 1], [0, 1], [0, 1], [4, 5], [0, 1], [4, 5])
    # Schoolbook for B1d: col [6]
    schoolbook_2x2_2x1([0, 1], [0, 1], [0, 1], [6], [0, 1], [6])
    # Schoolbook for A2*B2
    schoolbook_2x1_1x7([0, 1], [2], [2], list(range(7)), [0, 1], list(range(7)))
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
