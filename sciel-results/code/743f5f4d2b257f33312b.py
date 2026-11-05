"""Matrix multiplication 2x5 by 5x5 - Strassen-based block decomposition."""
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x5 matrix by a 5x5 matrix using Strassen's algorithm for 2x2 blocks.
    
    Strategy:
    - Split A (2x5) into A1 (2x2, cols 0-1), A2 (2x2, cols 2-3), A3 (2x1, col 4)
    - Split B (5x5) into 3x3 blocks of sizes (2,2,1) x (2,2,1)
    - Use Strassen (rank 7) for 2x2 * 2x2 sub-blocks
    - Use schoolbook for other cases
    """
    n, m, p = 2, 5, 5
    
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (10, rank)
    # W index convention: W[k*n + i, r] where i is row of output (0..1), k is col of output (0..4)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_entry, v_entry, w_entry):
        """Add one rank-1 term. u_entry=(row, coeff), v_entry=(row, coeff), w_entry=(row, coeff)"""
        U_rows.append([0.0] * 0)  # placeholder, we'll build columns
        # Actually, let's use a different approach: build the matrices column by column
        pass
    
    # Let's build the matrices as lists of columns
    # Each column c has: U[:, c], V[:, c], W[:, c]
    # U has 10 rows, V has 25 rows, W has 10 rows
    
    u_cols = []  # list of length-rank vectors of 10 elements
    v_cols = []  # list of length-rank vectors of 25 elements
    w_cols = []  # list of length-rank vectors of 10 elements
    
    def add_term(u_vec, v_vec, w_vec):
        """u_vec, v_vec, w_vec are dicts mapping row_index -> coefficient"""
        u_col = [0.0] * 10
        v_col = [0.0] * 25
        w_col = [0.0] * 10
        for r, c in u_vec.items():
            u_col[r] = c
        for r, c in v_vec.items():
            v_col[r] = c
        for r, c in w_vec.items():
            w_col[r] = c
        u_cols.append(u_col)
        v_cols.append(v_col)
        w_cols.append(w_col)
    
    def schoolbook_2x2_2x2(a_row_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Schoolbook multiplication of a 2x2 block of A by a 2x2 block of B.
        A block: rows a_row_offset..a_row_offset+1 (but A only has 2 rows, so this is the full 2x2 sub-block)
        Actually for A (2x5), the "2x2" block means:
          - A1: rows 0-1, cols 0-1 of A -> in U, these correspond to indices:
            A[i,j] maps to U[i*m + j] = U[i*5 + j]
            So A[0,0]->U[0], A[0,1]->U[1], A[1,0]->U[5], A[1,1]->U[6]
        
        B (5x5), block B[i,j]: rows i..i+1, cols j..j+1
          B[k,l] maps to V[k*p + l] = V[k*5 + l]
        
        C (2x5), block C[i,j]: rows i..i+1, cols j..j+1
          C[row, col] maps to W[col*n + row] = W[col*2 + row]
        """
        # For 2x2 * 2x2 schoolbook, we need 4 terms
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        
        # Each term is a product A[a,b]*B[c,d]
        # In U: row a*5+b
        # In V: row c*5+d
        # In W: row d*2+a (since W[k*n+i] where k=d is col, i=a is row)
        
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        # This gives A[a,b]*B[c,d] which contributes to C[a, b+c] 
                        # Wait, no. For standard matrix multiplication:
                        # C[i,j] = sum_k A[i,k]*B[k,j]
                        # For 2x2 blocks: C_block[i,j] = sum_k A_block[i,k]*B_block[k,j]
                        # The full C index: row = a (since A block starts at row 0 for A1), col = c_col_offset + j
                        # where j is the local col in the B block... 
                        
                        # Let me re-think. The B block is at rows (b_row_offset, b_row_offset+1) and cols (b_col_offset, b_col_offset+1)
                        # The A block is at rows (0,1) [since A only has 2 rows] and cols (a_col_offset, a_col_offset+1)
                        # C_block[i,j] = sum_k A_block[i,k] * B_block[k,j]
                        # Full C: row = i (0 or 1), col = b_col_offset + j
                        # W index: W[col*n + row] = W[(b_col_offset+j)*2 + i]
                        
                        u_row = a * 5 + (a_col_offset + b)
                        v_row = (b_row_offset + c) * 5 + (b_col_offset + d)
                        w_row = (b_col_offset + d) * 2 + a
                        
                        add_term({u_row: 1.0}, {v_row: 1.0}, {w_row: 1.0})
    
    def strassen_2x2_2x2(a_col_offset, b_row_offset, b_col_offset):
        """
        Strassen's algorithm for 2x2 * 2x2.
        A_block: A rows 0-1, cols a_col_offset to a_col_offset+1
        B_block: B rows b_row_offset to b_row_offset+1, cols b_col_offset to b_col_offset+1
        C_block: C rows 0-1, cols b_col_offset to b_col_offset+1
        
        Strassen formulas:
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
        
        Where A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
        B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
        
        For each Mi, we need to expand the products into rank-1 terms.
        Each Mi is a sum of products of linear combinations.
        """
        # A block elements:
        # A11 = A[0, a_col_offset] -> U[0*5 + a_col_offset]
        # A12 = A[0, a_col_offset+1] -> U[0*5 + a_col_offset+1]
        # A21 = A[1, a_col_offset] -> U[1*5 + a_col_offset]
        # A22 = A[1, a_col_offset+1] -> U[1*5 + a_col_offset+1]
        
        # B block elements:
        # B11 = B[b_row_offset, b_col_offset] -> V[b_row_offset*5 + b_col_offset]
        # B12 = B[b_row_offset, b_col_offset+1] -> V[b_row_offset*5 + b_col_offset+1]
        # B21 = B[b_row_offset+1, b_col_offset] -> V[(b_row_offset+1)*5 + b_col_offset]
        # B22 = B[b_row_offset+1, b_col_offset+1] -> V[(b_row_offset+1)*5 + b_col_offset+1]
        
        # W indices for C block:
        # C[0,0] -> W[b_col_offset*2 + 0]
        # C[0,1] -> W[(b_col_offset+1)*2 + 0]
        # C[1,0] -> W[b_col_offset*2 + 1]
        # C[1,1] -> W[(b_col_offset+1)*2 + 1]
        
        def make_u(a11_coeff, a12_coeff, a21_coeff, a22_coeff):
            vec = {}
            if a11_coeff != 0:
                vec[0*5 + a_col_offset] = vec.get(0*5 + a_col_offset, 0) + a11_coeff
            if a12_coeff != 0:
                vec[0*5 + a_col_offset+1] = vec.get(0*5 + a_col_offset+1, 0) + a12_coeff
            if a21_coeff != 0:
                vec[1*5 + a_col_offset] = vec.get(1*5 + a_col_offset, 0) + a21_coeff
            if a22_coeff != 0:
                vec[1*5 + a_col_offset+1] = vec.get(1*5 + a_col_offset+1, 0) + a22_coeff
            return vec
        
        def make_v(b11_coeff, b12_coeff, b21_coeff, b22_coeff):
            vec = {}
            if b11_coeff != 0:
                vec[b_row_offset*5 + b_col_offset] = vec.get(b_row_offset*5 + b_col_offset, 0) + b11_coeff
            if b12_coeff != 0:
                vec[b_row_offset*5 + b_col_offset+1] = vec.get(b_row_offset*5 + b_col_offset+1, 0) + b12_coeff
            if b21_coeff != 0:
                vec[(b_row_offset+1)*5 + b_col_offset] = vec.get((b_row_offset+1)*5 + b_col_offset, 0) + b21_coeff
            if b22_coeff != 0:
                vec[(b_row_offset+1)*5 + b_col_offset+1] = vec.get((b_row_offset+1)*5 + b_col_offset+1, 0) + b22_coeff
            return vec
        
        # M1 = (A11+A22)(B11+B22)
        add_term(make_u(1,0,0,1), make_v(1,0,0,1), {b_col_offset*2+0: 1, (b_col_offset+1)*2+1: 1})
        
        # M2 = (A21+A22)B11
        add_term(make_u(0,0,1,1), make_v(1,0,0,0), {b_col_offset*2+1: 1})
        
        # M3 = A11(B21-B22)
        add_term(make_u(1,0,0,0), make_v(0,0,1,-1), {(b_col_offset+1)*2+0: 1})
        
        # M4 = A22(B12-B11)
        add_term(make_u(0,0,0,1), make_v(-1,1,0,0), {b_col_offset*2+0: 1, (b_col_offset+1)*2+1: 1})
        
        # M5 = (A11+A12)B22
        add_term(make_u(1,1,0,0), make_v(0,0,0,1), {(b_col_offset+1)*2+0: -1})
        
        # M6 = (A21-A11)(B11+B12)
        add_term(make_u(-1,0,1,0), make_v(1,1,0,0), {(b_col_offset+1)*2+1: 1})
        
        # M7 = (A12-A22)(B21+B22)
        add_term(make_u(0,1,0,-1), make_v(0,0,1,1), {b_col_offset*2+0: 1, (b_col_offset+1)*2+0: 1})
    
    # Now let's build the full decomposition
    # A is 2x5, B is 5x5
    # Split A into: A1 (cols 0-1), A2 (cols 2-3), A3 (col 4)
    # Split B into 3x3 blocks:
    #   B[i,j] where i is row-block (0:rows 0-1, 1:rows 2-3, 2:row 4)
    #   j is col-block (0:cols 0-1, 1:cols 2-3, 2:col 4)
    
    # C = A*B, C is 2x5
    # C[:, 0:2] = A1*B[:,0:2] + A2*B[:,2:4] ... wait, no.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # C[:, 0:2] = sum over k=0..4 of A[:,k]*B[k,0:2]
    #           = A[:,0:2]*B[0:2,0:2] + A[:,2:4]*B[2:4,0:2] + A[:,4]*B[4,0:2]
    # C[:, 2:4] = A[:,0:2]*B[0:2,2:4] + A[:,2:4]*B[2:4,2:4] + A[:,4]*B[4,2:4]
    # C[:, 4]   = A[:,0:2]*B[0:2,4] + A[:,2:4]*B[2:4,4] + A[:,4]*B[4,4]
    
    # So the blocks are:
    # For output cols 0-1:
    #   A1 (2x2) * B11 (2x2) -> Strassen
    #   A2 (2x2) * B21 (2x2) -> Strassen
    #   A3 (2x1) * B31 (1x2) -> schoolbook (2*1*2=4 terms)
    # For output cols 2-3:
    #   A1 (2x2) * B12 (2x2) -> Strassen
    #   A2 (2x2) * B22 (2x2) -> Strassen
    #   A3 (2x1) * B32 (1x2) -> schoolbook (4 terms)
    # For output col 4:
    #   A1 (2x2) * B13 (2x1) -> schoolbook (2*2*1=4 terms)
    #   A2 (2x2) * B23 (2x1) -> schoolbook (4 terms)
    #   A3 (2x1) * B33 (1x1) -> schoolbook (2 terms)
    
    # Total: 7+7+4+7+7+4+4+4+2 = 46
    
    # Let's implement each block:
    
    # Block 1: A1 * B11 -> Strassen
    # A1: cols 0-1 of A, B11: rows 0-1, cols 0-1 of B
    strassen_2x2_2x2(a_col_offset=0, b_row_offset=0, b_col_offset=0)
    
    # Block 2: A2 * B21 -> Strassen
    # A2: cols 2-3 of A, B21: rows 2-3, cols 0-1 of B
    strassen_2x2_2x2(a_col_offset=2, b_row_offset=2, b_col_offset=0)
    
    # Block 3: A3 * B31 -> schoolbook
    # A3: col 4 of A (2x1), B31: row 4, cols 0-1 of B (1x2)
    # C[:,0:2] += A[:,4] * B[4,0:2]
    # C[0,0] += A[0,4]*B[4,0], C[0,1] += A[0,4]*B[4,1]
    # C[1,0] += A[1,4]*B[4,0], C[1,1] += A[1,4]*B[4,1]
    for i in range(2):
        for j in range(2):
            u_row = i * 5 + 4
            v_row = 4 * 5 + j
            w_row = j * 2 + i
            add_term({u_row: 1.0}, {v_row: 1.0}, {w_row: 1.0})
    
    # Block 4: A1 * B12 -> Strassen
    # A1: cols 0-1 of A, B12: rows 0-1, cols 2-3 of B
    strassen_2x2_2x2(a_col_offset=0, b_row_offset=0, b_col_offset=2)
    
    # Block 5: A2 * B22 -> Strassen
    # A2: cols 2-3 of A, B22: rows 2-3, cols 2-3 of B
    strassen_2x2_2x2(a_col_offset=2, b_row_offset=2, b_col_offset=2)
    
    # Block 6: A3 * B32 -> schoolbook
    # A3: col 4 of A, B32: row 4, cols 2-3 of B
    for i in range(2):
        for j in range(2):
            u_row = i * 5 + 4
            v_row = 4 * 5 + (2 + j)
            w_row = (2 + j) * 2 + i
            add_term({u_row: 1.0}, {v_row: 1.0}, {w_row: 1.0})
    
    # Block 7: A1 * B13 -> schoolbook
    # A1: cols 0-1 of A (2x2), B13: rows 0-1, col 4 of B (2x1)
    # C[:,4] += A[:,0:2] * B[0:2,4]
    # C[0,4] += A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] += A[1,0]*B[0,4] + A[1,1]*B[1,4]
    for a in range(2):
        for b in range(2):
            u_row = a * 5 + b
            v_row = b * 5 + 4
            w_row = 4 * 2 + a
            add_term({u_row: 1.0}, {v_row: 1.0}, {w_row: 1.0})
    
    # Block 8: A2 * B23 -> schoolbook
    # A2: cols 2-3 of A (2x2), B23: rows 2-3, col 4 of B (2x1)
    for a in range(2):
        for b in range(2):
            u_row = a * 5 + (2 + b)
            v_row = (2 + b) * 5 + 4
            w_row = 4 * 2 + a
            add_term({u_row: 1.0}, {v_row: 1.0}, {w_row: 1.0})
    
    # Block 9: A3 * B33 -> schoolbook
    # A3: col 4 of A (2x1), B33: row 4, col 4 of B (1x1)
    for i in range(2):
        u_row = i * 5 + 4
        v_row = 4 * 5 + 4
        w_row = 4 * 2 + i
        add_term({u_row: 1.0}, {v_row: 1.0}, {w_row: 1.0})
    
    # Convert to matrix format
    rank = len(u_cols)
    U = [[0] * rank for _ in range(10)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(10)]
    
    for c in range(rank):
        for r in range(10):
            U[r][c] = u_cols[c][r]
        for r in range(25):
            V[r][c] = v_cols[c][r]
        for r in range(10):
            W[r][c] = w_cols[c][r]
    
    # Convert to integers where possible
    def to_int_if_close(x):
        ix = round(x)
        if abs(x - ix) < 1e-10:
            return int(ix)
        return x
    
    U = [[to_int_if_close(x) for x in row] for row in U]
    V = [[to_int_if_close(x) for x in row] for row in V]
    W = [[to_int_if_close(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
