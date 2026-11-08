"""Rewritten program using a hierarchical Strassen decomposition for 3x5x7 matrix multiplication."""

def construct():
    n, m, p = 3, 5, 7
    # We'll build U, V, W as lists of columns (each column is a list of floats/ints)
    # U has n*m = 15 rows, V has m*p = 35 rows, W has p*n = 21 rows
    # Each column corresponds to a rank-1 term.
    
    # We'll partition the dimensions into blocks and apply Strassen where possible.
    # Partition of n=3: [0,1], [2] -> sizes [2, 1]
    # Partition of m=5: [0,1], [2,3], [4] -> sizes [2, 2, 1]
    # Partition of p=7: [0,1], [2,3], [4,5], [6] -> sizes [2, 2, 2, 1]
    
    # Block definitions:
    # A rows: 0-1 (size 2), 2 (size 1)
    # A cols (= intermediate m): 0-1 (size 2), 2-3 (size 2), 4 (size 1)
    # B rows (= intermediate m): 0-1 (size 2), 2-3 (size 2), 4 (size 1)
    # B cols (= output p): 0-1 (size 2), 2-3 (size 2), 4-5 (size 2), 6 (size 1)
    
    # For each combination of (A_row_block, A_col_block, B_row_block, B_col_block)
    # where A_col_block matches B_row_block (same intermediate dimension block),
    # we compute the product of that sub-block and add its decomposition.
    
    # A row blocks: [(0, 2, 2), (2, 3, 1)]  -> (start, end, size)
    A_row_blocks = [(0, 2, 2), (2, 3, 1)]
    # A col blocks (intermediate): [(0, 2, 2), (2, 4, 2), (4, 5, 1)]
    A_col_blocks = [(0, 2, 2), (2, 4, 2), (4, 5, 1)]
    # B row blocks (intermediate): same as A col blocks
    B_row_blocks = [(0, 2, 2), (2, 4, 2), (4, 5, 1)]
    # B col blocks (output): [(0, 2, 2), (2, 4, 2), (4, 6, 2), (6, 7, 1)]
    B_col_blocks = [(0, 2, 2), (2, 4, 2), (4, 6, 2), (6, 7, 1)]
    
    U_cols = []  # Each is a list of length n*m = 15
    V_cols = []  # Each is a list of length m*p = 35
    W_cols = []  # Each is a list of length p*n = 21
    
    def add_strassen_2x2x2(U_col_base, V_col_base, W_col_base, a_row_start, a_col_start, b_row_start, b_col_start):
        """Add Strassen decomposition for a 2x2x2 sub-block.
        U_col_base: index in U where the 4 elements of the 2x2 A block start (row-major)
        V_col_base: index in V where the 4 elements of the 2x2 B block start
        W_col_base: index in W where the 4 elements of the 2x2 C block start (k*n+i convention)
        """
        # A block: a00, a01, a10, a11 at U_col_base + 0, 1, 2, 3
        # B block: b00, b01, b10, b11 at V_col_base + 0, 1, 2, 3
        # C block: c00, c01, c10, c11 at W_col_base + 0, 1, 2, 3
        # But W uses k*n+i convention: c_{k,i} for k=0,1 and i=0,1
        # W_col_base + k*2 + i for local k,i
        
        # Strassen's 7 products:
        # M1 = (A10 + A01) * (B00 + B11)
        # M2 = (A10 + A11) * B01
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A11) * (B00 + B11)
        # M6 = (A01 - A11) * (B10 + B11)
        # M7 = (A00 - A10) * (B00 + B10)
        
        # U indices for A elements:
        # A00: U_col_base + 0
        # A01: U_col_base + 1
        # A10: U_col_base + 2
        # A11: U_col_base + 3
        
        # V indices for B elements:
        # B00: V_col_base + 0
        # B01: V_col_base + 1
        # B10: V_col_base + 2
        # B11: V_col_base + 3
        
        # W indices for C elements (k*n+i, local n=2):
        # C00 (k=0,i=0): W_col_base + 0
        # C01 (k=0,i=1): W_col_base + 1
        # C10 (k=1,i=0): W_col_base + 2
        # C11 (k=1,i=1): W_col_base + 3
        
        # Each Strassen term is a rank-1 contribution:
        # We need to express C = U * V^T * W^T where each term is a column.
        # For each Strassen product M_i = (linear combo of A) * (linear combo of B),
        # this contributes to C via: c += M_i * (some pattern in W)
        
        # Actually, the standard Strassen decomposition works as:
        # C = U_strassen * diag(M1..M7) * V_strassen
        # where U_strassen and V_strassen are 4x7 matrices.
        
        # Let's define the 7 columns for U, V, W:
        # U_col[j] has non-zeros at A00, A01, A10, A11 positions
        # V_col[j] has non-zeros at B00, B01, B10, B11 positions
        # W_col[j] has non-zeros at C00, C01, C10, C11 positions
        
        # Term 1: M1 = (A10 + A01)(B00 + B11)
        # U1: A01=1, A10=1
        # V1: B00=1, B11=1
        # C00 += M1, C01 -= M2... wait, let me use the standard formulas.
        
        # Standard Strassen:
        # C00 = M5 + M4 - M2 - M1  (wait, let me look this up properly)
        # 
        # Actually:
        # C00 = M5 + M4 - M2 - M1
        # C01 = M1 + M2
        # C10 = M3 + M5
        # C11 = M5 - M4 + M6 - M7
        
        # So for each M_i, we add its contribution to the C elements:
        # M1: C00 += -M1, C01 += M1
        # M2: C00 += -M2, C01 += M2
        # M3: C10 += M3
        # M4: C00 += M4, C11 += -M4
        # M5: C00 += M5, C10 += M5, C11 += M5
        # M6: C11 += M6
        # M7: C11 += -M7
        
        # So the W columns are:
        # W1: C00=-1, C01=1
        # W2: C00=-1, C01=1
        # W3: C10=1
        # W4: C00=1, C11=-1
        # W5: C00=1, C10=1, C11=1
        # W6: C11=1
        # W7: C11=-1
        
        # And U columns:
        # U1: A01=1, A10=1
        # U2: A10=1, A11=1
        # U3: A00=1
        # U4: A11=1
        # U5: A00=1, A11=1
        # U6: A01=1, A11=-1
        # U7: A00=1, A10=-1
        
        # V columns:
        # V1: B00=1, B11=1
        # V2: B01=1
        # V3: B01=1, B11=-1
        # V4: B10=1, B00=-1
        # V5: B00=1, B11=1
        # V6: B10=1, B11=1
        # V7: B00=1, B10=1
        
        u_vals = [
            [0, 1, 1, 0],  # U1: A01, A10
            [0, 0, 1, 1],  # U2: A10, A11
            [1, 0, 0, 0],  # U3: A00
            [0, 0, 0, 1],  # U4: A11
            [1, 0, 0, 1],  # U5: A00, A11
            [0, 1, 0, -1], # U6: A01, -A11
            [1, 0, -1, 0], # U7: A00, -A10
        ]
        v_vals = [
            [1, 0, 0, 1],  # V1: B00, B11
            [0, 1, 0, 0],  # V2: B01
            [0, 1, 0, -1], # V3: B01, -B11
            [-1, 0, 1, 0], # V4: -B00, B10
            [1, 0, 0, 1],  # V5: B00, B11
            [0, 0, 1, 1],  # V6: B10, B11
            [1, 0, 1, 0],  # V7: B00, B10
        ]
        w_vals = [
            [-1, 1, 0, 0],  # W1: -C00, C01
            [-1, 1, 0, 0],  # W2: -C00, C01
            [0, 0, 1, 0],   # W3: C10
            [1, 0, 0, -1],  # W4: C00, -C11
            [1, 0, 1, 1],   # W5: C00, C10, C11
            [0, 0, 0, 1],   # W6: C11
            [0, 0, 0, -1],  # W7: -C11
        ]
        
        for t in range(7):
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            # Set U values
            for local_j, val in enumerate(u_vals[t]):
                if val != 0:
                    U_col[U_col_base + local_j] = val
            
            # Set V values
            for local_k, val in enumerate(v_vals[t]):
                if val != 0:
                    V_col[V_col_base + local_k] = val
            
            # Set W values (local indices: 0=C00, 1=C01, 2=C10, 3=C11)
            for local_idx, val in enumerate(w_vals[t]):
                if val != 0:
                    W_col[W_col_base + local_idx] = val
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    def add_schoolbook(U_col_base, V_col_base, W_col_base, h, w, q):
        """Add schoolbook decomposition for an h x w times w x q sub-block."""
        for local_i in range(h):
            for local_j in range(w):
                for local_k in range(q):
                    U_col = [0] * (n * m)
                    V_col = [0] * (m * p)
                    W_col = [0] * (p * n)
                    U_col[U_col_base + local_i * w + local_j] = 1
                    V_col[V_col_base + local_j * q + local_k] = 1
                    W_col[W_col_base + local_k * h + local_i] = 1
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
    
    # Iterate over all combinations
    for a_row_idx, (a_row_start, a_row_end, a_row_size) in enumerate(A_row_blocks):
        for a_col_idx, (a_col_start, a_col_end, a_col_size) in enumerate(A_col_blocks):
            # B row block must match A col block
            if a_col_idx < len(B_row_blocks):
                b_row_start, b_row_end, b_row_size = B_row_blocks[a_col_idx]
                if a_col_start == b_row_start and a_col_end == b_row_end:
                    for b_col_idx, (b_col_start, b_col_end, b_col_size) in enumerate(B_col_blocks):
                        h = a_row_size
                        w = a_col_size
                        q = b_col_size
                        
                        # U base index: a_row_start * m + a_col_start
                        U_base = a_row_start * m + a_col_start
                        # V base index: b_row_start * p + b_col_start
                        V_base = b_row_start * p + b_col_start
                        # W base index: b_col_start * n + a_row_start
                        W_base = b_col_start * n + a_row_start
                        
                        if h == 2 and w == 2 and q == 2:
                            add_strassen_2x2x2(U_base, V_base, W_base, a_row_start, a_col_start, b_row_start, b_col_start)
                        else:
                            add_schoolbook(U_base, V_base, W_base, h, w, q)
    
    # Convert to the required format: lists of lists (rows x columns)
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r in range(n * m):
            U[r][c] = U_cols[c][r]
        for r in range(m * p):
            V[r][c] = V_cols[c][r]
        for r in range(p * n):
            W[r][c] = W_cols[c][r]
    
    return {"U": U, "V": V, "W": W}
