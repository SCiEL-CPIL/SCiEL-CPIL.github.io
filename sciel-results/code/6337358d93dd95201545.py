import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=3, m=4, p=6
    # U rows: i*m+j
    # V rows: j*p+k
    # W rows: k*n+i
    
    U = []
    V = []
    W = []
    
    def add_term(u_indices, v_indices, w_indices):
        """
        u_indices: list of (row, coeff) for U
        v_indices: list of (row, coeff) for V
        w_indices: list of (row, coeff) for W
        """
        # Create zero vectors
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        for r, c in u_indices:
            u_col[r] += c
        for r, c in v_indices:
            v_col[r] += c
        for r, c in w_indices:
            w_col[r] += c
            
        U.append(u_col)
        V.append(v_col)
        W.append(w_col)
        
    def add_block_1x4x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """
        Adds decomposition for 1x4 A * 4x2 B -> 1x2 C.
        Rank 8.
        a_row_start: row index in A (size 1)
        a_col_start: col index in A (size 4)
        b_row_start: row index in B (size 4)
        b_col_start: col index in B (size 2)
        c_row_start: row index in C (size 1)
        c_col_start: col index in C (size 2)
        """
        # A is 1x4. Row is a_row_start. Cols a_col_start .. a_col_start+3.
        # B is 4x2. Rows b_row_start .. b_row_start+3. Cols b_col_start .. b_col_start+1.
        # C is 1x2. Row is c_row_start. Cols c_col_start .. c_col_start+1.
        
        for i in range(4): # A col / B row
            for k in range(2): # B col / C col
                # U index: a_row_start * m + (a_col_start + i)
                u_idx = a_row_start * m + (a_col_start + i)
                # V index: (b_row_start + i) * p + (b_col_start + k)
                v_idx = (b_row_start + i) * p + (b_col_start + k)
                # W index: (c_col_start + k) * n + c_row_start
                w_idx = (c_col_start + k) * n + c_row_start
                
                add_term([(u_idx, 1.0)], [(v_idx, 1.0)], [(w_idx, 1.0)])
                
    def add_block_2x4x2(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """
        Adds decomposition for 2x4 A * 4x2 B -> 2x2 C.
        Rank 14 (2 * 7).
        a_row_start: row index in A (size 2)
        a_col_start: col index in A (size 4)
        b_row_start: row index in B (size 4)
        b_col_start: col index in B (size 2)
        c_row_start: row index in C (size 2)
        c_col_start: col index in C (size 2)
        """
        # Split inner dimension 4 into two 2s.
        # Part 1: A[0:2, 0:2] * B[0:2, 0:2]
        add_strassen_2x2x2(
            a_row_start + 0, a_col_start + 0, 
            b_row_start + 0, b_col_start + 0, 
            c_row_start + 0, c_col_start + 0
        )
        # Part 2: A[0:2, 2:4] * B[2:4, 0:2]
        add_strassen_2x2x2(
            a_row_start + 0, a_col_start + 2, 
            b_row_start + 2, b_col_start + 0, 
            c_row_start + 0, c_col_start + 0
        )
        
    def add_strassen_2x2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Adds Strassen decomposition for 2x2 A * 2x2 B -> 2x2 C.
        a_r, a_c: start row/col of A
        b_r, b_c: start row/col of B
        c_r, c_c: start row/col of C
        """
        # Indices in local 2x2:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        def u_idx(r, c): return (a_r + r) * m + (a_c + c)
        def v_idx(r, c): return (b_r + r) * p + (b_c + c)
        def w_idx(r, c): return (c_c + c) * n + (c_r + r)
        
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        add_term(
            [(u_idx(0,0), 1.0), (u_idx(1,1), 1.0)],
            [(v_idx(0,0), 1.0), (v_idx(1,1), 1.0)],
            [(w_idx(0,0), 1.0), (w_idx(1,1), 1.0)]
        )
        
        # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        add_term(
            [(u_idx(1,0), 1.0), (u_idx(1,1), 1.0)],
            [(v_idx(0,0), 1.0)],
            [(w_idx(1,0), 1.0), (w_idx(1,1), -1.0)]
        )
        
        # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        add_term(
            [(u_idx(0,0), 1.0)],
            [(v_idx(0,1), 1.0), (v_idx(1,1), -1.0)],
            [(w_idx(0,1), 1.0), (w_idx(1,1), 1.0)]
        )
        
        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        add_term(
            [(u_idx(1,1), 1.0)],
            [(v_idx(1,0), 1.0), (v_idx(0,0), -1.0)],
            [(w_idx(0,0), 1.0), (w_idx(1,0), 1.0)]
        )
        
        # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        add_term(
            [(u_idx(0,0), 1.0), (u_idx(0,1), 1.0)],
            [(v_idx(1,1), 1.0)],
            [(w_idx(0,0), -1.0), (w_idx(0,1), 1.0)]
        )
        
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        add_term(
            [(u_idx(1,0), 1.0), (u_idx(0,0), -1.0)],
            [(v_idx(0,0), 1.0), (v_idx(0,1), 1.0)],
            [(w_idx(1,1), 1.0)]
        )
        
        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        add_term(
            [(u_idx(0,1), 1.0), (u_idx(1,1), -1.0)],
            [(v_idx(1,0), 1.0), (v_idx(1,1), 1.0)],
            [(w_idx(0,0), 1.0)]
        )

    # Partition:
    # n=3: rows 0 (size 1), 1-2 (size 2)
    # m=4: cols 0-1 (size 2), 2-3 (size 2)
    # p=6: cols 0-1 (size 2), 2-3 (size 2), 4-5 (size 2)
    
    # Block 1: C[0, 0:2] = A[0, 0:4] * B[0:4, 0:2]
    add_block_1x4x2(0, 0, 0, 0, 0, 0)
    
    # Block 2: C[0, 2:4] = A[0, 0:4] * B[0:4, 2:4]
    add_block_1x4x2(0, 0, 0, 2, 0, 2)
    
    # Block 3: C[0, 4:6] = A[0, 0:4] * B[0:4, 4:6]
    add_block_1x4x2(0, 0, 0, 4, 0, 4)
    
    # Block 4: C[1:3, 0:2] = A[1:3, 0:4] * B[0:4, 0:2]
    add_block_2x4x2(1, 0, 0, 0, 1, 0)
    
    # Block 5: C[1:3, 2:4] = A[1:3, 0:4] * B[0:4, 2:4]
    add_block_2x4x2(1, 0, 0, 2, 1, 2)
    
    # Block 6: C[1:3, 4:6] = A[1:3, 0:4] * B[0:4, 4:6]
    add_block_2x4x2(1, 0, 0, 4, 1, 4)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick verification
    result = construct()
    n, m, p = DIMENSIONS
    import numpy as np
    U = np.array(result['U']).T # shape (rank, n*m)
    V = np.array(result['V']).T # shape (rank, m*p)
    W = np.array(result['W']).T # shape (rank, p*n)
    
    # Construct tensor T[i, j, k]
    # T is rank decomposition: sum_r U[i, r] V[j, r] W[k, r]
    # i in n*m, j in m*p, k in p*n
    # We want to check if this equals the matrix multiplication tensor.
    # MM tensor: T[i, j, k] = 1 if i=j and j=k? No.
    # MM tensor for (n,m,p):
    # Index i (n*m) -> (i_n, i_m)
    # Index j (m*p) -> (j_m, j_p)
    # Index k (p*n) -> (k_p, k_n)
    # T[i, j, k] = 1 if i_m == j_m and j_p == k_p and i_n == k_n.
    
    tensor = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    expected = np.zeros((n*m, m*p, p*n))
    for i_n in range(n):
        for i_m in range(m):
            for j_p in range(p):
                # i = i_n * m + i_m
                # j = i_m * p + j_p
                # k = j_p * n + i_n
                i = i_n * m + i_m
                j = i_m * p + j_p
                k = j_p * n + i_n
                expected[i, j, k] = 1.0
                
    if np.allclose(tensor, expected):
        print("Decomposition is valid.")
        print(f"Rank: {U.shape[0]}")
    else:
        print("Decomposition is INVALID.")
        diff = np.abs(tensor - expected)
        print(f"Max diff: {np.max(diff)}")
