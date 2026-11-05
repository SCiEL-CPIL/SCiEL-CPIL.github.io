"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def is_valid(U, V, W, rank):
        """Check if U,V,W with given rank exactly decompose the tensor."""
        T = np.zeros((nm, mp, pn))
        for c in range(rank):
            T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        T_target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m + j, j*p + k, k*n + i] = 1.0
        
        return np.allclose(T, T_target, atol=1e-9)
    
    def add_term(U, V, W, col, a_dict, b_dict, c_dict):
        """Add a rank-1 term. a_dict: {(i,j): coeff}, b_dict: {(j',k): coeff}, c_dict: {(i',k'): coeff}"""
        for (i, j), val in a_dict.items():
            U[i*m + j, col] += val
        for (j2, k), val in b_dict.items():
            V[j2*p + k, col] += val
        for (i2, k2), val in c_dict.items():
            W[k2*n + i2, col] += val
    
    def strassen_2x2(U, V, W, col, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add Strassen decomposition for 2x2 block mult.
        a_rows: 2 row indices for A, a_cols: 2 col indices for A
        b_rows: 2 row indices for B, b_cols: 2 col indices for B
        c_rows: 2 row indices for C, c_cols: 2 col indices for C
        Returns new col index after adding 7 terms.
        """
        a00, a01 = a_rows[0], a_rows[1]
        a10, a11 = a_cols[0], a_cols[1]
        # A[a00, a10], A[a00, a11], A[a01, a10], A[a01, a11]
        
        b00, b01 = b_rows[0], b_rows[1]
        b10, b11 = b_cols[0], b_cols[1]
        # B[b00, b10], B[b00, b11], B[b01, b10], B[b01, b11]
        
        c00, c01 = c_rows[0], c_rows[1]
        c10, c11 = c_cols[0], c_cols[1]
        # C[c00, c10], C[c00, c11], C[c01, c10], C[c01, c11]
        
        # P1 = (A[00]+A[11])(B[00]+B[11])
        a = {(a00, a10): 1, (a01, a11): 1}
        b = {(b00, b10): 1, (b01, b11): 1}
        c = {(c00, c10): 1, (c01, c11): 1}
        add_term(U, V, W, col, a, b, c)
        col += 1
        
        # P2 = (A[10]+A[11])B[00]
        a = {(a01, a10): 1, (a01, a11): 1}
        b = {(b00, b10): 1}
        c = {(c01, c10): 1, (c01, c11): -1}
        add_term(U, V, W, col, a, b, c)
        col += 1
        
        # P3 = A[00](B[01]-B[11])
        a = {(a00, a10): 1}
        b = {(b00, b11): 1, (b01, b11): -1}
        c = {(c00, c11): 1, (c01, c11): 1}
        add_term(U, V, W, col, a, b, c)
        col += 1
        
        # P4 = A[11](B[10]-B[00])
        a = {(a01, a11): 1}
        b = {(b01, b10): 1, (b00, b10): -1}
        c = {(c00, c10): 1, (c01, c10): 1}
        add_term(U, V, W, col, a, b, c)
        col += 1
        
        # P5 = (A[00]+A[01])B[11]
        a = {(a00, a10): 1, (a00, a11): 1}
        b = {(b01, b11): 1}
        c = {(c00, c10): -1, (c00, c11): 1}
        add_term(U, V, W, col, a, b, c)
        col += 1
        
        # P6 = (A[10]-A[00])(B[00]+B[10])
        a = {(a01, a10): 1, (a00, a10): -1}
        b = {(b00, b10): 1, (b01, b10): 1}
        c = {(c01, c11): 1}
        add_term(U, V, W, col, a, b, c)
        col += 1
        
        # P7 = (A[01]-A[11])(B[01]+B[11])
        a = {(a00, a11): 1, (a01, a11): -1}
        b = {(b00, b11): 1, (b01, b11): 1}
        c = {(c00, c10): 1}
        add_term(U, V, W, col, a, b, c)
        col += 1
        
        return col
    
    def multiply_1x2_2x2(U, V, W, col, a_row, a_cols, b_rows, b_cols, c_row, c_cols):
        """Add decomposition for 1x2 * 2x2 multiplication (4 scalar mults).
        a_row: row index for A
        a_cols: 2 col indices for A
        b_rows: 2 row indices for B
        b_cols: 2 col indices for B
        c_row: row index for C
        c_cols: 2 col indices for C
        """
        # C[c_row, c_cols[0]] = A[a_row, a_cols[0]]*B[b_rows[0], c_cols[0]] + A[a_row, a_cols[1]]*B[b_rows[1], c_cols[0]]
        # C[c_row, c_cols[1]] = A[a_row, a_cols[0]]*B[b_rows[0], c_cols[1]] + A[a_row, a_cols[1]]*B[b_rows[1], c_cols[1]]
        
        # Term 1: A[a_row, a_cols[0]] * B[b_rows[0], c_cols[0]] -> C[c_row, c_cols[0]]
        add_term(U, V, W, col, {(a_row, a_cols[0]): 1}, {(b_rows[0], c_cols[0]): 1}, {(c_row, c_cols[0]): 1})
        col += 1
        
        # Term 2: A[a_row, a_cols[1]] * B[b_rows[1], c_cols[0]] -> C[c_row, c_cols[0]]
        add_term(U, V, W, col, {(a_row, a_cols[1]): 1}, {(b_rows[1], c_cols[0]): 1}, {(c_row, c_cols[0]): 1})
        col += 1
        
        # Term 3: A[a_row, a_cols[0]] * B[b_rows[0], c_cols[1]] -> C[c_row, c_cols[1]]
        add_term(U, V, W, col, {(a_row, a_cols[0]): 1}, {(b_rows[0], c_cols[1]): 1}, {(c_row, c_cols[1]): 1})
        col += 1
        
        # Term 4: A[a_row, a_cols[1]] * B[b_rows[1], c_cols[1]] -> C[c_row, c_cols[1]]
        add_term(U, V, W, col, {(a_row, a_cols[1]): 1}, {(b_rows[1], c_cols[1]): 1}, {(c_row, c_cols[1]): 1})
        col += 1
        
        return col
    
    # Build Strassen-based decomposition
    # A is 3x4, B is 4x4, C is 3x4
    # Split A: rows [0,1] and [2], cols [0,1] and [2,3]
    # Split B: rows [0,1] and [2,3], cols [0,1] and [2,3]
    # C: rows [0,1] and [2], cols [0,1] and [2,3]
    
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    # C[2:3, 0:2] = A[2:3, 0:2]*B[0:2, 0:2] + A[2:3, 2:4]*B[2:4, 0:2]
    # C[2:3, 2:4] = A[2:3, 0:2]*B[0:2, 2:4] + A[2:3, 2:4]*B[2:4, 2:4]
    
    # Total rank: 4*7 + 4*4 = 28 + 16 = 44
    
    max_rank = 44
    U = np.zeros((nm, max_rank))
    V = np.zeros((mp, max_rank))
    W = np.zeros((pn, max_rank))
    
    col = 0
    
    # C[0:2, 0:2] = A1*B1 + A2*B3
    # A1: rows [0,1], cols [0,1]
    # B1: rows [0,1], cols [0,1]
    # C1: rows [0,1], cols [0,1]
    col = strassen_2x2(U, V, W, col, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # A2: rows [0,1], cols [2,3]
    # B3: rows [2,3], cols [0,1]
    # C1: rows [0,1], cols [0,1]
    col = strassen_2x2(U, V, W, col, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # C[0:2, 2:4] = A1*B2 + A2*B4
    # A1: rows [0,1], cols [0,1]
    # B2: rows [0,1], cols [2,3]
    # C2: rows [0,1], cols [2,3]
    col = strassen_2x2(U, V, W, col, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # A2: rows [0,1], cols [2,3]
    # B4: rows [2,3], cols [2,3]
    # C2: rows [0,1], cols [2,3]
    col = strassen_2x2(U, V, W, col, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # C[2:3, 0:2] = A3*B1 + A4*B3
    # A3: row [2], cols [0,1]
    # B1: rows [0,1], cols [0,1]
    # C3: row [2], cols [0,1]
    col = multiply_1x2_2x2(U, V, W, col, 2, [0, 1], [0, 1], [0, 1], 2, [0, 1])
    
    # A4: row [2], cols [2,3]
    # B3: rows [2,3], cols [0,1]
    # C3: row [2], cols [0,1]
    col = multiply_1x2_2x2(U, V, W, col, 2, [2, 3], [2, 3], [0, 1], 2, [0, 1])
    
    # C[2:3, 2:4] = A3*B2 + A4*B4
    # A3: row [2], cols [0,1]
    # B2: rows [0,1], cols [2,3]
    # C4: row [2], cols [2,3]
    col = multiply_1x2_2x2(U, V, W, col, 2, [0, 1], [0, 1], [2, 3], 2, [2, 3])
    
    # A4: row [2], cols [2,3]
    # B4: rows [2,3], cols [2,3]
    # C4: row [2], cols [2,3]
    col = multiply_1x2_2x2(U, V, W, col, 2, [2, 3], [2, 3], [2, 3], 2, [2, 3])
    
    rank = col
    print(f"Constructed rank {rank} decomposition")
    
    # Verify
    if is_valid(U, V, W, rank):
        print("Strassen decomposition verified!")
    else:
        print("ERROR: Strassen decomposition failed verification!")
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        rank = rank_sb
    
    result = {"U": U[:, :rank].tolist(), "V": V[:, :rank].tolist(), "W": W[:, :rank].tolist()}
    return result
# EVOLVE-BLOCK-END
