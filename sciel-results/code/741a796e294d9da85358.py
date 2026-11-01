import numpy as np
import time

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0

def _generate_strassen_decomposition():
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Applied to 4 blocks of 2x2 matrices
    # Total rank 4 * 7 = 28
    
    # The 7 Strassen columns for a 2x2 multiplication:
    # Each column is defined by coefficients for A, B, and C parts.
    # A is 2x2, B is 2x2, C is 2x2.
    # local_idx 0..3 for 2x2 matrices (row-major: 00, 01, 10, 11)
    
    strassen_cols = [
        # M1
        {0: [(0, 1), (3, 1)], 1: [(0, 1), (3, 1)], 2: [(0, 1), (3, 1)]},
        # M2
        {0: [(2, 1), (3, 1)], 1: [(0, 1)], 2: [(2, 1), (3, -1)]},
        # M3
        {0: [(0, 1)], 1: [(1, 1), (3, -1)], 2: [(1, 1), (3, 1)]},
        # M4
        {0: [(3, 1)], 1: [(2, 1), (0, -1)], 2: [(0, 1), (2, 1)]},
        # M5
        {0: [(0, 1), (1, 1)], 1: [(3, 1)], 2: [(0, -1), (1, 1)]},
        # M6
        {0: [(2, 1), (0, -1)], 1: [(0, 1), (1, 1)], 2: [(3, 1)]},
        # M7
        {0: [(1, 1), (3, -1)], 1: [(2, 1), (3, 1)], 2: [(0, 1)]}
    ]
    
    # Dimensions: n=2, m=4, p=4
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    U = np.zeros((8, 28), dtype=int)
    V = np.zeros((16, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    def map_A(local_idx, col_block_offset):
        # local_idx is 0..3 for 2x2 block
        # col_block_offset is 0 or 1 (since m=4, split into two 2-wide blocks? 
        # Wait, m=4. A is 2x4. We split A into two 2x2 blocks horizontally?
        # A = [A_left, A_right]
        # B = [[B_tl, B_tr], [B_bl, B_br]]
        # C = A B = [A_left B_tl + A_right B_bl, A_left B_tr + A_right B_br]
        # This requires 4 multiplications of 2x2 matrices.
        # The current code splits m into 2 blocks of size 2?
        # map_A(local_idx, offset):
        # r = local_idx // 2, c = local_idx % 2
        # return r * 4 + (offset * 2 + c)
        # If offset=0, cols 0,1. If offset=1, cols 2,3.
        # Yes, this splits A into two 2x2 blocks.
        r = local_idx // 2
        c = local_idx % 2
        return r * 4 + (offset * 2 + c)
        
    def map_B(local_idx, row_block_offset, col_block_offset):
        # B is 4x4. Split into 2x2 blocks.
        # row_block_offset 0 or 1, col_block_offset 0 or 1.
        r = local_idx // 2
        c = local_idx % 2
        return (row_block_offset * 2 + r) * 4 + (col_block_offset * 2 + c)
        
    def map_C(local_idx, col_block_offset):
        # C is 2x4. Split into two 2x2 blocks horizontally.
        r = local_idx // 2
        c = local_idx % 2
        return (col_block_offset * 2 + c) * 2 + r

    # We have 4 combinations of B blocks: (0,0), (1,0), (0,1), (1,1)
    # And 2 combinations of A blocks: (0), (1)
    # Wait, the logic in the current code:
    # Loop 1: A_offset=0, B_row=0, B_col=0. C_offset=0.
    # Loop 2: A_offset=1, B_row=1, B_col=0. C_offset=0.
    # Loop 3: A_offset=0, B_row=0, B_col=1. C_offset=1.
    # Loop 4: A_offset=1, B_row=1, B_col=1. C_offset=1.
    
    # This corresponds to:
    # C_left = A_left B_tl + A_right B_bl
    # C_right = A_left B_tr + A_right B_br
    
    # Each term is a 2x2 multiplication.
    # Term 1: A_left (offset 0) * B_tl (row 0, col 0) -> C_left (offset 0)
    # Term 2: A_right (offset 1) * B_bl (row 1, col 0) -> C_left (offset 0)
    # Term 3: A_left (offset 0) * B_tr (row 0, col 1) -> C_right (offset 1)
    # Term 4: A_right (offset 1) * B_br (row 1, col 1) -> C_right (offset 1)
    
    # The code iterates j from 0 to 6 for each of the 4 blocks.
    
    for block_idx, (a_off, b_row, b_col, c_off) in enumerate([
        (0, 0, 0, 0),
        (1, 1, 0, 0),
        (0, 0, 1, 1),
        (1, 1, 1, 1)
    ]):
        for j, col_def in enumerate(strassen_cols):
            col_idx = block_idx * 7 + j
            
            for loc_idx, coeff in col_def[0]:
                U[map_A(loc_idx, a_off), col_idx] += coeff
            for loc_idx, coeff in col_def[1]:
                V[map_B(loc_idx, b_row, b_col), col_idx] += coeff
            for loc_idx, coeff in col_def[2]:
                W[map_C(loc_idx, c_off), col_idx] += coeff
                
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U, V, W = _generate_strassen_decomposition()
    
    if time.time() - start_time > DEADLINE:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Mismatch")
        print(C)
        print(Cf)
    else:
        print("OK rank", U.shape[1])
