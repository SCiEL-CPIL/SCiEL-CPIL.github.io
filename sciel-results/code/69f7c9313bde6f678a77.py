import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for matrix multiplication M_{3,3,8}.
    Uses a partitioned approach:
    - M_{2,2,8} decomposed into two M_{2,2,4} blocks, each using two Strassen M_{2,2,2} blocks (rank 14 each, total 28).
    - M_{2,1,8} schoolbook (rank 16).
    - M_{1,2,8} schoolbook (rank 16).
    - M_{1,1,8} schoolbook (rank 8).
    Total rank: 28 + 16 + 16 + 8 = 68.
    """
    n, m, p = DIMENSIONS
    rank = 68
    
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a Strassen M_{2,2,2} decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # Indices in A: (0,0), (0,1), (1,0), (1,1)
    # Indices in B: (0,0), (0,1), (1,0), (1,1)
    # Indices in C: (0,0), (0,1), (1,0), (1,1)
    # Strassen products:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    
    def add_strassen(a_base_i, a_base_j, b_base_j, c_base_i, c_base_k, start_col):
        # a_base_i, a_base_j: base indices for A (2x2 block)
        # b_base_j: base index for B rows (2x2 block, cols are 2)
        # c_base_i, c_base_k: base indices for C
        # Actually, B is 2x2, so b_base_j is row start, col start is implicit?
        # In M_{2,2,2}, B is 2x2.
        # In our M_{3,3,8}, we are taking a 2x2 submatrix of B.
        # Let's say we are processing B rows r0, r1 and cols c0, c1.
        # A rows i0, i1 and cols j0, j1.
        # C rows i0, i1 and cols c0, c1.
        
        a00_idx = a_base_i * m + a_base_j
        a01_idx = a_base_i * m + (a_base_j + 1)
        a10_idx = (a_base_i + 1) * m + a_base_j
        a11_idx = (a_base_i + 1) * m + (a_base_j + 1)
        
        # B indices: row j, col k
        # In V, index is j * p + k
        # We are using B rows a_base_j, a_base_j+1 and cols c_base_k, c_base_k+1
        b00_idx = a_base_j * p + c_base_k
        b01_idx = a_base_j * p + (c_base_k + 1)
        b10_idx = (a_base_j + 1) * p + c_base_k
        b11_idx = (a_base_j + 1) * p + (c_base_k + 1)
        
        # C indices: row i, col k
        # In W, index is k * n + i
        c00_idx = c_base_k * n + a_base_i
        c01_idx = (c_base_k + 1) * n + a_base_i
        c10_idx = c_base_k * n + (a_base_i + 1)
        c11_idx = (c_base_k + 1) * n + (a_base_i + 1)
        
        # m1
        U[a00_idx][start_col] += 1.0
        U[a11_idx][start_col] += 1.0
        V[b00_idx][start_col] += 1.0
        V[b11_idx][start_col] += 1.0
        W[c00_idx][start_col] += 1.0
        W[c11_idx][start_col] += 1.0
        W[c01_idx][start_col] -= 1.0
        W[c10_idx][start_col] -= 1.0
        start_col += 1
        
        # m2
        U[a10_idx][start_col] += 1.0
        U[a11_idx][start_col] += 1.0
        V[b00_idx][start_col] += 1.0
        W[c10_idx][start_col] += 1.0
        W[c11_idx][start_col] += 1.0
        start_col += 1
        
        # m3
        U[a00_idx][start_col] += 1.0
        V[b01_idx][start_col] += 1.0
        V[b11_idx][start_col] -= 1.0
        W[c01_idx][start_col] += 1.0
        W[c11_idx][start_col] -= 1.0
        start_col += 1
        
        # m4
        U[a11_idx][start_col] += 1.0
        V[b10_idx][start_col] += 1.0
        V[b00_idx][start_col] -= 1.0
        W[c10_idx][start_col] += 1.0
        W[c11_idx][start_col] -= 1.0
        start_col += 1
        
        # m5
        U[a00_idx][start_col] += 1.0
        U[a01_idx][start_col] += 1.0
        V[b11_idx][start_col] += 1.0
        W[c00_idx][start_col] += 1.0
        W[c10_idx][start_col] += 1.0
        start_col += 1
        
        # m6
        U[a10_idx][start_col] += 1.0
        U[a00_idx][start_col] -= 1.0
        V[b00_idx][start_col] += 1.0
        V[b01_idx][start_col] += 1.0
        W[c00_idx][start_col] -= 1.0
        W[c10_idx][start_col] += 1.0
        start_col += 1
        
        # m7
        U[a01_idx][start_col] += 1.0
        U[a11_idx][start_col] -= 1.0
        V[b10_idx][start_col] += 1.0
        V[b11_idx][start_col] += 1.0
        W[c01_idx][start_col] += 1.0
        W[c11_idx][start_col] -= 1.0
        start_col += 1
        
        return start_col

    # 1. M_{2,2,4} for cols 0-3
    # A rows 0,1 cols 0,1. B rows 0,1 cols 0,1. C rows 0,1 cols 0,1.
    col_idx = add_strassen(0, 0, 0, 0, 0, col_idx)
    # A rows 0,1 cols 0,1. B rows 0,1 cols 2,3. C rows 0,1 cols 2,3.
    col_idx = add_strassen(0, 0, 2, 0, 2, col_idx)
    
    # 2. M_{2,2,4} for cols 4-7
    # A rows 0,1 cols 0,1. B rows 0,1 cols 4,5. C rows 0,1 cols 4,5.
    col_idx = add_strassen(0, 0, 4, 0, 4, col_idx)
    # A rows 0,1 cols 0,1. B rows 0,1 cols 6,7. C rows 0,1 cols 6,7.
    col_idx = add_strassen(0, 0, 6, 0, 6, col_idx)
    
    # 3. M_{2,1,8}: A rows 0,1 col 2. B row 2 cols 0-7. C rows 0,1 cols 0-7.
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col_idx] = 1.0
            V[v_idx][col_idx] = 1.0
            W[w_idx][col_idx] = 1.0
            col_idx += 1
            
    # 4. M_{1,2,8}: A row 2 cols 0,1. B rows 0,1 cols 0-7. C row 2 cols 0-7.
    for j in range(2):
        for k in range(8):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U[u_idx][col_idx] = 1.0
            V[v_idx][col_idx] = 1.0
            W[w_idx][col_idx] = 1.0
            col_idx += 1
            
    # 5. M_{1,1,8}: A row 2 col 2. B row 2 cols 0-7. C row 2 cols 0-7.
    for k in range(8):
        u_idx = 2 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 2
        U[u_idx][col_idx] = 1.0
        V[v_idx][col_idx] = 1.0
        W[w_idx][col_idx] = 1.0
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
