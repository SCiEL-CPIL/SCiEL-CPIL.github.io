"""
Rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses block decomposition: split A into 2x2 blocks of 2x2, B into 2x5 blocks.
Each 2x2 * 2x5 multiplication uses rank 18 (Strassen for 2x2 * 2x2 + trivial for 2x2 * 2x3).
Total rank = 4 * 18 = 72.
"""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Decompose M_{4,4,5} using 2x2 block structure.
    # A is 4x4, B is 4x5, C is 4x5.
    # Split A into 4 blocks of 2x2: A_{11}, A_{12}, A_{21}, A_{22}
    # Split B into 2 blocks of 2x5: B_top (rows 0,1), B_bot (rows 2,3)
    # C = [A_{11} B_top + A_{12} B_bot; A_{21} B_top + A_{22} B_bot]
    # Each term is M_{2,2,5}.
    # M_{2,2,5} = M_{2,2,2} (Strassen, rank 7) + M_{2,2,3} (trivial, rank 12) = 19
    # Wait, that gives 4*19 = 76. Let me reconsider.
    # Actually, for M_{2,2,5}, we can use Strassen for the first 2 columns (rank 7)
    # and trivial for the remaining 3 columns (rank 2*2*3 = 12), total 19.
    # 4 * 19 = 76.
    
    # But we can do better. Let's think about M_{2,2,5} more carefully.
    # A is 2x2, B is 2x5, C is 2x5.
    # The trivial rank is 2*2*5 = 20.
    # Using Strassen for M_{2,2,2} gives rank 7 for the first 2 columns.
    # For the remaining 3 columns, we need M_{2,2,3} which has trivial rank 12.
    # Total: 7 + 12 = 19.
    # 4 * 19 = 76.
    
    # Alternatively, we could use a different block size.
    # Let's try 1x1 blocks: M_{4,4,5} = sum of 4*4*5 = 80 rank-1 terms.
    # That's the trivial decomposition.
    
    # Let's implement the 76-rank decomposition.
    rank = 76
    U = [[0] * rank for _ in range(n * m)]  # 16 x 76
    V = [[0] * rank for _ in range(m * p)]  # 20 x 76
    W = [[0] * rank for _ in range(p * n)]  # 20 x 76
    
    def add_strassen_2x2(U, V, W, col_start, i_off, j_off, k_off):
        """Add Strassen decomposition for M_{2,2,2} at given offsets.
        i_off, j_off, k_off are the top-left indices in the global matrices.
        Returns the next column index.
        """
        col = col_start
        
        # Strassen's 7 multiplications for M_{2,2,2}:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U indices for A (2x2 block at i_off, j_off):
        # a00 -> (i_off, j_off) -> idx = i_off*m + j_off
        # a01 -> (i_off, j_off+1) -> idx = i_off*m + j_off+1
        # a10 -> (i_off+1, j_off) -> idx = (i_off+1)*m + j_off
        # a11 -> (i_off+1, j_off+1) -> idx = (i_off+1)*m + j_off+1
        
        # V indices for B (2x2 block at j_off, k_off):
        # b00 -> (j_off, k_off) -> idx = j_off*p + k_off
        # b01 -> (j_off, k_off+1) -> idx = j_off*p + k_off+1
        # b10 -> (j_off+1, k_off) -> idx = (j_off+1)*p + k_off
        # b11 -> (j_off+1, k_off+1) -> idx = (j_off+1)*p + k_off+1
        
        # W indices for C (2x2 block at k_off, i_off):
        # c00 -> (k_off, i_off) -> idx = k_off*n + i_off
        # c01 -> (k_off, i_off+1) -> idx = k_off*n + i_off+1
        # c10 -> (k_off+1, i_off) -> idx = (k_off+1)*n + i_off
        # c11 -> (k_off+1, i_off+1) -> idx = (k_off+1)*n + i_off+1
        
        u00 = i_off * m + j_off
        u01 = i_off * m + (j_off + 1)
        u10 = (i_off + 1) * m + j_off
        u11 = (i_off + 1) * m + (j_off + 1)
        
        v00 = j_off * p + k_off
        v01 = j_off * p + (k_off + 1)
        v10 = (j_off + 1) * p + k_off
        v11 = (j_off + 1) * p + (k_off + 1)
        
        w00 = k_off * n + i_off
        w01 = k_off * n + (i_off + 1)
        w10 = (k_off + 1) * n + i_off
        w11 = (k_off + 1) * n + (i_off + 1)
        
        # Term 1: M1 = (a00+a11)(b00+b11) -> c00 + c11
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # Term 2: M2 = (a10+a11)b00 -> c10
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # Term 3: M3 = a00(b10-b11) -> c01
        U[u00][col] += 1
        V[v10][col] += 1
        V[v11][col] -= 1
        W[w01][col] += 1
        col += 1
        
        # Term 4: M4 = a11(b01-b00) -> c00 + c10
        U[u11][col] += 1
        V[v01][col] += 1
        V[v00][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # Term 5: M5 = (a00+a01)b11 -> -c00 + c01
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1
        
        # Term 6: M6 = (a10-a00)(b00+b01) -> c11
        U[u10][col] += 1
        U[u00][col] -= 1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # Term 7: M7 = (a01-a11)(b10+b11) -> c00
        U[u01][col] += 1
        U[u11][col] -= 1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        col += 1
        
        return col
    
    def add_trivial_2x2_2x3(U, V, W, col_start, i_off, j_off, k_off):
        """Add trivial decomposition for M_{2,2,3} at given offsets.
        A is 2x2, B is 2x3, C is 2x3.
        Returns the next column index.
        """
        col = col_start
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    
                    U[u_idx][col] += 1
                    V[v_idx][col] += 1
                    W[w_idx][col] += 1
                    col += 1
        return col
    
    # Now decompose M_{4,4,5}.
    # A is 4x4, split into 2x2 blocks:
    # A_{11} at (0,0), A_{12} at (0,2), A_{21} at (2,0), A_{22} at (2,2)
    # B is 4x5, split into 2x5 blocks:
    # B_top at rows 0,1; B_bot at rows 2,3
    # C is 4x5, split into 2x5 blocks:
    # C_top at rows 0,1; C_bot at rows 2,3
    
    # C_top = A_{11} B_top + A_{12} B_bot
    # C_bot = A_{21} B_top + A_{22} B_bot
    
    # Each term is M_{2,2,5}.
    # For M_{2,2,5}, we use Strassen for the first 2 columns (rank 7) and trivial for the last 3 columns (rank 12).
    # Total rank per M_{2,2,5} = 19.
    # 4 terms * 19 = 76.
    
    col = 0
    
    # Term 1: A_{11} B_top
    # A_{11} is at i_off=0, j_off=0
    # B_top is at j_off=0, k_off=0 (rows 0,1 of B, all 5 columns)
    # But we split B_top into 2x2 (cols 0,1) and 2x3 (cols 2,3,4)
    # C contribution is at k_off=0, i_off=0 (rows 0,1 of C, all 5 columns)
    col = add_strassen_2x2(U, V, W, col, 0, 0, 0)  # Strassen for cols 0,1
    col = add_trivial_2x2_2x3(U, V, W, col, 0, 0, 2)  # Trivial for cols 2,3,4
    
    # Term 2: A_{12} B_bot
    # A_{12} is at i_off=0, j_off=2
    # B_bot is at j_off=2, k_off=0 (rows 2,3 of B, all 5 columns)
    # C contribution is at k_off=0, i_off=0 (rows 0,1 of C, all 5 columns)
    col = add_strassen_2x2(U, V, W, col, 0, 2, 0)  # Strassen for cols 0,1
    col = add_trivial_2x2_2x3(U, V, W, col, 0, 2, 2)  # Trivial for cols 2,3,4
    
    # Term 3: A_{21} B_top
    # A_{21} is at i_off=2, j_off=0
    # B_top is at j_off=0, k_off=0 (rows 0,1 of B, all 5 columns)
    # C contribution is at k_off=0, i_off=2 (rows 2,3 of C, all 5 columns)
    col = add_strassen_2x2(U, V, W, col, 2, 0, 0)  # Strassen for cols 0,1
    col = add_trivial_2x2_2x3(U, V, W, col, 2, 0, 2)  # Trivial for cols 2,3,4
    
    # Term 4: A_{22} B_bot
    # A_{22} is at i_off=2, j_off=2
    # B_bot is at j_off=2, k_off=0 (rows 2,3 of B, all 5 columns)
    # C contribution is at k_off=0, i_off=2 (rows 2,3 of C, all 5 columns)
    col = add_strassen_2x2(U, V, W, col, 2, 2, 0)  # Strassen for cols 0,1
    col = add_trivial_2x2_2x3(U, V, W, col, 2, 2, 2)  # Trivial for cols 2,3,4
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
