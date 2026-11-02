import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    deadline = start_time + 110
    
    # Block decomposition:
    # A (4x5): rows split as 2+2, cols split as 2+2+1
    # B (5x6): rows split as 2+2+1, cols split as 2+2+2
    # C (4x6): rows split as 2+2, cols split as 2+2+2
    #
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j]
    # where i in {0,1}, j in {0,1,2}
    #
    # A[i,0]*B[0,j]: 2x2 * 2x2 -> Strassen (rank 7)
    # A[i,1]*B[1,j]: 2x2 * 2x2 -> Strassen (rank 7)
    # A[i,2]*B[2,j]: 2x1 * 1x2 -> trivial (rank 4)
    #
    # Total rank = 6 * (7 + 7 + 4) = 108
    
    rank = 108
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, c in u_entries:
            U[idx, col] += c
        for idx, c in v_entries:
            V[idx, col] += c
        for idx, c in w_entries:
            W[idx, col] += c
        col += 1
    
    def strassen_2x2x2(u_row_off, u_col_off, v_row_off, v_col_off, w_row_off, w_col_off):
        """Strassen for 2x2 * 2x2 -> 2x2"""
        nonlocal col
        
        def u_idx(i, j):
            return (u_row_off + i) * m + (u_col_off + j)
        
        def v_idx(i, j):
            return (v_row_off + i) * p + (v_col_off + j)
        
        def w_idx(row, col_c):
            return col_c * n + row
        
        # M1 = (P00+P11)*(Q00+Q11)
        add_term(
            [(u_idx(0,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,0), 1), (v_idx(1,1), 1)],
            [(w_idx(0,0), 1), (w_idx(1,1), 1)]
        )
        
        # M2 = (P10+P11)*Q00
        add_term(
            [(u_idx(1,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,0), 1)],
            [(w_idx(1,0), 1)]
        )
        
        # M3 = P00*(Q01-Q11)
        add_term(
            [(u_idx(0,0), 1)],
            [(v_idx(0,1), 1), (v_idx(1,1), -1)],
            [(w_idx(0,1), 1)]
        )
        
        # M4 = P11*(Q10-Q00)
        add_term(
            [(u_idx(1,1), 1)],
            [(v_idx(1,0), 1), (v_idx(0,0), -1)],
            [(w_idx(0,0), 1), (w_idx(1,0), 1)]
        )
        
        # M5 = (P00+P01)*Q11
        add_term(
            [(u_idx(0,0), 1), (u_idx(0,1), 1)],
            [(v_idx(1,1), 1)],
            [(w_idx(0,0), -1), (w_idx(0,1), 1)]
        )
        
        # M6 = (P10-P00)*(Q00+Q01)
        add_term(
            [(u_idx(1,0), 1), (u_idx(0,0), -1)],
            [(v_idx(0,0), 1), (v_idx(0,1), 1)],
            [(w_idx(1,1), 1)]
        )
        
        # M7 = (P01-P11)*(Q10+Q11)
        add_term(
            [(u_idx(0,1), 1), (u_idx(1,1), -1)],
            [(v_idx(1,0), 1), (v_idx(1,1), 1)],
            [(w_idx(0,0), 1)]
        )
    
    def outer_2x1x2(u_row_off, u_col_off, v_row_off, v_col_off, w_row_off, w_col_off):
        """Trivial for 2x1 * 1x2 -> 2x2"""
        nonlocal col
        
        def u_idx(i, j):
            return (u_row_off + i) * m + (u_col_off + j)
        
        def v_idx(i, j):
            return (v_row_off + i) * p + (v_col_off + j)
        
        def w_idx(row, col_c):
            return col_c * n + row
        
        # C[0,0] = A[0,0]*B[0,0]
        add_term([(u_idx(0,0), 1)], [(v_idx(0,0), 1)], [(w_idx(0,0), 1)])
        # C[0,1] = A[0,0]*B[0,1]
        add_term([(u_idx(0,0), 1)], [(v_idx(0,1), 1)], [(w_idx(0,1), 1)])
        # C[1,0] = A[1,0]*B[0,0]
        add_term([(u_idx(1,0), 1)], [(v_idx(0,0), 1)], [(w_idx(1,0), 1)])
        # C[1,1] = A[1,0]*B[0,1]
        add_term([(u_idx(1,0), 1)], [(v_idx(0,1), 1)], [(w_idx(1,1), 1)])
    
    # Build decomposition
    for i in range(2):
        for j in range(3):
            # A[i,0] * B[0,j]: 2x2 * 2x2
            strassen_2x2x2(i*2, 0, 0, j*2, i*2, j*2)
            # A[i,1] * B[1,j]: 2x2 * 2x2
            strassen_2x2x2(i*2, 2, 2, j*2, i*2, j*2)
            # A[i,2] * B[2,j]: 2x1 * 1x2
            outer_2x1x2(i*2, 4, 4, j*2, i*2, j*2)
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    if time.time() > deadline:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
