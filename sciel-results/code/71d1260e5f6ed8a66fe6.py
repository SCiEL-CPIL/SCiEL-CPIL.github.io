import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 6 blocks * 7 Strassen terms = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M1: (x00+x11)(y00+y11)
    # M2: (x10+x11)y00
    # M3: x00(y10-y11)
    # M4: x11(y01-y00)
    # M5: (x00+x01)y11
    # M6: (x10-x00)(y00+y01)
    # M7: (x01-x11)(y10+y11)
    
    # Output reconstruction:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M1 - M2 + M6
    # Z10 = M2 + M4 - M1 + M5
    # Z11 = M3 + M7
    
    # Blocks definitions:
    # (x_idx_offset, y_row_offset, y_col_offset)
    # x_idx_offset: 0 for A1 (cols 0,1), 2 for A2 (cols 2,3)
    # y_row_offset: 0 for B1 (rows 0,1), 2 for B2 (rows 2,3)
    # y_col_offset: 0, 2, 4
    
    blocks = [
        (0, 0, 0), (0, 0, 2), (0, 0, 4),
        (2, 2, 0), (2, 2, 2), (2, 2, 4)
    ]
    
    term_id = 0
    for x_off, y_row_off, y_col_off in blocks:
        # Map local x indices (0,1) to global A indices
        # A is 2x4. Row i, Col j. Global index i*4 + j.
        # x00 -> A[0, x_off] -> 0*4 + x_off = x_off
        # x01 -> A[0, x_off+1] -> x_off+1
        # x10 -> A[1, x_off] -> 4 + x_off
        # x11 -> A[1, x_off+1] -> 4 + x_off + 1
        
        x_idx = [
            x_off,       # x00
            x_off + 1,   # x01
            4 + x_off,   # x10
            4 + x_off + 1 # x11
        ]
        
        # Map local y indices (0,1) to global B indices
        # B is 4x6. Row j, Col k. Global index j*6 + k.
        # y_row_off is the start row (0 or 2). y_col_off is start col.
        # y00 -> B[y_row_off, y_col_off] -> y_row_off*6 + y_col_off
        # y01 -> B[y_row_off, y_col_off+1] -> y_row_off*6 + y_col_off + 1
        # y10 -> B[y_row_off+1, y_col_off] -> (y_row_off+1)*6 + y_col_off
        # y11 -> B[y_row_off+1, y_col_off+1] -> (y_row_off+1)*6 + y_col_off + 1
        
        y_idx = [
            y_row_off * 6 + y_col_off,       # y00
            y_row_off * 6 + y_col_off + 1,   # y01
            (y_row_off + 1) * 6 + y_col_off,       # y10
            (y_row_off + 1) * 6 + y_col_off + 1   # y11
        ]
        
        # Map local z indices to global C indices
        # C is 2x6. W index is k*n + i = 2*k + i.
        # Z00 -> C[0, y_col_off] -> 2*y_col_off + 0
        # Z01 -> C[0, y_col_off+1] -> 2*(y_col_off+1) + 0
        # Z10 -> C[1, y_col_off] -> 2*y_col_off + 1
        # Z11 -> C[1, y_col_off+1] -> 2*(y_col_off+1) + 1
        
        z_idx = [
            2 * y_col_off,       # Z00
            2 * (y_col_off + 1), # Z01
            2 * y_col_off + 1,   # Z10
            2 * (y_col_off + 1) + 1 # Z11
        ]
        
        # Add 7 Strassen terms
        # M1: (x00+x11)(y00+y11)
        U[x_idx[0]][term_id] += 1
        U[x_idx[3]][term_id] += 1
        V[y_idx[0]][term_id] += 1
        V[y_idx[3]][term_id] += 1
        # Z00 += 1*M1, Z01 += 1*M1, Z10 += -1*M1
        W[z_idx[0]][term_id] += 1
        W[z_idx[1]][term_id] += 1
        W[z_idx[2]][term_id] -= 1
        term_id += 1
        
        # M2: (x10+x11)y00
        U[x_idx[2]][term_id] += 1
        U[x_idx[3]][term_id] += 1
        V[y_idx[0]][term_id] += 1
        # Z01 += -1*M2, Z10 += 1*M2
        W[z_idx[1]][term_id] -= 1
        W[z_idx[2]][term_id] += 1
        term_id += 1
        
        # M3: x00(y10-y11)
        U[x_idx[0]][term_id] += 1
        V[y_idx[2]][term_id] += 1
        V[y_idx[3]][term_id] -= 1
        # Z01 += 1*M3, Z11 += 1*M3
        W[z_idx[1]][term_id] += 1
        W[z_idx[3]][term_id] += 1
        term_id += 1
        
        # M4: x11(y01-y00)
        U[x_idx[3]][term_id] += 1
        V[y_idx[1]][term_id] += 1
        V[y_idx[0]][term_id] -= 1
        # Z00 += 1*M4, Z10 += 1*M4
        W[z_idx[0]][term_id] += 1
        W[z_idx[2]][term_id] += 1
        term_id += 1
        
        # M5: (x00+x01)y11
        U[x_idx[0]][term_id] += 1
        U[x_idx[1]][term_id] += 1
        V[y_idx[3]][term_id] += 1
        # Z00 += -1*M5, Z10 += 1*M5
        W[z_idx[0]][term_id] -= 1
        W[z_idx[2]][term_id] += 1
        term_id += 1
        
        # M6: (x10-x00)(y00+y01)
        U[x_idx[2]][term_id] += 1
        U[x_idx[0]][term_id] -= 1
        V[y_idx[0]][term_id] += 1
        V[y_idx[1]][term_id] += 1
        # Z01 += 1*M6
        W[z_idx[1]][term_id] += 1
        term_id += 1
        
        # M7: (x01-x11)(y10+y11)
        U[x_idx[1]][term_id] += 1
        U[x_idx[3]][term_id] -= 1
        V[y_idx[2]][term_id] += 1
        V[y_idx[3]][term_id] += 1
        # Z00 += 1*M7, Z11 += 1*M7
        W[z_idx[0]][term_id] += 1
        W[z_idx[3]][term_id] += 1
        term_id += 1
        
    return {"U": U, "V": V, "W": W}
