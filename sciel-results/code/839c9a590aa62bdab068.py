import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Total rank will be 34
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # i_off, j_off, k_off are the global offsets (0-indexed) for the 2x2x2 block
    # i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
    def add_strassen_block(i_off, j_off, k_off, start_col):
        # m1 = (a00 + a11)(b00 + b11)
        # U: a00, a11
        # V: b00, b11
        # W: c00, c11 (with signs)
        
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        # We need to map local indices (0,1) to global.
        # a_ii = A[i_off+i_loc, j_off+j_loc] -> U row (i_off+i_loc)*m + (j_off+j_loc)
        # b_jj = B[j_off+j_loc, k_off+k_loc] -> V row (j_off+j_loc)*p + (k_off+k_loc)
        # c_kk = C[k_off+k_loc, i_off+i_loc] -> W row (k_off+k_loc)*n + (i_off+i_loc)
        
        # m1
        U[(i_off+0)*m + (j_off+0)][start_col+0] += 1
        U[(i_off+1)*m + (j_off+1)][start_col+0] += 1
        V[(j_off+0)*p + (k_off+0)][start_col+0] += 1
        V[(j_off+1)*p + (k_off+1)][start_col+0] += 1
        W[(k_off+0)*n + (i_off+0)][start_col+0] += 1
        W[(k_off+1)*n + (i_off+1)][start_col+0] += 1
        
        # m2 = (a10 + a11)b00
        U[(i_off+1)*m + (j_off+0)][start_col+1] += 1
        U[(i_off+1)*m + (j_off+1)][start_col+1] += 1
        V[(j_off+0)*p + (k_off+0)][start_col+1] += 1
        W[(k_off+0)*n + (i_off+1)][start_col+1] += 1
        W[(k_off+1)*n + (i_off+1)][start_col+1] += 1 # Wait, c10 and c11?
        # c10 = m2 + m4. c11 = ... - m2 ...
        # m2 contributes to c10 (+1) and c11 (-1)
        W[(k_off+0)*n + (i_off+1)][start_col+1] += 1 # c10
        W[(k_off+1)*n + (i_off+1)][start_col+1] -= 1 # c11
        
        # m3 = a00(b01 - b11)
        U[(i_off+0)*m + (j_off+0)][start_col+2] += 1
        V[(j_off+0)*p + (k_off+1)][start_col+2] += 1
        V[(j_off+1)*p + (k_off+1)][start_col+2] -= 1
        W[(k_off+0)*n + (i_off+0)][start_col+2] += 1 # c01
        W[(k_off+1)*n + (i_off+1)][start_col+2] += 1 # c11
        
        # m4 = a11(b10 - b00)
        U[(i_off+1)*m + (j_off+1)][start_col+3] += 1
        V[(j_off+1)*p + (k_off+0)][start_col+3] += 1
        V[(j_off+0)*p + (k_off+0)][start_col+3] -= 1
        W[(k_off+0)*n + (i_off+0)][start_col+3] += 1 # c00
        W[(k_off+0)*n + (i_off+1)][start_col+3] += 1 # c10
        
        # m5 = (a00 + a01)b11
        U[(i_off+0)*m + (j_off+0)][start_col+4] += 1
        U[(i_off+0)*m + (j_off+1)][start_col+4] += 1
        V[(j_off+1)*p + (k_off+1)][start_col+4] += 1
        W[(k_off+0)*n + (i_off+0)][start_col+4] -= 1 # c00
        W[(k_off+0)*n + (i_off+0)][start_col+4] -= 1 # c01? No.
        # c01 = m3 + m5. c00 = ... - m5 ...
        # m5 contributes to c01 (+1) and c00 (-1)
        W[(k_off+0)*n + (i_off+0)][start_col+4] -= 1 # c00
        W[(k_off+1)*n + (i_off+0)][start_col+4] += 1 # c01
        # Wait, c01 is at k=1, i=0? No.
        # c_{i_loc, k_loc}.
        # c01 is i=0, k=1. W index: k=1, i=0.
        # c00 is i=0, k=0. W index: k=0, i=0.
        
        # m6 = (a10 - a00)(b00 + b01)
        U[(i_off+1)*m + (j_off+0)][start_col+5] += 1
        U[(i_off+0)*m + (j_off+0)][start_col+5] -= 1
        V[(j_off+0)*p + (k_off+0)][start_col+5] += 1
        V[(j_off+0)*p + (k_off+1)][start_col+5] += 1
        W[(k_off+1)*n + (i_off+1)][start_col+5] += 1 # c11
        
        # m7 = (a01 - a11)(b10 + b11)
        U[(i_off+0)*m + (j_off+1)][start_col+6] += 1
        U[(i_off+1)*m + (j_off+1)][start_col+6] -= 1
        V[(j_off+1)*p + (k_off+0)][start_col+6] += 1
        V[(j_off+1)*p + (k_off+1)][start_col+6] += 1
        W[(k_off+0)*n + (i_off+0)][start_col+6] += 1 # c00
        
        return start_col + 7

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    col = add_strassen_block(0, 0, 0, col)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    col = add_strassen_block(0, 0, 2, col)
    
    # Schoolbook for remaining terms
    # S_rem = { (i,j,k) | (i in {0,1,2}, j=2, k in {0,1,2,3}) U (i=2, j in {0,1}, k in {0,1,2,3}) }
    
    # Part 1: i in {0,1,2}, j=2, k in {0,1,2,3}
    for i in range(3):
        for k in range(4):
            # j=2
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Part 2: i=2, j in {0,1}, k in {0,1,2,3}
    for j in range(2):
        for k in range(4):
            # i=2
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
