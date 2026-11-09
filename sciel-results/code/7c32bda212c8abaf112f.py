import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Helper to add a Strassen term for 2x2x2
    # X is 2x2 submatrix of A, Y is 2x2 submatrix of B
    # x_idx: indices in U for X elements [00, 01, 10, 11]
    # y_idx: indices in V for Y elements [00, 01, 10, 11]
    # c_idx: indices in W for C elements [00, 01, 10, 11]
    def add_strassen(x_idx, y_idx, c_idx, start_rank):
        # M1 = (x00+x11)(y00+y11) -> C00 += 1, C11 += 1
        # M2 = (x10+x11)y00 -> C10 += 1, C11 -= 1
        # M3 = x00(y01-y11) -> C01 += 1, C11 += 1
        # M4 = x11(y10-y00) -> C00 += 1, C10 += 1
        # M5 = (x00+x01)y11 -> C00 -= 1, C01 += 1
        # M6 = (x10-x00)(y00+y01) -> C11 += 1
        # M7 = (x01-x11)(y10+y11) -> C00 += 1
        
        # Term 1: M1
        U[x_idx[0], start_rank] = 1
        U[x_idx[3], start_rank] = 1
        V[y_idx[0], start_rank] = 1
        V[y_idx[3], start_rank] = 1
        W[c_idx[0], start_rank] = 1
        W[c_idx[3], start_rank] = 1
        
        # Term 2: M2
        U[x_idx[2], start_rank+1] = 1
        U[x_idx[3], start_rank+1] = 1
        V[y_idx[0], start_rank+1] = 1
        W[c_idx[2], start_rank+1] = 1
        W[c_idx[3], start_rank+1] = -1
        
        # Term 3: M3
        U[x_idx[0], start_rank+2] = 1
        V[y_idx[1], start_rank+2] = 1
        V[y_idx[3], start_rank+2] = -1
        W[c_idx[1], start_rank+2] = 1
        W[c_idx[3], start_rank+2] = 1
        
        # Term 4: M4
        U[x_idx[3], start_rank+3] = 1
        V[y_idx[2], start_rank+3] = 1
        V[y_idx[0], start_rank+3] = -1
        W[c_idx[0], start_rank+3] = 1
        W[c_idx[2], start_rank+3] = 1
        
        # Term 5: M5
        U[x_idx[0], start_rank+4] = 1
        U[x_idx[1], start_rank+4] = 1
        V[y_idx[3], start_rank+4] = 1
        W[c_idx[0], start_rank+4] = -1
        W[c_idx[1], start_rank+4] = 1
        
        # Term 6: M6
        U[x_idx[2], start_rank+5] = 1
        U[x_idx[0], start_rank+5] = -1
        V[y_idx[0], start_rank+5] = 1
        V[y_idx[1], start_rank+5] = 1
        W[c_idx[3], start_rank+5] = 1
        
        # Term 7: M7
        U[x_idx[1], start_rank+6] = 1
        U[x_idx[3], start_rank+6] = -1
        V[y_idx[2], start_rank+6] = 1
        V[y_idx[3], start_rank+6] = 1
        W[c_idx[0], start_rank+6] = 1
        
    # Define indices for A[0:2, :]
    # A is 3x4. Rows 0,1.
    # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2], A[0,3]=U[3]
    # A[1,0]=U[4], A[1,1]=U[5], A[1,2]=U[6], A[1,3]=U[7]
    
    # Product 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # X = A[0:2, 0:2] -> U[0], U[1], U[4], U[5]
    x_idx_1 = [0, 1, 4, 5]
    # Y = B[0:2, 0:2] -> V[0], V[1], V[4], V[5]
    y_idx_1 = [0, 1, 4, 5]
    # C[0:2, 0:2] -> C[0,0]=W[0], C[0,1]=W[3], C[1,0]=W[1], C[1,1]=W[4]
    # W index = k*3 + i
    # C[0,0]: k=0, i=0 -> 0
    # C[0,1]: k=1, i=0 -> 3
    # C[1,0]: k=0, i=1 -> 1
    # C[1,1]: k=1, i=1 -> 4
    c_idx_1 = [0, 3, 1, 4]
    add_strassen(x_idx_1, y_idx_1, c_idx_1, 0)
    
    # Product 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # X = A[0:2, 2:4] -> U[2], U[3], U[6], U[7]
    x_idx_2 = [2, 3, 6, 7]
    # Y = B[2:4, 0:2] -> V[8], V[9], V[12], V[13]
    # B row 2 col 0 = V[2*4+0]=8
    # B row 2 col 1 = V[2*4+1]=9
    # B row 3 col 0 = V[3*4+0]=12
    # B row 3 col 1 = V[3*4+1]=13
    y_idx_2 = [8, 9, 12, 13]
    # C[0:2, 0:2] -> same c_idx_1
    add_strassen(x_idx_2, y_idx_2, c_idx_1, 7)
    
    # Product 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # X = A[0:2, 0:2] -> x_idx_1
    # Y = B[0:2, 2:4] -> V[2], V[3], V[6], V[7]
    y_idx_3 = [2, 3, 6, 7]
    # C[0:2, 2:4] -> C[0,2]=W[2*3+0]=6, C[0,3]=W[3*3+0]=9, C[1,2]=W[2*3+1]=7, C[1,3]=W[3*3+1]=10
    c_idx_3 = [6, 9, 7, 10]
    add_strassen(x_idx_1, y_idx_3, c_idx_3, 14)
    
    # Product 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # X = A[0:2, 2:4] -> x_idx_2
    # Y = B[2:4, 2:4] -> V[10], V[11], V[14], V[15]
    y_idx_4 = [10, 11, 14, 15]
    # C[0:2, 2:4] -> same c_idx_3
    add_strassen(x_idx_2, y_idx_4, c_idx_3, 21)
    
    # Part 2: Row 2 of A * B
    # A[2, j] is U[2*4+j] = U[8+j]
    # B[j, k] is V[j*4+k]
    # C[2, k] is W[k*3+2]
    
    current_rank = 28
    for j in range(4):
        for k in range(4):
            u_idx = 8 + j
            v_idx = j * 4 + k
            w_idx = k * 3 + 2
            U[u_idx, current_rank] = 1
            V[v_idx, current_rank] = 1
            W[w_idx, current_rank] = 1
            current_rank += 1
            
    return {'U': U, 'V': V, 'W': W}
