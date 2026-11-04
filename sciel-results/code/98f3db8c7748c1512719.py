import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def strassen_222(i_off, j_off, k_off, col_start):
        # Returns next column index after filling
        col = col_start
        # m1 = (a00+a11)(b00+b11)
        U[i_off*m+j_off, col] += 1
        U[(i_off+1)*m+(j_off+1), col] += 1
        V[j_off*p+k_off, col] += 1
        V[(j_off+1)*p+(k_off+1), col] += 1
        W[k_off*n+i_off, col] += 1
        W[(k_off+1)*n+(i_off+1), col] += 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[(i_off+1)*m+j_off, col] += 1
        U[(i_off+1)*m+(j_off+1), col] += 1
        V[j_off*p+k_off, col] += 1
        W[k_off*n+(i_off+1), col] += 1
        W[(k_off+1)*n+(i_off+1), col] += -1
        col += 1
        
        # m3 = a00(b01-b11)
        U[i_off*m+j_off, col] += 1
        V[j_off*p+(k_off+1), col] += 1
        V[(j_off+1)*p+(k_off+1), col] += -1
        W[(k_off+1)*n+i_off, col] += 1
        W[(k_off+1)*n+(i_off+1), col] += 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[(i_off+1)*m+(j_off+1), col] += 1
        V[(j_off+1)*p+k_off, col] += 1
        V[j_off*p+k_off, col] += -1
        W[k_off*n+i_off, col] += 1
        W[k_off*n+(i_off+1), col] += 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[i_off*m+j_off, col] += 1
        U[i_off*m+(j_off+1), col] += 1
        V[(j_off+1)*p+(k_off+1), col] += 1
        W[k_off*n+i_off, col] += -1
        W[(k_off+1)*n+i_off, col] += 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[(i_off+1)*m+j_off, col] += 1
        U[i_off*m+j_off, col] += -1
        V[j_off*p+k_off, col] += 1
        V[j_off*p+(k_off+1), col] += 1
        W[(k_off+1)*n+(i_off+1), col] += 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[i_off*m+(j_off+1), col] += 1
        U[(i_off+1)*m+(j_off+1), col] += -1
        V[(j_off+1)*p+k_off, col] += 1
        V[(j_off+1)*p+(k_off+1), col] += 1
        W[k_off*n+i_off, col] += 1
        col += 1
        
        return col
    
    def schoolbook_2x2x1(i_off, j_off, k_idx, col_start):
        col = col_start
        for i in range(2):
            for j in range(2):
                U[(i_off+i)*m+(j_off+j), col] = 1
                V[(j_off+j)*p+k_idx, col] = 1
                W[k_idx*n+(i_off+i), col] = 1
                col += 1
        return col
    
    col = 0
    # Block A1 (j=0,1) with k=0,1
    col = strassen_222(0, 0, 0, col)
    # Block A1 (j=0,1) with k=2,3
    col = strassen_222(0, 0, 2, col)
    # A1 (j=0,1) with k=4: schoolbook
    col = schoolbook_2x2x1(0, 0, 4, col)
    # Block A2 (j=2,3) with k=0,1
    col = strassen_222(0, 2, 0, col)
    # Block A2 (j=2,3) with k=2,3
    col = strassen_222(0, 2, 2, col)
    # A2 (j=2,3) with k=4: schoolbook
    col = schoolbook_2x2x1(0, 2, 4, col)
    
    # Verify
    # Tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want T[i,j,k] = 1 for all i,j,k
    # Reshape for efficient verification
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # For each i, j, k: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a 3-index sum. Let's verify a few entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1) > 1e-9:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = np.zeros((n * m, rank_fb))
                    V_fb = np.zeros((m * p, rank_fb))
                    W_fb = np.zeros((p * n, rank_fb))
                    idx = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U_fb[i2*m+j2, idx] = 1
                                V_fb[j2*p+k2, idx] = 1
                                W_fb[k2*n+i2, idx] = 1
                                idx += 1
                    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
