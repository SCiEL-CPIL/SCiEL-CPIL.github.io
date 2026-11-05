import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # We will use a rank-24 decomposition for the 3x3x3 part
    # and rank-9 for the 3x3x1 part.
    # Total rank = 24 + 24 + 9 = 57.
    
    U = np.zeros((n * m, 57))
    V = np.zeros((m * p, 57))
    W = np.zeros((p * n, 57))
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook(U, V, W, i_start, i_end, j_start, j_end, k_start, k_end, col):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return col
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i_start, j_start, k_start, col):
        # i_start, j_start, k_start are the top-left indices of the 2x2x2 block
        # U indices: i*m+j
        # V indices: j*p+k
        # W indices: k*n+i
        
        # M1
        U[i_start*m + j_start, col] = 1
        U[(i_start+1)*m + (j_start+1), col] = 1
        V[j_start*p + k_start, col] = 1
        V[(j_start+1)*p + (k_start+1), col] = 1
        W[k_start*n + i_start, col] = 1
        W[(k_start+1)*n + (i_start+1), col] = 1
        W[(k_start+1)*n + i_start, col] = -1
        W[k_start*n + (i_start+1), col] = -1
        col += 1
        
        # M2
        U[i_start*m + (j_start+1), col] = 1
        U[(i_start+1)*m + (j_start+1), col] = 1
        V[j_start*p + k_start, col] = 1
        W[(k_start+1)*n + i_start, col] = 1
        W[(k_start+1)*n + (i_start+1), col] = 1
        col += 1
        
        # M3
        U[i_start*m + j_start, col] = 1
        V[(j_start+1)*p + k_start, col] = 1
        V[(j_start+1)*p + (k_start+1), col] = -1
        W[k_start*n + i_start, col] = 1
        W[(k_start+1)*n + i_start, col] = 1
        col += 1
        
        # M4
        U[(i_start+1)*m + (j_start+1), col] = 1
        V[j_start*p + (k_start+1), col] = 1
        V[j_start*p + k_start, col] = -1
        W[(k_start+1)*n + (i_start+1), col] = 1
        W[k_start*n + (i_start+1), col] = 1
        col += 1
        
        # M5
        U[i_start*m + j_start, col] = 1
        U[i_start*m + (j_start+1), col] = 1
        V[(j_start+1)*p + (k_start+1), col] = 1
        W[(k_start+1)*n + (i_start+1), col] = 1
        W[k_start*n + (i_start+1), col] = 1
        col += 1
        
        # M6
        U[(i_start+1)*m + j_start, col] = 1
        U[i_start*m + j_start, col] = -1
        V[j_start*p + k_start, col] = 1
        V[j_start*p + (k_start+1), col] = 1
        W[k_start*n + i_start, col] = 1
        W[(k_start+1)*n + i_start, col] = 1
        col += 1
        
        # M7
        U[i_start*m + (j_start+1), col] = 1
        U[(i_start+1)*m + (j_start+1), col] = -1
        V[(j_start+1)*p + k_start, col] = 1
        V[(j_start+1)*p + (k_start+1), col] = 1
        W[(k_start+1)*n + (i_start+1), col] = 1
        W[(k_start+1)*n + i_start, col] = 1
        col += 1
        
        return col

    # Block 1: 3x3x3 (columns 0-2)
    # Use Strassen for 0:2, 0:2, 0:2
    col = add_strassen(U, V, W, 0, 0, 0, col)
    # Use schoolbook for the rest
    # i=0,1; j=0,1; k=2
    col = add_schoolbook(U, V, W, 0, 2, 0, 2, 2, 3, col)
    # i=0,1; j=2; k=0,1,2
    col = add_schoolbook(U, V, W, 0, 2, 2, 3, 0, 3, col)
    # i=2; j=0,1,2; k=0,1,2
    col = add_schoolbook(U, V, W, 2, 3, 0, 3, 0, 3, col)
    
    # Block 2: 3x3x3 (columns 3-5)
    col = add_strassen(U, V, W, 0, 0, 3, col)
    col = add_schoolbook(U, V, W, 0, 2, 0, 2, 5, 6, col)
    col = add_schoolbook(U, V, W, 0, 2, 2, 3, 3, 6, col)
    col = add_schoolbook(U, V, W, 2, 3, 0, 3, 3, 6, col)
    
    # Block 3: 3x3x1 (column 6)
    col = add_schoolbook(U, V, W, 0, 3, 0, 3, 6, 7, col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
