import time
import numpy as np

def construct():
    n, m, p = 4, 4, 9
    rank = 102
    
    U = np.zeros((n*m, rank))  # (16, 102)
    V = np.zeros((m*p, rank))  # (36, 102)
    W = np.zeros((p*n, rank))  # (36, 102)
    
    # Strassen 2x2x2 terms
    # Each term: (U_2x2, V_2x2, W_2x2) where each is a flat 4-vector
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),   # M3
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),   # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),   # M7
    ]
    
    def add_strassen_4x4(U, V, W, rank_offset, b_col_offset, c_col_offset):
        """Add a 4x4x4 Strassen decomposition starting at rank_offset."""
        # A blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
        # B blocks: B11=B[0:2,0:2+b_off], B12=B[0:2,2:4+b_off], B21=B[2:4,0:2+b_off], B22=B[2:4,2:4+b_off]
        # C blocks: C11=C[0:2,0:2+c_off], C12=C[0:2,2:4+c_off], C21=C[2:4,0:2+c_off], C22=C[2:4,2:4+c_off]
        
        # Define the 2x2 blocks as index arrays
        # A[i,j] -> U index i*4+j
        A11 = [[0*4+0, 0*4+1], [1*4+0, 1*4+1]]  # A[0,0],A[0,1]; A[1,0],A[1,1]
        A12 = [[0*4+2, 0*4+3], [1*4+2, 1*4+3]]  # A[0,2],A[0,3]; A[1,2],A[1,3]
        A21 = [[2*4+0, 2*4+1], [3*4+0, 3*4+1]]  # A[2,0],A[2,1]; A[3,0],A[3,1]
        A22 = [[2*4+2, 2*4+3], [3*4+2, 3*4+3]]  # A[2,2],A[2,3]; A[3,2],A[3,3]
        
        # B[j,k] -> V index j*9+k
        B11 = [[0*9+0+b_col_offset, 0*9+1+b_col_offset], [1*9+0+b_col_offset, 1*9+1+b_col_offset]]
        B12 = [[0*9+2+b_col_offset, 0*9+3+b_col_offset], [1*9+2+b_col_offset, 1*9+3+b_col_offset]]
        B21 = [[2*9+0+b_col_offset, 2*9+1+b_col_offset], [3*9+0+b_col_offset, 3*9+1+b_col_offset]]
        B22 = [[2*9+2+b_col_offset, 2*9+3+b_col_offset], [3*9+2+b_col_offset, 3*9+3+b_col_offset]]
        
        # C[i,k] -> W index k*4+i
        C11 = [[0*4+0, 0*4+1], [1*4+0, 1*4+1]]  # C[0,0],C[0,1]; C[1,0],C[1,1]
        C12 = [[2*4+0, 2*4+1], [3*4+0, 3*4+1]]  # C[0,2],C[0,3]; C[1,2],C[1,3]
        C21 = [[0*4+2, 0*4+3], [1*4+2, 1*4+3]]  # C[2,0],C[2,1]; C[3,0],C[3,1]
        C22 = [[2*4+2, 2*4+3], [3*4+2, 3*4+3]]  # C[2,2],C[2,3]; C[3,2],C[3,3]
        
        # Strassen combinations for 4x4:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Mi, expand into 7 rank-1 terms
        # Mi = X_i * Y_i where X_i is a 2x2 matrix (linear combo of A blocks) and Y_i is a 2x2 matrix
        
        # Define the 7 Strassen sub-terms for each Mi
        # Each sub-term t has: U_t (4-vector for X), V_t (4-vector for Y), W_t (4-vector for Z)
        
        def expand_2x2(U_mat, V_mat, W_mat, sign=1):
            """Expand a 2x2x2x2 product into 7 rank-1 terms."""
            contributions = []
            for t in range(7):
                U_t, V_t, W_t = strassen_terms[t]
                # U_t is a 4-vector: [x00, x01, x10, x11]
                # V_t is a 4-vector: [y00, y01, y10, y11]
                # W_t is a 4-vector: [z00, z01, z10, z11]
                
                U_contrib = np.zeros(n*m)
                V_contrib = np.zeros(m*p)
                W_contrib = np.zeros(p*n)
                
                # Map X indices to U
                for xi in range(2):
                    for xj in range(2):
                        idx = xi*2 + xj  # 0,1,2,3
                        U_idx = U_mat[xi][xj]
                        U_contrib[U_idx] += sign * U_t[idx]
                
                # Map Y indices to V
                for yi in range(2):
                    for yj in range(2):
                        idx = yi*2 + yj
                        V_idx = V_mat[yi][yj]
                        V_contrib[V_idx] += sign * V_t[idx]
                
                # Map Z indices to W
                for zi in range(2):
                    for zj in range(2):
                        idx = zi*2 + zj
                        W_idx = W_mat[zi][zj]
                        W_contrib[W_idx] += sign * W_t[idx]
                
                contributions.append((U_contrib, V_contrib, W_contrib))
            
            return contributions
        
        col = rank_offset
        # M1 = (A11+A22)(B11+B22)
        X1 = [[A11[0][0]+A22[0][0], A11[0][1]+A22[0][1]],
              [A11[1][0]+A22[1][0], A11[1][1]+A22[1][1]]]
        Y1 = [[B11[0][0], B11[0][1]],
              [B11[1][0], B11[1][1]]]  # Wait, B11+B22
        Y1 = [[B11[0][0]+B22[0][0], B11[0][1]+B22[0][1]],
              [B11[1][0]+B22[1][0], B11[1][1]+B22[1][1]]]
        
        for U_c, V_c, W_c in expand_2x2(X1, Y1, C11, sign=1):
            U[:, col] += U_c
            V[:, col] += V_c
            W[:, col] += W_c
            col += 1
        # M1 also contributes to C22 with sign +1
        for U_c, V_c, W_c in expand_2x2(X1, Y1, C22, sign=1):
            U[:, col] += U_c
            V[:, col] += V_c
            W[:, col] += W_c
            col += 1
        
        # Hmm, this doesn't work because M1 contributes to both C11 and C22.
        # I need to handle this differently.
        
        return col
    
    # This approach is getting complicated. Let me restructure.
    pass
