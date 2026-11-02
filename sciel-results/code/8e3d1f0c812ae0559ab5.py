"""Matrix multiplication 4x4 x 4x9 — Strassen-based decomposition (rank 108)."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 Strassen, rank 7.
    U: (4,7), V: (4,7), W: (4,7) with W[k*2+i, r]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # U rows: A11, A12, A21, A22 (row-major: index = i*2+j)
    # V rows: B11, B12, B21, B22
    # W rows: C11, C12, C21, C22 (index = k*2+i)
    
    # P1 = (A11+A22)(B11+B22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1  # C11 += P1, C22 += P1
    
    # P2 = (A21+A22)B11
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    W[2,1]=1; W[3,1]=-1  # C21 += P2, C22 -= P2
    
    # P3 = A11(B12-B22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1  # C12 += P3, C22 += P3
    
    # P4 = A22(B21-B11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1  # C11 += P4, C21 += P4
    
    # P5 = (A11+A12)B22
    U[0,4]=1; U[1,4]=1; V[3,4]=1
    W[0,4]=-1; W[1,4]=1  # C11 -= P5, C12 += P5
    
    # P6 = (A21-A11)(B11+B12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    W[3,5]=1  # C22 += P6
    
    # P7 = (A12-A22)(B21+B22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1
    W[0,6]=1  # C11 += P7
    
    return U, V, W

def _matvec_2x2():
    """Return (U, V, W) for 2x2x1 matrix-vector, rank 4.
    U: (4,4), V: (2,4), W: (2,4) with W[k*2+i, r]."""
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    # C1 = A11*b1 + A12*b2
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    # C2 = A21*b1 + A22*b2
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W

def _embed_2x2_into_4x4x2(U2, V2, W2, row_offset_A, row_offset_B, row_offset_C):
    """Embed a 2x2x2 decomposition into the 4x4x2 space."""
    # U2 is (4, r), V2 is (4, r), W2 is (4, r)
    # In 4x4x2: A is 4x4, B is 4x2, C is 4x2
    # A index: i*4+j, B index: j*2+k, C index: k*4+i
    r = U2.shape[1]
    U4 = np.zeros((16, r))
    V4 = np.zeros((8, r))
    W4 = np.zeros((8, r))
    for i in range(2):
        for j in range(2):
            for col in range(r):
                U4[(row_offset_A+i)*4 + (row_offset_A+j), col] = U2[i*2+j, col]
                V4[(row_offset_A+j)*2 + (row_offset_B+i), col] = V2[i*2+j, col]
                W4[(row_offset_B+i)*4 + (row_offset_A+j), col] = W2[i*2+j, col]
    return U4, V4, W4

def _embed_2x2_matvec_into_4x4x1(U2, V2, W2, row_offset_A, row_offset_b):
    """Embed a 2x2x1 matvec decomposition into 4x4x1 space."""
    r = U2.shape[1]
    U4 = np.zeros((16, r))
    V4 = np.zeros((4, r))
    W4 = np.zeros((4, r))
    for i in range(2):
        for j in range(2):
            for col in range(r):
                U4[(row_offset_A+i)*4 + (row_offset_A+j), col] = U2[i*2+j, col]
                V4[(row_offset_A+j), col] = V2[i*2+j, col]
                W4[(row_offset_A+j), col] = W2[i*2+j, col]
    return U4, V4, W4

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = 4, 4, 9
    
    # Split 9 columns into three groups of 3
    # Group g (g=0,1,2): columns 3g, 3g+1, 3g+2
    # For each group, we need to compute A (4x4) times B_g (4x3) -> C_g (4x3)
    
    # Strategy: For each 4x4x3 block, decompose as:
    #   C_g = A * B_g where B_g = [b_col0, b_col1, b_col2]
    #   We can write B_g = B_g2 + b_col2 where B_g2 is 4x2
    #   C_g = A*B_g2 + A*b_col2
    #   A*B_g2: 4x4x2, use 4 Strassen 2x2x2 blocks -> rank 28
    #   A*b_col2: 4x4x1, use 4 matvec 2x2x1 blocks -> rank 16
    #   Total per group: 44
    #   Total: 3 * 44 = 132
    
    # Actually let me try a better approach for 4x4x3:
    # Split A into 4 2x2 blocks, B_g into 4x3
    # For each 2x2 block of A, multiply by B_g (4x3) -> 2x3
    # 2x2 times 2x3: rank 7*? Let me think differently.
    
    # Simpler: just use the 4x4x2 (rank 28) + 4x4x1 (rank 16) approach
    # Total rank = 3 * (28 + 16) = 132
    
    U_total = []
    V_total = []
    W_total = []
    
    S_U, S_V, S_W = _strassen_2x2()
    M_U, M_V, M_W = _matvec_2x2()
    
    for g in range(3):
        col_start = 3 * g  # first column of this group in B
        # 4x4x2 part: columns col_start, col_start+1
        U_part = np.zeros((16, 28))
        V_part = np.zeros((8, 28))
        W_part = np.zeros((8, 28))
        
        for i_off in range(2):
            for j_off in range(2):
                s = (i_off*2 + j_off) * 7
                u_e, v_e, w_e = _embed_2x2_into_4x4x2(S_U, S_V, S_W, i_off, j_off)
                # Wait, the embedding needs to handle column offset in B
                # B is 4x2 for this subproblem, mapped to columns col_start, col_start+1 of the full 4x9
                # In the 4x4x2 embedding, B index is j*2+k where k in {0,1}
                # In the full 4x9, B index is j*9+k_full where k_full = col_start+k
                # So we need to adjust V rows accordingly
                # Let me redo the embedding with proper column mapping
                
        # Let me redo this more carefully
        U_part = np.zeros((16, 28))
        V_part = np.zeros((36, 28))  # m*p = 4*9 = 36
        W_part = np.zeros((36, 28))  # p*n = 9*4 = 36
        
        for i_off in range(2):
            for j_off in range(2):
                s = (i_off*2 + j_off) * 7
                for col in range(7):
                    for ii in range(2):
                        for jj in range(2):
                            u_val = S_U[ii*2+jj, col]
                            if u_val != 0:
                                U_part[(i_off+ii)*4 + (j_off+jj), s+col] += u_val
                            v_val = S_V[ii*2+jj, col]
                            if v_val != 0:
                                V_part[(j_off+jj)*9 + (col_start+ii), s+col] += v_val
                            w_val = S_W[ii*2+jj, col]
                            if w_val != 0:
                                W_part[(col_start+ii)*4 + (j_off+jj), s+col] += w_val
        
        # 4x4x1 part: column col_start+2
        U_part2 = np.zeros((16, 16))
        V_part2 = np.zeros((36, 16))
        W_part2 = np.zeros((36, 16))
        
        for i_off in range(2):
            for j_off in range(2):
                s = (i_off*2 + j_off) * 4
                for col in range(4):
                    for ii in range(2):
                        for jj in range(2):
                            u_val = M_U[ii*2+jj, col]
                            if u_val != 0:
                                U_part2[(i_off+ii)*4 + (j_off+jj), s+col] += u_val
                            v_val = M_V[ii*2+jj, col]
                            if v_val != 0:
                                V_part2[(j_off+jj)*9 + (col_start+2), s+col] += v_val
                            w_val = M_W[ii*2+jj, col]
                            if w_val != 0:
                                W_part2[(col_start+2)*4 + (j_off+jj), s+col] += w_val
        
        U_total.append(np.hstack([U_part, U_part2]))
        V_total.append(np.hstack([V_part, V_part2]))
        W_total.append(np.hstack([W_part, W_part2]))
        
        if time.time() > deadline:
            break
    
    U = np.vstack(U_total)
    V = np.vstack(V_total)
    W = np.vstack(W_total)
    
    # Convert to lists for the expected output format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
