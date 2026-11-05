"""Matrix multiplication 4x4x7 with block decomposition using Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix mult with Strassen, rank 7.
    U: shape (4, 7) for A[0,0],A[0,1],A[1,0],A[1,1]
    V: shape (4, 7) for B[0,0],B[0,1],B[1,0],B[1,1]
    W: shape (4, 7) for C[0,0],C[0,1],C[1,0],C[1,1] (k*n+i with n=2, so C[0*2+0]=C[0,0], C[1*2+0]=C[1,0], C[0*2+1]=C[0,1], C[1*2+1]=C[1,1])
    Wait: W[k*n+i] with n=2: index = k*2+i
    C[0,0] -> k=0,i=0 -> 0
    C[0,1] -> k=1,i=0 -> 2  (this is C[1,0] in row-major? No.)
    
    Actually the convention is W[k*n+i, r] where the output is C[i,j] but indexed as W[k*n+i].
    For 2x2x2: n=2, m=2, p=2. Output C is 2x2. W has p*n = 4 rows.
    W[k*n+i, r] means row index = k*n + i.
    So W[0] = k=0,i=0 -> contributes to C[0,?]
    W[1] = k=0,i=1 -> contributes to C[1,?]
    W[2] = k=1,i=0 -> contributes to C[0,1]
    W[3] = k=1,i=1 -> contributes to C[1,1]
    
    Hmm, this is the "official trace convention W[k*n+i,r]". Let me think about what this means.
    
    The tensor for matrix mult: T[i,j,k] = delta[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k].
    So the tensor T has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    
    W[k*n+i, r] suggests the output index is k*n+i, which for a p×n output C[k,i]... 
    Actually for 4x4x7: A is 4x4, B is 4x7, C is 4x7. C[i,k] = sum_j A[i,j]*B[j,k].
    So C has dimensions n x p = 4 x 7. The output index in W is k*n+i, so for C[i,k], the W row is k*n+i.
    
    For 2x2x2: n=2, p=2. C[i,k] for i in {0,1}, k in {0,1}. W row = k*n+i = k*2+i.
    W[0] = k=0,i=0 -> C[0,0]
    W[1] = k=0,i=1 -> C[1,0]
    W[2] = k=1,i=0 -> C[0,1]
    W[3] = k=1,i=1 -> C[1,1]
    """
    # Strassen's 7 factors for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U rows: A[0,0], A[0,1], A[1,0], A[1,1]
    # V rows: B[0,0], B[0,1], B[1,0], B[1,1]
    # W rows: C[0,0](k=0,i=0), C[1,0](k=0,i=1), C[0,1](k=1,i=0), C[1,1](k=1,i=1)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[1, 1] = 1  # C10
    
    # M3: A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1; V[3, 2] = -1  # B10, -B11
    W[2, 2] = 1  # C01
    
    # M4: A11(B01-B00)
    U[3, 3] = 1  # A11
    V[1, 3] = 1; V[0, 3] = -1  # B01, -B00
    W[0, 3] = 1  # C00
    W[1, 3] = 1  # C10
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1  # C00
    W[2, 4] = 1  # C01
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1  # A10, -A00
    V[0, 5] = 1; V[2, 5] = 1  # B00, B10
    W[3, 5] = 1  # C11
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01, -A11
    V[1, 6] = 1; V[3, 6] = 1  # B01, B11
    W[0, 6] = 1  # C00
    W[2, 6] = 1  # C01
    
    return U, V, W


def _schoolbook_2x2x1():
    """Return (U, V, W) for 2x2x1 matrix mult (C[i,0] = sum_j A[i,j]*B[j,0]), rank 2.
    U: shape (4, 2) for A[0,0],A[0,1],A[1,0],A[1,1]
    V: shape (2, 2) for B[0,0],B[1,0]
    W: shape (2, 2) for C[0,0](k=0,i=0), C[1,0](k=0,i=1)
    """
    U = np.zeros((4, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    # Factor 0: A[*,0]*B[0,0]
    U[0, 0] = 1  # A[0,0]
    U[2, 0] = 1  # A[1,0]
    V[0, 0] = 1  # B[0,0]
    W[0, 0] = 1  # C[0,0] (k=0,i=0)
    W[1, 0] = 1  # C[1,0] (k=0,i=1)
    
    # Factor 1: A[*,1]*B[1,0]
    U[1, 1] = 1  # A[0,1]
    U[3, 1] = 1  # A[1,1]
    V[1, 1] = 1  # B[1,0]
    W[0, 1] = 1  # C[0,0]
    W[1, 1] = 1  # C[1,0]
    
    return U, V, W


def _schoolbook_2x2x3():
    """Return (U, V, W) for 2x2x3 matrix mult using Strassen(2x2x2) + schoolbook(2x2x1), rank 9.
    U: shape (4, 9)
    V: shape (6, 9)
    W: shape (6, 9)
    """
    U_s, V_s, W_s = _strassen_2x2x2()  # rank 7
    U_sb, V_sb, W_sb = _schoolbook_2x2x1()  # rank 2
    
    U = np.zeros((4, 9))
    V = np.zeros((6, 9))
    W = np.zeros((6, 9))
    
    # Strassen part (columns 0-6)
    U[:, 0:7] = U_s
    V[0:4, 0:7] = V_s  # B[0,0],B[0,1],B[1,0],B[1,1] -> V rows 0-3
    # For 2x2x2, V has 4 rows (m*p = 2*2 = 4), for 2x2x3, V has 6 rows (m*p = 2*3 = 6)
    # B indices: B[j,k] for j in {0,1}, k in {0,1,2}. V row = j*p+k = j*3+k
    # B[0,0]=V[0], B[0,1]=V[1], B[0,2]=V[2], B[1,0]=V[3], B[1,1]=V[4], B[1,2]=V[5]
    # Strassen uses B[0,0],B[0,1],B[1,0],B[1,1] -> V[0],V[1],V[3],V[4]
    V_new = np.zeros((6, 7))
    V_new[0, :] = V_s[0, :]  # B[0,0]
    V_new[1, :] = V_s[1, :]  # B[0,1]
    V_new[3, :] = V_s[2, :]  # B[1,0]
    V_new[4, :] = V_s[3, :]  # B[1,1]
    V[:, 0:7] = V_new
    
    # W for Strassen: C[i,k] for i in {0,1}, k in {0,1}. W row = k*n+i = k*2+i
    # W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1]
    # For 2x2x3, W has 6 rows (p*n = 3*2 = 6). W row = k*n+i = k*2+i
    # W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1], W[4]=C[0,2], W[5]=C[1,2]
    W[:, 0:7] = W_s  # First 4 rows get Strassen W
    
    # Schoolbook 2x2x1 part (columns 7-8) for the 3rd column
    U[:, 7:9] = U_sb
    # V for schoolbook: B[0,0],B[1,0] -> but we need B[0,2],B[1,2] for the 3rd column
    # V_sb has 2 rows: B[0,0]=V[0], B[1,0]=V[1]
    # We need to map to V[2] (B[0,2]) and V[5] (B[1,2])
    V_new2 = np.zeros((6, 2))
    V_new2[2, :] = V_sb[0, :]  # B[0,2]
    V_new2[5, :] = V_sb[1, :]  # B[1,2]
    V[:, 7:9] = V_new2
    
    # W for schoolbook: C[0,0],C[1,0] -> but we need C[0,2],C[1,2]
    # W_sb has 2 rows: C[0,0]=W[0], C[1,0]=W[1]
    # We need W[4] (C[0,2]) and W[5] (C[1,2])
    W_new2 = np.zeros((6, 2))
    W_new2[4, :] = W_sb[0, :]  # C[0,2]
    W_new2[5, :] = W_sb[1, :]  # C[1,2]
    W[:, 7:9] = W_new2
    
    return U, V, W


def _schoolbook_2x2x1_generic():
    """Return (U, V, W) for 2x2x1, rank 2, but with correct V and W row indexing.
    U: shape (4, 2)
    V: shape (2, 2)
    W: shape (2, 2)
    """
    U = np.zeros((4, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    
    U[0, 0] = 1; U[2, 0] = 1
    U[1, 1] = 1; U[3, 1] = 1
    V[0, 0] = 1; V[1, 1] = 1
    W[0, 0] = 1; W[1, 0] = 1
    W[0, 1] = 1; W[1, 1] = 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Block decomposition:
    # i: {0,1}, {2,3}
    # j: {0,1}, {2,3}
    # k: {0,1,2}, {3,4,5}, {6}
    
    # For each (i_block, j_block, k_block):
    #   if k_block size 3: use 2x2x3 (rank 9)
    #   if k_block size 1: use 2x2x1 (rank 2)
    
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1, 2), (3, 4, 5), (6)]
    
    total_rank = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                if len(kb) == 3:
                    total_rank += 9
                else:
                    total_rank += 2
    
    rank = total_rank
    
    U = np.zeros((n * m, rank))  # 16 x rank
    V = np.zeros((m * p, rank))  # 28 x rank
    W = np.zeros((p * n, rank))  # 28 x rank
    
    col = 0
    for ib_start in [0, 2]:
        for jb_start in [0, 2]:
            for kb in [(0, 3), (3, 6), (6, 7)]:
                kb_start, kb_end = kb
                kb_size = kb_end - kb_start
                
                if kb_size == 3:
                    U_b, V_b, W_b = _schoolbook_2x2x3()
                    # U_b: 4x9, V_b: 6x9, W_b: 6x9
                    # Map to global indices
                    # U global row: i*m + j, where i = ib_start + i_local, j = jb_start + j_local
                    for i_local in range(2):
                        for j_local in range(2):
                            global_row = (ib_start + i_local) * m + (jb_start + j_local)
                            U[global_row, col:col+9] = U_b[i_local * 2 + j_local, :]
                    
                    # V global row: j*p + k, where j = jb_start + j_local, k = kb_start + k_local
                    for j_local in range(2):
                        for k_local in range(3):
                            global_row = (jb_start + j_local) * p + (kb_start + k_local)
                            V[global_row, col:col+9] = V_b[j_local * 3 + k_local, :]
                    
                    # W global row: k*n + i, where k = kb_start + k_local, i = ib_start + i_local
                    for k_local in range(3):
                        for i_local in range(2):
                            global_row = (kb_start + k_local) * n + (ib_start + i_local)
                            W[global_row, col:col+9] = W_b[k_local * 2 + i_local, :]
                    
                    col += 9
                else:
                    U_b, V_b, W_b = _schoolbook_2x2x1()
                    # U_b: 4x2, V_b: 2x2, W_b: 2x2
                    for i_local in range(2):
                        for j_local in range(2):
                            global_row = (ib_start + i_local) * m + (jb_start + j_local)
                            U[global_row, col:col+2] = U_b[i_local * 2 + j_local, :]
                    
                    for j_local in range(2):
                        global_row = (jb_start + j_local) * p + kb_start
                        V[global_row, col:col+2] = V_b[j_local, :]
                    
                    for i_local in range(2):
                        global_row = kb_start * n + (ib_start + i_local)
                        W[global_row, col:col+2] = W_b[i_local, :]
                    
                    col += 2
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
