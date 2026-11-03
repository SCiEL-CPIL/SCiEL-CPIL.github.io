import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 6

    # --- Helper: build a 2x2x2 Strassen factor (7 terms) ---
    # Returns (U_local[4][7], V_local[4][7], W_local[4][7])
    # where indices are A[i*2+j], B[j*2+k], C[k*2+i]
    def strassen_222():
        U_l = [[0]*7 for _ in range(4)]
        V_l = [[0]*7 for _ in range(4)]
        W_l = [[0]*7 for _ in range(4)]
        
        # Term 0: M1=(A00+A11)(B00+B11) -> C00+C11
        U_l[0][0]=1; U_l[3][0]=1
        V_l[0][0]=1; V_l[3][0]=1
        W_l[0][0]=1; W_l[3][0]=1
        
        # Term 1: M2=(A10+A11)B00 -> C10-C11
        U_l[2][1]=1; U_l[3][1]=1
        V_l[0][1]=1
        W_l[2][1]=1; W_l[3][1]=-1
        
        # Term 2: M3=A00(B01-B11) -> C01+C11
        U_l[0][2]=1
        V_l[1][2]=1; V_l[3][2]=-1
        W_l[1][2]=1; W_l[3][2]=1
        
        # Term 3: M4=A11(B10-B00) -> C00+C10
        U_l[3][3]=1
        V_l[2][3]=1; V_l[0][3]=-1
        W_l[0][3]=1; W_l[2][3]=1
        
        # Term 4: M5=(A00+A01)B11 -> C01-C00
        U_l[0][4]=1; U_l[1][4]=1
        V_l[3][4]=1
        W_l[1][4]=1; W_l[0][4]=-1
        
        # Term 5: M6=(A10-A00)(B00+B01) -> C11
        U_l[2][5]=1; U_l[0][5]=-1
        V_l[0][5]=1; V_l[1][5]=1
        W_l[3][5]=1
        
        # Term 6: M7=(A01-A11)(B10+B11) -> C00
        U_l[1][6]=1; U_l[3][6]=-1
        V_l[2][6]=1; V_l[3][6]=1
        W_l[0][6]=1
        
        return U_l, V_l, W_l

    # --- Helper: build a 2x2x1 schoolbook factor (4 terms) ---
    def schoolbook_221():
        U_l = [[0]*4 for _ in range(4)]
        V_l = [[0]*4 for _ in range(4)]
        W_l = [[0]*4 for _ in range(4)]
        
        U_l[0][0]=1; V_l[0][0]=1; W_l[0][0]=1
        U_l[1][1]=1; V_l[2][1]=1; W_l[0][1]=1
        U_l[2][2]=1; V_l[0][2]=1; W_l[2][2]=1
        U_l[3][3]=1; V_l[2][3]=1; W_l[2][3]=1
        
        return U_l, V_l, W_l

    # --- Global construction ---
    # Block decomposition: 2x2 blocks of A (2x2), 2x3 blocks of B (2x3), 2x3 blocks of C (2x3)
    # For each block (rb, cb):
    #   A_block is 2x2 (indices rb*2..rb*2+1, cb*2..cb*2+1)
    #   B_block is 2x3 (indices rb*2..rb*2+1, cb*3..cb*3+2)
    #   C_block is 2x3 (indices rb*2..rb*2+1, cb*3..cb*3+2)
    #
    # For each block, we compute C_block = A_block * B_block
    # We split B_block into B[:,0:2] (2x2) and B[:,2] (2x1)
    # C_block[:,0:2] = A_block * B_block[:,0:2]  -> use Strassen (7 terms)
    # C_block[:,2]   = A_block * B_block[:,2]    -> use Schoolbook (4 terms)
    # Total per block: 11 terms
    # Total: 4 blocks * 11 terms = 44 terms

    U_g = [[0]*44 for _ in range(n*m)]  # 16 x 44
    V_g = [[0]*44 for _ in range(m*p)]  # 24 x 44
    W_g = [[0]*44 for _ in range(p*n)]  # 24 x 44

    U_l, V_l, W_l = strassen_222()
    U_s, V_s, W_s = schoolbook_221()

    for rb in range(2):
        for cb in range(2):
            base = (rb*2 + cb) * 11
            
            # Strassen part (first 2 columns)
            for t in range(7):
                col = base + t
                for i in range(2):
                    for j in range(2):
                        if U_l[i*2+j][t] != 0:
                            I = rb*2 + i
                            J = cb*2 + j
                            U_g[I*m + J][col] += U_l[i*2+j][t]
                        if V_l[i*2+j][t] != 0:
                            I = rb*2 + i
                            J = cb*3 + j  # j=0,1 for first 2 cols
                            V_g[I*p + J][col] += V_l[i*2+j][t]
                        if W_l[i*2+j][t] != 0:
                            I = rb*2 + i
                            J = cb*3 + j
                            W_g[J*n + I][col] += W_l[i*2+j][t]
            
            # Schoolbook part (3rd column, j=2)
            for t in range(4):
                col = base + 7 + t
                for i in range(2):
                    for j in range(2):
                        if U_s[i*2+j][t] != 0:
                            I = rb*2 + i
                            J = cb*2 + j
                            U_g[I*m + J][col] += U_s[i*2+j][t]
                        if V_s[i*2+j][t] != 0:
                            I = rb*2 + i
                            J = cb*3 + 2  # 3rd column
                            V_g[I*p + J][col] += V_s[i*2+j][t]
                        if W_s[i*2+j][t] != 0:
                            I = rb*2 + i
                            J = cb*3 + 2
                            W_g[J*n + I][col] += W_s[i*2+j][t]

    # Verify
    def verify(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if verify(U_g, V_g, W_g, 44):
        return {"U": U_g, "V": V_g, "W": W_g}

    # Fallback: schoolbook (rank 96)
    rank_fb = n * m * p
    U_fb = [[0]*rank_fb for _ in range(n*m)]
    V_fb = [[0]*rank_fb for _ in range(m*p)]
    W_fb = [[0]*rank_fb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
