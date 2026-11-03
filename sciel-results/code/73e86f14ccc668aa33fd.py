"""Matrix multiplication 3x4x6 with block Strassen decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    # We'll build the decomposition using block structure:
    # A: 3x4, split rows as [2, 1]
    # B: 4x6, split cols as [2, 2, 2]
    # 
    # For each (A_row_block, B_col_block):
    #   A[0:2,:] x B[:,c:c+2] -> 2x4x2, split middle 4 into 2+2:
    #     A[0:2,0:2] x B[0:2,c:c+2] -> 2x2x2, Strassen rank 7
    #     A[0:2,2:4] x B[2:4,c:c+2] -> 2x2x2, Strassen rank 7
    #   A[2:3,:] x B[:,c:c+2] -> 1x4x2, split middle 4 into 2+2:
    #     A[2:3,0:2] x B[0:2,c:c+2] -> 1x2x2, schoolbook rank 4
    #     A[2:3,2:4] x B[2:4,c:c+2] -> 1x2x2, schoolbook rank 4
    #
    # Total: 3 col_blocks * (2*7 + 2*4) = 3 * 22 = 66
    
    # Global index tracking
    all_U = []  # Each entry: (nm, rank_block)
    all_V = []
    all_W = []
    
    def add_block(U_block, V_block, W_block):
        """Add a block decomposition. U_block: (4, r), V_block: (4, r), W_block: (4, r) for 2x2x2
        or U_block: (2, r), V_block: (4, r), W_block: (2, r) for 1x2x2"""
        all_U.append(U_block)
        all_V.append(V_block)
        all_W.append(W_block)
    
    # Strassen decomposition for 2x2x2
    # T[a,b,c] = sum_d A[a,d] * B[d,c]
    # U: (4, 7) where row index = a*2+d
    # V: (4, 7) where row index = d*2+c
    # W: (4, 7) where row index = c*2+a
    def strassen_2x2x2():
        # 7 Strassen terms
        # Mr = (sum_ad alpha_mr[a,d] * A[a,d]) * (sum_dc beta_mr[d,c] * B[d,c])
        # Contribution to C[a,c] from Mr: gamma_mr[a,c] * Mr
        
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        # A part: A00 + A11 -> alpha[0,0]=1, alpha[1,1]=1
        U_s[0*2+0, 0] = 1; U_s[1*2+1, 0] = 1
        # B part: B00 + B11 -> beta[0,0]=1, beta[1,1]=1
        V_s[0*2+0, 0] = 1; V_s[1*2+1, 0] = 1
        # C part: C00 += M1, C11 += M1 -> gamma[0,0]=1, gamma[1,1]=1
        W_s[0*2+0, 0] = 1; W_s[1*2+1, 0] = 1
        
        # M2 = (A10+A11)B00
        U_s[1*2+0, 1] = 1; U_s[1*2+1, 1] = 1
        V_s[0*2+0, 1] = 1
        W_s[1*2+0, 1] = 1  # C10 += M2
        
        # M3 = A00(B01-B11)
        U_s[0*2+0, 2] = 1
        V_s[0*2+1, 2] = 1; V_s[1*2+1, 2] = -1
        W_s[0*2+1, 2] = 1  # C01 += M3
        
        # M4 = A11(B10-B00)
        U_s[1*2+1, 3] = 1
        V_s[1*2+0, 3] = 1; V_s[0*2+0, 3] = -1
        W_s[0*2+0, 3] = 1  # C00 += M4
        W_s[1*2+0, 3] = 1  # C10 += M4
        
        # M5 = (A00+A10)B11
        U_s[0*2+0, 4] = 1; U_s[1*2+0, 4] = 1
        V_s[1*2+1, 4] = 1
        W_s[0*2+0, 4] = -1  # C00 -= M5
        W_s[1*2+1, 4] = 1   # C11 += M5
        
        # M6 = (A10-A00)(B00+B10)
        U_s[1*2+0, 5] = 1; U_s[0*2+0, 5] = -1
        V_s[0*2+0, 5] = 1; V_s[1*2+0, 5] = 1
        W_s[1*2+1, 5] = 1   # C11 += M6
        
        # M7 = (A01-A11)(B01+B11)
        U_s[0*2+1, 6] = 1; U_s[1*2+1, 6] = -1
        V_s[0*2+1, 6] = 1; V_s[1*2+1, 6] = 1
        W_s[0*2+0, 6] = 1   # C00 += M7
        
        return U_s, V_s, W_s
    
    # Schoolbook decomposition for 1x2x2
    # T[a,b,c] = sum_d A[a,d] * B[d,c] where a is fixed (1 row)
    # U: (1*2, 4) = (2, 4), row index = d
    # V: (2*2, 4) = (4, 4), row index = d*2+c
    # W: (2*1, 4) = (2, 4), row index = c*1+a = c (since a=0)
    def schoolbook_1x2x2():
        U_s = np.zeros((2, 4))
        V_s = np.zeros((4, 4))
        W_s = np.zeros((2, 4))
        
        col = 0
        for d in range(2):
            for c in range(2):
                U_s[d, col] = 1
                V_s[d*2+c, col] = 1
                W_s[c*1+0, col] = 1  # W[c*1 + a] with a=0
                col += 1
        
        return U_s, V_s, W_s
    
    # Now build the full decomposition
    # We need to map local block indices to global indices
    # Global U: (3*4, R) = (12, R), index = i*4 + j
    # Global V: (4*6, R) = (24, R), index = j*6 + k
    # Global W: (6*3, R) = (18, R), index = k*3 + i
    
    # For each column block of B (c_start in [0, 2, 4]):
    for c_start in [0, 2, 4]:
        # Top 2 rows of A (i in [0,2])
        #   Middle split: j in [0,2) and j in [2,4)
        for j_start in [0, 2]:
            U_s, V_s, W_s = strassen_2x2x2()
            # Map to global indices
            # U local row: a*2+d where a in [0,2) is local row (i-a_offset), d in [0,2) is local col (j-j_offset)
            # Global U row: i*4 + j where i = a (since a_offset=0 for top block), j = j_start + d
            U_g = np.zeros((12, 7))
            V_g = np.zeros((24, 7))
            W_g = np.zeros((18, 7))
            
            for a in range(2):
                for d in range(2):
                    local_row = a * 2 + d
                    global_i = a  # top 2 rows
                    global_j = j_start + d
                    U_g[global_i * 4 + global_j, :] = U_s[local_row, :]
            
            for d in range(2):
                for c in range(2):
                    local_row = d * 2 + c
                    global_j = j_start + d
                    global_k = c_start + c
                    V_g[global_j * 6 + global_k, :] = V_s[local_row, :]
            
            for c in range(2):
                for a in range(2):
                    local_row = c * 2 + a
                    global_k = c_start + c
                    global_i = a
                    W_g[global_k * 3 + global_i, :] = W_s[local_row, :]
            
            add_block(U_g, V_g, W_g)
        
        # Bottom 1 row of A (i=2)
        for j_start in [0, 2]:
            U_s, V_s, W_s = schoolbook_1x2x2()
            U_g = np.zeros((12, 4))
            V_g = np.zeros((24, 4))
            W_g = np.zeros((18, 4))
            
            # U local row: d (0 or 1)
            # Global U row: i*4 + j where i=2, j = j_start + d
            for d in range(2):
                global_i = 2
                global_j = j_start + d
                U_g[global_i * 4 + global_j, :] = U_s[d, :]
            
            for d in range(2):
                for c in range(2):
                    local_row = d * 2 + c
                    global_j = j_start + d
                    global_k = c_start + c
                    V_g[global_j * 6 + global_k, :] = V_s[local_row, :]
            
            for c in range(2):
                local_row = c
                global_k = c_start + c
                global_i = 2
                W_g[global_k * 3 + global_i, :] = W_s[local_row, :]
            
            add_block(U_g, V_g, W_g)
    
    # Combine all blocks
    total_rank = sum(U.shape[1] for U in all_U)
    U_final = np.zeros((12, total_rank))
    V_final = np.zeros((24, total_rank))
    W_final = np.zeros((18, total_rank))
    
    col_offset = 0
    for U_b, V_b, W_b in zip(all_U, all_V, all_W):
        r = U_b.shape[1]
        U_final[:, col_offset:col_offset+r] = U_b
        V_final[:, col_offset:col_offset+r] = V_b
        W_final[:, col_offset:col_offset+r] = W_b
        col_offset += r
    
    # Verify the decomposition
    def check_decomposition(U, V, W):
        n, m, p = DIMENSIONS
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not check_decomposition(U_final, V_final, W_final):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n*m, rank))
        V_fb = np.zeros((m*p, rank))
        W_fb = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
