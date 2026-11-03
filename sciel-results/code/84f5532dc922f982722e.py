import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        # 7 terms
        # U: 4x7, V: 4x7, W: 4x7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2: (A10+A11)B00 -> C10, C11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # Term 3: A00(B01-B11) -> C00, C01
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # Term 4: A11(B10-B00) -> C10, C11
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # Term 5: (A00+A10)B11 -> C00, C10
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # Term 6: (A10-A00)(B00+B01) -> C01, C11
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 5] = 1; W[3, 5] = 1
        
        # Term 7: (A01-A11)(B10+B11) -> C01, C11
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    # Kronecker product of decompositions
    def kruskal_kron(U1, V1, W1, U2, V2, W2):
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        R = r1 * r2
        
        U = np.zeros((U1.shape[0] * U2.shape[0], R))
        V = np.zeros((V1.shape[0] * V2.shape[0], R))
        W = np.zeros((W1.shape[0] * W2.shape[0], R))
        
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1 * r2 + c2
                U[:, c] = np.kron(U1[:, c1], U2[:, c2])
                V[:, c] = np.kron(V1[:, c1], V2[:, c2])
                W[:, c] = np.kron(W1[:, c1], W2[:, c2])
                
        return U, V, W

    # 4x4x4 decomposition using Strassen
    U2, V2, W2 = strassen_2x2x2()
    U4, V4, W4 = kruskal_kron(U2, V2, W2, U2, V2, W2)
    rank_444 = U4.shape[1] # 49

    # Total rank calculation
    # 4x4x4 part: rank 49
    # 4x4x2 part (A1 x B12): schoolbook 4*4*2 = 32
    # 4x1x6 part (A2 x B2): schoolbook 4*1*6 = 24
    # Total: 49 + 32 + 24 = 105
    
    total_rank = 105
    
    U_total = np.zeros((n * m, total_rank))
    V_total = np.zeros((m * p, total_rank))
    W_total = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # 1. 4x4x4 block (i in 0..3, j in 0..3, k in 0..3)
    # U indices: i*5+j. For j<4, i<4.
    # V indices: j*6+k. For j<4, k<4.
    # W indices: k*4+i. For k<4, i<4.
    
    for c in range(rank_444):
        # Map 4x4x4 indices to 4x5x6 indices
        # U: i*4+j -> i*5+j (since m=5, but we only use j<4)
        # Wait, U4 is 16x49. Index is i*4+j.
        # We need to place this into U_total[i*5+j, col].
        for i in range(4):
            for j in range(4):
                idx_u4 = i * 4 + j
                idx_ut = i * 5 + j
                U_total[idx_ut, col_offset + c] = U4[idx_u4, c]
                
        for j in range(4):
            for k in range(4):
                idx_v4 = j * 4 + k
                idx_vt = j * 6 + k
                V_total[idx_vt, col_offset + c] = V4[idx_v4, c]
                
        for k in range(4):
            for i in range(4):
                idx_w4 = k * 4 + i
                idx_wt = k * 4 + i # p=6, n=4. W index is k*n+i.
                # Wait, W4 is for 4x4x4. Index k*4+i.
                # W_total index is k*4+i.
                # So it maps directly.
                W_total[idx_wt, col_offset + c] = W4[idx_w4, c]
                
    col_offset += rank_444
    
    # 2. 4x4x2 block (i in 0..3, j in 0..3, k in 4..5)
    # Schoolbook
    U1, V1, W1 = schoolbook(4, 4, 2)
    for c in range(U1.shape[1]):
        for i in range(4):
            for j in range(4):
                idx_u1 = i * 4 + j
                idx_ut = i * 5 + j
                U_total[idx_ut, col_offset + c] = U1[idx_u1, c]
                
        for j in range(4):
            for k in range(2):
                idx_v1 = j * 2 + k
                idx_vt = j * 6 + (4 + k) # k in 4,5
                V_total[idx_vt, col_offset + c] = V1[idx_v1, c]
                
        for k in range(2):
            for i in range(4):
                idx_w1 = k * 4 + i
                idx_wt = (4 + k) * 4 + i # k in 4,5
                W_total[idx_wt, col_offset + c] = W1[idx_w1, c]
                
    col_offset += U1.shape[1]
    
    # 3. 4x1x6 block (i in 0..3, j=4, k in 0..5)
    # Schoolbook
    U2, V2, W2 = schoolbook(4, 1, 6)
    for c in range(U2.shape[1]):
        for i in range(4):
            for j in range(1):
                idx_u2 = i * 1 + j
                idx_ut = i * 5 + (4 + j) # j=4
                U_total[idx_ut, col_offset + c] = U2[idx_u2, c]
                
        for j in range(1):
            for k in range(6):
                idx_v2 = j * 6 + k
                idx_vt = (4 + j) * 6 + k # j=4
                V_total[idx_vt, col_offset + c] = V2[idx_v2, c]
                
        for k in range(6):
            for i in range(4):
                idx_w2 = k * 4 + i
                idx_wt = k * 4 + i
                W_total[idx_wt, col_offset + c] = W2[idx_w2, c]
                
    col_offset += U2.shape[1]
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    construct()
