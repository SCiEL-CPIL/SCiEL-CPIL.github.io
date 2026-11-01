"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def check_decomposition(U, V, W):
        """Check if U,V,W form a valid decomposition."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Strategy: Decompose M_{2,3,8} into 4 blocks of M_{2,3,2}
    # For each M_{2,3,2} block, use a rank-11 decomposition
    # (Strassen 7 for the 2x2 part + 4 extra rank-1 terms for the 3rd row of B)
    # Total rank: 4 * 11 = 44
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
        # Standard Strassen for 2x2 matrices:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-g)
        # m4 = d(h-g)
        # m5 = (a+b)h
        # m6 = (c-d)(g+h)
        # m7 = (a-b)(e+f)
        # p11 = m1 + m4 - m5 + m7
        # p12 = m3 + m5
        # p21 = m2 + m4
        # p22 = m1 - m2 + m3 + m6
        
        # U: 4x7 (a,b,c,d)
        U_s = [
            [1, 0, 1, 0, 1, 0, 1],   # a
            [0, 0, 0, 0, 1, 0, -1],  # b
            [0, 1, 0, 0, 0, 1, 0],   # c
            [1, 1, 0, 1, 0, -1, 0],  # d
        ]
        # V: 4x7 (e,f,g,h)
        V_s = [
            [1, 1, 0, 0, 0, 0, 1],   # e
            [0, 0, 1, 0, 0, 0, 1],   # f
            [0, 0, -1, 0, 0, 1, 0],  # g
            [1, 0, 0, 1, 1, 1, 0],   # h
        ]
        # W: 4x7 (p11,p12,p21,p22)
        # W[k*n+i, c] convention: n=2
        # p11: i=0,k=0 -> W[0]
        # p12: i=0,k=1 -> W[1]
        # p21: i=1,k=0 -> W[2]
        # p22: i=1,k=1 -> W[3]
        W_s = [
            [1, 0, 0, 0, -1, 0, 1],   # p11: m1+m4-m5+m7
            [0, 0, 1, 0, 1, 0, 0],    # p12: m3+m5
            [0, 1, 0, 1, 0, 0, 0],    # p21: m2+m4
            [1, -1, 1, 0, 0, 1, 0],   # p22: m1-m2+m3+m6
        ]
        
        return U_s, V_s, W_s
    
    # Build M_{2,3,2} decomposition with rank 11
    # Split the 3 rows of B into: first 2 rows handled by Strassen (rank 7), 
    # 3rd row handled by 4 rank-1 terms
    def m_2x3x2_decomposition():
        """Return U (6x11), V (6x11), W (4x11) for M_{2,3,2}."""
        U_s, V_s, W_s = strassen_2x2x2()
        
        # U_s is 4x7 for (a,b,c,d) where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # For M_{2,3,2}: A is 2x3, B is 3x2
        # U has shape (n*m, rank) = (6, rank)
        # U[i*m+j, c] where i=0,1 (row of A), j=0,1,2 (col of A)
        
        # Strassen handles the 2x2 part (j=0,1 of A, k=0,1 of B)
        # We need to extend to j=2 (3rd column of A)
        
        # For the 3rd column of A (j=2), we need 4 rank-1 terms:
        # T[i,2,k] = A[i,2] * B[2,k]
        # This can be done with 4 terms: one for each (i,k) pair
        # But we can be smarter: use 2 terms
        # T[0,2,0] = A[0,2]*B[2,0], T[0,2,1] = A[0,2]*B[2,1]
        # T[1,2,0] = A[1,2]*B[2,0], T[1,2,1] = A[1,2]*B[2,1]
        # This requires 4 rank-1 terms (or 2 if we can combine, but we can't easily)
        
        # Actually, let's use 4 rank-1 terms for the j=2 part
        # Total rank: 7 + 4 = 11
        
        U_full = [[0] * 11 for _ in range(6)]
        V_full = [[0] * 11 for _ in range(6)]  # m*p = 3*2 = 6
        W_full = [[0] * 11 for _ in range(4)]  # p*n = 2*2 = 4
        
        # Fill in Strassen part (columns 0-6)
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U_full[i*3+j][c] = U_s[i*2+j][c]
        
        for k in range(2):
            for j in range(2):
                for c in range(7):
                    V_full[j*2+k][c] = V_s[j*2+k][c]
        
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W_full[k*2+i][c] = W_s[k*2+i][c]
        
        # Add 4 rank-1 terms for j=2 (3rd column of A)
        # Term 7: i=0, k=0
        U_full[0*3+2][7] = 1  # A[0,2]
        V_full[2*2+0][7] = 1  # B[2,0]
        W_full[0*2+0][7] = 1  # result[0,0]
        
        # Term 8: i=0, k=1
        U_full[0*3+2][8] = 1  # A[0,2]
        V_full[2*2+1][8] = 1  # B[2,1]
        W_full[1*2+0][8] = 1  # result[0,1]
        
        # Term 9: i=1, k=0
        U_full[1*3+2][9] = 1  # A[1,2]
        V_full[2*2+0][9] = 1  # B[2,0]
        W_full[0*2+1][9] = 1  # result[1,0]
        
        # Term 10: i=1, k=1
        U_full[1*3+2][10] = 1  # A[1,2]
        V_full[2*2+1][10] = 1  # B[2,1]
        W_full[1*2+1][10] = 1  # result[1,1]
        
        return U_full, V_full, W_full
    
    # Now build the full M_{2,3,8} decomposition
    # Split 8 columns into 4 blocks of 2
    # Each block uses the M_{2,3,2} decomposition
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    U_232, V_232, W_232 = m_2x3x2_decomposition()
    
    for block in range(4):
        # Each block handles columns 2*block and 2*block+1 of B
        # U is the same for all blocks (A is shared)
        # V and W need to be shifted for the block offset
        
        U_block = [row[:] for row in U_232]
        V_block = [[0] * 11 for _ in range(24)]  # m*p = 3*8 = 24
        W_block = [[0] * 11 for _ in range(16)]  # p*n = 8*2 = 16
        
        # V: for block b, the k indices are 2*b and 2*b+1
        # V[j*p+k, c] where j=0,1,2 and k=0,...,7
        for j in range(3):
            for local_k in range(2):
                global_k = 2 * block + local_k
                for c in range(11):
                    V_block[j*8 + global_k][c] = V_232[j*2 + local_k][c]
        
        # W: for block b, the k indices are 2*b and 2*b+1
        # W[k*n+i, c] where k=0,...,7 and i=0,1
        for local_k in range(2):
            global_k = 2 * block + local_k
            for i in range(2):
                for c in range(11):
                    W_block[global_k*2 + i][c] = W_232[local_k*2 + i][c]
        
        U_blocks.append(U_block)
        V_blocks.append(V_block)
        W_blocks.append(W_block)
    
    # Concatenate all blocks
    total_rank = 4 * 11
    U_final = [[0] * total_rank for _ in range(6)]
    V_final = [[0] * total_rank for _ in range(24)]
    W_final = [[0] * total_rank for _ in range(16)]
    
    for block in range(4):
        offset = block * 11
        for i in range(6):
            for c in range(11):
                U_final[i][offset + c] = U_blocks[block][i][c]
        for i in range(24):
            for c in range(11):
                V_final[i][offset + c] = V_blocks[block][i][c]
        for i in range(16):
            for c in range(11):
                W_final[i][offset + c] = W_blocks[block][i][c]
    
    # Verify the decomposition
    if check_decomposition(U_final, V_final, W_final):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # If verification failed, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
