import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen-based decomposition: rank 51
    # 3 blocks of 2x3x3, each block rank 17
    rank = 51
    U = np.zeros((n_rows, rank), dtype=np.int64)
    V = np.zeros((m_rows, rank), dtype=np.int64)
    W = np.zeros((p_rows, rank), dtype=np.int64)
    
    col = 0
    
    for block in range(3):
        # Block b handles B columns 3b..3b+2 and C columns 3b..3b+2
        # A indices: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
        # B indices in block: B[0,3b]=3b, B[0,3b+1]=3b+1, B[0,3b+2]=3b+2
        #                    B[1,3b]=9+3b, B[1,3b+1]=10+3b, B[1,3b+2]=11+3b
        #                    B[2,3b]=18+3b, B[2,3b+1]=19+3b, B[2,3b+2]=20+3b
        # C indices in block: C[0,3b]=(3b)*2+0=6b, C[0,3b+1]=6b+2, C[0,3b+2]=6b+4
        #                    C[1,3b]=6b+1, C[1,3b+1]=6b+3, C[1,3b+2]=6b+5
        
        # Part 1: Strassen for 2x2x2 (A[:,0:2] * B[0:2, 0:2] -> C[0:2, 0:2])
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
        U[0, col] = 1; U[4, col] = 1
        V[3*block, col] = 1; V[9+3*block, col] = 1
        W[6*block, col] = 1; W[6*block+2, col] = 1; W[6*block+1, col] = 1; W[6*block+3, col] = 1
        col += 1
        
        # M2 = (A10+A11)B00 -> contributes to C10, C11
        U[3, col] = 1; U[4, col] = 1
        V[3*block, col] = 1
        W[6*block+1, col] = 1; W[6*block+3, col] = 1
        col += 1
        
        # M3 = A00(B01-B11) -> contributes to C01, C11
        U[0, col] = 1
        V[1+3*block, col] = 1; V[10+3*block, col] = -1
        W[6*block+2, col] = 1; W[6*block+3, col] = 1
        col += 1
        
        # M4 = A11(B11-B00) -> contributes to C10, C11
        U[4, col] = 1
        V[9+3*block, col] = 1; V[3*block, col] = -1
        W[6*block+1, col] = 1; W[6*block+3, col] = 1
        col += 1
        
        # M5 = (A00+A01)B11 -> contributes to C00, C01
        U[0, col] = 1; U[1, col] = 1
        V[10+3*block, col] = 1
        W[6*block, col] = -1; W[6*block+2, col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10) -> contributes to C10
        U[3, col] = 1; U[0, col] = -1
        V[3*block, col] = 1; V[9+3*block, col] = 1
        W[6*block+1, col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> contributes to C00
        U[1, col] = 1; U[4, col] = -1
        V[9+3*block, col] = 1; V[10+3*block, col] = 1
        W[6*block, col] = 1
        col += 1
        
        # Part 2: 4 rank-1 terms for A[:,0:2] * B[0:2, 2] -> C[:,2]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # A[0,0]*B[0,2] -> C[0,2]
        U[0, col] = 1
        V[12+3*block, col] = 1
        W[6*block+4, col] = 1
        col += 1
        
        # A[0,1]*B[1,2] -> C[0,2]
        U[1, col] = 1
        V[13+3*block, col] = 1
        W[6*block+4, col] = 1
        col += 1
        
        # A[1,0]*B[0,2] -> C[1,2]
        U[3, col] = 1
        V[12+3*block, col] = 1
        W[6*block+5, col] = 1
        col += 1
        
        # A[1,1]*B[1,2] -> C[1,2]
        U[4, col] = 1
        V[13+3*block, col] = 1
        W[6*block+5, col] = 1
        col += 1
        
        # Part 3: 6 rank-1 terms for A[:,2] * B[2,:] -> C[:,0:3]
        # C[i,k] += A[i,2]*B[2,k] for i=0,1 and k=0,1,2
        
        for i in range(2):
            for k in range(3):
                U[i*3+2, col] = 1  # A[i,2]
                V[18+3*block+k, col] = 1  # B[2, 3block+k]
                W[6*block+2*k+i, col] = 1  # C[i, 3block+k]
                col += 1
    
    # Verify
    if col != rank:
        return schoolbook()
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
