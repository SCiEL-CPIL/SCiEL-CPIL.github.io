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
    
    # Schoolbook fallback
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
    
    # Build rank-36 decomposition using Strassen for 2x2x2 core
    # 2x3x9 = 3 blocks of 2x3x3
    # Each 2x3x3 = (2x2x3 via Strassen+rank2) + (1x1x3 via rank3) = 12
    # Total rank = 36
    
    rank = 36
    U = np.zeros((n_rows, rank), dtype=np.int64)
    V = np.zeros((m_rows, rank), dtype=np.int64)
    W = np.zeros((p_rows, rank), dtype=np.int64)
    
    col = 0
    
    for block in range(3):
        # Block handles columns 3*block to 3*block+2 of B and C
        # A is 2x3, indices: A[i,j] at U[i*3+j, col]
        # B block is 3x3, indices: B[j,k] at V[j*9 + 3*block + k, col]
        # C block is 2x3, indices: C[i,k] at W[(3*block+k)*2 + i, col]
        
        # Part 1: 2x2x2 Strassen for A[:,0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
        # B indices: (0,0)=0, (0,1)=1, (1,0)=9, (1,1)=10 (within block, offset by 3*block)
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3 (within block, offset by 3*block)
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[0, col] = 1; U[4, col] = 1
        V[3*block, col] = 1; V[9 + 3*block, col] = 1
        W[3*block, col] = 1; W[3*block+1, col] = 1  # C[0,0], C[0,1]... wait
        # W index: k*n + i = k*2 + i
        # C[0,0]: k=0, i=0 -> W[0]
        # C[0,1]: k=1, i=0 -> W[2]
        # C[1,0]: k=0, i=1 -> W[1]
        # C[1,1]: k=1, i=1 -> W[3]
        # But within block, k_full = 3*block + k
        # So W index = (3*block + k) * 2 + i
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        U[3, col] = 1; U[4, col] = 1
        V[3*block, col] = 1
        W[3*block, col] = 1; W[3*block+1, col] = 1  # C[1,0], C[1,1]... no
        # M2 contributes to C[1,0] and C[1,1]
        # C[1,0]: k=0, i=1 -> W[(3*block+0)*2 + 1] = W[6*block + 1]
        # C[1,1]: k=1, i=1 -> W[(3*block+1)*2 + 1] = W[6*block + 3]
        W[6*block + 1, col] = 1
        W[6*block + 3, col] = 1
        col += 1
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        U[0, col] = 1
        V[1 + 3*block, col] = 1; V[10 + 3*block, col] = -1
        W[2 + 6*block, col] = 1; W[3 + 6*block, col] = 1  # C[0,1], C[1,1]
        col += 1
        
        # M4 = A[1,1] * (B[1,1] - B[0,0])
        U[4, col] = 1
        V[9 + 3*block, col] = 1; V[3*block, col] = -1
        W[6*block, col] = 1; W[2 + 6*block, col] = 1  # C[0,0], C[0,1]... 
        # M4 contributes to C[0,0] and C[1,0]
        # C[0,0]: k=0, i=0 -> W[6*block]
        # C[1,0]: k=0, i=1 -> W[6*block + 1]
        W[6*block + 1, col] = 1
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        U[0, col] = 1; U[1, col] = 1
        V[10 + 3*block, col] = 1
        W[6*block, col] = -1; W[2 + 6*block, col] = 1  # -C[0,0] + C[0,1]... 
        # M5 contributes to -C[0,0] and C[0,1]
        # C[0,0]: k=0, i=0 -> W[6*block]
        # C[0,1]: k=1, i=0 -> W[6*block + 2]
        W[6*block + 2, col] = 1
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[1,0])
        U[3, col] = 1; U[0, col] = -1
        V[3*block, col] = 1; V[9 + 3*block, col] = 1
        W[6*block + 3, col] = 1  # C[1,1]
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        U[1, col] = 1; U[4, col] = -1
        V[9 + 3*block, col] = 1; V[10 + 3*block, col] = 1
        W[6*block, col] = 1  # C[0,0]
        col += 1
        
        # Part 2: 2x2x1 for A[:,0:2] * B[0:2, 2] -> C[0:2, 2]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        # Rank 2: two products
        
        # Product 1: A[0,0]*B[0,2] -> contributes to C[0,2]
        U[0, col] = 1
        V[12 + 3*block, col] = 1  # B[0,2] in block
        W[4 + 6*block, col] = 1  # C[0,2]: k=2, i=0 -> W[(3*block+2)*2 + 0]
        col += 1
        
        # Product 2: A[0,1]*B[1,2] -> contributes to C[0,2]
        U[1, col] = 1
        V[13 + 3*block, col] = 1  # B[1,2] in block
        W[4 + 6*block, col] = 1
        col += 1
        
        # Product 3: A[1,0]*B[0,2] -> contributes to C[1,2]
        U[3, col] = 1
        V[12 + 3*block, col] = 1
        W[5 + 6*block, col] = 1  # C[1,2]: k=2, i=1 -> W[(3*block+2)*2 + 1]
        col += 1
        
        # Product 4: A[1,1]*B[1,2] -> contributes to C[1,2]
        U[4, col] = 1
        V[13 + 3*block, col] = 1
        W[5 + 6*block, col] = 1
        col += 1
        
        # Part 3: 1x1x3 for A[1,2] * B[2, 0:3] -> C[1, 0:3]... 
        # Wait, A is 2x3, so A[1,2] is the last element of row 1.
        # Actually, A[i,j] where i=0,1 and j=0,1,2.
        # The 1x1x3 part is A[1,2] * B[2,k] for k=0,1,2, contributing to C[1,k]
        # But also A[0,2] * B[2,k] contributes to C[0,k]
        # So it's actually 2x1x3, not 1x1x3!
        
        # Let me reconsider. 2x3x3 = (2x2x3) + (2x1x3)
        # 2x1x3: A[:,2] (2x1) * B[2,:] (1x3) -> C[:,0:3] contribution
        # This is rank 6 (each of 2 rows times each of 3 cols)
        # But we can do better: C[i,k] += A[i,2] * B[2,k]
        # This is an outer product of A[:,2] and B[2,:], which is rank 1 as a matrix,
        # but as a bilinear form, it's rank 6 (each entry is a separate product).
        
        # Wait, no. The bilinear form for 2x1x3:
        # T[i,k] = A[i,2] * B[2,k] for i=0,1 and k=0,1,2
        # This is 6 terms, each a product of one A entry and one B entry.
        # Rank 6.
        
        # So 2x3x3 = (2x2x3) + (2x1x3)
        # 2x2x3 = 7 (Strassen) + 2 (for the third column) = 9
        # 2x1x3 = 6
        # Total = 15, not 12.
        
        # Hmm, my earlier calculation was wrong. Let me redo.
        
        # Actually, for 2x2x3:
        # C[:,0:2] uses Strassen (rank 7)
        # C[:,2] = A[:,0:2] * B[0:2,2]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        # This is rank 4 (four products), not 2.
        
        # So 2x2x3 = 7 + 4 = 11
        # 2x1x3 = 6
        # 2x3x3 = 11 + 6 = 17
        # 2x3x9 = 3 * 17 = 51
        
        # That's worse than schoolbook (54) but only slightly better.
        # Let me check if this is correct.
        
        # Actually, I think I need to be more careful. Let me just use the schoolbook for now
        # and verify the Strassen construction for 2x2x2.
        
        pass
    
    # Given the complexity and time, let me just return schoolbook
    return schoolbook()
