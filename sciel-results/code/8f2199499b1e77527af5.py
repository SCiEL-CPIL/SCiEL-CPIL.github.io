import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # We will use a decomposition of the 7 columns into 3, 3, and 1.
    # The tensor is a sum of three tensors:
    # 1. 3x3x3 (Winograd rank 12)
    # 2. 3x3x3 (Winograd rank 12)
    # 3. 3x3x1 (Schoolbook rank 9)
    # Total rank = 12 + 12 + 9 = 33.
    
    # Helper to construct Winograd 3x3x3 tensor
    def winograd_3x3x3(offset_u, offset_v, offset_w):
        # Winograd algorithm for 3x3 matrix multiplication
        # Rank 12
        # U: 9x12, V: 9x12, W: 9x12
        U = [[0]*12 for _ in range(9)]
        V = [[0]*12 for _ in range(9)]
        W = [[0]*12 for _ in range(9)]
        
        # Winograd 3x3 decomposition
        # This is a known decomposition.
        # We can use the standard Winograd minimal inverse for 3x3.
        # The matrices are:
        # U = [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
        # ... 
        # Actually, let's use a simpler approach: just use schoolbook for 3x3x3?
        # Schoolbook rank is 27.
        # Winograd rank is 12.
        # Let's use the known Winograd 3x3 decomposition.
        
        # The Winograd 3x3 decomposition has rank 12.
        # The factor matrices are:
        # U (9x12), V (9x12), W (9x12)
        
        # We can construct it using the following pattern:
        # The 12 multiplications correspond to the 12 non-zero entries in the transformed matrices.
        
        # For simplicity and correctness, let's use the schoolbook for 3x3x3 if Winograd is too complex to hardcode correctly in this time.
        # But 12 is much better than 27.
        # Let's try to hardcode the Winograd 3x3 matrices.
        
        # Winograd 3x3:
        # U:
        # [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
        # [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
        # [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
        # [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        # [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
        # [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        # [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        
        # This is not correct. Let's use a known correct Winograd 3x3 decomposition.
        # Actually, the Winograd 3x3 decomposition is not trivial to hardcode.
        # Let's use the schoolbook for 3x3x3 (rank 27) and 3x3x1 (rank 9).
        # Total rank = 27 + 27 + 9 = 63. This is the same as the seed.
        
        # We need a better approach.
        # Let's use the Winograd algorithm for 3x3x3 with rank 12.
        # The matrices are:
        # U = [[1,0,0, 1,0,0, 0,1,0, 1,0,0],
        #      [0,1,0, 0,1,0, 1,0,0, 0,1,0],
        #      [0,0,1, 0,0,1, 0,0,1, 0,0,1],
        #      [1,0,0, 0,1,0, 0,0,1, 0,0,1],
        #      [0,1,0, 0,0,1, 0,0,0, 0,1,0],
        #      [0,0,1, 0,0,0, 0,1,0, 0,0,1],
        #      [1,0,0, 0,1,0, 0,0,1, 0,0,1],
        #      [0,1,0, 0,0,1, 0,0,0, 0,1,0],
        #      [0,0,1, 0,0,0, 0,1,0, 0,0,1]]
        
        # This is still not correct. Let's use a library or a known decomposition.
        # Since we can't use external libraries for the decomposition itself, we must hardcode it.
        
        # Let's use the fact that the Winograd 3x3 decomposition is equivalent to using 12 multiplications.
        # We can construct it using the following method:
        # The Winograd algorithm for 3x3 uses 2-point DFTs.
        # The rank is 12.
        
        # For the purpose of this task, let's use the schoolbook for 3x3x3 (rank 27) and 3x3x1 (rank 9).
        # Total rank 63.
        
        # To improve, we can use the Winograd 3x3 decomposition.
        # Here is a correct Winograd 3x3 decomposition (rank 12):
        # U:
        # [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
        # [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
        # [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
        # [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        # [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
        # [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        # [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        
        # V:
        # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
        # [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        # [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        # [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        # [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        # [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        # [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        # [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        # [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        
        # W:
        # [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
        # [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
        # [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
        # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
        # [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        # [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        # [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
        # [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        # [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        
        # This is not correct. Let's use the schoolbook for 3x3x3.
        # Rank 27.
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][k] = 1
                    V[j*3+k][k] = 1
                    W[k*3+i][k] = 1
        return U, V, W

    # Helper to construct schoolbook 3x3x1 tensor
    def schoolbook_3x3x1(offset_u, offset_v, offset_w):
        U = [[0]*9 for _ in range(9)]
        V = [[0]*9 for _ in range(9)]
        W = [[0]*9 for _ in range(9)]
        for i in range(3):
            for j in range(3):
                for k in range(1):
                    U[i*3+j][k] = 1
                    V[j*3+k][k] = 1
                    W[k*3+i][k] = 1
        return U, V, W

    # Total rank = 27 + 27 + 9 = 63.
    # This is the same as the seed.
    
    # To improve, we need a better decomposition.
    # Let's use the Winograd 3x3x3 decomposition.
    # The correct Winograd 3x3x3 decomposition has rank 12.
    # We can find it online or derive it.
    
    # For now, let's return the schoolbook decomposition.
    rank = 63
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
