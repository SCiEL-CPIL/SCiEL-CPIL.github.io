import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def build_2x3x3_rank11():
        """Build a rank-11 decomposition for 2x3x3 matrix multiplication.
        
        U: (6, 11), V: (9, 11), W: (6, 11)
        Constraint: for all i in {0,1}, j in {0,1,2}, k in {0,1,2}:
            sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
        """
        # This is a known rank-11 decomposition for 2x3x3
        # I'll use a construction based on partitioning
        
        # Actually, let me use the block approach:
        # Partition A (2x3) columns as (1,2):
        # A1 (2x1), A2 (2x2)
        # B1 (1x3), B2 (2x3)
        # C = A1*B1 + A2*B2
        # A1*B1: 2x1x3, rank 6
        # A2*B2: 2x2x3, rank 11 (using 2x2x2 Strassen + 2x2x1)
        
        # For 2x2x3: partition B (2x3) columns as (2,1):
        # B1 (2x2), B2 (2x1)
        # A*B1: 2x2x2, rank 7 (Strassen)
        # A*B2: 2x2x1, rank 4
        # Total: rank 11
        
        # So total for 2x3x3: 6 + 11 = 17, not 11.
        
        # Let me try a different approach. Maybe there's a direct rank-11 construction.
        # For now, let me use the rank-17 construction.
        
        U = np.zeros((6, 17))
        V = np.zeros((9, 17))
        W = np.zeros((6, 17))
        
        # Block 1: A1*B1 where A1 is 2x1 (col 0), B1 is 1x3 (row 0)
        # This is 2x1x3: T[i,0,k] = 1
        # U indices: i*3+0 for i in {0,1}
        # V indices: 0*3+k for k in {0,1,2}
        # W indices: k*2+i for k in {0,1,2}, i in {0,1}
        col = 0
        for i in range(2):
            for k in range(3):
                U[i*3+0, col] = 1
                V[0*3+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
        # col = 6
        
        # Block 2: A2*B2 where A2 is 2x2 (cols 1,2), B2 is 2x3 (rows 1,2)
        # This is 2x2x3: T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
        # But in the original indexing:
        # A2 corresponds to A[:, 1:3], so j_global = j_local + 1
        # B2 corresponds to B[1:3, :], so k_global = k_local (wait, B is 3x3, rows 1,2)
        # Actually B2 is 2x3, rows 1,2 of B, all 3 columns
        # So for the sub-problem:
        # U indices: i*3 + (j_local+1) for i in {0,1}, j_local in {0,1}
        # V indices: (j_local+1)*3 + k for j_local in {0,1}, k in {0,1,2}
        # W indices: k*2 + i for k in {0,1,2}, i in {0,1}
        
        # Sub-block 2a: A2*B2a where B2a is 2x2 (cols 0,1 of B2, i.e., cols 0,1 of B)
        # This is 2x2x2, use Strassen rank 7
        # Strassen decomposition for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # In our indexing for the 2x2x2 sub-problem:
        # i in {0,1}, j_local in {0,1}, k_local in {0,1}
        # U[i*3 + (j_local+1), r]
        # V[(j_local+1)*3 + k_local, r]  (but B2's rows are 1,2 of B, so V index is (1+j_local)*3 + k_local)
        # W[k_local*2 + i, r]  (but this is for the full 2x3x3, k goes 0,1,2)
        # Wait, for the 2x2x2 sub-problem, k_local goes 0,1 (cols 0,1 of B2)
        # But in the full problem, these are cols 0,1 of B, so k_global = k_local
        
        # Strassen terms for 2x2x2:
        strassen_U = [
            [1, 1],  # M1: A00, A11
            [0, 1],  # M2: A10, A11
            [1, 0],  # M3: A00
            [1, 1],  # M4: A11
            [1, 1],  # M5: A00, A01
            [1, -1], # M6: A10, A00
            [1, -1], # M7: A01, A11
        ]
        strassen_V = [
            [1, 1],  # M1: B00, B11
            [1, 0],  # M2: B00
            [1, -1], # M3: B10, B11
            [1, -1], # M4: B01, B00
            [0, 1],  # M5: B11
            [1, 1],  # M6: B00, B10
            [1, 1],  # M7: B01, B11
        ]
        strassen_W = [
            [1, 0, 0, 0],  # M1: C00, -C11
            [0, 0, 1, 0],  # M2: -C10, -C11
            [0, 0, 0, 1],  # M3: C01, C11
            [0, 1, 0, 0],  # M4: C00, C10
            [0, 1, 0, -1], # M5: -C00, C01
            [0, 0, 1, 0],  # M6: C10
            [1, 0, 0, 1],  # M7: C00, C11
        ]
        # W is indexed as [k*2+i], so:
        # C00: k=0,i=0 -> W[0]
        # C01: k=1,i=0 -> W[2]
        # C10: k=0,i=1 -> W[1]
        # C11: k=1,i=1 -> W[3]
        
        # Let me re-index strassen_W properly:
        # M1: C00 += 1, C11 -= 1 -> W[0]+=1, W[3]-=1
        # M2: C10 -= 1, C11 -= 1 -> W[1]-=1, W[3]-=1
        # M3: C01 += 1, C11 += 1 -> W[2]+=1, W[3]+=1
        # M4: C00 += 1, C10 += 1 -> W[0]+=1, W[1]+=1
        # M5: C00 -= 1, C01 += 1 -> W[0]-=1, W[2]+=1
        # M6: C10 += 1 -> W[1]+=1
        # M7: C00 += 1, C11 += 1 -> W[0]+=1, W[3]+=1
        
        strassen_W_correct = [
            [1, 0, 0, -1],  # M1
            [0, -1, 0, -1], # M2
            [0, 0, 1, 1],   # M3
            [1, 1, 0, 0],   # M4
            [-1, 0, 1, 0],  # M5
            [0, 1, 0, 0],   # M6
            [1, 0, 0, 1],   # M7
        ]
        
        for r in range(7):
            # U: i in {0,1}, j_local in {0,1}
            # j_global = j_local + 1
            U[0*3 + 1, col+r] = strassen_U[r][0]  # i=0, j_local=0
            U[1*3 + 1, col+r] = strassen_U[r][1]  # i=1, j_local=0
            U[0*3 + 2, col+r] = strassen_U[r][0]  # i=0, j_local=1
            U[1*3 + 2, col+r] = strassen_U[r][1]  # i=1, j_local=1
            
            # V: j_local in {0,1}, k_local in {0,1}
            # j_global = j_local + 1
            V[1*3 + 0, col+r] = strassen_V[r][0]  # j_local=0, k_local=0
            V[1*3 + 1, col+r] = strassen_V[r][1]  # j_local=0, k_local=1
            V[2*3 + 0, col+r] = strassen_V[r][0]  # j_local=1, k_local=0
            V[2*3 + 1, col+r] = strassen_V[r][1]  # j_local=1, k_local=1
            
            # W: k_local in {0,1}, i in {0,1}
            # W index = k_global*2 + i = k_local*2 + i
            W[0*2 + 0, col+r] = strassen_W_correct[r][0]  # k=0, i=0 (C00)
            W[0*2 + 1, col+r] = strassen_W_correct[r][1]  # k=0, i=1 (C10)
            W[1*2 + 0, col+r] = strassen_W_correct[r][2]  # k=1, i=0 (C01)
            W[1*2 + 1, col+r] = strassen_W_correct[r][3]  # k=1, i=1 (C11)
        
        col += 7
        # col = 13
        
        # Sub-block 2b: A2*B2b where B2b is 2x1 (col 2 of B2, i.e., col 2 of B)
        # This is 2x2x1: T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k=2
        # Schoolbook rank 4
        for i in range(2):
            for j_local in range(2):
                U[i*3 + (j_local+1), col] = 1
                V[(j_local+1)*3 + 2, col] = 1
                W[2*2 + i, col] = 1
                col += 1
        
        # col should be 17
        assert col == 17, f"Expected col=17, got {col}"
        
        return U, V, W
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the 2x3x6 multiplication tensor."""
        R = U.shape[1]
        if V.shape[1] != R or W.shape[1] != R:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Build the 2x3x6 decomposition using two 2x3x3 blocks
    U1, V1, W1 = build_2x3x3_rank11()  # Actually rank 17
    U2, V2, W2 = build_2x3x3_rank11()
    
    # For the first block (k=0,1,2):
    # U is same: (6, 17)
    # V: first 9 rows of V1, last 9 rows are 0
    # W: first 6 rows of W1, last 6 rows are 0
    
    # For the second block (k=3,4,5):
    # U is same: (6, 17)
    # V: first 9 rows are 0, last 9 rows are V2
    # W: first 6 rows are 0, last 6 rows are W2
    
    R = 17
    U = np.zeros((6, 2*R))
    V = np.zeros((18, 2*R))
    W = np.zeros((12, 2*R))
    
    # First block
    U[:, 0:R] = U1
    V[0:9, 0:R] = V1
    W[0:6, 0:R] = W1
    
    # Second block
    U[:, R:2*R] = U2
    V[9:18, R:2*R] = V2
    W[6:12, R:2*R] = W2
    
    if verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
