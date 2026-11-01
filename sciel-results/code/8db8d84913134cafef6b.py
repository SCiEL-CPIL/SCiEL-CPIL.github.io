import numpy as np

def construct():
    n, m, p = 2, 2, 6
    
    # Rank 7 Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices:
    # U: i (0..3) corresponds to A[i//2, i%2]
    # V: j (0..3) corresponds to B[j//2, j%2]
    # W: k (0..3) corresponds to C[k//2, k%2]
    
    U_strassen = np.array([
        [1, 0, 0, 1, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, -1]
    ], dtype=int)
    
    V_strassen = np.array([
        [1, 1, 1, -1, 0, 1, 1],
        [1, 0, -1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1]
    ], dtype=int)
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, -1, 0, 1, 0, 1, -1]
    ], dtype=int)
    
    # Verify Strassen:
    # C = A * B
    # A is 2x2, B is 2x2
    # We need to ensure the indices match the task's convention.
    # Task: U shape (n*m, rank) -> (4, 7)
    # V shape (m*p, rank) -> (4, 7) for 2x2x2
    # W shape (p*n, rank) -> (4, 7)
    
    # The task uses W[k*n + i, r].
    # My W_strassen is indexed by k (output row) and i (output col)?
    # Standard: C[i, j] = sum_k A[i, k] B[k, j].
    # Tensor index: (i, k, j).
    # U corresponds to A[i, k]. Index in U: i*m + k? Or i?
    # Task says U shape (n*m, rank). So U index is i*m + k.
    # For 2x2, U index is 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # My U_strassen row 0 is [1, 0, 0, 1, 1, -1, 0].
    # This corresponds to A[0,0] + A[1,1]?
    # Let's check Strassen P1 = (A00+A11)(B00+B11).
    # U for P1 should pick A00 and A11.
    # A00 is index 0. A11 is index 3.
    # So U[0] should be 1, U[3] should be 1.
    # My U_strassen[0] is 1, U_strassen[3] is 1. Correct.
    
    # V corresponds to B[k, j]. Index in V: k*p + j.
    # For 2x2x2, p=2. V index is k*2 + j.
    # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # P1 uses B00 and B11.
    # B00 is index 0. B11 is index 3.
    # V_strassen[0] should be 1, V_strassen[3] should be 1.
    # My V_strassen[0] is 1, V_strassen[3] is 1. Correct.
    
    # W corresponds to C[i, j]. Index in W: j*n + i.
    # Task: W[k*n + i, r]. Here k is the output column index (j in standard), i is output row index.
    # Wait, task says "output factor follows the official trace convention W[k*n+i,r]".
    # Usually C is n x p. So C[i, j] with i in 0..n-1, j in 0..p-1.
    # If W index is k*n + i, and k corresponds to the column j?
    # Let's assume W index = j*n + i.
    # For 2x2x2, n=2. W index = j*2 + i.
    # 0: j=0, i=0 -> C[0,0]
    # 1: j=0, i=1 -> C[1,0]
    # 2: j=1, i=0 -> C[0,1]
    # 3: j=1, i=1 -> C[1,1]
    
    # P1 = (A00+A11)(B00+B11) contributes to C00, C01, C10, C11?
    # No, Strassen P1 is used to compute C00 and C11?
    # C00 = P1 - P2 - P4 + P7?
    # Let's check standard Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 - P2 - P4 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5 - P6
    
    # Let's check my W_strassen for P1 (col 0).
    # W_strassen[:, 0] = [1, 0, 0, 1].
    # Indices: 0(C00)=1, 1(C10)=0, 2(C01)=0, 3(C11)=1.
    # So P1 contributes +1 to C00 and +1 to C11.
    # This matches C00 = P1... and C11 = P1...
    
    # Now construct the full 2x2x6 decomposition.
    # We have 3 blocks of B, each 2x2.
    # B is 2x6. Columns 0,1 are Block 0. 2,3 are Block 1. 4,5 are Block 2.
    # C is 2x6. Columns 0,1 are C0. 2,3 are C1. 4,5 are C2.
    
    rank = 21
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((12, rank), dtype=int) # m*p = 2*6 = 12
    W = np.zeros((12, rank), dtype=int) # p*n = 6*2 = 12
    
    for block in range(3):
        base_col = block * 7
        # U is the same for all blocks (shared A)
        U[:, base_col:base_col+7] = U_strassen
        
        # V is 2x6. Block `block` uses columns 2*block and 2*block+1.
        # V index is k*6 + j.
        # For block `block`, j_local is 0,1. j_global is 2*block + j_local.
        # V_strassen rows are k*2 + j_local.
        # We need to map V_strassen row `r` to V row `k*6 + j_global`.
        # V_strassen row `r` corresponds to k = r//2, j_local = r%2.
        # V_global index = k*6 + (2*block + j_local).
        
        for r in range(4):
            k = r // 2
            j_local = r % 2
            j_global = 2 * block + j_local
            v_global_idx = k * 6 + j_global
            V[v_global_idx, base_col:base_col+7] = V_strassen[r, :]
            
        # W is 6x2. Block `block` uses columns 2*block and 2*block+1 of C.
        # W index is j*n + i = j*2 + i.
        # For block `block`, j_global is 2*block + j_local.
        # W_strassen rows are j_local*2 + i.
        # W_strassen row `r` corresponds to j_local = r//2, i = r%2.
        # W_global index = j_global*2 + i.
        
        for r in range(4):
            j_local = r // 2
            i = r % 2
            j_global = 2 * block + j_local
            w_global_idx = j_global * 2 + i
            W[w_global_idx, base_col:base_col+7] = W_strassen[r, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
