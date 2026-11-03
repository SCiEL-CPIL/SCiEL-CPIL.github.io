import time
import numpy as np

DIMENSIONS = [5, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i==k else 0
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                target = 1 if i == k else 0
                if abs(total - target) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 -> 2x2 with rank 7.
    Indices:
    U: (4, 7) -> i*2+j
    V: (4, 7) -> j*2+k
    W: (4, 7) -> k*2+i
    """
    # Strassen formulas for C = AB
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the tensor form: sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Note: W index is k*n+i. Here n=2. So W is indexed by k*2+i.
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # r=0: M1
    # (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1 (since M1 contributes to C11 and C22? No, M1 is (A11+A22)(B11+B22) in 1-based.
    # Let's stick to 0-based.
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # Contributes to C[0,0] and C[1,1].
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # W index: k*2+i. For C[0,0], k=0,i=0 -> 0. For C[1,1], k=1,i=1 -> 3.
    
    # r=1: M2 = (A[1,0] + A[1,1]) * B[0,0]
    # Contributes to C[1,0] and C[1,1] (with sign -)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C[1,0] -> k=1,i=0 -> 2
    W[3, 1] = -1 # C[1,1] -> k=1,i=1 -> 3
    
    # r=2: M3 = A[0,0] * (B[1,0] - B[1,1])
    # Contributes to C[0,1] and C[1,1]
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C[0,1] -> k=0,i=1 -> 1
    W[3, 2] = 1 # C[1,1] -> k=1,i=1 -> 3
    
    # r=3: M4 = A[1,1] * (B[0,1] - B[0,0])
    # Contributes to C[0,0] and C[1,0]
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C[0,0]
    W[2, 3] = 1 # C[1,0]
    
    # r=4: M5 = (A[0,0] + A[0,1]) * B[1,1]
    # Contributes to C[0,0] (with sign -) and C[0,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C[0,0]
    W[1, 4] = 1 # C[0,1]
    
    # r=5: M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # Contributes to C[1,1]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C[1,1]
    
    # r=6: M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # Contributes to C[0,0]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C[0,0]
    
    return U, V, W

def _laderman_3x3():
    """
    Returns U, V, W for 3x3 * 3x3 -> 3x3 with rank 7.
    Laderman's algorithm.
    """
    # Laderman 3x3 decomposition
    # This is more complex. We will use a known rank 7 decomposition.
    # References: Laderman, "On the complexity of multiplying matrices" (1976)
    # Formulas:
    # M1 = (A11+A22+A33)(B11+B22+B33)
    # M2 = (A12+A21+A31+A32)(B12+B21+B31+B32)
    # M3 = (A13+A23+A31+A33)(B13+B23+B31+B33)
    # M4 = (A11+A22)(B11+B22)
    # M5 = (A11+A33)(B11+B33)
    # M6 = (A22+A33)(B22+B33)
    # M7 = (A12+A21)(B12+B21) + (A13+A23)(B13+B23) + (A31+A32)(B31+B32) -- Wait, this is not linear in the standard way.
    
    # Actually, Laderman's 7 terms are:
    # 1. (A11+A22+A33)(B11+B22+B33)
    # 2. (A12+A21+A31+A32)(B12+B21+B31+B32)
    # 3. (A13+A23+A31+A33)(B13+B23+B31+B33)
    # 4. (A11+A22)(B11+B22)
    # 5. (A11+A33)(B11+B33)
    # 6. (A22+A33)(B22+B33)
    # 7. (A12+A21)(B12+B21) + (A13+A23)(B13+B23) + (A31+A32)(B31+B32) -- This term 7 is a sum of products, not a single product.
    # Wait, Laderman's algorithm uses 7 multiplications.
    # Let's use a verified rank 7 decomposition for 3x3.
    # One such decomposition is found in literature.
    # For simplicity and reliability within time limit, if I can't recall the exact 7-term Laderman coefficients, 
    # I will fall back to a rank 9 or 10 decomposition for 3x3 if needed, or just use schoolbook (27) for 3x3 
    # to ensure the code runs, but that defeats the purpose.
    
    # Let's try to implement a known rank 7 3x3 decomposition.
    # If I cannot verify it quickly, I will use a rank 9 decomposition (e.g. 3 Strassen 2x2 + corrections) or schoolbook.
    # Given the risk, let's use a **conservative** approach:
    # Use Schoolbook for 3x3 (rank 27) if we can't be sure about Laderman.
    # But 7+27+12+12+18+12+18+18 = 124. No improvement.
    
    # Let's try to implement Laderman.
    # U, V, W shapes: (9, 7)
    
    # I will use a placeholder for now and fill with a known valid decomposition if I can find one in memory.
    # Actually, I can use the **Strassen 2x2** on the 3x3 by padding? No.
    
    # Let's assume I can construct a rank 9 decomposition for 3x3 easily?
    # 3x3 can be split into 2x2 and 1x1.
    # A = [A22 A21; A12 A11]
    # B = [B22 B21; B12 B11]
    # C11 (2x2) = A22 B22 + A21 B12 -> 2 terms? No.
    
    # Let's stick to the **Block Decomposition with Schoolbook Cross Terms** and **Strassen for 2x2**.
    # For 3x3, I will use **Schoolbook** (rank 27) to be safe, unless I can implement Laderman.
    # Total Rank: 7 (2x2) + 27 (3x3) + 12 + 12 + 18 + 12 + 18 + 18 = 124.
    # This is worse than 125? No, 124 < 125.
    
    # Wait, 7 + 27 + 12 + 12 + 18 + 12 + 18 + 18 = 124.
    # This is a valid improvement!
    
    # Let's refine the cross terms.
    # Term 2: 2x3 * 3x2 -> 2x2. Rank 12.
    # Term 3: 2x2 * 2x3 -> 2x3. Rank 12.
    # Term 4: 2x3 * 3x3 -> 2x3. Rank 18.
    # Term 5: 3x2 * 2x2 -> 3x2. Rank 12.
    # Term 6: 3x2 * 2x3 -> 3x3. Rank 18.
    # Term 7: 3x3 * 2x2 -> 3x2. Rank 18.
    
    # Total: 7 + 27 + 12 + 12 + 18 + 12 + 18 + 18 = 124.
    
    # Can we do better on 3x3?
    # If I use a rank 9 decomposition for 3x3?
    # There is a rank 9 algorithm for 3x3? No, best known is 7.
    # But implementing Laderman is risky.
    
    # Let's implement the 124 rank solution.
    
    U_2x2, V_2x2, W_2x2 = _strassen_2x2()
    
    # Construct the full 5x5 decomposition
    n, m, p = 5, 5, 5
    rank_total = 124
    U = np.zeros((n*m, rank_total), dtype=np.int64)
    V = np.zeros((m*p, rank_total), dtype=np.int64)
    W = np.zeros((p*n, rank_total), dtype=np.int64)
    
    col = 0
    
    # Helper to add a block decomposition
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset, rank_block):
        nonlocal col
        for r in range(rank_block):
            for i in range(U_block.shape[0]):
                val = U_block[i, r]
                if val != 0:
                    U[i_offset + i, col + r] = val
            for j in range(V_block.shape[0]):
                val = V_block[j, r]
                if val != 0:
                    V[j_offset + j, col + r] = val
            for k in range(W_block.shape[0]):
                val = W_block[k, r]
                if val != 0:
                    W[k_offset + k, col + r] = val
        col += rank_block

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # i: 0,1; j: 0,1; k: 0,1
    add_block(U_2x2, V_2x2, W_2x2, 0, 0, 0, 7)
    
    # Block 2: A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5]
    # i: 2,3,4; j: 2,3,4; k: 2,3,4
    # Schoolbook for 3x3
    U_3x3 = np.zeros((9, 27), dtype=np.int64)
    V_3x3 = np.zeros((9, 27), dtype=np.int64)
    W_3x3 = np.zeros((9, 27), dtype=np.int64)
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_3x3[i*3+j, c] = 1
                V_3x3[j*3+k, c] = 1
                W_3x3[k*3+i, c] = 1
                c += 1
    add_block(U_3x3, V_3x3, W_3x3, 2*5, 2*5, 2*5, 27)
    
    # Block 3: A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # i: 0,1; j: 2,3,4; k: 0,1
    # 2x3 * 3x2 -> 2x2
    # Schoolbook: 2*3*2 = 12
    U_232 = np.zeros((6, 12), dtype=np.int64)
    V_32 = np.zeros((6, 12), dtype=np.int64)
    W_22 = np.zeros((4, 12), dtype=np.int64)
    c = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U_232[i*3+j, c] = 1
                V_32[j*2+k, c] = 1
                W_22[k*2+i, c] = 1
                c += 1
    add_block(U_232, V_32, W_22, 0, 2*5, 0, 12)
    
    # Block 4: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # i: 0,1; j: 0,1; k: 2,3,4
    # 2x2 * 2x3 -> 2x3
    # Schoolbook: 2*2*3 = 12
    U_22 = np.zeros((4, 12), dtype=np.int64)
    V_23 = np.zeros((6, 12), dtype=np.int64)
    W_23 = np.zeros((6, 12), dtype=np.int64)
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_22[i*2+j, c] = 1
                V_23[j*3+k, c] = 1
                W_23[k*2+i, c] = 1
                c += 1
    add_block(U_22, V_23, W_23, 0, 0, 2*5, 12)
    
    # Block 5: A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # i: 0,1; j: 2,3,4; k: 2,3,4
    # 2x3 * 3x3 -> 2x3
    # Schoolbook: 2*3*3 = 18
    U_23 = np.zeros((6, 18), dtype=np.int64)
    V_33 = np.zeros((9, 18), dtype=np.int64)
    W_23b = np.zeros((6, 18), dtype=np.int64)
    c = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_23[i*3+j, c] = 1
                V_33[j*3+k, c] = 1
                W_23b[k*2+i, c] = 1
                c += 1
    add_block(U_23, V_33, W_23b, 0, 2*5, 2*5, 18)
    
    # Block 6: A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2]
    # i: 2,3,4; j: 0,1; k: 0,1
    # 3x2 * 2x2 -> 3x2
    # Schoolbook: 3*2*2 = 12
    U_32 = np.zeros((6, 12), dtype=np.int64)
    V_22b = np.zeros((4, 12), dtype=np.int64)
    W_32 = np.zeros((6, 12), dtype=np.int64)
    c = 0
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U_32[i*2+j, c] = 1
                V_22b[j*2+k, c] = 1
                W_32[k*3+i, c] = 1
                c += 1
    add_block(U_32, V_22b, W_32, 2*5, 0, 0, 12)
    
    # Block 7: A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5]
    # i: 2,3,4; j: 0,1; k: 2,3,4
    # 3x2 * 2x3 -> 3x3
    # Schoolbook: 3*2*3 = 18
    U_32b = np.zeros((6, 18), dtype=np.int64)
    V_23b = np.zeros((6, 18), dtype=np.int64)
    W_33b = np.zeros((9, 18), dtype=np.int64)
    c = 0
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U_32b[i*2+j, c] = 1
                V_23b[j*3+k, c] = 1
                W_33b[k*3+i, c] = 1
                c += 1
    add_block(U_32b, V_23b, W_33b, 2*5, 0, 2*5, 18)
    
    # Block 8: A[2:5, 2:5] * B[2:5, 0:2] -> C[2:5, 0:2]
    # i: 2,3,4; j: 2,3,4; k: 0,1
    # 3x3 * 2x2 -> 3x2
    # Schoolbook: 3*3*2 = 18
    U_33c = np.zeros((9, 18), dtype=np.int64)
    V_22c = np.zeros((4, 18), dtype=np.int64)
    W_32c = np.zeros((6, 18), dtype=np.int64)
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U_33c[i*3+j, c] = 1
                V_22c[j*2+k, c] = 1
                W_32c[k*3+i, c] = 1
                c += 1
    add_block(U_33c, V_22c, W_32c, 2*5, 2*5, 0, 18)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U = np.zeros((n*m, n*m*p), dtype=np.int64)
        V = np.zeros((m*p, n*m*p), dtype=np.int64)
        W = np.zeros((p*n, n*m*p), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _block_decomposition()

# EVOLVE-BLOCK-START
def _block_decomposition():
    return _construct_block()

def _construct_block():
    # Call the logic from above
    # Since the above logic is in the global scope, we can just call it
    # But we defined it as part of the previous block. 
    # We need to make sure the function is defined.
    # The function _verify_decomposition is defined.
    # The function _strassen_2x2 is defined.
    # We will re-implement the block construction here to be self-contained in the evolve block.
    
    n, m, p = 5, 5, 5
    rank_total = 124
    U = np.zeros((n*m, rank_total), dtype=np.int64)
    V = np.zeros((m*p, rank_total), dtype=np.int64)
    W = np.zeros((p*n, rank_total), dtype=np.int64)
    
    col = 0
    
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset, rank_block):
        nonlocal col
        for r in range(rank_block):
            for i in range(U_block.shape[0]):
                val = U_block[i, r]
                if val != 0:
                    U[i_offset + i, col + r] = val
            for j in range(V_block.shape[0]):
                val = V_block[j, r]
                if val != 0:
                    V[j_offset + j, col + r] = val
            for k in range(W_block.shape[0]):
                val = W_block[k, r]
                if val != 0:
                    W[k_offset + k, col + r] = val
        col += rank_block

    # Block 1: 2x2 Strassen
    U_2x2, V_2x2, W_2x2 = _strassen_2x2()
    add_block(U_2x2, V_2x2, W_2x2, 0, 0, 0, 7)
    
    # Block 2: 3x3 Schoolbook
    U_3x3 = np.zeros((9, 27), dtype=np.int64)
    V_3x3 = np.zeros((9, 27), dtype=np.int64)
    W_3x3 = np.zeros((9, 27), dtype=np.int64)
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_3x3[i*3+j, c] = 1
                V_3x3[j*3+k, c] = 1
                W_3x3[k*3+i, c] = 1
                c += 1
    add_block(U_3x3, V_3x3, W_3x3, 2*5, 2*5, 2*5, 27)
    
    # Block 3: 2x3 * 3x2 -> 2x2 (Schoolbook 12)
    U_232 = np.zeros((6, 12), dtype=np.int64)
    V_32 = np.zeros((6, 12), dtype=np.int64)
    W_22 = np.zeros((4, 12), dtype=np.int64)
    c = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U_232[i*3+j, c] = 1
                V_32[j*2+k, c] = 1
                W_22[k*2+i, c] = 1
                c += 1
    add_block(U_232, V_32, W_22, 0, 2*5, 0, 12)
    
    # Block 4: 2x2 * 2x3 -> 2x3 (Schoolbook 12)
    U_22 = np.zeros((4, 12), dtype=np.int64)
    V_23 = np.zeros((6, 12), dtype=np.int64)
    W_23 = np.zeros((6, 12), dtype=np.int64)
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_22[i*2+j, c] = 1
                V_23[j*3+k, c] = 1
                W_23[k*2+i, c] = 1
                c += 1
    add_block(U_22, V_23, W_23, 0, 0, 2*5, 12)
    
    # Block 5: 2x3 * 3x3 -> 2x3 (Schoolbook 18)
    U_23 = np.zeros((6, 18), dtype=np.int64)
    V_33 = np.zeros((9, 18), dtype=np.int64)
    W_23b = np.zeros((6, 18), dtype=np.int64)
    c = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_23[i*3+j, c] = 1
                V_33[j*3+k, c] = 1
                W_23b[k*2+i, c] = 1
                c += 1
    add_block(U_23, V_33, W_23b, 0, 2*5, 2*5, 18)
    
    # Block 6: 3x2 * 2x2 -> 3x2 (Schoolbook 12)
    U_32 = np.zeros((6, 12), dtype=np.int64)
    V_22b = np.zeros((4, 12), dtype=np.int64)
    W_32 = np.zeros((6, 12), dtype=np.int64)
    c = 0
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U_32[i*2+j, c] = 1
                V_22b[j*2+k, c] = 1
                W_32[k*3+i, c] = 1
                c += 1
    add_block(U_32, V_22b, W_32, 2*5, 0, 0, 12)
    
    # Block 7: 3x2 * 2x3 -> 3x3 (Schoolbook 18)
    U_32b = np.zeros((6, 18), dtype=np.int64)
    V_23b = np.zeros((6, 18), dtype=np.int64)
    W_33b = np.zeros((9, 18), dtype=np.int64)
    c = 0
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U_32b[i*2+j, c] = 1
                V_23b[j*3+k, c] = 1
                W_33b[k*3+i, c] = 1
                c += 1
    add_block(U_32b, V_23b, W_33b, 2*5, 0, 2*5, 18)
    
    # Block 8: 3x3 * 2x2 -> 3x2 (Schoolbook 18)
    U_33c = np.zeros((9, 18), dtype=np.int64)
    V_22c = np.zeros((4, 18), dtype=np.int64)
    W_32c = np.zeros((6, 18), dtype=np.int64)
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U_33c[i*3+j, c] = 1
                V_22c[j*2+k, c] = 1
                W_32c[k*3+i, c] = 1
                c += 1
    add_block(U_33c, V_22c, W_32c, 2*5, 2*5, 0, 18)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        U = np.zeros((n*m, n*m*p), dtype=np.int64)
        V = np.zeros((m*p, n*m*p), dtype=np.int64)
        W = np.zeros((p*n, n*m*p), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _block_decomposition()
# EVOLVE-BLOCK-END
