"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy:
    1. Start with schoolbook rank-96 decomposition.
    2. Use the Kronecker product structure: 4x4x6 = (2x2x3) ⊗ (2x2x2).
       - 2x2x2 has rank 7 (Strassen)
       - 2x2x3 has rank 12 (schoolbook for 2x2*2x3)
       - Kronecker gives rank 7*12 = 84
    3. Try to reduce rank further via local optimization / rank reduction.
    
    The tensor T[i,j,k] for multiplying A (4x4) by B (4x6) to get C (4x6):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In the trace convention W[k*n+i, r], where n=4, m=4, p=6:
    U has shape (n*m, rank) = (16, rank)
    V has shape (m*p, rank) = (24, rank)  
    W has shape (p*n, rank) = (24, rank)
    
    The decomposition satisfies: for all i,j,k:
    sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]  ... wait, that's not right.
    
    Actually the tensor decomposition means:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    where T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C.
    
    For matrix multiplication, T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}... 
    Actually T[i,j,k] = 1 for all valid i,j,k (it's the multiplication tensor).
    
    The condition is: for each triple (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for all other combinations (which don't exist in the tensor), it should be 0.
    
    But since i ranges over n=4, j over m=4, k over p=6, all combinations (i,j,k) with i in [0,4), j in [0,4), k in [0,6) are part of the tensor.
    
    So we need: for all i in [0,4), j in [0,4), k in [0,6):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Strassen decomposition for 2x2x2
    # U indices: (i,j) -> i*2+j, V indices: (j,k) -> j*2+k, W indices: (k,i) -> k*2+i
    # 7 terms for 2x2 matrix multiplication
    strassen_U = [
        [1, 1, 0, 0, 0, 0, 0],  # (0,0)
        [0, 0, 1, 0, 0, 0, 0],  # (0,1)
        [0, 0, 0, 1, 0, 0, 0],  # (1,0)
        [1, 1, 0, 0, 1, 1, 0],  # (1,1)
    ]
    strassen_V = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 1, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 1, 0],  # (1,1)
    ]
    strassen_W = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 0],  # (1,0)
        [0, 0, 0, 1, 1, 1, 0],  # (1,1)
    ]
    
    # Verify Strassen: for 2x2, C[i,k] = sum_j A[i,j]*B[j,k]
    # Term 0: A[0,0]*B[0,0] -> contributes to C[0,0]
    # Let me use the standard Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M6 - M7
    
    # U[i,j,r]: coefficient of A[i,j] in term r
    # V[j,k,r]: coefficient of B[j,k] in term r
    # W[k,i,r]: coefficient of term r in C[i,k]  (note: W index is k*n+i for n=2, so k*2+i)
    
    strassen_U = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # A00 in M1, M3, M5
        [0, 0, 0, 0, 1, 0, 0],  # A01 in M5
        [0, 1, 0, 0, 0, 1, 0],  # A10 in M2, M6
        [1, 1, 0, 1, 0, 0, 0],  # A11 in M1, M2, M4
    ])
    strassen_V = np.array([
        [1, 1, 0, 0, 0, 1, 0],  # B00 in M1, M2, M6
        [0, 0, 0, -1, 0, 1, 0], # B01 in M4, M6
        [0, 0, 1, 0, 0, 0, 1],  # B10 in M3, M7
        [1, 0, -1, 0, 1, 0, 1], # B11 in M1, M3, M5, M7
    ])
    strassen_W = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # C00: M1
        [0, 1, 0, 0, 0, 0, 0],  # C01: M4  wait, C01 = M3+M4
        [0, 0, 1, 0, 0, 0, 0],  # C10: M2
        [0, 0, 0, 1, 0, 0, 0],  # C11: M1
    ])
    # Let me redo this carefully.
    # C[i,k] uses W[k*2+i, r]
    # C[0,0] = M1 - M4 + M5 - M7 => W[0*2+0, r] = W[0, r]
    # C[0,1] = M3 + M4 => W[1*2+0, r] = W[2, r]
    # C[1,0] = M2 + M5 => W[0*2+1, r] = W[1, r]
    # C[1,1] = M1 - M3 + M6 - M7 => W[1*2+1, r] = W[3, r]
    
    strassen_W = np.array([
        [1, 0, 0, -1, 1, 0, -1],  # W[0, r] for C[0,0]
        [0, 1, 0, 0, 1, 0, 0],    # W[1, r] for C[1,0]
        [0, 0, 1, 1, 0, 0, 0],    # W[2, r] for C[0,1]
        [1, 0, -1, 0, 0, 1, -1],  # W[3, r] for C[1,1]
    ])
    
    # Verify Strassen decomposition
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = sum(strassen_U[i*2+j, r] * strassen_V[j*2+k, r] * strassen_W[k*2+i, r] for r in range(7))
                    if val != 1:
                        return False
        return True
    
    if not verify_strassen():
        # Fallback to schoolbook
        return best_result
    
    # Schoolbook for 2x2x3: rank 12
    # A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 2*2*3 = 12 terms
    sb223_U = np.zeros((4, 12))
    sb223_V = np.zeros((6, 12))
    sb223_W = np.zeros((6, 12))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                sb223_U[i*2+j, col] = 1
                sb223_V[j*3+k, col] = 1
                sb223_W[k*2+i, col] = 1
                col += 1
    
    # Kronecker product: 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # Index mapping:
    # A is 4x4: A[i,j] where i = 2*i1 + i2, j = 2*j1 + j2
    # B is 4x6: B[j,k] where j = 2*j1 + j2, k = 3*k1 + k2
    # C is 4x6: C[i,k] where i = 2*i1 + i2, k = 3*k1 + k2
    # 
    # Term r = (r1, r2) where r1 in [0,7), r2 in [0,12)
    # U[i*m+j, r] = strassen_U[i2*2+j2, r1] * sb223_U[i2*2+j2, r2]... 
    # Wait, the Kronecker structure:
    # A[i,j] = A[i1,i2; j1,j2] where i=2*i1+i2, j=2*j1+j2
    # B[j,k] = B[j1,j2; k1,k2] where j=2*j1+j2, k=3*k1+k2
    # C[i,k] = C[i1,i2; k1,k2]
    #
    # C[i1,i2; k1,k2] = sum_{j1,j2} A[i1,i2; j1,j2] * B[j1,j2; k1,k2]
    # = sum_{j1,j2} (sum_{r1} U1[i2,j2,r1] * ... ) 
    #
    # Actually the Kronecker product of tensors:
    # T[(i1,i2),(j1,j2),(k1,k2)] = sum_{r1,r2} U1[i2,j2,r1] * V1[j2,k2,r1] * W1[k2,i2,r1] * U2[i1,j1,r2] * V2[j1,k1,r2] * W2[k1,i1,r2]
    #
    # But this isn't quite right either. Let me think differently.
    #
    # The 4x4x6 multiplication tensor can be written as:
    # T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,6)
    #
    # Using block structure: i = 2*i1 + i2, j = 2*j1 + j2, k = 3*k1 + k2
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_{j1,j2} A[2*i1+i2, 2*j1+j2] * B[2*j1+j2, 3*k1+k2]
    #
    # This is a sum over j1 in {0,1} and j2 in {0,1} of 4 terms.
    # For each (j1,j2), we have a 2x2x3 sub-problem... no.
    #
    # Actually, let's use the fact that 4 = 2*2 and 6 = 2*3.
    # The multiplication (4x4)*(4x6) can be seen as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Write i = 2*i1 + i2, j = 2*j1 + j2, k = 3*k1 + k2
    # Then:
    # C[2*i1+i2, 3*k1+k2] = sum_{j1=0}^1 sum_{j2=0}^1 A[2*i1+i2, 2*j1+j2] * B[2*j1+j2, 3*k1+k2]
    #
    # This is a 2x2 sum over (j1,j2) of products.
    # Each term A[2*i1+i2, 2*j1+j2] * B[2*j1+j2, 3*k1+k2] is a scalar product.
    #
    # The Kronecker approach: if we had 2x2x2 and 2x2x3, the Kronecker would give
    # (2*2)x(2*2)x(2*3) = 4x4x6. 
    #
    # The tensor for 4x4x6 is the Kronecker product of the 2x2x2 tensor and 2x2x3 tensor?
    # Not exactly, because the middle dimension is shared.
    #
    # Let me try a different approach: direct rank reduction from schoolbook.
    
    # Alternative: use the fact that for matrix multiplication, we can sometimes
    # reduce rank by exploiting symmetries.
    
    # For now, let's try the Kronecker construction properly.
    # 4x4x6: n=4, m=4, p=6
    # Factor as n=2*2, m=2*2, p=2*3
    # The Kronecker product of (2x2x2) and (2x2x3) tensors:
    # T[(i1,i2),(j1,j2),(k1,k2)] = T1[i1,j1,k1] * T2[i2,j2,k2]
    # But T1[i1,j1,k1] = 1 and T2[i2,j2,k2] = 1, so T = 1.
    # The decomposition is:
    # sum_{r1,r2} U1[i1,j1,r1] * U2[i2,j2,r2] * V1[j1,k1,r1] * V2[j2,k2,r2] * W1[k1,i1,r1] * W2[k2,i2,r2]
    # = (sum_{r1} U1[i1,j1,r1] * V1[j1,k1,r1] * W1[k1,i1,r1]) * (sum_{r2} U2[i2,j2,r2] * V2[j2,k2,r2] * W2[k2,i2,r2])
    # = 1 * 1 = 1 (if both decompositions are valid)
    #
    # So the Kronecker decomposition has rank = rank1 * rank2 = 7 * 12 = 84.
    #
    # But we need to map indices correctly.
    # U[i*m+j, r] where i = 2*i1+i2, j = 2*j1+j2, so i*m+j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    # r = 7*r2 + r1 (or some ordering)
    
    rank_kron = 7 * 12
    U_kron = [[0] * rank_kron for _ in range(16)]
    V_kron = [[0] * rank_kron for _ in range(24)]
    W_kron = [[0] * rank_kron for _ in range(24)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    u_idx = i * 4 + j
                    for r1 in range(7):
                        for r2 in range(12):
                            r = r1 * 12 + r2
                            U_kron[u_idx][r] = strassen_U[i2*2+j2, r1] * sb223_U[i2*2+j2, r2]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(3):
                    j = 2*j1 + j2
                    k = 3*k1 + k2
                    v_idx = j * 6 + k
                    for r1 in range(7):
                        for r2 in range(12):
                            r = r1 * 12 + r2
                            V_kron[v_idx][r] = strassen_V[j2*2+k2, r1] * sb223_V[j2*3+k2, r2]
    
    for k1 in range(2):
        for k2 in range(3):
            for i1 in range(2):
                for i2 in range(2):
                    k = 3*k1 + k2
                    i = 2*i1 + i2
                    w_idx = k * 4 + i
                    for r1 in range(7):
                        for r2 in range(12):
                            r = r1 * 12 + r2
                            W_kron[w_idx][r] = strassen_W[k2*2+i2, r1] * sb223_W[k2*2+i2, r2]
    
    # Verify Kronecker decomposition
    def verify_decomp(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(len(U[0])))
                    if val != 1:
                        return False
        return True
    
    if verify_decomp(U_kron, V_kron, W_kron, n, m, p):
        if rank_kron < best_rank:
            best_rank = rank_kron
            best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
    
    # Try to reduce rank by zeroing out columns
    # Check if any column can be removed (if it's linearly dependent in a certain sense)
    # Simple approach: try removing each column and check if the decomposition still holds
    
    def try_reduce_rank(U, V, W, n, m, p):
        rank = len(U[0])
        # Try removing each column
        for col_to_remove in range(rank):
            if time.time() > deadline:
                break
            # Create new matrices without this column
            U_new = [[U[i][r] for r in range(rank) if r != col_to_remove] for i in range(len(U))]
            V_new = [[V[i][r] for r in range(rank) if r != col_to_remove] for i in range(len(V))]
            W_new = [[W[i][r] for r in range(rank) if r != col_to_remove] for i in range(len(W))]
            if verify_decomp(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, True
        return U, V, W, False
    
    U_cur, V_cur, W_cur = best_result["U"], best_result["V"], best_result["W"]
    rank_cur = best_rank
    
    # Iteratively try to reduce rank
    max_iterations = 50
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        U_new, V_new, W_new, success = try_reduce_rank(U_cur, V_cur, W_cur, n, m, p)
        if success:
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            rank_cur -= 1
            if rank_cur < best_rank:
                best_rank = rank_cur
                best_result = {"U": U_cur, "V": V_cur, "W": W_cur}
        else:
            break
    
    return best_result
# EVOLVE-BLOCK-END
