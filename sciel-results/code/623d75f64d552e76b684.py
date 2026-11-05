import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 8]
DEADLINE = 140.0  # safety margin before 150s hard kill

def _check_time():
    """Return True if we're running out of time."""
    if 'start_time' not in globals():
        return False
    return (time.time() - globals()['start_time']) > DEADLINE

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    Convention: W[k*n+i, r]
    U[i*m+j, r], V[j*p+k, r]
    
    The product tensor T_{i,j,k} should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    For matrix multiplication: T_{i,j,k} = delta_{i,i'} delta_{j,j''} delta_{k',k} ... 
    Actually, the standard decomposition is:
    C[i,k] = sum_j A[i,j] B[j,k]
    Using factors: C = sum_r (U_r^T A) (V_r^T B) W_r  -- no, that's not right.
    
    The correct form: for each r, we have rank-1 terms.
    A is n x m, B is m x p, C is n x p.
    
    The tensor for matrix multiplication has indices (i, j, k) where:
    T[i,j,k] = 1 if this combination appears, 0 otherwise
    Specifically, T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k].
    
    So the decomposition must satisfy:
    For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for all (i,j,k) not corresponding to valid matrix mult terms... 
    Wait, actually the matrix multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    because C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) triple where j is the shared index
    contributes A[i,j]*B[j,k] to C[i,k].
    
    So the constraint is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i in [0,n), j in [0,m), k in [0,p)
    
    And that's it! The tensor is just all ones on the valid domain.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Check all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition with rank = n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    
    Using standard Strassen formulas with 0-based indexing.
    A is 2x2, B is 2x2, C is 2x2.
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M6 + M7
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # For Strassen, we need to be careful about the tensor convention.
    # The rank-1 term r contributes: U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for the appropriate (i,j,k)
    # 
    # Let me think about this differently. In the tensor decomposition:
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # 
    # For 2x2x2, T[i,j,k] = 1 for all i,j,k in {0,1} (it's the all-ones tensor for matmul).
    #
    # Strassen computes 7 products. Each product M_r is a bilinear form in A and B.
    # M_r = sum_{i,j} a_{r,i,j} * A[i,j] = sum_{j,k} b_{r,j,k} * B[j,k]
    # And C[i,k] = sum_r c_{r,i,k} * M_r
    #
    # So U[i*m+j, r] = a_{r,i,j}, V[j*p+k, r] = b_{r,j,k}, W[k*n+i, r] = c_{r,i,k}
    #
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # a1 = [1,0,0,1] (for A00,A01,A10,A11)
    # b1 = [1,1,0,0] (for B00,B01,B10,B11) -- wait, B is 2x2, indices (j,k) with j,k in {0,1}
    # b1[j*p+k]: B00=1, B01=1, B10=0, B11=0? No...
    # M1 = (A00+A11)(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # So a1 = [1, 0, 0, 1], b1 = [1, 0, 0, 1] (since B00 and B11 are involved)
    # W1[i,k] such that C00 += M1, C11 += M1, so W1[0*n+0]=1, W1[1*n+1]=1, i.e., W1 = diag(1,1)
    
    # Let me define all 7:
    # M1 = (A00+A11)(B00+B11)
    a1 = [1,0,0,1]; b1 = [1,0,0,1]; c1 = [1,0,0,1]  # c1[k*n+i]: c1[0]=C00, c1[3]=C11
    # M2 = (A10+A11)B00
    a2 = [0,0,1,1]; b2 = [1,0,0,0]; c2 = [0,0,1,0]  # C10 += M2
    # M3 = A00(B10-B11)
    a3 = [1,0,0,0]; b3 = [0,0,1,-1]; c3 = [0,1,0,0]  # C01 += M3
    # M4 = A11(B01-B00)
    a4 = [0,0,0,1]; b4 = [-1,1,0,0]; c4 = [1,0,1,0]  # C00 += M4, C10 += M4
    # M5 = (A00+A01)B11
    a5 = [1,1,0,0]; b5 = [0,0,0,1]; c5 = [0,1,0,-1]  # C01 += M5, C00 -= M5
    # M6 = (A10-A00)(B00+B01)
    a6 = [-1,0,1,0]; b6 = [1,1,0,0]; c6 = [0,0,0,1]  # C11 += M6
    # M7 = (A01-A11)(B10+B11)
    a7 = [0,1,0,-1]; b7 = [0,0,1,1]; c7 = [1,0,0,1]  # C00 += M7, C11 += M7
    
    U[0,0] = a1[0]; U[1,0] = a1[1]; U[2,0] = a1[2]; U[3,0] = a1[3]
    V[0,0] = b1[0]; V[1,0] = b1[1]; V[2,0] = b1[2]; V[3,0] = b1[3]
    W[0,0] = c1[0]; W[1,0] = c1[1]; W[2,0] = c1[2]; W[3,0] = c1[3]
    
    U[0,1] = a2[0]; U[1,1] = a2[1]; U[2,1] = a2[2]; U[3,1] = a2[3]
    V[0,1] = b2[0]; V[1,1] = b2[1]; V[2,1] = b2[2]; V[3,1] = b2[3]
    W[0,1] = c2[0]; W[1,1] = c2[1]; W[2,1] = c2[2]; W[3,1] = c2[3]
    
    U[0,2] = a3[0]; U[1,2] = a3[1]; U[2,2] = a3[2]; U[3,2] = a3[3]
    V[0,2] = b3[0]; V[1,2] = b3[1]; V[2,2] = b3[2]; V[3,2] = b3[3]
    W[0,2] = c3[0]; W[1,2] = c3[1]; W[2,2] = c3[2]; W[3,2] = c3[3]
    
    U[0,3] = a4[0]; U[1,3] = a4[1]; U[2,3] = a4[2]; U[3,3] = a4[3]
    V[0,3] = b4[0]; V[1,3] = b4[1]; V[2,3] = b4[2]; V[3,3] = b4[3]
    W[0,3] = c4[0]; W[1,3] = c4[1]; W[2,3] = c4[2]; W[3,3] = c4[3]
    
    U[0,4] = a5[0]; U[1,4] = a5[1]; U[2,4] = a5[2]; U[3,4] = a5[3]
    V[0,4] = b5[0]; V[1,4] = b5[1]; V[2,4] = b5[2]; V[3,4] = b5[3]
    W[0,4] = c5[0]; W[1,4] = c5[1]; W[2,4] = c5[2]; W[3,4] = c5[3]
    
    U[0,5] = a6[0]; U[1,5] = a6[1]; U[2,5] = a6[2]; U[3,5] = a6[3]
    V[0,5] = b6[0]; V[1,5] = b6[1]; V[2,5] = b6[2]; V[3,5] = b6[3]
    W[0,5] = c6[0]; W[1,5] = c6[1]; W[2,5] = c6[2]; W[3,5] = c6[3]
    
    U[0,6] = a7[0]; U[1,6] = a7[1]; U[2,6] = a7[2]; U[3,6] = a7[3]
    V[0,6] = b7[0]; V[1,6] = b7[1]; V[2,6] = b7[2]; V[3,6] = b7[3]
    W[0,6] = c7[0]; W[1,6] = c7[1]; W[2,6] = c7[2]; W[3,6] = c7[3]
    
    return U, V, W

def _verify_strassen():
    """Verify Strassen 2x2x2 decomposition."""
    n, m, p = 2, 2, 2
    U, V, W = _strassen_2x2x2()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    print(f"Strassen verification failed at ({i},{j},{k}): got {val}", file=sys.stderr)
                    return False
    return True

def _embed_2x2x2_in_3x4x8(row_offset, col_offset, U, V, W, offset, rank):
    """Embed a 2x2x2 Strassen decomposition into a 3x4x8 decomposition.
    
    row_offset: which rows of the 3x4 matrix A are used (0-based, 2 rows)
    col_offset: which cols of the 4x8 matrix B are used (0-based, 2 cols)
    The 2x2x2 block multiplies A[row_offset:row_offset+2, col_offset:col_offset+2] 
    (wait, A is 3x4, B is 4x8, so A has 4 cols and B has 4 rows, they share the 4-dim)
    
    Actually, A is n x m = 3 x 4, B is m x p = 4 x 8.
    The shared dimension is m=4.
    
    A 2x2x2 block corresponds to:
    - 2 rows of A (i indices)
    - 2 columns of A = 2 rows of B (j indices)  
    - 2 columns of B (k indices)
    
    So row_offset selects 2 of the 3 row indices of A
    col_offset selects 2 of the 4 column/row indices (shared)
    And we need 2 column indices of B (out of 8)
    """
    pass  # This function is getting complicated; let me use a different approach

def _construct_block_decomposition():
    """Construct a block-based decomposition.
    
    Strategy: Partition the 3x4x8 problem into 2x2x2 blocks where possible.
    A is 3x4, B is 4x8.
    
    We can partition:
    - A into row blocks: [0,1] and [2] (sizes 2 and 1)
    - A columns / B rows: [0,1], [2,3] (sizes 2, 2)
    - B columns: [0,1], [2,3], [4,5], [6,7] (sizes 2,2,2,2)
    
    For each combination of (A row block, shared dim block, B col block):
    - If A row block has size 2 and B col block has size 2: use Strassen (rank 7)
    - Otherwise: use schoolbook (rank = 2*2*2=8 or smaller)
    
    A row blocks: [0,1], [2] -> sizes 2, 1
    Shared blocks: [0,1], [2,3] -> sizes 2, 2
    B col blocks: [0,1], [2,3], [4,5], [6,7] -> sizes 2, 2, 2, 2
    
    Total rank:
    - A rows [0,1], shared [0,1], B cols [0,1]: 2x2x2 -> Strassen rank 7
    - A rows [0,1], shared [0,1], B cols [2,3]: 2x2x2 -> Strassen rank 7
    - A rows [0,1], shared [0,1], B cols [4,5]: 2x2x2 -> Strassen rank 7
    - A rows [0,1], shared [0,1], B cols [6,7]: 2x2x2 -> Strassen rank 7
    - A rows [0,1], shared [2,3], B cols [0,1]: 2x2x2 -> Strassen rank 7
    - A rows [0,1], shared [2,3], B cols [2,3]: 2x2x2 -> Strassen rank 7
    - A rows [0,1], shared [2,3], B cols [4,5]: 2x2x2 -> Strassen rank 7
    - A rows [0,1], shared [2,3], B cols [6,7]: 2x2x2 -> Strassen rank 7
    - A rows [2], shared [0,1], B cols [0,1]: 1x2x2 -> schoolbook rank 4
    - A rows [2], shared [0,1], B cols [2,3]: 1x2x2 -> schoolbook rank 4
    - A rows [2], shared [0,1], B cols [4,5]: 1x2x2 -> schoolbook rank 4
    - A rows [2], shared [0,1], B cols [6,7]: 1x2x2 -> schoolbook rank 4
    - A rows [2], shared [2,3], B cols [0,1]: 1x2x2 -> schoolbook rank 4
    - A rows [2], shared [2,3], B cols [2,3]: 1x2x2 -> schoolbook rank 4
    - A rows [2], shared [2,3], B cols [4,5]: 1x2x2 -> schoolbook rank 4
    - A rows [2], shared [2,3], B cols [6,7]: 1x2x2 -> schoolbook rank 4
    
    Total: 8 * 7 + 8 * 4 = 56 + 32 = 88
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Define blocks
    a_row_blocks = [(0, 1), (2,)]  # row index ranges for A (and C)
    shared_blocks = [(0, 1), (2, 3)]  # column indices of A = row indices of B
    b_col_blocks = [(0, 1), (2, 3), (4, 5), (6, 7)]  # column indices of B (and C)
    
    total_rank = 0
    for a_rows in a_row_blocks:
        for s_cols in shared_blocks:
            for b_cols in b_col_blocks:
                if len(a_rows) == 2 and len(b_cols) == 2:
                    total_rank += 7  # Strassen
                elif len(a_rows) == 1 and len(s_cols) == 2 and len(b_cols) == 2:
                    total_rank += 4  # schoolbook 1x2x2
                elif len(a_rows) == 2 and len(s_cols) == 2 and len(b_cols) == 2:
                    total_rank += 7
                else:
                    total_rank += len(a_rows) * len(s_cols) * len(b_cols)
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for a_rows in a_row_blocks:
        for s_cols in shared_blocks:
            for b_cols in b_col_blocks:
                na = len(a_rows)
                ns = len(s_cols)
                nb = len(b_cols)
                
                if na == 2 and ns == 2 and nb == 2:
                    # Use Strassen
                    U_s, V_s, W_s = _strassen_2x2x2()
                    for local_i, global_i in enumerate(a_rows):
                        for local_j, global_j in enumerate(s_cols):
                            U[global_i*m + global_j, col:col+7] = U_s[local_i*2 + local_j, :]
                    for local_j, global_j in enumerate(s_cols):
                        for local_k, global_k in enumerate(b_cols):
                            V[global_j*p + global_k, col:col+7] = V_s[local_j*2 + local_k, :]
                    for local_k, global_k in enumerate(b_cols):
                        for local_i, global_i in enumerate(a_rows):
                            W[global_k*n + global_i, col:col+7] = W_s[local_k*2 + local_i, :]
                    col += 7
                elif na == 1 and ns == 2 and nb == 2:
                    # Schoolbook 1x2x2: rank 4
                    for local_j in range(ns):
                        for local_k in range(nb):
                            global_i = a_rows[0]
                            global_j = s_cols[local_j]
                            global_k = b_cols[local_k]
                            U[global_i*m + global_j, col] = 1
                            V[global_j*p + global_k, col] = 1
                            W[global_k*n + global_i, col] = 1
                            col += 1
                else:
                    # General schoolbook
                    for local_i in range(na):
                        for local_j in range(ns):
                            for local_k in range(nb):
                                global_i = a_rows[local_i]
                                global_j = s_cols[local_j]
                                global_k = b_cols[local_k]
                                U[global_i*m + global_j, col] = 1
                                V[global_j*p + global_k, col] = 1
                                W[global_k*n + global_i, col] = 1
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify the full decomposition."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j, :], V[j*p+k, :]), W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    global start_time
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    best_rank = n * m * p  # 96
    best_U, best_V, best_W = _schoolbook()
    
    # First, verify Strassen works
    if not _verify_strassen():
        print("Strassen verification failed, using schoolbook only", file=sys.stderr)
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try block decomposition
    try:
        U_block, V_block, W_block = _construct_block_decomposition()
        if _verify_decomposition(U_block, V_block, W_block):
            best_U, best_V, best_W = U_block, V_block, W_block
            best_rank = U_block.shape[1]
            print(f"Block decomposition verified with rank {best_rank}", file=sys.stderr)
        else:
            print("Block decomposition failed verification", file=sys.stderr)
    except Exception as e:
        print(f"Block decomposition error: {e}", file=sys.stderr)
    
    # Try to improve further with greedy rank reduction
    # For now, return the best we have
    if (time.time() - start_time) > DEADLINE - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
