"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Returns the Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    The 7 rank-1 terms are indexed 0..6.
    
    For C = A * B where A, B are 2x2:
    C[0,0] = M1 - M4 + M5 + M7
    C[0,1] = M3 + M4
    C[1,0] = M1 + M2
    C[1,1] = M1 - M3 + M6 + M7
    
    Where:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    We return U, V, W matrices of shape (4,7), (4,7), (4,7)
    where U[i,j] gives coefficient of A[i,j] in term t,
    V[i,j] gives coefficient of B[i,j] in term t,
    W[i,j] gives coefficient of C[i,j] in term t.
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] += M1, C[1,1] += M1
    
    # Term 1: M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C[1,0] += M2
    
    # Term 2: M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1  # C[0,1] += M3, C[1,1] -= M3
    
    # Term 3: M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1  # C[0,0] -= M4, C[0,1] += M4
    
    # Term 4: M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1  # C[0,0] += M5
    
    # Term 5: M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1] += M6
    
    # Term 6: M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1  # C[0,0] += M7, C[1,1] += M7
    
    return U, V, W

def _strassen_2x2_2x4():
    """
    Decomposition for 2x2 * 2x4 matrix multiplication.
    This is equivalent to two independent 2x2*2x2 multiplications,
    so rank = 14 (two copies of the 2x2 Strassen).
    
    But we can do better by noting the structure.
    Actually, let's just use two independent 2x2 Strassen decompositions
    for the two 2x2 blocks of the 2x4 matrix.
    
    A is 2x2, B is 2x4, C is 2x4.
    C[:, 0:2] = A * B[:, 0:2]  (2x2 * 2x2)
    C[:, 2:4] = A * B[:, 2:4]  (2x2 * 2x2)
    
    So rank = 7 + 7 = 14.
    """
    rank = 14
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    U1, V1, W1 = _strassen_2x2()
    
    # First half: columns 0-1 of B and C
    U[:, 0:7] = U1
    V[0:4, 0:7] = V1
    W[0:4, 0:7] = W1
    
    # Second half: columns 2-3 of B and C
    U[:, 7:14] = U1
    V[4:8, 7:14] = V1
    W[4:8, 7:14] = W1
    
    return U, V, W

def _strassen_4x4_4x2():
    """
    Decomposition for 4x4 * 4x2 matrix multiplication.
    Split into four 2x2 * 2x2 subproblems.
    Rank = 4 * 7 = 28.
    
    A is 4x4, B is 4x2, C is 4x2.
    C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]
    C[0:2, 2:4] = A[0:2, 2:4] * B[0:2, 0:2]
    C[2:4, 0:2] = A[2:4, 0:2] * B[2:4, 0:2]
    C[2:4, 2:4] = A[2:4, 2:2] * B[2:4, 0:2]
    
    Wait, B is 4x2, so it has 2 columns. Let me reconsider.
    
    A is 4x4, B is 4x2.
    C[i,j] = sum_k A[i,k] * B[k,j] for i in 0..3, j in 0..1.
    
    Split A into four 2x2 blocks: A00, A01, A10, A11
    Split B into two 2x2 blocks (vertically): B_top (2x2), B_bottom (2x2)
    C has four 2x1 blocks... no, C is 4x2.
    
    C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    
    Hmm, this is getting complicated. Let me just do it directly.
    
    Actually, the simplest approach: treat 4x4 * 4x2 as a direct tensor.
    Schoolbook rank = 4*4*2 = 32.
    Using Strassen on 2x2 blocks:
    
    Write A as [[A00, A01], [A10, A11]] where each is 2x2.
    Write B as [[B0], [B1]] where B0, B1 are 2x2.
    C = [[A00*B0 + A01*B1], [A10*B0 + A11*B1]]
    
    Each 2x2*2x2 product has rank 7. We have 4 such products.
    Total rank = 4 * 7 = 28.
    """
    rank = 28
    U = np.zeros((16, rank))  # 4x4 = 16 entries
    V = np.zeros((8, rank))   # 4x2 = 8 entries
    W = np.zeros((8, rank))   # 4x2 = 8 entries
    
    U1, V1, W1 = _strassen_2x2()
    
    # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # U indices for A[0:2,0:2]: rows 0-3 in flattened 4x4
    # V indices for B[0:2,0:2]: rows 0-3 in flattened 4x2
    # W indices for C[0:2,0:2]: rows 0-3 in flattened 4x2
    
    U[0:4, 0:7] = U1
    V[0:4, 0:7] = V1
    W[0:4, 0:7] = W1
    
    # Block (0,1): A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    # U indices for A[0:2,2:4]: rows 4-7 in flattened 4x4
    # V indices for B[2:4,0:2]: rows 4-7 in flattened 4x2
    # W indices for C[0:2,0:2]: rows 0-3 in flattened 4x2
    
    U[4:8, 7:14] = U1
    V[4:8, 7:14] = V1
    W[0:4, 7:14] = W1
    
    # Block (1,0): A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
    # U indices for A[2:4,0:2]: rows 8-11 in flattened 4x4
    # V indices for B[0:2,0:2]: rows 0-3 in flattened 4x2
    # W indices for C[2:4,0:2]: rows 4-7 in flattened 4x2
    
    U[8:12, 14:21] = U1
    V[0:4, 14:21] = V1
    W[4:8, 14:21] = W1
    
    # Block (1,1): A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
    # U indices for A[2:4,2:4]: rows 12-15 in flattened 4x4
    # V indices for B[2:4,0:2]: rows 4-7 in flattened 4x2
    # W indices for C[2:4,0:2]: rows 4-7 in flattened 4x2
    
    U[12:16, 21:28] = U1
    V[4:8, 21:28] = V1
    W[4:8, 21:28] = W1
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Split the 4x6 matrix B into three 4x2 matrices B0, B1, B2.
    Compute C0 = A * B0, C1 = A * B1, C2 = A * B2 using the 4x4 * 4x2 decomposition.
    Total rank = 3 * 28 = 84.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Get the 4x4 * 4x2 decomposition
    U_sub, V_sub, W_sub = _strassen_4x4_4x2()
    rank_sub = U_sub.shape[1]  # 28
    
    total_rank = 3 * rank_sub  # 84
    
    U = np.zeros((n * m, total_rank))  # (16, 84)
    V = np.zeros((m * p, total_rank))  # (24, 84)
    W = np.zeros((p * n, total_rank))  # (24, 84)
    
    for block in range(3):
        col_start = block * rank_sub
        col_end = (block + 1) * rank_sub
        
        # U is the same for each block (it's the A matrix part)
        U[:, col_start:col_end] = U_sub
        
        # V: B is 4x6, split into three 4x2 blocks
        # B[0:4, block*2:(block+1)*2] corresponds to rows block*4:(block+1)*4 in flattened 4x6
        V[block * 4:(block + 1) * 4, col_start:col_end] = V_sub
        
        # W: C is 4x6, split into three 4x2 blocks
        # C[0:4, block*2:(block+1)*2] corresponds to rows block*4:(block+1)*4 in flattened 4x6
        # But wait, the trace convention is W[k*n+i, r] where k is the column index of C
        # C is 4x6, so k ranges over 0..5, i ranges over 0..3
        # W[k*4+i, r]
        # For block b, the columns of C are 2*b and 2*b+1
        # So k = 2*b and k = 2*b+1
        # W indices: (2*b)*4 + i and (2*b+1)*4 + i for i in 0..3
        
        # W_sub has shape (8, 28) where row index is k*2+i for the 4x2 output
        # In the 4x2 output, row index r = k_local * 2 + i_local where k_local in 0..1, i_local in 0..3
        # Wait, the 4x2 output has 4 rows and 2 columns.
        # Flattened as row-major: index = row * 2 + col
        # So row 0: indices 0, 1; row 1: indices 2, 3; row 2: indices 4, 5; row 3: indices 6, 7
        
        # In the full 4x6 output, the block b occupies columns 2*b and 2*b+1.
        # The trace convention is W[k*n + i, r] where k is the column index and i is the row index.
        # So for column k and row i, the index is k*4 + i.
        
        # In the subproblem, the 4x2 output has columns 0 and 1 (local).
        # W_sub[k_local * 4 + i, r] where k_local in 0..1, i in 0..3.
        # But wait, W_sub has shape (8, 28), not (8, 28). Let me recheck.
        
        # In _strassen_4x4_4x2, W has shape (8, 28).
        # The 4x2 output C has 4 rows and 2 columns = 8 entries.
        # Flattened row-major: index = row * 2 + col.
        # But the trace convention requires W[k*n + i] where k is column, i is row.
        # So for the subproblem, k_local in 0..1, i in 0..3.
        # Index in W_sub = k_local * 4 + i.
        
        # So W_sub[k_local * 4 + i, r] gives the coefficient for C[i, k_local] in the subproblem.
        
        # In the full problem, the subproblem's column k_local corresponds to global column 2*b + k_local.
        # So the global W index is (2*b + k_local) * 4 + i.
        
        # Let's verify: W_sub has 8 rows. For k_local=0: rows 0,1,2,3 (i=0,1,2,3)
        # For k_local=1: rows 4,5,6,7 (i=0,1,2,3)
        # This matches k_local * 4 + i.
        
        # So we need to map W_sub[k_local * 4 + i] -> W[(2*b + k_local) * 4 + i]
        
        for k_local in range(2):
            for i in range(4):
                src_idx = k_local * 4 + i
                dst_idx = (2 * block + k_local) * 4 + i
                W[dst_idx, col_start:col_end] = W_sub[src_idx, :]
    
    # Convert to lists of lists for the required interface
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    # Verify the decomposition
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Verify correctness
        U_np = np.array(U_list)
        V_np = np.array(V_list)
        W_np = np.array(W_list)
        
        # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i==k and j==j (trivially)
        # Actually, the tensor T[i,j,k] = delta_{i,k} * delta_{j,j} = 1 for all valid i,j,k
        # Wait, no. The matrix multiplication tensor T[i,j,k] = 1 if the (i,k) entry of A*B uses A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        
        # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify a few entries
        n, m, p = 4, 4, 6
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(total_rank):
                        val += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                    expected = 1.0
                    err = abs(val - expected)
                    max_err = max(max_err, err)
        
        if max_err > 1e-10:
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U_fb = [[0] * rank for _ in range(n * m)]
            V_fb = [[0] * rank for _ in range(m * p)]
            W_fb = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
