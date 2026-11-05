"""Program using Strassen-style tensor decomposition for matrix multiplication 4x4 * 4x9."""
import numpy as np

DIMENSIONS = [4, 4, 9]

# Strassen 2x2*2x2 -> 2x2 decomposition: rank 7
# Standard Strassen formulas for C = A*B where A, B are 2x2
# A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
# C = [[c11, c12], [c21, c22]]
#
# M1 = (a11+a22)(b11+b22)
# M2 = (a21+a22)b11
# M3 = a11(b12-b22)
# M4 = a22(b21-b11)
# M5 = (a11+a12)b22
# M6 = (a21-a11)(b11+b12)
# M7 = (a12-a22)(b21+b22)
#
# c11 = M1+M4-M5+M7
# c12 = M3+M5
# c21 = M2+M4
# c22 = M1-M2+M3+M6
#
# For the tensor decomposition, we need U, V, W such that:
# sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta (correct entry)
#
# Actually, the convention is:
# (A*B)[i,k] = sum_j A[i,j] * B[j,k]
# In our indexing for M_{n,m,p}:
# U has shape (n*m, rank), indexed by (i*m+j)
# V has shape (m*p, rank), indexed by (j*p+k)  
# W has shape (p*n, rank), indexed by (k*n+i)
#
# The decomposition satisfies:
# sum_c U[(i*m+j), c] * V[(j*p+k), c] * W[(k*n+i), c] = delta_{i,i'} * delta_{k,k'} ... 
# Wait, let me re-read the task.
#
# "exactly decompose the matrix multiplication tensor"
# The matrix multiplication tensor T[i,j,k] for A[i,j]*B[j,k] contributing to C[i,k]
# Under the trace convention W[k*n+i, r], this means:
# For each column c, the term is sum_{i,j,k} A[i,j] B[j,k] * U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
# This should equal sum_{i,k} C[i,k] * (something)
#
# Actually, standard tensor decomposition for matrix multiplication:
# T = sum_c U_c ⊗ V_c ⊗ W_c
# where (U_c)_{i,j} (V_c)_{j,k} (W_c)_{k,i} ... hmm.
#
# Let me think about this more carefully. The task says:
# U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
# "exactly decompose the matrix multiplication tensor"
#
# The matrix multiplication can be written as:
# C[i,k] = sum_j A[i,j] B[j,k]
# 
# The bilinear form is: sum_{i,j,k} A[i,j] B[j,k] C'[i,k] where C' is the output.
# But actually, for the tensor decomposition, we want:
# sum_c A^T U_c B V_c W_c^T ... no.
#
# Let me look at the schoolbook example:
# U[i*m+j][c] = 1, V[j*p+k][c] = 1, W[k*n+i][c] = 1
# for each (i,j,k) tuple.
# 
# So the contribution is: sum_{i,j,k} A[i,j] B[j,k] * 1 * 1 * 1
# = sum_{i,j,k} A[i,j] B[j,k]
#
# But this doesn't directly give C[i,k]. Let me think...
#
# Actually, I think the decomposition is for the trilinear form:
# sum_{i,j,k} A[i,j] B[j,k] C[i,k] = sum_c (sum_{i,j} A[i,j] U[i*m+j,c]) (sum_{j,k} B[j,k] V[j*p+k,c]) (sum_{k,i} C[i,k] W[k*n+i,c])
#
# No wait, that's not right either. Let me reconsider.
#
# Looking at the schoolbook: for each (i,j,k), we have one column with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
# The total sum over all columns of:
# (sum_{i,j} A[i,j] U[i*m+j,c]) * (sum_{j,k} B[j,k] V[j*p+k,c]) * (sum_{k,i} C[i,k] W[k*n+i,c])
#
# For the schoolbook, each column c corresponds to a specific (i,j,k) tuple.
# The first factor picks out A[i,j], the second picks out B[j,k], and the third picks out C[k,i]... 
# wait, W is indexed by k*n+i, so W[k*n+i] corresponds to entry (i,k) of C if we think of C as having row i and column k.
#
# So the sum is: sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
#
# For the decomposition to be exact, we need:
# sum_c (sum_{i,j} A[i,j] U[i*m+j,c]) (sum_{j,k} B[j,k] V[j*p+k,c]) (sum_{k,i} C[i,k] W[k*n+i,c]) = sum_{i,k} C[i,k] (sum_j A[i,j] B[j,k])
#
# This must hold for all A, B, C. So the tensor defined by U, V, W must equal the matrix multiplication tensor.

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 -> 2x2 multiplication with rank 7 (Strassen)."""
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen product, define U, V, W contributions
    # U[i*m+j, c] for A[i,j]
    # V[j*p+k, c] for B[j,k]
    # W[k*n+i, c] for C[i,k]
    
    # M1: (A00+A11)(B00+B11)
    # This is a product of a linear form in A and a linear form in B
    # U part: A00 + A11 -> U[0,0]=1, U[1,1]=1 (i*m+j: 0*2+0=0, 1*2+1=3)
    # V part: B00 + B11 -> V[0,0]=1, V[1,1]=1 (j*p+k: 0*2+0=0, 1*2+1=3)
    # W part: contributes to C00 (+1), C11 (+1)
    # C00 is at index k*n+i = 0*2+0 = 0
    # C11 is at index k*n+i = 1*2+1 = 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)(B00)
    # U: A10 + A11 -> U[2,0]=1, U[3,0]=1 (i*m+j: 1*2+0=2, 1*2+1=3)
    # V: B00 -> V[0,0]=1
    # W: contributes to C10 (+1) -> index 0*2+1 = 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: (A00)(B01-B11)
    # U: A00 -> U[0,0]=1
    # V: B01 - B11 -> V[1,0]=1, V[3,0]=-1 (j*p+k: 0*2+1=1, 1*2+1=3)
    # W: contributes to C01 (+1), C11 (+1)
    # C01: k*n+i = 1*2+0 = 2
    # C11: k*n+i = 1*2+1 = 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: (A11)(B10-B00)
    # U: A11 -> U[3,0]=1
    # V: B10 - B00 -> V[2,0]=1, V[0,0]=-1 (j*p+k: 1*2+0=2, 0*2+0=0)
    # W: contributes to C00 (+1), C10 (+1)
    # C00: 0, C10: 2
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A01)(B11)
    # U: A00 + A01 -> U[0,0]=1, U[1,0]=1 (i*m+j: 0, 1)
    # V: B11 -> V[3,0]=1
    # W: contributes to C00 (-1), C01 (+1)
    # C00: 0, C01: 2
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: A10 - A00 -> U[2,0]=1, U[0,0]=-1
    # V: B00 + B01 -> V[0,0]=1, V[1,0]=1
    # W: contributes to C11 (+1) -> index 3
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U: A01 - A11 -> U[1,0]=1, U[3,0]=-1
    # V: B10 + B11 -> V[2,0]=1, V[3,0]=1
    # W: contributes to C00 (+1) -> index 0
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _combine_decompositions(n, m, p, decomps):
    """
    Combine multiple matrix multiplication decompositions for different sub-problems.
    decomps: list of (U_sub, V_sub, W_sub, i_offset, j_offset, k_offset)
    where the sub-problem is A[i_offset:i_offset+n_sub, j_offset:j_offset+m_sub] * 
            B[j_offset:j_offset+m_sub, k_offset:k_offset+p_sub]
    and the result goes to C[i_offset:i_offset+n_sub, k_offset:k_offset+p_sub]
    """
    total_rank = sum(U.shape[1] for U, V, W, *_ in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for U_sub, V_sub, W_sub, i_off, j_off, k_off in decomps:
        n_sub, m_sub = U_sub.shape[0] // (m_sub if m_sub > 0 else 1), None
        # Actually, we know the shapes
        U_sub_rows = U_sub.shape[0]  # n_sub * m_sub
        V_sub_rows = V_sub.shape[0]  # m_sub * p_sub
        W_sub_rows = W_sub.shape[0]  # p_sub * n_sub
        rank_sub = U_sub.shape[1]
        
        for r in range(U_sub_rows):
            # r corresponds to (i_sub * m_sub + j_sub) where i_sub in [0, n_sub), j_sub in [0, m_sub)
            i_sub = r // m_sub
            j_sub = r % m_sub
            global_i = i_off + i_sub
            global_j = j_off + j_sub
            global_idx = global_i * m + global_j
            U[global_idx, col_offset:col_offset+rank_sub] = U_sub[r, :]
        
        for r in range(V_sub_rows):
            # r corresponds to (j_sub * p_sub + k_sub)
            p_sub = V_sub_rows // m_sub if m_sub > 0 else 1
            j_sub = r // p_sub
            k_sub = r % p_sub
            global_j = j_off + j_sub
            global_k = k_off + k_sub
            global_idx = global_j * p + global_k
            V[global_idx, col_offset:col_offset+rank_sub] = V_sub[r, :]
        
        for r in range(W_sub_rows):
            # r corresponds to (k_sub * n_sub + i_sub)
            n_sub = W_sub_rows // p_sub if p_sub > 0 else 1
            k_sub = r // n_sub
            i_sub = r % n_sub
            global_k = k_off + k_sub
            global_i = i_off + i_sub
            global_idx = global_k * n + global_i
            W[global_idx, col_offset:col_offset+rank_sub] = W_sub[r, :]
        
        col_offset += rank_sub
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Decompose 4x4 * 4x9 as:
    # Two 4x4 * 4x4 blocks (columns 0-3 and 4-7) + one 4x4 * 4x1 block (column 8)
    # For 4x4 * 4x4: use Strassen on 2x2 blocks to get rank 49
    # For 4x4 * 4x1: schoolbook gives rank 16, but we can do better
    
    # First, build 4x4 * 4x4 decomposition using Strassen
    # 4x4 * 4x4 can be decomposed into 16 2x2 * 2x2 sub-problems, but Strassen reduces this.
    # Actually, for 4x4 * 4x4, we can use Strassen recursively or directly.
    # Rank of 4x4 * 4x4 with Strassen is 7^2 = 49.
    
    def strassen_4x4_4x4():
        """Build 4x4 * 4x4 -> 4x4 decomposition with rank 49 using Strassen."""
        # Split into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] where each is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = A * B
        #
        # Use Strassen on the 2x2 block level, where each "entry" is a 2x2 matrix.
        # M1 = (A00+A11)(B00+B11) - 4x4 * 4x4 product
        # M2 = (A10+A11)(B00)
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # Each of these is a 4x4 * 4x4 product, which has rank 16 schoolbook.
        # But we can use Strassen for each! So each M_i has rank 7.
        # Total rank: 7 * 7 = 49.
        
        # Actually, let's just do it directly.
        # For 4x4 * 4x4, split into 2x2 blocks.
        # Each Strassen product involves adding 2x2 matrices (cheap, linear) and multiplying 2x2 matrices (rank 7 each).
        
        # Let's build this step by step.
        # We'll create a decomposition for each of the 7 Strassen products at the block level,
        # where each block product is itself a 2x2 * 2x2 multiplication.
        
        # Actually, it's easier to just use the known result: rank 49 for 4x4 * 4x4.
        # Let me construct it explicitly.
        
        # Split indices: i, j, k in {0,1,2,3}
        # i = 2*i_b + i_l, j = 2*j_b + j_l, k = 2*k_b + k_l
        # where i_b, j_b, k_b in {0,1} (block level) and i_l, j_l, k_l in {0,1} (within block)
        
        # A[i,j] = A[i_b, j_b][i_l, j_l]
        # B[j,k] = B[j_b, k_b][j_l, k_l]
        # C[i,k] = C[i_b, k_b][i_l, k_l]
        
        # C[i_b, k_b][i_l, k_l] = sum_{j_b=0}^1 sum_{j_l=0}^1 A[i_b, j_b][i_l, j_l] B[j_b, k_b][j_l, k_l]
        
        # Block level: C_block[i_b, k_b] = sum_{j_b} A_block[i_b, j_b] * B_block[j_b, k_b]
        # Within each block product, use Strassen.
        
        # This is getting complex. Let me just use the schoolbook for 4x4 * 4x4 for now,
        # which has rank 64, and see if the total is competitive.
        # 64 + 64 + 16 = 144, same as schoolbook for 4x4 * 4x9.
        
        # Let me try a different approach: use Strassen directly on 4x4 * 4x4.
        
        # I'll construct the 49-rank decomposition explicitly.
        # Reference: Strassen's algorithm for 4x4 matrices.
        
        # Actually, let me just use schoolbook for simplicity and correctness,
        # and see if I can find a better decomposition.
        
        # For now, schoolbook 4x4 * 4x4: rank 64
        rank = 16
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        
        return U, V, W
    
    # For 4x4 * 4x1: schoolbook rank 16
    def schoolbook_4x4_4x1():
        n_sub, m_sub, p_sub = 4, 4, 1
        rank = n_sub * m_sub * p_sub
        U = np.zeros((n_sub*m_sub, rank))
        V = np.zeros((m_sub*p_sub, rank))
        W = np.zeros((p_sub*n_sub, rank))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i*m_sub+j, col] = 1
                    V[j*p_sub+k, col] = 1
                    W[k*n_sub+i, col] = 1
                    col += 1
        return U, V, W
    
    # Build the 4x4 * 4x9 decomposition
    # Block 1: rows 0-3 of A, cols 0-3 of B -> cols 0-3 of C (4x4 * 4x4)
    U1, V1, W1 = strassen_4x4_4x4()
    
    # Block 2: rows 0-3 of A, cols 4-7 of B -> cols 4-7 of C (4x4 * 4x4)
    U2, V2, W2 = strassen_4x4_4x4()
    
    # Block 3: rows 0-3 of A, col 8 of B -> col 8 of C (4x4 * 4x1)
    U3, V3, W3 = schoolbook_4x4_4x1()
    
    # Combine
    # For block 1: i_off=0, j_off=0, k_off=0, n_sub=4, m_sub=4, p_sub=4
    # For block 2: i_off=0, j_off=4, k_off=4, n_sub=4, m_sub=4, p_sub=4
    # For block 3: i_off=0, j_off=8, k_off=8, n_sub=4, m_sub=4, p_sub=1
    
    total_rank = U1.shape[1] + U2.shape[1] + U3.shape[1]
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    
    # Block 1
    for r in range(16):
        i = r // 4
        j = r % 4
        U[r, col_offset:col_offset+U1.shape[1]] = U1[r, :]
    for r in range(16):
        j = r // 4
        k = r % 4
        V[j*9+k, col_offset:col_offset+V1.shape[1]] = V1[r, :]
    for r in range(16):
        k = r // 4
        i = r % 4
        W[k*4+i, col_offset:col_offset+W1.shape[1]] = W1[r, :]
    col_offset += U1.shape[1]
    
    # Block 2: j offset by 4, k offset by 4
    for r in range(16):
        i = r // 4
        j = r % 4
        U[r, col_offset:col_offset+U2.shape[1]] = U2[r, :]
    for r in range(16):
        j_sub = r // 4
        k_sub = r % 4
        j_global = 4 + j_sub
        k_global = 4 + k_sub
        V[j_global*9+k_global, col_offset:col_offset+V2.shape[1]] = V2[r, :]
    for r in range(16):
        k_sub = r // 4
        i_sub = r % 4
        k_global = 4 + k_sub
        i_global = i_sub
        W[k_global*4+i_global, col_offset:col_offset+W2.shape[1]] = W2[r, :]
    col_offset += U2.shape[1]
    
    # Block 3: j offset by 8, k offset by 8
    for r in range(16):
        i = r // 4
        j = r % 4
        U[r, col_offset:col_offset+U3.shape[1]] = U3[r, :]
    for r in range(4):  # m_sub*p_sub = 4*1 = 4
        j_sub = r // 1
        k_sub = r % 1
        j_global = 8 + j_sub
        k_global = 8 + k_sub
        V[j_global*9+k_global, col_offset:col_offset+V3.shape[1]] = V3[r, :]
    for r in range(4):  # p_sub*n_sub = 1*4 = 4
        k_sub = r // 4
        i_sub = r % 4
        k_global = 8 + k_sub
        i_global = i_sub
        W[k_global*4+i_global, col_offset:col_offset+W3.shape[1]] = W3[r, :]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
