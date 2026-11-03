"""Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS

def _build_target_tensor():
    T = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_idx = i * m + j
                V_idx = j * p + k
                W_idx = k * n + i
                T[U_idx, V_idx, W_idx] = 1
    return T

def _schoolbook():
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, T):
    r = U.shape[1]
    T_decomp = np.zeros((n * m, m * p, p * n), dtype=object)
    for a in range(r):
        for i in range(n * m):
            if U[i, a] == 0:
                continue
            for j in range(m * p):
                if V[j, a] == 0:
                    continue
                for k in range(p * n):
                    if W[k, a] == 0:
                        continue
                    T_decomp[i, j, k] += U[i, a] * V[j, a] * W[k, a]
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                if T_decomp[i, j, k] != T[i, j, k]:
                    return False
    return True

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U, V, W have shape (4, 7) each.
    Indexing: row = i*m + j for U, j*p + k for V, k*n + i for W.
    For n=m=p=2.
    """
    nn, mm, pp = 2, 2, 2
    U = np.zeros((nn * mm, 7), dtype=object)
    V = np.zeros((mm * pp, 7), dtype=object)
    W = np.zeros((pp * nn, 7), dtype=object)
    
    # Strassen formulas:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_t is a rank-1 term: U[:,t] * V[:,t] * W[:,t]
    # M1: (A11+A22)*(B11+B22) -> contributes to C11 and C22
    # We need to decompose each M_t into contributions to C entries.
    # Actually, each M_t is a single product of a linear form in A and a linear form in B,
    # contributing to specific C entries.
    
    # Let's build it properly.
    # For term t, U[:,t] has nonzeros at A-indices, V[:,t] at B-indices, W[:,t] at C-indices.
    
    # M1 = (A11+A22)(B11+B22), contributes +1 to C11 and +1 to C22
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2 = (A21+A22)B11, contributes +1 to C21 and -1 to C22
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1               # B11
    W[2, 1] = 1; W[3, 1] = -1 # C21, C22
    
    # M3 = A11(B12-B22), contributes +1 to C12 and +1 to C22
    U[0, 2] = 1               # A11
    V[1, 2] = 1; V[3, 2] = -1 # B12, B22
    W[1, 2] = 1; W[3, 2] = 1  # C12, C22
    
    # M4 = A22(B21-B11), contributes +1 to C11 and +1 to C21
    U[3, 3] = 1               # A22
    V[2, 3] = 1; V[0, 3] = -1 # B21, B11
    W[0, 3] = 1; W[2, 3] = 1  # C11, C21
    
    # M5 = (A11+A12)B22, contributes -1 to C11 and +1 to C12
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1               # B22
    W[0, 4] = -1; W[1, 4] = 1 # C11, C12
    
    # M6 = (A21-A11)(B11+B12), contributes +1 to C22
    U[2, 5] = 1; U[0, 5] = -1 # A21, A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1               # C22
    
    # M7 = (A12-A22)(B21+B22), contributes +1 to C11
    U[1, 6] = 1; U[3, 6] = -1 # A12, A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1               # C11
    
    return U, V, W

def _embed_2x2_into_4x4x4(U2, V2, W2):
    """
    Embed a 2x2 Strassen decomposition into 4x4 * 4x4 -> 4x4.
    We split the 4x4 matrices into 2x2 blocks.
    A = [[A11, A12], [A21, A22]] where each block is 2x2.
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    The 2x2 block multiplication C11 = A11*B11 + A12*B21 etc.
    We apply Strassen to the 2x2 block multiplication.
    
    This gives a rank 7*4=28 decomposition for 4x4*4x4 (since each 2x2 Strassen
    term needs to be applied to each 2x2 block multiplication, and there are 4 block
    multiplications... wait, no. Strassen computes all 4 block products with 7 scalar
    multiplications. So the total rank is 7 * (2*2) = 28? No.
    
    Actually, for block Strassen: we treat each 2x2 block as a scalar. The 7 products
    are of 2x2 matrices. Each 2x2 matrix product has rank 4 (schoolbook). So total
    rank = 7 * 4 = 28.
    
    But we can do better by noting that the Strassen formula is linear, so we can
    directly write the rank-1 terms.
    
    Let me think again. The 2x2 Strassen decomposition has 7 terms. Each term is a
    product of a linear combination of A-blocks and a linear combination of B-blocks,
    contributing to C-blocks. Since each block is 2x2, each "scalar" multiplication
    of blocks is actually a 2x2 matrix multiplication, which itself has rank 4.
    
    So total rank = 7 * 4 = 28 for 4x4*4x4.
    """
    nn, mm, pp = 4, 4, 4
    rank = U2.shape[1] * (2 * 2)  # 7 * 4 = 28
    U4 = np.zeros((nn * mm, rank), dtype=object)
    V4 = np.zeros((mm * pp, rank), dtype=object)
    W4 = np.zeros((pp * nn, rank), dtype=object)
    
    # Block structure: 
    # A row index: i*4+j, where i,j in 0..1 for block row/col, and within block i2,j2 in 0..1
    # Block (I,J) occupies rows I*2+i2, cols J*2+j2, so global index (I*2+i2)*4 + (J*2+j2)
    
    for t in range(U2.shape[1]):
        for i2 in range(2):
            for j2 in range(2):
                col = t * 4 + i2 * 2 + j2
                # U: for each nonzero in U2[:,t] at block position (I,J), 
                # place U2[I*2+J, t] at global position (I*2+i2)*4 + (J*2+j2)
                for I in range(2):
                    for J in range(2):
                        val = U2[I * 2 + J, t]
                        if val != 0:
                            U4[(I * 2 + i2) * 4 + (J * 2 + j2), col] = val
                # V: similarly for V2[:,t]
                for J in range(2):
                    for K in range(2):
                        val = V2[J * 2 + K, t]
                        if val != 0:
                            V4[(J * 2 + j2) * 4 + (K * 2 + i2), col] = val
                # W: similarly for W2[:,t]
                for K in range(2):
                    for I in range(2):
                        val = W2[K * 2 + I, t]
                        if val != 0:
                            W4[(K * 2 + i2) * 4 + (I * 2 + j2), col] = val
    
    return U4, V4, W4

def _embed_4x4_into_4x4x5(U4, V4, W4):
    """
    Embed a 4x4*4x4 decomposition into 4x4*4x5.
    We pad the 4x4 B matrix to 4x5 (add a zero column) and C from 4x4 to 4x5... 
    wait, the output is 4x5? No, C is n x p = 4 x 5.
    
    Actually, for 4x4 * 4x5 -> 4x5, we can view B as having 5 columns.
    We can split B into B[:, :4] and B[:, 4]. The first part is 4x4, the second is 4x1.
    C[:, :4] = A * B[:, :4] and C[:, 4] = A * B[:, 4].
    
    So we can use the 4x4*4x4 decomposition for the first 4 columns of C,
    and a separate decomposition for the 5th column.
    
    A * B[:, 4] is a 4x1 vector, which is just a matrix-vector product.
    This has rank 4 (schoolbook: each output element is a dot product).
    
    Total rank = 28 + 4 = 32.
    
    But wait, can we do better? The 4x4*4x4 part has rank 28 via Strassen.
    The 4x4*4x1 part: C_i = sum_j A_ij * B_j4. This is rank 4.
    Total = 32.
    
    Alternatively, we can treat the whole thing as 4x4 * 4x5 directly.
    Let me just pad.
    """
    nn, mm, pp = 4, 4, 5
    rank4 = U4.shape[1]
    # The 4x4*4x4 part handles C[:, :4]
    # We need to add terms for C[:, 4]
    # C_i4 = sum_j A_ij * B_j4, rank 4
    
    total_rank = rank4 + nn  # 28 + 4 = 32
    U = np.zeros((nn * mm, total_rank), dtype=object)
    V = np.zeros((mm * pp, total_rank), dtype=object)
    W = np.zeros((pp * nn, total_rank), dtype=object)
    
    # Copy the 4x4*4x4 part (handles C columns 0..3)
    for a in range(rank4):
        for i in range(nn * mm):
            U[i, a] = U4[i, a]
        for j in range(mm * 4):  # B has only first 4 columns used
            V[j, a] = V4[j, a]
        for k in range(4 * nn):  # C has first 4 columns
            W[k, a] = W4[k, a]
    
    # Add terms for C[:, 4]
    # C_i4 = sum_j A_ij * B_j4
    for i in range(nn):
        for j in range(mm):
            col = rank4 + i * mm + j
            U[i * mm + j, col] = 1
            V[j * pp + 4, col] = 1  # B[j, 4]
            W[4 * nn + i, col] = 1  # C[4, i] -> index 4*4+i = 16+i
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    T = _build_target_tensor()
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook()
    best_rank = U_sb.shape[1]
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try Strassen-based decomposition
    if time.time() < deadline:
        U2, V2, W2 = _strassen_2x2()
        U4, V4, W4 = _embed_2x2_into_4x4x4(U2, V2, W2)
        U_try, V_try, W_try = _embed_4x4_into_4x4x5(U4, V4, W4)
        
        if _verify(U_try, V_try, W_try, T):
            if U_try.shape[1] < best_rank:
                best_rank = U_try.shape[1]
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
    
    # Try random search for lower rank
    if time.time() < deadline:
        remaining = deadline - time.time() - 10
        if remaining > 5:
            # Try ranks from best_rank-1 down
            for target_rank in range(best_rank - 1, max(best_rank - 20, 20)):
                if time.time() > deadline - 10:
                    break
                found = False
                for trial in range(20):
                    if time.time() > deadline - 10:
                        break
                    U_r = np.random.randint(-2, 3, (n * m, target_rank))
                    V_r = np.random.randint(-2, 3, (m * p, target_rank))
                    W_r = np.random.randint(-2, 3, (p * n, target_rank))
                    if _verify(U_r, V_r, W_r, T):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                        found = True
                        break
                if found:
                    break
    
    # Convert to lists
    def to_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                if isinstance(val, np.integer):
                    r.append(int(val))
                elif isinstance(val, np.floating):
                    r.append(float(val))
                elif isinstance(val, Fraction):
                    r.append(val)
                elif hasattr(val, '__int__'):
                    try:
                        r.append(int(val))
                    except:
                        r.append(float(val))
                else:
                    try:
                        r.append(int(val))
                    except:
                        try:
                            r.append(float(val))
                        except:
                            r.append(val)
            result.append(r)
        return result
    
    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)
    
    return {"U": U_final, "V": V_final, "W": W_final}
