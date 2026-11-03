"""Matrix multiplication 4,4,7 rank-minimal decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] where
    C[i,k] = sum_j A[i,j] B[j,k].
    T is a 3-tensor of shape (n*p, m, n*p) or we use the standard form:
    T[i*m*p + j*n*p + k] ... 
    Actually, let's use the standard 3-tensor form:
    T[a, b, c] where a indexes (i, k) for output, b indexes j for inner dim, c indexes (i, j) for A and (j, k) for B.
    
    More precisely, the tensor is T in V* ⊗ W* ⊗ X* where:
    - First factor: output index (i, k) with i in [0,n), k in [0,p) -> size n*p
    - Second factor: input A index (i, j) with i in [0,n), j in [0,m) -> size n*m
    - Third factor: input B index (j, k) with j in [0,m), k in [0,p) -> size m*p
    
    T[(i,k), (i,j), (j,k)] = 1 if indices match, else 0.
    """
    tensor = np.zeros((n * p, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                out_idx = i * p + k
                a_idx = i * m + j
                b_idx = j * p + k
                tensor[out_idx, a_idx, b_idx] = 1.0
    return tensor

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    exactly decompose the matrix multiplication tensor.
    
    The decomposition means: for all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Wait, let me re-read the contract. The output factor follows W[k*n+i, r].
    So the reconstruction is:
    C[i, k] = sum_j A[i, j] B[j, k]
    
    In tensor form: T[i, j, k] where we sum over the decomposition rank r:
    sum_r U[i, j, r] * V[j, k, r] * W[i, k, r] = delta (the tensor)
    
    But the shapes are:
    U: (n*m, rank) -> U[i*m + j, r]
    V: (m*p, rank) -> V[j*p + k, r]
    W: (p*n, rank) -> W[k*n + i, r]
    
    So the reconstruction is:
    For each (i, k), the coefficient of A[i,j]*B[j,k] in the product should be 1 if we're computing C[i,k].
    
    Actually, the tensor decomposition of matrix multiplication says:
    sum_r U[i, j, r] * V[j, k, r] * W[i, k, r] = T[i, j, k]
    
    where T[i, j, k] = 1 (since C[i,k] = sum_j A[i,j] B[j,k], the tensor has 1s at all (i,j,k)).
    
    Wait no. The matrix multiplication tensor M_{n,m,p} is defined such that:
    M[i, j, k] = 1 for all i, j, k.
    
    And the rank-r decomposition means:
    M[i, j, k] = sum_{r=1}^{rank} U[i, j, r] * V[j, k, r] * W[i, k, r]
    
    With the index conventions:
    U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    So we need: for all i, j, k:
    sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the left-hand side tensor
    # LHS[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    lhs = np.zeros((n, m, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        # w_r is (p, n), so w_r.T is (n, p)
        # We need to combine: for each (i,j,k), u_r[i,j] * v_r[j,k] * w_r.T[i,k]
        # This is like an element-wise product after broadcasting
        # u_r[i,j] * v_r[j,k] gives (n, m, p) via broadcasting
        # Then multiply by w_r.T[i,k] which is (n, p) -> (n, 1, p)
        lhs += u_r[:, :, np.newaxis] * v_r[np.newaxis, :, :] * w_r.T[:, np.newaxis, :]
    
    # Check if lhs == ones
    return np.allclose(lhs, np.ones((n, m, p)), atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_strassen_4x4x7(n, m, p):
    """
    Try to construct a low-rank decomposition using Strassen-like ideas.
    
    For 2x2x2, Strassen gives rank 7.
    For 4x4x4, we can use Kronecker products or block Strassen.
    
    Let's try a block approach: view 4x4 as 2x2 blocks of 2x2 matrices.
    A = [[A11, A12], [A21, A22]] where each block is 2x2.
    B = [[B11, B12], [B21, B22]] where each block is 2x2.
    
    Using Strassen on the 2x2 block structure:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a 2x2 matrix product. If we use Strassen for each 2x2 product (rank 7),
    and we have 7 such products, the total rank would be 7 * 7 = 49.
    
    But wait, the products are not independent 2x2 matrix multiplications in the usual sense.
    Actually, each Mi involves products of 2x2 matrices, so each can be done with rank 7.
    Total rank: 7 * 7 = 49.
    
    For 4x4x7, we can extend B to 4x7 by padding to 4x8 and taking first 7 columns.
    Let's just handle 4x4x4 first with rank 49, then adapt for 4x4x7.
    
    Actually, let me think more carefully. The tensor for 4x4x4 has a known rank.
    The best known upper bound for 4x4x4 is 49 (from Strassen).
    
    For 4x4x7, we can use the 4x4x4 decomposition and pad.
    
    Let's construct the 4x4x4 Strassen decomposition explicitly.
    """
    # This is complex. Let me try a different approach.
    # Use the fact that rank(4x4x4) <= 49 via Strassen.
    # Construct it explicitly.
    
    # Strassen for 2x2x2:
    # The 7 terms are:
    # S1 = (A11+A22)(B11+B22)
    # S2 = (A21+A22)B11
    # S3 = A11(B12-B22)
    # S4 = A22(B21-B11)
    # S5 = (A11+A12)B22
    # S6 = (A21-A11)(B11+B12)
    # S7 = (A12-A22)(B21+B22)
    
    # For the 2x2x2 tensor, the decomposition is:
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # Let's build the U, V, W for 2x2x2 with rank 7.
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # For each Strassen term t in 1..7:
    # U and V encode the linear combinations of A and B.
    # W encodes the linear combination to get C.
    
    # Term 1: (A11+A22)(B11+B22)
    # U: A11 coeff 1, A22 coeff 1 -> U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
    # V: B11 coeff 1, B22 coeff 1 -> V[0,0]=1, V[3,0]=1 (j=0,k=0 and j=1,k=1)
    # W: contributes to C11 (+1), C22 (+1) -> W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    
    # Let me be more systematic. I'll build the 2x2x2 Strassen decomposition.
    
    n2, m2, p2 = 2, 2, 2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A11+A22)(B11+B22)
    # A indices: (0,0) and (1,1) -> U2[0,0]=1, U2[3,0]=1
    # B indices: (0,0) and (1,1) -> V2[0,0]=1, V2[3,0]=1
    # C: C[0,0] += 1, C[1,1] += 1 -> W2[0,0]=1, W2[3,0]=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 2: (A21+A22)B11
    # A: (1,0) and (1,1) -> U2[2,1]=1, U2[3,1]=1
    # B: (0,0) -> V2[0,1]=1
    # C: C[1,0] += 1, C[1,1] -= 1 -> W2[2,1]=1 (k=0,i=1), W2[3,1]=-1 (k=1,i=1)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 3: A11(B12-B22)
    # A: (0,0) -> U2[0,2]=1
    # B: (0,1) and (1,1) -> V2[1,2]=1, V2[3,2]=-1
    # C: C[0,1] += 1, C[1,1] += 1 -> W2[1,2]=1 (k=1,i=0), W2[3,2]=1 (k=1,i=1)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 4: A22(B21-B11)
    # A: (1,1) -> U2[3,3]=1
    # B: (1,0) and (0,0) -> V2[2,3]=1, V2[0,3]=-1
    # C: C[0,0] += 1, C[1,0] += 1 -> W2[0,3]=1 (k=0,i=0), W2[2,3]=1 (k=0,i=1)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 5: (A11+A12)B22
    # A: (0,0) and (0,1) -> U2[0,4]=1, U2[1,4]=1
    # B: (1,1) -> V2[3,4]=1
    # C: C[0,0] -= 1, C[0,1] += 1 -> W2[0,4]=-1 (k=0,i=0), W2[1,4]=1 (k=1,i=0)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 6: (A21-A11)(B11+B12)
    # A: (1,0) and (0,0) -> U2[2,5]=1, U2[0,5]=-1
    # B: (0,0) and (0,1) -> V2[0,5]=1, V2[1,5]=1
    # C: C[1,1] += 1 -> W2[3,5]=1 (k=1,i=1)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 7: (A12-A22)(B21+B22)
    # A: (0,1) and (1,1) -> U2[1,6]=1, U2[3,6]=-1
    # B: (1,0) and (1,1) -> V2[2,6]=1, V2[3,6]=1
    # C: C[0,0] += 1 -> W2[0,6]=1 (k=0,i=0)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Verify 2x2x2 decomposition
    if not _check_decomposition(U2, V2, W2, 2, 2, 2):
        return None
    
    # Now for 4x4x4, use Kronecker product of the 2x2x2 Strassen decomposition with itself.
    # The rank would be 7 * 7 = 49.
    # 
    # For the Kronecker product:
    # U4[i*4+j, r1*7+r2] = U2[i*2+i2, r1] * U2[j*2+j2, r2] ... 
    # This is getting complicated. Let me think differently.
    #
    # Actually, the Kronecker product of two matrix multiplication tensors:
    # If T1 is the 2x2x2 tensor with decomposition (U1, V1, W1) of rank 7,
    # and T2 is the 2x2x2 tensor with decomposition (U2, V2, W2) of rank 7,
    # then T1 ⊗ T2 is the 4x4x4 tensor with decomposition of rank 49.
    #
    # The Kronecker product decomposition:
    # U4 = U1 ⊗ U2 (in some appropriate sense)
    # 
    # More precisely, if we view the 4x4 matrix as a 2x2 block matrix of 2x2 blocks,
    # then the 4x4x4 tensor is the tensor product of two 2x2x2 tensors.
    #
    # Let me construct it directly.
    # For 4x4x4, indices i, j, k in [0,4).
    # Write i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2 where i1,i2,j1,j2,k1,k2 in [0,2).
    # 
    # The 4x4x4 tensor T4[i,j,k] = 1 for all i,j,k.
    # This can be factored as T4[i,j,k] = T2[i1,j1,k1] * T2[i2,j2,k2] * (cross terms?)
    # 
    # Actually, T4[i,j,k] = 1 = 1*1 = T2[i1,j1,k1] * T2[i2,j2,k2] since both are 1.
    # But that's not a tensor product in the right sense.
    #
    # The correct tensor product: The 4x4x4 matrix multiplication tensor is the
    # Kronecker product of two 2x2x2 tensor products.
    #
    # Let me just use the block Strassen approach directly.
    # View A as 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Using Strassen on the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 matrix product. Using Strassen for each (rank 7), total rank = 7*7 = 49.
    #
    # Now I need to construct U, V, W for this.
    # For each Strassen term t (1-7) and each inner Strassen term s (1-7):
    # The combined term has rank index (t-1)*7 + (s-1).
    #
    # U4[i*4+j, (t-1)*7+(s-1)] = U2_block[i, t] * U2_inner[j_block, s]
    # where i is the block row (0 or 1), j_block is the inner row, etc.
    #
    # This is complex but doable. Let me code it up.
    
    # For the block Strassen, each term t involves a linear combination of A blocks.
    # Let me define the block-level coefficients.
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) for 2x2 blocks.
    # A block (a, b) means A[a*2:(a+1)*2, b*2:(b+1)*2]
    
    # For each Strassen term t, the A-side is a linear combination of A blocks.
    # A_coeff[t][block] = coefficient
    
    A_coeff = np.zeros((7, 4))  # 7 terms, 4 blocks
    B_coeff = np.zeros((7, 4))
    C_coeff = np.zeros((7, 4))  # How each term contributes to C blocks
    
    # Term 1: (A00+A11)(B00+B11)
    A_coeff[0, 0] = 1; A_coeff[0, 3] = 1
    B_coeff[0, 0] = 1; B_coeff[0, 3] = 1
    C_coeff[0, 0] = 1; C_coeff[0, 3] = 1  # C00 += M1, C11 += M1
    
    # Term 2: (A10+A11)B00
    A_coeff[1, 2] = 1; A_coeff[1, 3] = 1
    B_coeff[1, 0] = 1
    C_coeff[1, 2] = 1; C_coeff[1, 3] = -1  # C10 += M2, C11 -= M2
    
    # Term 3: A00(B01-B11)
    A_coeff[2, 0] = 1
    B_coeff[2, 1] = 1; B_coeff[2, 3] = -1
    C_coeff[2, 1] = 1; C_coeff[2, 3] = 1  # C01 += M3, C11 += M3
    
    # Term 4: A11(B10-B00)
    A_coeff[3, 3] = 1
    B_coeff[3, 2] = 1; B_coeff[3, 0] = -1
    C_coeff[3, 0] = 1; C_coeff[3, 2] = 1  # C00 += M4, C10 += M4
    
    # Term 5: (A00+A01)B11
    A_coeff[4, 0] = 1; A_coeff[4, 1] = 1
    B_coeff[4, 3] = 1
    C_coeff[4, 0] = -1; C_coeff[4, 1] = 1  # C00 -= M5, C01 += M5
    
    # Term 6: (A10-A00)(B00+B01)
    A_coeff[5, 2] = 1; A_coeff[5, 0] = -1
    B_coeff[5, 0] = 1; B_coeff[5, 1] = 1
    C_coeff[5, 3] = 1  # C11 += M6
    
    # Term 7: (A01-A11)(B10+B11)
    A_coeff[6, 1] = 1; A_coeff[6, 3] = -1
    B_coeff[6, 2] = 1; B_coeff[6, 3] = 1
    C_coeff[6, 0] = 1  # C00 += M7
    
    # Now construct U4, V4, W4 for 4x4x4.
    rank4 = 49
    U4 = np.zeros((16, rank4))
    V4 = np.zeros((16, rank4))
    W4 = np.zeros((16, rank4))
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            # U4: for each element (i, j) in 4x4, 
            # i = i_block*2 + i_inner, j = j_block*2 + j_inner
            # U4[i*4+j, r] = A_coeff[t, i_block*2+j_block] * U2[i_inner*2+j_inner, s]
            for i_block in range(2):
                for j_block in range(2):
                    block_idx = i_block * 2 + j_block
                    a_coeff = A_coeff[t, block_idx]
                    if a_coeff == 0:
                        continue
                    for i_inner in range(2):
                        for j_inner in range(2):
                            i = i_block * 2 + i_inner
                            j = j_block * 2 + j_inner
                            u2_val = U2[i_inner * 2 + j_inner, s]
                            U4[i * 4 + j, r] = a_coeff * u2_val
            
            # V4: for each element (j, k) in 4x4,
            # j = j_block*2 + j_inner, k = k_block*2 + k_inner
            # V4[j*4+k, r] = B_coeff[t, j_block*2+k_block] * V2[j_inner*2+k_inner, s]
            for j_block in range(2):
                for k_block in range(2):
                    block_idx = j_block * 2 + k_block
                    b_coeff = B_coeff[t, block_idx]
                    if b_coeff == 0:
                        continue
                    for j_inner in range(2):
                        for k_inner in range(2):
                            j = j_block * 2 + j_inner
                            k = k_block * 2 + k_inner
                            v2_val = V2[j_inner * 2 + k_inner, s]
                            V4[j * 4 + k, r] = b_coeff * v2_val
            
            # W4: for each element (k, i) in 4x4 (W[k*4+i, r]),
            # k = k_block*2 + k_inner, i = i_block*2 + i_inner
            # W4[k*4+i, r] = C_coeff[t, i_block*2+k_block] * W2[k_inner*2+i_inner, s]
            for i_block in range(2):
                for k_block in range(2):
                    block_idx = i_block * 2 + k_block
                    c_coeff = C_coeff[t, block_idx]
                    if c_coeff == 0:
                        continue
                    for i_inner in range(2):
                        for k_inner in range(2):
                            k = k_block * 2 + k_inner
                            i = i_block * 2 + i_inner
                            w2_val = W2[k_inner * 2 + i_inner, s]
                            W4[k * 4 + i, r] = c_coeff * w2_val
    
    # Verify 4x4x4 decomposition
    if not _check_decomposition(U4, V4, W4, 4, 4, 4):
        return None
    
    # Now extend to 4x4x7.
    # We have a 4x4x4 decomposition. For 4x4x7, we need to handle the extra 3 columns of B.
    # The 4x4x7 tensor T[i, j, k] = 1 for i in [0,4), j in [0,4), k in [0,7).
    # We can view this as 4x4x4 tensor for k in [0,4) plus 4x4x3 tensor for k in [4,7).
    # The 4x4x3 part can be done with schoolbook: rank 4*4*3 = 48.
    # Total rank: 49 + 48 = 97.
    
    # But we can do better. Let's use the 4x4x4 decomposition for the first 4 columns,
    # and schoolbook for the last 3 columns.
    
    rank_total = 49 + 4 * 4 * 3
    U_total = np.zeros((16, rank_total))
    V_total = np.zeros((28, rank_total))  # m*p = 4*7 = 28
    W_total = np.zeros((28, rank_total))  # p*n = 7*4 = 28
    
    # First 49 columns: from 4x4x4 decomposition
    U_total[:, :49] = U4
    V_total[:16, :49] = V4  # j*p+k for j in [0,4), k in [0,4) -> indices 0..15
    # For k in [4,7), V4 doesn't have entries, so V_total[16:28, :49] = 0 (already zero)
    W_total[:16, :49] = W4  # k*n+i for k in [0,4) -> indices 0..15
    # For k in [4,7), W_total[16:28, :49] = 0 (already zero)
    
    # Last 48 columns: schoolbook for k in [4,7)
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U_total[i * 4 + j, col] = 1.0
                V_total[j * 7 + k, col] = 1.0
                W_total[k * 4 + i, col] = 1.0
                col += 1
    
    # Verify 4x4x7 decomposition
    if not _check_decomposition(U_total, V_total, W_total, 4, 4, 7):
        return None
    
    return U_total, V_total, W_total

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try the Strassen-based construction
    try:
        result = _try_strassen_4x4x7(n, m, p)
        if result is not None:
            U, V, W = result
            # Convert to lists for the return format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
