import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: T[i*4+j, j*4+k, k*4+i] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1
    """
    # n=4, m=4, p=4
    # Rank 49
    
    # We define the decomposition by composing the 2x2 Strassen decomposition with itself.
    # Let S be the 2x2 Strassen tensor.
    # S has rank 7.
    # The 4x4 tensor is S \otimes S.
    # The rank is 7 * 7 = 49.
    
    # 2x2 Strassen factors (u, v, w)
    # Indices for 2x2: A is 2x2, B is 2x2, C is 2x2.
    # A indices: 0,1,2,3 (row-major: 00,01,10,11)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    # Strassen 7 multiplications:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We represent each of the 7 terms as vectors u_i, v_i, w_i in R^4.
    # u_i has non-zeros at A positions.
    # v_i has non-zeros at B positions.
    # w_i has non-zeros at C positions.
    
    # A layout: [[a, b], [c, d]] -> indices: a=0, b=1, c=2, d=3
    # B layout: [[e, f], [g, h]] -> indices: e=0, f=1, g=2, h=3
    # C layout: [[C11, C12], [C21, C22]] -> indices: C11=0, C12=1, C21=2, C22=3
    
    # Term 1: M1 = (a+d)(e+h)
    # u1: a=1, d=1 -> [1, 0, 0, 1]
    # v1: e=1, h=1 -> [1, 0, 0, 1]
    # w1: C11=1, C22=1 -> [1, 0, 0, 1]
    
    # Term 2: M2 = (c+d)e
    # u2: c=1, d=1 -> [0, 0, 1, 1]
    # v2: e=1 -> [1, 0, 0, 0]
    # w2: C21=1, C22=-1 -> [0, 0, 1, -1]
    
    # Term 3: M3 = a(f-h)
    # u3: a=1 -> [1, 0, 0, 0]
    # v3: f=1, h=-1 -> [0, 1, 0, -1]
    # w3: C12=1, C22=1 -> [0, 1, 0, 1]
    
    # Term 4: M4 = d(g-e)
    # u4: d=1 -> [0, 0, 0, 1]
    # v4: g=1, e=-1 -> [-1, 0, 1, 0]
    # w4: C11=1, C21=1 -> [1, 0, 1, 0]
    
    # Term 5: M5 = (a+b)h
    # u5: a=1, b=1 -> [1, 1, 0, 0]
    # v5: h=1 -> [0, 0, 0, 1]
    # w5: C11=-1, C12=1 -> [-1, 1, 0, 0]
    
    # Term 6: M6 = (c-a)(e+f)
    # u6: c=1, a=-1 -> [-1, 0, 1, 0]
    # v6: e=1, f=1 -> [1, 1, 0, 0]
    # w6: C22=1 -> [0, 0, 0, 1]
    
    # Term 7: M7 = (b-d)(g+h)
    # u7: b=1, d=-1 -> [0, 1, 0, -1]
    # v7: g=1, h=1 -> [0, 0, 1, 1]
    # w7: C11=1 -> [1, 0, 0, 0]
    
    U_2x2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]) # Shape (7, 4)
    
    V_2x2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]) # Shape (7, 4)
    
    W_2x2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]) # Shape (7, 4)
    
    # Verify 2x2 decomposition
    # T_2x2[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # We can check this quickly, but we trust Strassen.
    
    # Construct 4x4 decomposition using Kronecker product
    # The 4x4 tensor is the Kronecker product of the 2x2 tensor with itself.
    # U_4x4 = U_2x2 \otimes U_2x2
    # V_4x4 = V_2x2 \otimes V_2x2
    # W_4x4 = W_2x2 \otimes W_2x2
    
    # Note on index ordering:
    # The problem states:
    # U[i*4+j, r] corresponds to A[i, j]
    # V[j*4+k, r] corresponds to B[j, k]
    # W[k*4+i, r] corresponds to C[i, k] ?? 
    # "output factor follows the official trace convention W[k*n+i,r]"
    # n=4. W index is k*4 + i.
    # Standard C[i, j] = sum_k A[i, k] B[k, j].
    # Here output is C. Let's call it C[i, j].
    # The tensor is T[A_idx, B_idx, C_idx].
    # A_idx = i*4 + k? No, A is 4x4. A[i, k].
    # The prompt says: "U[i*4+j, r]". This implies A is indexed by i (row) and j (col).
    # "V[j*4+k, r]". B is indexed by j (row) and k (col).
    # "W[k*4+i, r]". C is indexed by... k and i?
    # If C = A B, then C[i, k] = sum_j A[i, j] B[j, k].
    # So the output index is (i, k).
    # The W index is k*4 + i. This is column-major for the output matrix C if we view i as row and k as col?
    # Or is it row-major for C^T?
    # Let's check the standard convention.
    # Usually C[i, j].
    # If W index is k*4 + i, and the sum is over j (which is the 'k' in V and 'j' in U), then:
    # Sum_j U[i*4+j, r] V[j*4+k, r] W[k*4+i, r]
    # This computes C[i, k] = sum_j A[i, j] B[j, k].
    # So the output matrix C has rows i and cols k.
    # The index in W is col * 4 + row.
    
    # The Kronecker product structure:
    # A 4x4 matrix can be viewed as a 2x2 matrix of 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # A11 is A[0:2, 0:2].
    # The index for A11 elements in A (4x4) is i*4+j.
    # If we map 2x2 block index (bi, bj) and internal index (ri, rj) to global (i, j):
    # i = 2*bi + ri
    # j = 2*bj + rj
    # Global index = (2*bi + ri)*4 + (2*bj + rj) = 8*bi + 4*ri + 2*bj + rj.
    
    # The Kronecker product U_2x2 \otimes U_2x2 will have shape (4*4, 7*7) = (16, 49).
    # The row index in the Kronecker product is usually (block_row * dim + internal_row).
    # If we use np.kron(U1, U2), the row index is r1 * 4 + r2.
    # Here r1 is the block row index (0 or 1) and r2 is the internal row index (0..3).
    # This matches the global index if we identify:
    # Block row 0 -> global rows 0,1. Block row 1 -> global rows 2,3.
    # Internal row 0,1,2,3 -> global cols 0,1,2,3?
    # Wait.
    # Global index = i*4 + j.
    # Kronecker index = r1 * 4 + r2.
    # If r1 = i_block and r2 = j_global? No.
    # Let's map carefully.
    # A is 4x4.
    # We split A into 2x2 blocks of size 2x2.
    # A = [[A00, A01], [A10, A11]]
    # A00 corresponds to i in {0,1}, j in {0,1}.
    # A01 corresponds to i in {0,1}, j in {2,3}.
    # A10 corresponds to i in {2,3}, j in {0,1}.
    # A11 corresponds to i in {2,3}, j in {2,3}.
    
    # In the 2x2 Strassen, the "elements" are the blocks.
    # The "rows" of the 2x2 matrix of blocks are the block rows (0, 1).
    # The "cols" of the 2x2 matrix of blocks are the block cols (0, 1).
    
    # The vector U_2x2 has length 4, corresponding to the 4 elements of the 2x2 block matrix.
    # Element 0: Block (0,0)
    # Element 1: Block (0,1)
    # Element 2: Block (1,0)
    # Element 3: Block (1,1)
    
    # When we take U_2x2 \otimes U_2x2:
    # The resulting vector has length 16.
    # The index is idx = idx_block * 4 + idx_internal.
    # idx_block corresponds to the position in the 2x2 block matrix (0,1,2,3).
    # idx_internal corresponds to the position within the 2x2 block (0,1,2,3).
    
    # Let's check if this matches the global index i*4 + j.
    # Block (bi, bj) contains elements (2*bi + ri, 2*bj + rj).
    # Global index = (2*bi + ri)*4 + (2*bj + rj) = 8*bi + 4*ri + 2*bj + rj.
    
    # Kronecker index:
    # The "block" index in the 2x2 matrix is $b = bi*2 + bj$?
    # In row-major 2x2:
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 2
    # (1,1) -> 3
    # So $b = bi*2 + bj$.
    # The "internal" index in the 2x2 block is $r = ri*2 + rj$.
    # Kronecker index $K = b*4 + r = (bi*2 + bj)*4 + (ri*2 + rj) = 8*bi + 4*bj + 2*ri + rj$.
    
    # Global index $G = 8*bi + 4*ri + 2*bj + rj$.
    
    # $K \neq G$.
    # $K = 8*bi + 4*bj + 2*ri + rj$
    # $G = 8*bi + 4*ri + 2*bj + rj$
    
    # The Kronecker product mixes the block column index $bj$ with the internal row index $ri$.
    # Specifically, the 4th bit in $K$ is $bj$, while in $G$ it is $ri$.
    # The 3rd bit in $K$ is $ri$, while in $G$ it is $bj$.
    
    # So we cannot just use np.kron directly on the vectors without permuting the indices.
    # We need to permute the rows of the Kronecker product matrix to match the global index order.
    
    # Let $P$ be the permutation such that $U_{4x4}[G, r] = (U_{2x2} \otimes U_{2x2})[K, r]$ where $K$ is derived from $G$.
    # Actually, it's easier to construct the 49 terms directly using the block structure, as the current code does, but cleaner.
    # Or, we can apply the permutation to the rows of the Kronecker product.
    
    # Let's find the mapping from Global Index $G$ to Kronecker Index $K$.
    # $G = 8*bi + 4*ri + 2*bj + rj$
    # $K = 8*bi + 4*bj + 2*ri + rj$
    
    # We can iterate $G$ from 0 to 15, extract bits, compute $K$, and build the permutation.
    
    U_kron = np.kron(U_2x2, U_2x2) # Shape (16, 49)
    V_kron = np.kron(V_2x2, V_2x2)
    W_kron = np.kron(W_2x2, W_2x2)
    
    # Permutation for U and V (Input matrices)
    # For W, the index convention is W[k*4+i, r].
    # C[i, k] = sum_j A[i, j] B[j, k].
    # C is 4x4.
    # Block structure of C:
    # C = [[C00, C01], [C10, C11]]
    # Cij corresponds to i_block=bi, k_block=bk.
    # Internal indices: i_internal=ri, k_internal=rk.
    # Global C index in W is $W_{idx} = k*4 + i$.
    # $k = 2*bk + rk$
    # $i = 2*bi + ri$
    # $W_{idx} = (2*bk + rk)*4 + (2*bi + ri) = 8*bk + 4*rk + 2*bi + ri$.
    
    # The W_kron index $K_w$ from np.kron(W_2x2, W_2x2):
    # The "block" index in the 2x2 result matrix (which is C viewed as 2x2 blocks) is $b_c = bi*2 + bk$?
    # Wait, the result of 2x2 Strassen is a 2x2 matrix of blocks.
    # The elements of the 2x2 result matrix are the blocks C00, C01, C10, C11.
    # In the 2x2 Strassen formula, the result is a 2x2 matrix.
    # Element (0,0) is C00. Element (0,1) is C01.
    # So the block index $b_c = bi*2 + bk$.
    # The internal index $r_c = ri*2 + rk$.
    # $K_w = b_c*4 + r_c = (bi*2 + bk)*4 + (ri*2 + rk) = 8*bi + 4*bk + 2*ri + rk$.
    
    # We need to map $W_{idx}$ to $K_w$.
    # $W_{idx} = 8*bk + 4*rk + 2*bi + ri$
    # $K_w = 8*bi + 4*bk + 2*ri + rk$
    
    # We will construct the permuted matrices.
    
    def get_permutation(n_bits=4):
        perm = [0] * (1 << n_bits)
        for idx in range(1 << n_bits):
            # Extract bits
            b0 = (idx >> 0) & 1
            b1 = (idx >> 1) & 1
            b2 = (idx >> 2) & 1
            b3 = (idx >> 3) & 1
            
            # For U/V:
            # G = 8*bi + 4*ri + 2*bj + rj
            # b3=bi, b2=ri, b1=bj, b0=rj
            bi, ri, bj, rj = b3, b2, b1, b0
            K = 8*bi + 4*bj + 2*ri + rj
            perm[idx] = K
            
        return perm
        
    perm_UV = get_permutation()
    
    # For W:
    # W_idx = 8*bk + 4*rk + 2*bi + ri
    # b3=bk, b2=rk, b1=bi, b0=ri
    def get_permutation_W():
        perm = [0] * 16
        for idx in range(16):
            b0 = (idx >> 0) & 1
            b1 = (idx >> 1) & 1
            b2 = (idx >> 2) & 1
            b3 = (idx >> 3) & 1
            
            bk, rk, bi, ri = b3, b2, b1, b0
            K_w = 8*bi + 4*bk + 2*ri + rk
            perm[idx] = K_w
        return perm
        
    perm_W = get_permutation_W()
    
    # Apply permutation
    # U_final[G, r] = U_kron[K, r] where K = perm_UV[G]
    U = U_kron[perm_UV, :]
    V = V_kron[perm_UV, :]
    W = W_kron[perm_W, :]
    
    # Convert to lists of lists of floats/ints
    # The problem asks for integer/half-integer entries.
    # np.kron of integers is integers.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
