"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2_tensors():
    """
    Return the 7 rank-1 terms of Strassen's 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 4-vectors (indices 00,01,10,11).
    The tensor T = sum_r u_r ⊗ v_r ⊗ w_r equals the matmul tensor.
    
    Convention: A is 2x2 with indices A[i,j] stored at position i*2+j
                B is 2x2 with indices B[j,k] stored at position j*2+k  
                C is 2x2 with indices C[i,k] stored at position k*2+i (trace convention W[k*n+i])
    
    Wait, let me re-read the convention: W[k*n+i, r]
    So for 2x2x2: W has shape (p*n, rank) = (2*2, rank) = (4, rank)
    W index: k*n + i where k is column of B / row of C, i is row of A / col of C
    
    Actually for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] = A[i,j] * B[j,k] * C[i,k]
    
    U has shape (n*m, rank): U[i*m+j, r]
    V has shape (m*p, rank): V[j*p+k, r]
    W has shape (p*n, rank): W[k*n+i, r]
    
    For 2x2x2:
    U indices: i*2+j for i,j in {0,1}
    V indices: j*2+k for j,k in {0,1}
    W indices: k*2+i for k,i in {0,1}
    
    Strassen's 7 products:
    M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    M2 = (A10+A11)*B00       -> contributes to C10, C11
    M3 = A00*(B10-B11)       -> contributes to C01, C11
    M4 = A11*(B01-B00)       -> contributes to C00, C10
    M5 = (A00+A01)*B11       -> contributes to C00, C01
    M6 = (A10-A00)*(B00+B01) -> contributes to C11
    M7 = (A01-A11)*(B10+B11) -> contributes to C00
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # For each of the 7 Strassen products, define u, v, w vectors of length 4
    # u: coefficients for A entries (indices i*2+j)
    # v: coefficients for B entries (indices j*2+k)
    # w: coefficients for C entries (indices k*2+i)
    
    terms = []
    
    # M1 = (A00+A11)*(B00+B11)
    # u: A00=1 (idx 0), A11=1 (idx 3)
    # v: B00=1 (idx 0), B11=1 (idx 3)
    # w: C00=1 (idx 0*2+0=0), C11=1 (idx 1*2+1=3)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A10+A11)*B00
    # u: A10=1 (idx 2), A11=1 (idx 3)
    # v: B00=1 (idx 0)
    # w: C10=1 (idx 0*2+1=2), C11=-1 (idx 1*2+1=3)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # M3 = A00*(B10-B11)
    # u: A00=1 (idx 0)
    # v: B10=1 (idx 2), B11=-1 (idx 3)
    # w: C01=1 (idx 1*2+0=2), C11=1 (idx 1*2+1=3)
    terms.append(([1,0,0,0], [0,0,1,-1], [0,0,1,1]))
    
    # M4 = A11*(B01-B00)
    # u: A11=1 (idx 3)
    # v: B01=1 (idx 1), B00=-1 (idx 0)
    # w: C00=1 (idx 0*2+0=0), C10=1 (idx 0*2+1=2)
    terms.append(([0,0,0,1], [-1,1,0,0], [1,0,1,0]))
    
    # M5 = (A00+A01)*B11
    # u: A00=1 (idx 0), A01=1 (idx 1)
    # v: B11=1 (idx 3)
    # w: C00=-1 (idx 0*2+0=0), C01=1 (idx 1*2+0=2)
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    
    # M6 = (A10-A00)*(B00+B01)
    # u: A10=1 (idx 2), A00=-1 (idx 0)
    # v: B00=1 (idx 0), B01=1 (idx 1)
    # w: C11=1 (idx 1*2+1=3)
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    
    # M7 = (A01-A11)*(B10+B11)
    # u: A01=1 (idx 1), A11=-1 (idx 3)
    # v: B10=1 (idx 2), B11=1 (idx 3)
    # w: C00=1 (idx 0*2+0=0)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms

def _verify_strassen_2x2x2():
    """Verify that the Strassen 2x2x2 tensors correctly decompose the matmul tensor."""
    terms = _strassen_2x2x2_tensors()
    # Build the tensor and verify
    T = np.zeros((4, 4, 4))  # T[i*2+j, j*2+k, k*2+i]
    for u, v, w in terms:
        T += np.einsum('a,b,c->abc', u, v, w)
    
    # Expected: T[i*2+j, j*2+k, k*2+i] = 1 if the indices are consistent
    # Actually the matmul tensor T[i,j,k] = 1 for all i,j,k (it's the identity in tensor space)
    # No wait. The tensor for matrix multiplication is:
    # T[i,j,k] = delta_{i,j} * delta_{j,k} ... no that's not right either.
    # 
    # The matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * (coefficient)
    # where the coefficient is 1 if the indices form a valid matmul term.
    # 
    # Actually, the standard matmul tensor for n x m x p is:
    # T[i,j,k] = 1 for all valid i,j,k (it's the "all ones" tensor in the appropriate space)
    # No, that's the schoolbook. The tensor is:
    # T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # So as a tensor, T[i,j,k] = 1 for all i,j,k.
    
    # Let me verify differently: check that sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k
    n = 2
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0
                for u, v, w in terms:
                    s += u[i*n+j] * v[j*n+k] * w[k*n+i]
                if s != 1:
                    print(f"FAIL: i={i}, j={j}, k={k}, sum={s}")
                    return False
    return True

def _kronecker_strassen_4x4x4():
    """
    Generate the 49 rank-1 terms for 4x4x4 matrix multiplication
    via Kronecker product of two Strassen 2x2x2 decompositions.
    
    A is 4x4, B is 4x4, C is 4x4.
    Block structure: A = [[A00, A01], [A10, A11]] where each block is 2x2.
    Similarly for B and C.
    
    C = A * B in blocks:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Using Strassen on the 2x2 block structure:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B10-B11)
    M4 = A11*(B01-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each M_r is a 2x2 * 2x2 = 2x2 multiplication, which itself uses Strassen (rank 7).
    So each M_r contributes 7 rank-1 terms, total 7*7 = 49 terms.
    
    For the Kronecker product:
    If M_r = sum_{s=1}^{7} u_s ⊗ v_s ⊗ w_s (each 4-vectors for 2x2x2)
    And the outer Strassen gives coefficient c_{r,block} for block position,
    then the full 4x4x4 term is:
    For outer term r and inner term s:
    u_full = c_{r,block_u} * (u_s ⊗ u_s)  ... this needs careful index mapping.
    
    Actually, let me think about this more carefully.
    
    The 4x4 matrix A has entries A[i,j] for i,j in 0..3.
    Block A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    For the outer Strassen, we have 7 "meta" terms. Each meta term r involves
    linear combinations of the 4 blocks of A, 4 blocks of B, producing 4 blocks of C.
    
    For meta term r, the contribution to C is:
    C += coeff_r * (A_comb_r) * (B_comb_r)
    where A_comb_r is a 4x4 matrix formed by combining the 2x2 blocks,
    and similarly for B_comb_r.
    
    But actually, each meta term is itself a 2x2x2 matmul at the block level,
    where each "entry" is a 2x2 matrix. So we need to expand each block-level
    matmul into scalar-level matmul using the inner Strassen.
    
    This is getting complex. Let me use a different approach:
    directly compute the Kronecker product of the two Strassen decompositions.
    
    For 4x4x4, we can think of it as two independent 2x2x2 problems in the
    row and column dimensions. The Kronecker product of two rank-7 decompositions
    gives a rank-49 decomposition.
    
    Specifically, if T_2 = sum_{r=1}^7 u_r ⊗ v_r ⊗ w_r is the 2x2x2 matmul tensor,
    then T_4 = T_2 ⊗ T_2 = sum_{r,s=1}^7 (u_r ⊗ u_s) ⊗ (v_r ⊗ v_s) ⊗ (w_r ⊗ w_s)
    
    But we need to be careful with the index conventions.
    
    Let me just build it computationally.
    """
    strassen_terms = _strassen_2x2x2_tensors()
    
    # For 4x4x4:
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    # U index: i*4+j for i,j in 0..3
    # V index: j*4+k for j,k in 0..3
    # W index: k*4+i for k,i in 0..3
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for r in range(7):
        u_r, v_r, w_r = strassen_terms[r]
        for s in range(7):
            u_s, v_s, w_s = strassen_terms[s]
            
            # u_full is length 16: u_full[i*4+j] for i,j in 0..3
            # This is the Kronecker product u_r ⊗ u_s
            # But we need to map correctly.
            # 
            # The 4x4 matrix A can be seen as a 2x2 block matrix where each block is 2x2.
            # A[i,j] where i = 2*i_hi + i_lo, j = 2*j_hi + j_lo
            # 
            # The outer Strassen operates on the "hi" indices, inner on "lo" indices.
            # 
            # For the Kronecker product:
            # u_full[i*4+j] = u_r[i_hi*2+j_hi] * u_s[i_lo*2+j_lo]
            # where i = 2*i_hi + i_lo, j = 2*j_hi + j_lo
            # 
            # Wait, this isn't quite right. The Kronecker product of tensors
            # for nested matrix multiplication needs careful index handling.
            # 
            # Let me think of it differently. The 4x4x4 matmul tensor can be
            # factored as:
            # T_4[i,j,k] = sum_{r,s} (u_r[i_hi] * u_s[i_lo]) * (v_r[j_hi] * v_s[j_lo]) * (w_r[k_hi] * w_s[k_lo])
            # where i = 2*i_hi + i_lo, j = 2*j_hi + j_lo, k = 2*k_hi + k_lo
            # 
            # Hmm, but the Strassen decomposition doesn't factor this way simply.
            # 
            # Let me just verify by construction. I'll build the full tensor and check.
            
            u_full = [0] * 16
            v_full = [0] * 16
            w_full = [0] * 16
            
            for i_hi in range(2):
                for j_hi in range(2):
                    for i_lo in range(2):
                        for j_lo in range(2):
                            i = 2 * i_hi + i_lo
                            j = 2 * j_hi + j_lo
                            u_full[i * 4 + j] = u_r[i_hi * 2 + j_hi] * u_s[i_lo * 2 + j_lo]
            
            for j_hi in range(2):
                for k_hi in range(2):
                    for j_lo in range(2):
                        for k_lo in range(2):
                            j = 2 * j_hi + j_lo
                            k = 2 * k_hi + k_lo
                            v_full[j * 4 + k] = v_r[j_hi * 2 + k_hi] * v_s[j_lo * 2 + k_lo]
            
            for k_hi in range(2):
                for i_hi in range(2):
                    for k_lo in range(2):
                        for i_lo in range(2):
                            k = 2 * k_hi + k_lo
                            i = 2 * i_hi + i_lo
                            w_full[k * 4 + i] = w_r[k_hi * 2 + i_hi] * w_s[k_lo * 2 + i_lo]
            
            for idx in range(16):
                U[idx][col] = u_full[idx]
                V[idx][col] = v_full[idx]
                W[idx][col] = w_full[idx]
            
            col += 1
    
    return U, V, W

def _verify_4x4x4(U, V, W):
    """Verify that the 4x4x4 decomposition is correct."""
    n = 4
    rank = len(U[0])
    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0
                for r in range(rank):
                    s += U[i * n + j][r] * V[j * n + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    print(f"4x4x4 FAIL: i={i}, j={j}, k={k}, sum={s}")
                    return False
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # The 4x4 * 4x8 multiplication can be split into two 4x4 * 4x4 multiplications:
    # C_left = A * B_left (columns 0-3 of B)
    # C_right = A * B_right (columns 4-7 of B)
    # 
    # Each 4x4x4 uses rank 49 (Strassen Kronecker), total rank 98.
    
    U4, V4, W4 = _kronecker_strassen_4x4x4()
    
    # Verify the 4x4x4 decomposition
    if not _verify_4x4x4(U4, V4, W4):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now construct the 4x4x8 decomposition using two 4x4x4 blocks
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # shape (16, 98)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 98)
    W = [[0] * rank for _ in range(p * n)]  # shape (32, 98)
    
    # First half: columns 0-3 of B (B_left)
    # U: same as U4 (16 x 49)
    # V: V4 placed in rows 0-15 (j*4+k for k=0..3, so V indices 0..15)
    # W: W4 placed in rows 0-15 (k*4+i for k=0..3, so W indices 0..15)
    
    for idx in range(16):
        for r in range(49):
            U[idx][r] = U4[idx][r]
            V[idx][r] = V4[idx][r]
            W[idx][r] = W4[idx][r]
    
    # Second half: columns 4-7 of B (B_right)
    # U: same as U4 (16 x 49)
    # V: V4 placed in rows 16-31 (j*4+k for k=4..7, so V indices 16..31)
    #    V index = j*4 + k where k = 4 + k', so V index = j*4 + 4 + k' = (j+1)*4 + k'
    #    Wait, V has shape (m*p, rank) = (4*8, rank) = (32, rank)
    #    V index = j*p + k = j*8 + k
    #    For k = 0..3: V index = j*8 + k
    #    For k = 4..7: V index = j*8 + 4 + k' = j*8 + 4 + k'
    #    
    #    But V4 has indices j*4 + k' for j,k' in 0..3
    #    So for the second half, V[j*8 + 4 + k'] = V4[j*4 + k']
    
    # W: W4 placed in rows 16-31 (k*4+i for k=4..7, so W indices 16..31)
    #    W has shape (p*n, rank) = (8*4, rank) = (32, rank)
    #    W index = k*n + i = k*4 + i
    #    For k = 0..3: W index = k*4 + i
    #    For k = 4..7: W index = (4+k')*4 + i = 16 + k'*4 + i
    #    W4 has indices k'*4 + i for k',i in 0..3
    #    So W[16 + k'*4 + i] = W4[k'*4 + i]
    
    for idx in range(16):
        for r in range(49):
            U[idx][49 + r] = U4[idx][r]
    
    for j in range(4):
        for k_prime in range(4):
            v4_idx = j * 4 + k_prime
            v_idx = j * 8 + 4 + k_prime
            for r in range(49):
                V[v_idx][49 + r] = V4[v4_idx][r]
    
    for k_prime in range(4):
        for i in range(4):
            w4_idx = k_prime * 4 + i
            w_idx = (4 + k_prime) * 4 + i
            for r in range(49):
                W[w_idx][49 + r] = W4[w4_idx][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
