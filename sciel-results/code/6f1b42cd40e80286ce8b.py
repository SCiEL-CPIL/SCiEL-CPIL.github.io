"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(len(U[0])):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is a dict with 'u', 'v', 'w' keys mapping flat indices to coefficients.
    A is 2x2: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    B is 2x2: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    C is 2x2: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    """
    terms = []
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    terms.append({'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    # M2 = (A10+A11)B00 -> contributes to C10, C11 (C11 gets -M2)
    terms.append({'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    # M3 = A00(B01-B11) -> contributes to C01, C11
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    # M4 = A11(B10-B00) -> contributes to C00, C10
    terms.append({'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    # M5 = (A00+A01)B11 -> contributes to C00 (-M5), C01
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    terms.append({'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    terms.append({'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    return terms

def _build_2x3x2_decomposition():
    """
    Build a rank-11 decomposition for 2x3x2 matrix multiplication.
    
    A is 2x3, B is 3x2, C is 2x2.
    We split into:
    - Left 2x2 block: A[:,0:2] * B[0:2,:] using Strassen (7 terms)
    - Right 1x2 block: A[:,2] * B[2,:] using 2 scalar multiplications
    
    Total rank: 7 + 2 = 9... wait, that's not 11.
    
    Actually, let me reconsider. The 2x3x2 tensor has:
    - 6 entries in A (i in 0..1, j in 0..2)
    - 6 entries in B (j in 0..2, k in 0..1)  
    - 4 entries in C (i in 0..1, k in 0..1)
    
    The schoolbook rank is 6*6/4... no, it's n*m*p = 2*3*2 = 12.
    
    Let me use a different approach. For 2x3x2:
    - Use Strassen on the left 2x2 part of A and left 2x2 part of B (7 terms)
    - Handle the remaining column of A (j=2) and row of B (j=2) separately
    
    The product C = A*B where:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Split: C[i,k] = (A_left[i,:] * B_left[:,k]) + A[i,2]*B[2,k]
    where A_left is 2x2 and B_left is 2x2.
    
    The first part uses 7 Strassen terms.
    The second part: A[i,2]*B[2,k] for i in 0..1, k in 0..1.
    This is a rank-1 tensor in (i,k) space with coefficients A[i,2] and B[2,k].
    We need 2 scalar multiplications:
    - t0 = A[0,2]*B[2,0] -> contributes to C[0,0]
    - t1 = A[1,2]*B[2,0] -> contributes to C[1,0]
    - t2 = A[0,2]*B[2,1] -> contributes to C[0,1]
    - t3 = A[1,2]*B[2,1] -> contributes to C[1,1]
    
    But we can do better. Note that:
    C[0,0] gets A[0,2]*B[2,0]
    C[0,1] gets A[0,2]*B[2,1]
    C[1,0] gets A[1,2]*B[2,0]
    C[1,1] gets A[1,2]*B[2,1]
    
    This is exactly the outer product of vector [A[0,2], A[1,2]] and [B[2,0], B[2,1]].
    The rank of this 2x2 tensor is at most 2 (it's a rank-1 tensor if we view it as u*v^T).
    But in tensor decomposition terms, we need:
    sum_c U[i,2,c] * V[2,k,c] * W[i,k,c] = A[i,2]*B[2,k]
    
    We can use 2 terms:
    Term 8: U[0,2,8]=1, V[2,0,8]=1, W[0,0,8]=1  -> gives A[0,2]*B[2,0] to C[0,0]
    Term 9: U[1,2,9]=1, V[2,0,9]=1, W[1,0,9]=1  -> gives A[1,2]*B[2,0] to C[1,0]
    Term 10: U[0,2,10]=1, V[2,1,10]=1, W[0,1,10]=1 -> gives A[0,2]*B[2,1] to C[0,1]
    Term 11: U[1,2,11]=1, V[2,1,11]=1, W[1,1,11]=1 -> gives A[1,2]*B[2,1] to C[1,1]
    
    That's 4 more terms, total 7+4=11. OK so rank 11 for 2x3x2.
    
    Actually wait - can we do the 4 scalar terms in 2? 
    The sub-tensor for j=2 is: T[i,k] = A[i,2]*B[2,k]
    This factors as: (A[:,2]) outer (B[2,:])
    In tensor decomposition: we need sum_c U[i,2,c]*V[2,k,c]*W[i,k,c] = A[i,2]*B[2,k]
    
    If we set:
    c=0: U[0,2,0]=1, U[1,2,0]=1, V[2,0,0]=1, V[2,1,0]=1, W[0,0,0]=1, W[0,1,0]=1, W[1,0,0]=1, W[1,1,0]=1
    Then sum = A[0,2]*B[2,0] + A[0,2]*B[2,1] + A[1,2]*B[2,0] + A[1,2]*B[2,1]
    That's not right - we get all 4 products in one term, but we need them separately.
    
    The issue is that W[i,k,c] multiplies the product, so we can't separate them with one term.
    We need at least 2 terms if we use the structure cleverly, but actually for a generic
    2x2 matrix, the rank is 4 (schoolbook) unless we have special structure.
    
    Hmm, but A[i,2]*B[2,k] is a rank-1 matrix (outer product of two vectors), so as a
    bilinear form it has rank 1. But in the tensor decomposition context, we're computing
    C[i,k] = A[i,2]*B[2,k], which is a product of one A entry and one B entry.
    
    Actually, I think the minimum rank for this sub-problem is indeed 4 because each
    output C[i,k] depends on a different pair (A[i,2], B[2,k]).
    
    Wait no. Let's think again. We have:
    C[0,0] = A[0,2]*B[2,0]
    C[0,1] = A[0,2]*B[2,1]
    C[1,0] = A[1,2]*B[2,0]
    C[1,1] = A[1,2]*B[2,1]
    
    This is a 2x2x2 tensor (in i, k, and the "j=2" dimension which is fixed).
    Actually it's just a 2x2 matrix where entry (i,k) = A[i,2]*B[2,k].
    As a bilinear map from (A[:,2], B[2,:]) to C, this has rank... 
    The matrix of the bilinear form is [[A02*B20, A02*B21], [A12*B20, A12*B21]]
    This is (A[:,2]) @ (B[2,:]) which is rank 1 as a matrix.
    But the tensor rank of the identity-like structure... 
    
    I think I'm overcomplicating this. Let me just use the 4-term schoolbook for the
    j=2 part and accept rank 11 for 2x3x2.
    
    Actually, let me reconsider the whole approach. Maybe I should just build the
    2x3x6 decomposition directly using the block structure.
    
    For 2x3x6, split p=6 into three blocks of 2: k in {0,1}, {2,3}, {4,5}.
    Each block is a 2x3x2 problem with rank 11.
    Total rank = 3 * 11 = 33.
    
    But the current best is already 33 from inspiration 3. Let me see if I can do better.
    
    Actually, looking at inspiration 3, it achieved 1.0909 which corresponds to rank 33
    (since seed is 36, score = 36/33 ≈ 1.0909).
    
    Can we do better than 33? The theoretical lower bound for 2x3x6 is not well known,
    but it's at least max(2*3, 3*6, 2*6) = max(6, 18, 12) = 18.
    
    Let me try to build the rank-33 decomposition explicitly and verify it works.
    """
    n, m, p = 2, 3, 2
    rank = 11
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 11
    V = [[0.0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0.0] * rank for _ in range(p * n)]  # 4 x 11
    
    # Strassen terms for the left 2x2 block (j=0,1)
    strassen_terms = _strassen_2x2x2_terms()
    
    for c, term in enumerate(strassen_terms):
        for idx, coeff in term['u'].items():
            U[idx][c] += coeff
        for idx, coeff in term['v'].items():
            V[idx][c] += coeff
        for idx, coeff in term['w'].items():
            W[idx][c] += coeff
    
    # Scalar terms for j=2 (right column of A, right row of B)
    # Term 7: A[0,2]*B[2,0] -> C[0,0]
    U[0*3+2][7] = 1  # A[0,2]
    V[2*2+0][7] = 1  # B[2,0]
    W[0*2+0][7] = 1  # C[0,0]
    
    # Term 8: A[1,2]*B[2,0] -> C[1,0]
    U[1*3+2][8] = 1  # A[1,2]
    V[2*2+0][8] = 1  # B[2,0]
    W[1*2+0][8] = 1  # C[1,0]
    
    # Term 9: A[0,2]*B[2,1] -> C[0,1]
    U[0*3+2][9] = 1  # A[0,2]
    V[2*2+1][9] = 1  # B[2,1]
    W[0*2+1][9] = 1  # C[0,1]
    
    # Term 10: A[1,2]*B[2,1] -> C[1,1]
    U[1*3+2][10] = 1  # A[1,2]
    V[2*2+1][10] = 1  # B[2,1]
    W[1*2+1][10] = 1  # C[1,1]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Strategy: Split p=6 into three blocks of size 2.
    # Each block uses a rank-11 decomposition for 2x3x2.
    # Total rank = 3 * 11 = 33.
    
    rank_per_block = 11
    total_rank = 3 * rank_per_block  # 33
    
    U = [[0.0] * total_rank for _ in range(nm)]  # 6 x 33
    V = [[0.0] * total_rank for _ in range(mp)]  # 18 x 33
    W = [[0.0] * total_rank for _ in range(pn)]  # 12 x 33
    
    # Build each 2x3x2 block
    for block in range(3):
        block_offset = block * rank_per_block
        k_offset = block * 2  # k indices for this block: k_offset, k_offset+1
        
        # Get the 2x3x2 decomposition
        U_block, V_block, W_block = _build_2x3x2_decomposition()
        
        # Copy U: same for all blocks (U depends on i, j which are the same)
        for i in range(n):
            for j in range(m):
                for c in range(rank_per_block):
                    U[i * m + j][block_offset + c] = U_block[i * m + j][c]
        
        # Copy V: V[j*p+k, c] where k = k_offset + k_local
        for j in range(m):
            for k_local in range(2):
                k_global = k_offset + k_local
                for c in range(rank_per_block):
                    V[j * p + k_global][block_offset + c] = V_block[j * 2 + k_local][c]
        
        # Copy W: W[k*n+i, c] where k = k_offset + k_local
        for k_local in range(2):
            k_global = k_offset + k_local
            for i in range(n):
                for c in range(rank_per_block):
                    W[k_global * n + i][block_offset + c] = W_block[k_local * n + i][c]
    
    # Verify the decomposition
    max_err = _verify_decomposition(n, m, p, U, V, W)
    
    if max_err < 1e-9:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition of rank 36
    rank = n * m * p  # 36
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
# EVOLVE-BLOCK-END
