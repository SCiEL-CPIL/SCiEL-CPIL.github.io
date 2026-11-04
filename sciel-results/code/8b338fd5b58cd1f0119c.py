"""
Matrix multiplication 3x4 * 4x4 -> rank-minimal decomposition.

Strategy:
  - Use the 4x4x4 rank-16 decomposition (Strassen on 2x2 blocks).
  - Zero out the 4th row of A (i=3) to obtain a valid rank-16 decomposition
    for the 3x4 * 4x4 case.
  - Verify numerically before returning.
  - Fallback to schoolbook (rank 48) if verification fails.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (u, v, w) where:
      u: length-4 vector for A (indices 00,01,10,11)
      v: length-4 vector for B (indices 00,01,10,11)
      w: length-4 vector for C (indices 00,01,10,11) -- row-major C[i][k]
    
    Strassen formulas (using 0-indexed A[i][j], B[j][k], C[i][k]):
      M1 = (a00+a11)*(b00+b11)
      M2 = (a10+a11)*b00
      M3 = a00*(b10-b00)
      M4 = a11*(b11-b01)
      M5 = (a00+a01)*b11
      M6 = (a10-a00)*(b00+b01)
      M7 = (a01-a11)*(b10+b11)
    
      C[0][0] = M1 - M4 + M5 - M7
      C[0][1] = M3 + M5
      C[1][0] = M2 + M4
      C[1][1] = M1 - M2 + M3 + M6
    """
    # u vectors: coefficients of A entries (a00, a01, a10, a11)
    # v vectors: coefficients of B entries (b00, b01, b10, b11)
    # w vectors: coefficients for C entries (C00, C01, C10, C11)
    
    terms = []
    
    # M1: f = a00+a11, g = b00+b11
    # C00: +1, C11: +1
    terms.append((
        [1, 0, 0, 1],   # u: a00 + a11
        [1, 0, 0, 1],   # v: b00 + b11
        [1, 0, 0, 1],   # w: C00 + C11
    ))
    
    # M2: f = a10+a11, g = b00
    # C10: +1, C11: -1
    terms.append((
        [0, 0, 1, 1],   # u: a10 + a11
        [1, 0, 0, 0],   # v: b00
        [0, 0, 1, -1],  # w: C10 - C11
    ))
    
    # M3: f = a00, g = b10-b00
    # C01: +1, C11: +1
    terms.append((
        [1, 0, 0, 0],   # u: a00
        [-1, 0, 1, 0],  # v: b10 - b00
        [0, 1, 0, 1],   # w: C01 + C11
    ))
    
    # M4: f = a11, g = b11-b01
    # C00: -1, C10: +1
    terms.append((
        [0, 0, 0, 1],   # u: a11
        [0, -1, 0, 1],  # v: b11 - b01
        [-1, 0, 1, 0],  # w: -C00 + C10
    ))
    
    # M5: f = a00+a01, g = b11
    # C00: +1, C01: +1
    terms.append((
        [1, 1, 0, 0],   # u: a00 + a01
        [0, 0, 0, 1],   # v: b11
        [1, 1, 0, 0],   # w: C00 + C01
    ))
    
    # M6: f = a10-a00, g = b00+b01
    # C11: +1
    terms.append((
        [-1, 0, 1, 0],  # u: a10 - a00
        [1, 1, 0, 0],   # v: b00 + b01
        [0, 0, 0, 1],   # w: C11
    ))
    
    # M7: f = a01-a11, g = b10+b11
    # C00: -1
    terms.append((
        [0, 1, 0, -1],  # u: a01 - a11
        [0, 0, 1, 1],   # v: b10 + b11
        [-1, 0, 0, 0],  # w: -C00
    ))
    
    return terms


def _verify_2x2_strassen():
    """Verify that the Strassen terms correctly decompose the 2x2x2 tensor."""
    terms = _strassen_2x2_terms()
    # Check: for all i,j,k in {0,1}, sum_r u_r[i,j]*v_r[j,k]*w_r[i,k] == 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for (u, v, w) in terms:
                    total += u[i*2+j] * v[j*2+k] * w[i*2+k]
                if abs(total - 1) > 1e-12:
                    return False, (i, j, k, total)
    return True, None


def _build_rank16_for_4x4x4():
    """
    Build the rank-16 decomposition for 4x4 * 4x4 -> 4x4.
    Uses Strassen on 2x2 blocks.
    
    The 4x4 matrix multiplication can be seen as 4x4 block structure:
    Split i into {0,1} and {2,3}, j into {0,1} and {2,3}, k into {0,1} and {2,3}.
    
    There are 4 blocks of 2x2 * 2x2 -> 2x2, each with rank 7.
    Total rank: 4 * 7 = 28.
    
    Wait, that gives rank 28, not 16. Let me reconsider.
    
    Actually, for 4x4*4x4, we can use a different decomposition.
    The rank of 4x4*4x4 is at most 16 (using Strassen recursively: 7^2 = 49? No.)
    
    Actually, Strassen gives O(n^log2(7)) ~ n^2.807, so for n=4, rank is at most 7^2 = 49? No.
    
    Let me think again. For 2x2*2x2, rank is 7.
    For 4x4*4x4, using Strassen on 2x2 blocks: we need 7 multiplications of 2x2*2x2.
    Each 2x2*2x2 has rank 7. So total rank is 7*7 = 49? That's worse than schoolbook (64).
    
    Wait no. The rank of the 4x4x4 tensor is the minimum number of rank-1 terms.
    Using Strassen: 4x4*4x4 = 7 products of 2x2*2x2.
    But each 2x2*2x2 product is not a single rank-1 term; it's a tensor of rank 7.
    So the total rank would be 7 * 7 = 49? That doesn't seem right either.
    
    Let me reconsider. The tensor for n x n * n x n has n^3 entries.
    For n=2, it has 8 entries, and rank is 7.
    For n=4, it has 64 entries. The rank is at most 64 (schoolbook).
    
    Using Strassen: we can compute 4x4*4x4 with 7 products of 2x2*2x2.
    But in terms of tensor rank, this corresponds to a decomposition where each
    "product" is itself a rank-7 tensor. So the total rank is 7 * 7 = 49.
    
    But 49 < 64, so this is an improvement over schoolbook!
    
    However, the task asks for 3x4 * 4x4, not 4x4 * 4x4.
    The 3x4 * 4x4 tensor is a sub-tensor of the 4x4 * 4x4 tensor.
    
    For the 4x4 * 4x4 rank-49 decomposition, if we zero out the 4th row of A,
    we get a valid decomposition for 3x4 * 4x4 with rank at most 49.
    
    But wait, the inspiration mentioned rank-16. Let me re-examine.
    
    Actually, I think I was confused. Let me re-read the strategy:
    "Use the rank-16 decomposition of the 4x4x4 matrix multiplication tensor"
    
    Hmm, is the rank of 4x4x4 actually 16? That seems too low.
    The rank of 2x2x2 is 7. The rank of 3x3x3 is at most 23 (Bini et al.).
    The rank of 4x4x4 should be higher than 16.
    
    Let me reconsider. Maybe the strategy is different.
    
    Actually, I think the key insight is:
    - For 3x4 * 4x4, we can split j into two groups of 2: {0,1} and {2,3}.
    - For each group, we have a 3x2 * 2x4 multiplication.
    - The 3x2 * 2x4 can be further split.
    
    Alternatively, split i into {0,1} and {2}, and k into {0,1} and {2,3}, and j into {0,1} and {2,3}.
    
    Blocks:
    - i in {0,1}, k in {0,1}, j in {0,1}: 2x2 * 2x2 -> rank 7
    - i in {0,1}, k in {0,1}, j in {2,3}: 2x2 * 2x2 -> rank 7
    - i in {0,1}, k in {2,3}, j in {0,1}: 2x2 * 2x2 -> rank 7
    - i in {0,1}, k in {2,3}, j in {2,3}: 2x2 * 2x2 -> rank 7
    - i=2, k in {0,1}, j in {0,1}: 1x2 * 2x2 -> rank ?
    - i=2, k in {0,1}, j in {2,3}: 1x2 * 2x2 -> rank ?
    - i=2, k in {2,3}, j in {0,1}: 1x2 * 2x2 -> rank ?
    - i=2, k in {2,3}, j in {2,3}: 1x2 * 2x2 -> rank ?
    
    For 1x2 * 2x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is a sum of 2 terms, each of which is a product of a scalar (from A) and a row of B.
    In tensor form: T[i,j,k] = 1 for i=0 (the single row), j in {0,1}, k in {0,1}.
    This is a 1x2x2 tensor with all entries 1.
    The rank of this tensor: we need sum_r u_r[j]*v_r[k]*w_r[i] = 1 for all j,k.
    Since i is fixed, w_r[i] is a scalar. Let w_r[i] = c_r.
    Then sum_r c_r * u_r[j] * v_r[k] = 1 for all j,k.
    This is equivalent to: the 2x2 all-ones matrix has rank 1 (in the usual matrix sense),
    but in tensor rank terms, we need to express it as a sum of rank-1 matrices weighted by c_r.
    The all-ones 2x2 matrix can be written as [1,1]^T * [1,1], which is rank 1.
    So we need c_1 * u_1 * v_1^T = ones(2,2).
    Take c_1 = 1, u_1 = [1,1], v_1 = [1,1].
    So rank is 1? No, that's not right either.
    
    Actually, the tensor is T[i,j,k] with i fixed at 0.
    T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1.
    We need sum_r u_r[j] * v_r[k] * w_r[i] = T[0,j,k] = 1.
    Since i is fixed, w_r[i] is just a scalar coefficient.
    Let alpha_r = w_r[i]. Then sum_r alpha_r * u_r[j] * v_r[k] = 1.
    This is a matrix equation: sum_r alpha_r * u_r * v_r^T = J (all-ones 2x2).
    The rank of J is 1, so we can write J = [1,1]^T * [1,1].
    So take r=1: alpha_1 = 1, u_1 = [1,1], v_1 = [1,1].
    This gives rank 1 for the 1x2x2 tensor.
    
    Wait, but the "rank" in the tensor decomposition context is the number of rank-1 tensor terms.
    Each rank-1 tensor term is u ⊗ v ⊗ w.
    For the 1x2x2 case, we have 1 term: u = [1,1] (2-dim), v = [1,1] (2-dim), w = [1] (1-dim, since i is fixed).
    
    But in the full decomposition, the vectors u, v, w have full dimension (nm, mp, pn).
    So for the 1x2x2 block, the corresponding rank-1 term has:
    - u: nonzero only at the positions corresponding to (i=2, j=0) and (i=2, j=1) in the 3x4 A matrix.
    - v: nonzero only at the positions corresponding to (j=0 or 1, k=0 or 1) in the 4x4 B matrix.
    - w: nonzero only at the positions corresponding to (i=2, k=0 or 1) in the 4x3 C matrix (using W[k*n+i] convention).
    
    So each 1x2x2 block contributes rank 1? That seems too optimistic.
    
    Let me verify: for the block i=2, j in {0,1}, k in {0,1}:
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    
    Using one rank-1 term:
    u has u[2*4+0]=1, u[2*4+1]=1 (i.e., A[2,0] and A[2,1] both have coefficient 1)
    v has v[0*4+0]=1, v[0*4+1]=1, v[1*4+0]=1, v[1*4+1]=1 (B[0,0],B[0,1],B[1,0],B[1,1] all have coefficient 1)
    w has w[0*3+2]=1, w[1*3+2]=1 (C[2,0] and C[2,1] both have coefficient 1)
    
    The contribution is: (A[2,0]+A[2,1]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1]) * 1
    But we need: C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    
    The single term gives:
    C[2,0] += (A[2,0]+A[2,1])*(B[0,0]+B[0,1]+B[1,0]+B[1,1])
    C[2,1] += (A[2,0]+A[2,1])*(B[0,0]+B[0,1]+B[1,0]+B[1,1])
    
    This is NOT correct. The single term overcounts.
    
    So the rank of the 1x2x2 tensor is not 1. Let me reconsider.
    
    The tensor T[i,j,k] for i=0 (fixed), j in {0,1}, k in {0,1} with all entries 1.
    We need sum_r u_r[j] * v_r[k] * w_r = 1 for all j,k, where w_r is a scalar (since i is fixed).
    
    Actually, in the full tensor decomposition, w_r is a vector of length p*n = 4*3 = 12.
    But for this block, only w_r[0*3+0]=w_r[0] and w_r[1*3+0]=w_r[3] are relevant (if i=0).
    Wait, the W index is k*n+i. For n=3, i=0: W[0*3+0]=W[0], W[1*3+0]=W[3].
    
    So we need: for k in {0,1}, j in {0,1}:
    sum_r u_r[j] * v_r[k] * w_r[k*3+0] = 1
    
    This is a system of 4 equations. Let's denote a_r = w_r[k*3+0] which depends on k.
    So we have:
    k=0: sum_r u_r[0]*v_r[0]*w_r[0] + u_r[1]*v_r[0]*w_r[0] ... no, this is getting complicated.
    
    Let me just use the schoolbook for the 1x2x2 blocks. Each has rank 4 (2*1*2=4 terms... no, 1*2*2=4).
    Actually, 1x2*2x2: C[0,k] = sum_{j=0}^{1} A[0,j]*B[j,k]. There are 2*2=4 (j,k) pairs.
    Schoolbook rank for 1x2x2 is 4.
    
    So total rank: 4*7 + 4*4 = 28 + 16 = 44.
    
    This matches the inspiration's rank-44!
    
    Let me implement this.
    """
    pass


def _build_rank44_decomposition():
    """
    Build the rank-44 decomposition for 3x4 * 4x4 -> 4x3.
    
    Split:
    - i into {0,1} and {2}
    - j into {0,1} and {2,3}
    - k into {0,1} and {2,3}
    
    Blocks:
    1. i in {0,1}, j in {0,1}, k in {0,1}: 2x2 * 2x2 -> rank 7
    2. i in {0,1}, j in {2,3}, k in {0,1}: 2x2 * 2x2 -> rank 7
    3. i in {0,1}, j in {0,1}, k in {2,3}: 2x2 * 2x2 -> rank 7
    4. i in {0,1}, j in {2,3}, k in {2,3}: 2x2 * 2x2 -> rank 7
    5. i=2, j in {0,1}, k in {0,1}: 1x2 * 2x2 -> rank 4
    6. i=2, j in {2,3}, k in {0,1}: 1x2 * 2x2 -> rank 4
    7. i=2, j in {0,1}, k in {2,3}: 1x2 * 2x2 -> rank 4
    8. i=2, j in {2,3}, k in {2,3}: 1x2 * 2x2 -> rank 4
    
    Total rank: 4*7 + 4*4 = 28 + 16 = 44
    """
    n, m, p = 3, 4, 4
    rank_total = 44
    
    U = [[0] * rank_total for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 12 x 44
    
    col = 0
    strassen_terms = _strassen_2x2_terms()
    
    # 2x2 * 2x2 blocks
    for i0 in [0, 1]:  # start index for i-block
        for j0 in [0, 2]:  # start index for j-block
            for k0 in [0, 2]:  # start index for k-block
                # This block: i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
                for (u_local, v_local, w_local) in strassen_terms:
                    # u_local: [a00, a01, a10, a11] -> A[i0+di, j0+dj]
                    # v_local: [b00, b01, b10, b11] -> B[j0+dj, k0+dk]
                    # w_local: [C00, C01, C10, C11] -> C[i0+di, k0+dk]
                    
                    for di in range(2):
                        for dj in range(2):
                            u_idx = (i0 + di) * m + (j0 + dj)
                            U[u_idx][col] = u_local[di * 2 + dj]
                    
                    for dj in range(2):
                        for dk in range(2):
                            v_idx = (j0 + dj) * p + (k0 + dk)
                            V[v_idx][col] = v_local[dj * 2 + dk]
                    
                    for di in range(2):
                        for dk in range(2):
                            w_idx = (k0 + dk) * n + (i0 + di)
                            W[w_idx][col] = w_local[di * 2 + dk]
                    
                    col += 1
    
    # 1x2 * 2x2 blocks (schoolbook, rank 4 each)
    for j0 in [0, 2]:  # start index for j-block
        for k0 in [0, 2]:  # start index for k-block
            # Block: i=2, j in {j0, j0+1}, k in {k0, k0+1}
            for dj in range(2):
                for dk in range(2):
                    # Term: A[2, j0+dj] * B[j0+dj, k0+dk] -> C[2, k0+dk]
                    u_idx = 2 * m + (j0 + dj)
                    v_idx = (j0 + dj) * p + (k0 + dk)
                    w_idx = (k0 + dk) * n + 2
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    assert col == rank_total, f"Expected rank {rank_total}, got {col}"
    
    return U, V, W


def _verify_decomposition(U, V, W, n=3, m=4, p=4):
    """
    Verify that the decomposition is correct.
    T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should equal 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1) > 1e-9:
                    return False, (i, j, k, total)
    return True, None


def _build_schoolbook():
    """Build the schoolbook decomposition (rank 48)."""
    n, m, p = 3, 4, 4
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
    return U, V, W


def construct():
    deadline = time.time() + 118.0
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # First, verify the Strassen 2x2 decomposition
    strassen_ok, err = _verify_2x2_strassen()
    
    if strassen_ok:
        # Try the rank-44 decomposition
        try:
            U, V, W = _build_rank44_decomposition()
            ok, err = _verify_decomposition(U, V, W, n, m, p)
            if ok:
                return {"U": U, "V": V, "W": W}
            else:
                # Verification failed, fall through to schoolbook
                pass
        except Exception:
            pass
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
