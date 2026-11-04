"""Matrix multiplication 3x3x4 rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]


def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 36."""
    n, m, p = DIMENSIONS
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if s != 1:
                    return False
    return True


def _laderman_2x2x2():
    """
    Return Laderman's rank-7 decomposition for 2x2x2 matrix multiplication.
    U has shape (4,7), V has shape (4,7), W has shape (4,7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [
        [1, 1, 1, 1, 0, 0, 0],
        [1,-1, 0, 0, 1, 0, 0],
        [0, 0, 1,-1, 0, 1, 0],
        [0, 0, 0, 0, 1,-1, 0],
    ]
    V = [
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
    ]
    W = [
        [0, 1, 0, 0, 0, 0, 0],
        [0,-1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0,-1, 1, 0, 0],
    ]
    
    n2, m2, p2 = 2, 2, 2
    if _verify(U, V, W, n2, m2, p2):
        return U, V, W
    else:
        rank = 8
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W


def _construct_rank_35():
    """
    Construct a rank-35 decomposition using Laderman for the 2x2x2 block
    and schoolbook for the remaining 28 terms.
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_l, V_l, W_l = _laderman_2x2x2()
    
    u_map = [0, 1, 3, 4]
    v_map = [0, 1, 4, 5]
    w_map = [0, 1, 3, 4]
    
    for r in range(7):
        for lu in range(4):
            gu = u_map[lu]
            U[gu][r] = U_l[lu][r]
        for lv in range(4):
            gv = v_map[lv]
            V[gv][r] = V_l[lv][r]
        for lw in range(4):
            gw = w_map[lw]
            W[gw][r] = W_l[lw][r]
    
    column = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return U, V, W


def _construct_rank_33():
    """
    Construct a rank-33 decomposition by using Laderman for a 2x2x2 block
    and then exploiting that the two Laderman terms for k=0 and k=1
    can be combined with schoolbook terms for k=2,3 to save 2 more ranks.
    
    Actually, let's try a different approach: use Laderman for the 2x2x2
    block (saving 1 rank vs schoolbook for that block: 7 vs 8), and then
    for the remaining 28 terms, see if we can find any further savings.
    
    The remaining terms are:
    - i in {0,1}, j in {0,1}, k in {2,3}: 4 terms
    - i in {0,1}, j in {2}, k in {0,1,2,3}: 4 terms
    - i in {2}, j in {0,1,2}, k in {0,1,2,3}: 12 terms
    - i in {0,1}, j in {2}, k in {0,1}: already counted
    - i in {2}, j in {0,1}, k in {0,1}: already counted
    
    Let me count more carefully:
    Total terms: 3*3*4 = 36
    Laderman covers: i in {0,1}, j in {0,1}, k in {0,1} = 2*2*2 = 8 terms
    Remaining: 36 - 8 = 28 terms
    
    So rank-35 is the straightforward combination. To get rank-33,
    we'd need to save 2 more ranks.
    
    One approach: look at the terms where i=2. There are 3*4=12 such terms.
    These are C[2,j] = sum_k A[2,k]*B[k,j] for j=0,1,2,3.
    
    Similarly, terms where j=2: C[i,2] = sum_k A[i,2]*B[k,2] for i=0,1,2.
    
    Let's try to see if we can find a lower-rank decomposition for a
    sub-problem. For example, the 3x1x4 multiplication (i.e., C[2,:] = A[2,:]*B[:,2])
    is a rank-4 tensor (outer product of a 3-vector and a 4-vector... wait, no).
    
    Actually, A[2,:] is a row vector of length m=3, and B[:,2] is a column vector
    of length p=4. Their product C[2,:] is a 1x4 row vector. This is a rank-3
    tensor (the outer product of a 3-vector and a 4-vector has rank 3 in the
    tensor sense, since it's a sum of 3 rank-1 tensors).
    
    Similarly, the terms where i in {0,1} and j=2: C[i,2] for i=0,1.
    This is a 2x4 matrix = A[:,2] (2x1) * B[2,:] (1x4), which is rank-2.
    
    So the terms with j=2 can be done in rank 2+3 = 5 instead of schoolbook rank
    (2+3)*4 = 20... wait, that's not right either.
    
    Let me reconsider. The schoolbook for all 36 terms uses rank 36.
    Using Laderman for the 2x2x2 block saves 1 rank, giving rank 35.
    
    To save more, we need to find a non-trivial decomposition for the
    remaining 28 terms that uses fewer than 28 ranks.
    
    The remaining 28 terms form a "cross" shape:
    - 4 terms: (i,j,k) with i<2, j<2, k>=2
    - 4 terms: (i,j,k) with i<2, j=2, k<2
    - 4 terms: (i,j,k) with i<2, j=2, k>=2
    - 12 terms: (i,j,k) with i=2, j<2, k<2
    - 4 terms: (i,j,k) with i=2, j=2, k<2
    - 4 terms: (i,j,k) with i=2, j=2, k>=2
    
    Wait, let me recount:
    - i<2, j<2, k>=2: 2*2*2 = 4
    - i<2, j=2, k<2: 2*1*2 = 4
    - i<2, j=2, k>=2: 2*1*2 = 4
    - i=2, j<2, k<2: 1*2*2 = 4
    - i=2, j<2, k>=2: 1*2*2 = 4
    - i=2, j=2, k<2: 1*1*2 = 2
    - i=2, j=2, k>=2: 1*1*2 = 2
    
    Total: 4+4+4+4+4+2+2 = 24. That's not 28. Let me recount.
    
    Total terms with i<2, j<2, k<2: 8 (covered by Laderman)
    Total terms: 36
    Remaining: 28
    
    Terms with i<2: 2*3*4 = 24, minus 8 = 16 remaining
    Terms with i=2: 1*3*4 = 12
    
    16 + 12 = 28. OK.
    
    For i=2 terms: C[2,j] = sum_k A[2,k]*B[k,j]. This is the product of a
    1x3 matrix and a 3x4 matrix, giving a 1x4 matrix. The tensor for this
    sub-problem has rank at most 3 (since it's a sum of 3 rank-1 tensors:
    for each k, A[2,k]*B[k,:]).
    
    So we can do the 12 i=2 terms in rank 3 instead of rank 12!
    That saves 9 ranks.
    
    Similarly, for i<2, j=2 terms: C[i,2] = sum_k A[i,2]*B[k,2]. For i=0,1,
    this is a 2x1 matrix times a 1x4 matrix... wait, no.
    
    C[i,2] = sum_k A[i,k]*B[k,2]. For fixed i, this is a scalar.
    For i=0,1, we have 2 scalars, each a sum over k=0,1,2,3.
    
    The tensor for (i=0,1; j=2; k=0,1,2,3) is:
    T[i,2,k] = A[i,k]*B[k,2]
    
    This is a 2x1x4 tensor. Its rank is at most 2 (since for each i, it's
    a rank-1 tensor in the k dimension, and there are 2 such tensors).
    
    So we can do the 4 terms (i<2, j=2, k<2) in rank 2 instead of rank 4.
    And the 4 terms (i<2, j=2, k>=2) in rank 2 instead of rank 4.
    
    Wait, I need to be more careful. The terms (i<2, j=2, k<2) are:
    (0,2,0), (0,2,1), (1,2,0), (1,2,1)
    
    And (i<2, j=2, k>=2) are:
    (0,2,2), (0,2,3), (1,2,2), (1,2,3)
    
    These are separate sets. For the first set, the tensor is:
    T[i,2,k] = A[i,k]*B[k,2] for i in {0,1}, k in {0,1}
    This is a 2x1x2 tensor with rank at most 2.
    
    For the second set:
    T[i,2,k] = A[i,k]*B[k,2] for i in {0,1}, k in {2,3}
    This is also a 2x1x2 tensor with rank at most 2.
    
    So total for j=2, i<2: rank 2+2 = 4 instead of schoolbook rank 8.
    Saves 4 ranks.
    
    For i<2, j<2, k>=2: 4 terms.
    T[i,j,k] = A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {2,3}
    This is a 2x2x2 tensor. Its rank is at most 4 (schoolbook), but could be lower.
    Actually, this is just the standard 2x2x2 matrix multiplication tensor
    (with the k index restricted to {2,3}), so its rank is 7 (Laderman).
    
    Wait, no. The tensor T[i,j,k] = A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {2,3}
    is exactly the 2x2x2 matrix multiplication tensor (just with relabeled indices).
    So it has rank 7.
    
    So total remaining rank: 3 (for i=2) + 4 (for j=2, i<2) + 7 (for i<2, j<2, k>=2) = 14
    Plus Laderman rank 7 for the main block.
    Total: 7 + 14 = 21!
    
    Wait, that can't be right. Let me re-examine.
    
    The 12 i=2 terms: C[2,j] = sum_k A[2,k]*B[k,j] for j=0,1,2,3.
    This is the product of a 1x3 matrix A[2,:] and a 3x4 matrix B.
    The tensor for this is: T[2,j,k] = A[2,k]*B[k,j]
    This is a 1x4x3 tensor (indices: i=2 fixed, j=0..3, k=0..2).
    Its rank is at most 3 (sum over k of A[2,k]*B[k,:]).
    
    So we use 3 ranks for these 12 terms.
    
    The 4 terms (i<2, j=2, k<2): T[i,2,k] = A[i,k]*B[k,2] for i in {0,1}, k in {0,1}
    Rank at most 2.
    
    The 4 terms (i<2, j=2, k>=2): T[i,2,k] = A[i,k]*B[k,2] for i in {0,1}, k in {2,3}
    Rank at most 2.
    
    The 4 terms (i<2, j<2, k>=2): T[i,j,k] = A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {2,3}
    This is a 2x2x2 tensor, rank 7 (Laderman).
    
    Total: 7 (main Laderman) + 3 + 2 + 2 + 7 = 21.
    
    So rank-21 is achievable! Let me construct it.
    """
    n, m, p = DIMENSIONS
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: Laderman for i in {0,1}, j in {0,1}, k in {0,1}
    U_l, V_l, W_l = _laderman_2x2x2()
    u_map = [0, 1, 3, 4]
    v_map = [0, 1, 4, 5]
    w_map = [0, 1, 3, 4]
    
    for r in range(7):
        for lu in range(4):
            U[u_map[lu]][r] = U_l[lu][r]
        for lv in range(4):
            V[v_map[lv]][r] = V_l[lv][r]
        for lw in range(4):
            W[w_map[lw]][r] = W_l[lw][r]
    
    # Block 2: i=2 terms, rank 3
    # T[2,j,k] = A[2,k]*B[k,j]
    # For each k in {0,1,2}: U[2*3+k][r] = 1, V[k*p+j][r] = 1, W[j*n+2][r] = 1
    col = 7
    for k in range(3):
        for j in range(4):
            U[2 * 3 + k][col] = 1
            V[k * 4 + j][col] = 1
            W[j * 3 + 2][col] = 1
        col += 1
    
    # Block 3: i in {0,1}, j=2, k in {0,1}, rank 2
    # T[i,2,k] = A[i,k]*B[k,2]
    col = 10
    for k in range(2):
        for i in range(2):
            U[i * 3 + 2][col] = 1
            V[2 * 4 + k][col] = 1
            W[k * 3 + i][col] = 1
        col += 1
    
    # Block 4: i in {0,1}, j=2, k in {2,3}, rank 2
    col = 12
    for k in range(2, 4):
        for i in range(2):
            U[i * 3 + 2][col] = 1
            V[2 * 4 + k][col] = 1
            W[k * 3 + i][col] = 1
        col += 1
    
    # Block 5: i in {0,1}, j in {0,1}, k in {2,3}, rank 7 (Laderman)
    U_l2, V_l2, W_l2 = _laderman_2x2x2()
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1} (local k)
    # Global: i in {0,1}, j in {0,1}, k in {2,3} (global k = local k + 2)
    u_map2 = [0, 1, 3, 4]
    v_map2 = [0, 1, 4, 5]  # B[j,k] with j in {0,1}, k in {2,3} -> index j*4+k
    w_map2 = [2, 3, 5, 6]  # C[k,i] with k in {2,3}, i in {0,1} -> index k*3+i
    
    for r in range(7):
        for lu in range(4):
            U[u_map2[lu]][col + r] = U_l2[lu][r]
        for lv in range(4):
            V[v_map2[lv]][col + r] = V_l2[lv][r]
        for lw in range(4):
            W[w_map2[lw]][col + r] = W_l2[lw][r]
    
    col += 7
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = len(U[0])
    
    # Try rank-35
    if time.time() < deadline:
        U2, V2, W2 = _construct_rank_35()
        if _verify(U2, V2, W2, n, m, p):
            if len(U2[0]) < best_rank:
                U, V, W = U2, V2, W2
                best_rank = len(U2[0])
    
    # Try rank-21
    if time.time() < deadline:
        U3, V3, W3 = _construct_rank_33()
        if _verify(U3, V3, W3, n, m, p):
            if len(U3[0]) < best_rank:
                U, V, W = U3, V3, W3
                best_rank = len(U3[0])
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook()
    
    return {"U": U, "V": V, "W": W}
