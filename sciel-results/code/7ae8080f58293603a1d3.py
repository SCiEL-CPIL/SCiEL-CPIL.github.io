"""Matrix multiplication 4x4 x 4x8 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# Deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 140  # 10s safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-10)


def _strassen_2x2_factors():
    """Return the 7 Strassen rank-1 factors for 2x2*2x2.
    Returns (U, V, W) where each is a list of 7 vectors.
    U[r]: 4-vector for A element coefficients
    V[r]: 4-vector for B element coefficients
    W[r]: 4-vector for C element coefficients
    """
    # A elements indexed as (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B elements indexed as (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C elements indexed as (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U = [
        [1, 0, 0, 1],   # M0: A11+A22
        [0, 0, 1, 1],   # M1: A21+A22
        [1, 0, 0, 0],   # M2: A11
        [0, 0, 0, 1],   # M3: A22
        [1, 1, 0, 0],   # M4: A11+A12
        [-1, 0, 1, 0],  # M5: A21-A11
        [0, 1, 0, -1],  # M6: A12-A22
    ]
    V = [
        [1, 0, 0, 1],   # M0: B11+B22
        [1, 0, 0, 0],   # M1: B11
        [0, 1, 0, -1],  # M2: B12-B22
        [-1, 0, 1, 0],  # M3: B21-B11
        [0, 0, 0, 1],   # M4: B22
        [1, 1, 0, 0],   # M5: B11+B12
        [0, 0, 1, 1],   # M6: B21+B22
    ]
    # W: for each output C element, coefficients of M0..M6
    # C11 = M0 + M3 - M4 + M6
    # C12 = M2 + M4
    # C21 = M1 + M3
    # C22 = M0 - M1 + M2 + M5
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0],   # C22
    ]
    return U, V, W


def _build_strassen_4x4_4x8():
    """Build rank-98 Strassen decomposition for 4x4 * 4x8.
    
    Strategy: Split B (4x8) into two 4x4 blocks B1, B2.
    For each, apply 2-level Strassen: 7 block-level products, each a 2x2*2x2
    which uses 7 scalar Strassen products. Total: 2*7*7 = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank))  # (16, 98)
    V = np.zeros((m * p, rank))  # (32, 98)
    W = np.zeros((p * n, rank))  # (32, 98)
    
    sU, sV, sW = _strassen_2x2_factors()
    
    # For each subproblem t (0, 1), block-level product s (0..6), scalar product r (0..6):
    for t in range(2):
        for s in range(7):
            for r in range(7):
                col = t * 49 + s * 7 + r
                
                # U: A is 4x4, split into 2x2 blocks
                # Block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                # strassen_U[s] gives block coefficients: [A11, A12, A21, A22]
                # strassen_U[r] gives scalar coefficients: [(0,0), (0,1), (1,0), (1,1)]
                for bi in range(2):
                    for bj in range(2):
                        c_block = sU[s][bi * 2 + bj]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                c_scalar = sU[r][ei * 2 + ej]
                                if c_scalar == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j, col] += c_block * c_scalar
                
                # V: B is 4x8, subproblem t uses columns 4*t..4*t+3
                # Block (bi, bj) covers rows 2*bi..2*bi+1, cols 4*t+2*bj..4*t+2*bj+1
                for bi in range(2):
                    for bj in range(2):
                        c_block = sV[s][bi * 2 + bj]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                c_scalar = sV[r][ei * 2 + ej]
                                if c_scalar == 0:
                                    continue
                                j = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                V[j * p + k, col] += c_block * c_scalar
                
                # W: C is 4x8, subproblem t uses columns 4*t..4*t+3
                # C blocks for product s:
                c_block_coeffs = [
                    [(0, 0, 1), (1, 1, 1)],    # s=0: C11(1), C22(1)
                    [(1, 0, 1), (1, 1, -1)],   # s=1: C21(1), C22(-1)
                    [(0, 1, 1), (1, 1, 1)],    # s=2: C12(1), C22(1)
                    [(0, 0, 1), (1, 0, 1)],    # s=3: C11(1), C21(1)
                    [(0, 0, -1), (0, 1, 1)],   # s=4: C11(-1), C12(1)
                    [(1, 1, 1)],               # s=5: C22(1)
                    [(0, 0, 1)],               # s=6: C11(1)
                ]
                
                for ci, cj, c_cblock in c_block_coeffs[s]:
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = sW[ei * 2 + ej][r]  # W[c_elem][product_r]
                            if c_scalar == 0:
                                continue
                            i = 2 * ci + ei
                            k = 4 * t + 2 * cj + ej
                            W[k * n + i, col] += c_cblock * c_scalar
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _reduce_rank(U, V, W, max_time=60):
    """Try to reduce rank by finding and eliminating linearly dependent columns.
    
    A column r is redundant if there exist scalars alpha, beta such that
    for all (i,j): U[i,j,r] = alpha, V[j,k,r] = beta, W[k,i,r] = gamma
    and the column can be expressed as a linear combination of others.
    
    More practically: if two columns are proportional in their contribution
    pattern, we can merge them.
    
    This is a heuristic: check if any column's U-part is zero (then V and W
    don't matter), or if columns are linearly dependent in a way that allows
    elimination.
    
    For exact rank reduction, we need: for column r, there exist coefficients
    c_s (s != r) such that for ALL (i,j,k):
    sum_s c_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    This is a very hard constraint. Instead, we do a simpler check:
    if a column has all-zero U or all-zero V or all-zero W, it's removable.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check for trivially removable columns (zero in any factor)
    remove = set()
    for r in range(rank):
        if np.allclose(U_arr[:, r], 0, atol=1e-12):
            remove.add(r)
        elif np.allclose(V_arr[:, r], 0, atol=1e-12):
            remove.add(r)
        elif np.allclose(W_arr[:, r], 0, atol=1e-12):
            remove.add(r)
    
    if not remove:
        return U, V, W
    
    keep = [r for r in range(rank) if r not in remove]
    return U_arr[:, keep].tolist(), V_arr[:, keep].tolist(), W_arr[:, keep].tolist()


def _try_winslow_4x4():
    """Try to use a rank-49 decomposition for 4x4*4x4 if available.
    The current best known for 4x4*4x4 is 49 (Strassen). 
    For 4x4*4x8, we need at least 49*2=98 if we can't share.
    But maybe we can do better with a different decomposition.
    
    Actually, let's try a different approach: use the fact that B is 4x8
    and find a decomposition that exploits the rectangular shape.
    
    One idea: split A into 2x4 blocks (top and bottom), and B into 4x8.
    C = [A_top; A_bot] * B = [A_top*B; A_bot*B]
    Each A_top*B is 2x4 * 4x8 = 2x8, rank 2*4*8=64 schoolbook.
    That's worse.
    
    Another idea: Use the decomposition from the literature.
    For 4x4*4x4, rank 49 is optimal (Strassen).
    For 4x4*4x8, the lower bound is max(4*4, 4*8, 4*8) = 32, 
    but the tensor rank is at least 4*4*4/4 = 16... no that's not right.
    
    The tensor rank of matmul(n,m,p) is at least n*m*p / max(n,m,p) ... no.
    Actually, the rank is at least n*m (since fixing k gives a bilinear form
    on n*m inputs). And at least m*p. And at least n*p.
    So lower bound is max(n*m, m*p, n*p) = max(16, 32, 32) = 32.
    
    But Strassen gives 98. Can we do better?
    
    Let's try a different blocking: split the 4x4*4x8 problem as:
    A (4x4) * B (4x8)
    
    Split A into four 2x2 blocks, B into four 2x4 blocks (top-left, top-right,
    bottom-left, bottom-right where "right" means columns 4-7).
    
    Actually, let me try: split into 2x2 A-blocks and 2x4 B-blocks.
    A = [[A11, A12], [A21, A22]] (each 2x2)
    B = [[B11, B12], [B21, B22]] (each 2x4)
    C = [[C11, C12], [C21, C22]] (each 2x4)
    
    C11 = A11*B11 + A12*B21  (2x2 * 2x4 + 2x2 * 2x4)
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each term is a 2x2 * 2x4 multiplication, which has rank 2*2*4 = 16 schoolbook.
    But we can use Strassen-like tricks for 2x2 * 2x4.
    
    For 2x2 * 2x4: split the 2x4 into two 2x2 blocks.
    2x2 * 2x4 = (2x2 * 2x2) + (2x2 * 2x2), each with Strassen rank 7.
    Total: 14 per output block, 4 blocks = 56.
    
    But we can share! The A factors are shared between C11 and C12 (both use A11, A12),
    and between C21 and C22 (both use A21, A22).
    
    Let me think about this differently. The total number of "scalar multiplications"
    in the Strassen approach:
    - 4x4*4x4 has rank 49
    - 4x4*4x8 = 4x4*4x4 + 4x4*4x4 (two independent) = rank 98
    
    To do better, we need to find a decomposition where the two halves of B
    share some of the A-side computations.
    
    One approach: use a "batched" Strassen where we compute
    (A11+A22) * (B11+B22) where B11, B22 are 2x4 blocks.
    This is a 2x2 * 2x4 multiplication, rank 14 (two Strassen 2x2*2x2).
    
    Let me try to construct this explicitly.
    
    Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    Split B into 2x4 blocks: B11 (rows 0-1, cols 0-3), B12 (rows 0-1, cols 4-7),
                              B21 (rows 2-3, cols 0-3), B22 (rows 2-3, cols 4-7)
    Split C into 2x4 blocks: C11 (rows 0-1, cols 0-3), C12 (rows 0-1, cols 4-7),
                              C21 (rows 2-3, cols 0-3), C22 (rows 2-3, cols 4-7)
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Now, for each 2x2 * 2x4 product, we can use a rank-14 decomposition
    (two Strassen 2x2*2x2 side by side).
    
    But there's sharing: A11 appears in both C11 and C12 computations.
    Specifically, in C11 = A11*B11 + A12*B21, A11 is multiplied by B11 (2x4).
    In C12 = A11*B12 + A12*B22, A11 is multiplied by B12 (2x4).
    
    If we write A11*B11 and A11*B12 as A11*[B11|B12] = A11*B_top where B_top is 2x8...
    no, that doesn't help directly.
    
    Let me try a different decomposition. Use the 7 Strassen products at the
    2x2 block level, but where each "product" is a 2x2 * 2x4 multiplication
    (rank 14 each).
    
    M0 = (A11+A22) * (B11+B21)  -- 2x2 * 2x4, rank 14
    M1 = (A21+A22) * B11        -- 2x2 * 2x4, rank 14
    M2 = A11 * (B12-B22)        -- 2x2 * 2x4, rank 14
    M3 = A22 * (B21-B11)        -- 2x2 * 2x4, rank 14
    M4 = (A11+A12) * B22        -- 2x2 * 2x4, rank 14
    M5 = (A21-A11) * (B11+B12)  -- 2x2 * 2x4, rank 14
    M6 = (A12-A22) * (B21+B22)  -- 2x2 * 2x4, rank 14
    
    Wait, this doesn't look right. Let me reconsider.
    
    Actually, the standard Strassen for 4x4*4x4 splits into 2x2 blocks and
    does 7 products of 2x2*2x2. For 4x4*4x8, if we split B into 2x4 blocks,
    the products become 2x2*2x4.
    
    But the Strassen formula is designed for square blocks. Let me just stick
    with the rank-98 construction and try to reduce it.
    """
    pass


def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_rank_reduction(U, V, W, time_budget=40):
    """Attempt to reduce rank by finding linear dependencies among columns.
    
    For a column r to be removable, we need to find coefficients c_s for s != r
    such that for all (i,j,k):
    U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_{s!=r} c_s * U[i,j,s]*V[j,k,s]*W[k,i,s]
    
    This is equivalent to saying the rank-1 "tensor" formed by the r-th column
    is in the span of the other columns' contributions.
    
    A simpler sufficient condition: if column r's U-vector is a linear combination
    of other columns' U-vectors, AND the corresponding V,W vectors align properly.
    
    In practice, for Strassen decompositions, some columns may be redundant
    due to the specific structure. Let's try a numerical approach:
    
    1. Build the "contribution matrix" for each (i,j,k) triple.
    2. For each column r, check if its contribution vector is in the span
       of the other columns' contribution vectors.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The contribution of column r to entry (i,j,k) is:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For rank reduction, we need to find a subset of columns that spans
    # all the (i,j,k) contributions.
    
    # Total number of (i,j,k) triples: n*m*p = 128
    # Rank is 98. We need to check if we can remove any columns.
    
    # Build a matrix where each row is a (i,j,k) triple and each column r
    # has the value U[i,j,r]*V[j,k,r]*W[k,i,r].
    # Then the column space of this matrix tells us which columns are redundant.
    
    # But this matrix is 128 x 98. If its rank is < 98, some columns are redundant.
    # However, this is a nonlinear relationship (product of three factors),
    # so standard linear algebra doesn't directly apply.
    
    # Instead, let's try a different approach: use the fact that for exact
    # matrix multiplication, the tensor is fixed. We can try to find a
    # lower-rank decomposition by solving a system of equations.
    
    # For now, let's just check if the current decomposition is valid and
    # return it. Rank reduction is complex and may not yield improvements
    # within the time budget.
    
    return U, V, W


def construct():
    """Construct the tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    if _elapsed() > DEADLINE:
        # Emergency fallback
        return _schoolbook()
    
    # Build the Strassen-based rank-98 decomposition
    result = _build_strassen_4x4_4x8()
    
    # Verify
    if not _verify(result["U"], result["V"], result["W"]):
        # Fallback to schoolbook if Strassen construction has a bug
        return _schoolbook()
    
    # Try rank reduction
    if _elapsed() < DEADLINE - 20:
        U, V, W = _reduce_rank(result["U"], result["V"], result["W"])
        if _verify(U, V, W):
            result = {"U": U, "V": V, "W": W}
    
    # Final time check
    if _elapsed() > DEADLINE:
        return _schoolbook()
    
    return result
