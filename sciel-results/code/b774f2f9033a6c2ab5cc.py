"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication.

Strategy: Use Strassen's rank-7 decomposition for 2x2x2 sub-blocks combined
with schoolbook for smaller sub-blocks. Split the 3x4x4 multiplication into
two 3x2x4 halves (j=0,1 and j=2,3). Each 3x2x4 is further split into
2x2x2 (Strassen, rank 7) and 1x2x2 (schoolbook, rank 4) pieces.
Total rank = 2*(7+7+4+4) = 44, below schoolbook rank 48.
"""
import time

DIMENSIONS = [3, 4, 4]
_START = time.time()
_DEADLINE = 150.0 - 5.0  # safety margin before the 150s wall-clock kill


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition of the n x m x p multiplication tensor."""
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


def _strassen_222_terms():
    """Return the 7 Strassen rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u,v,w are length-4 vectors indexed [i*2+j],
    [j*2+k], [k*2+i] respectively."""
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U vector: A[i,j] flattened [i*2+j] = [a11, a12, a21, a22]
    # V vector: B[j,k] flattened [j*2+k] = [b11, b12, b21, b22]
    # W vector: C[k,i] flattened [k*2+i] = [c11, c12, c21, c22]
    
    def vec4(pairs):
        a = [0.0] * 4
        for idx, val in pairs:
            a[idx] = val
        return a
    
    terms = []
    # M1: (a11+a22)(b11+b22) -> contributes to C11 and C22
    # C11 += 1, C22 += 1
    terms.append((
        vec4([(0, 1), (3, 1)]),   # a11 + a22
        vec4([(0, 1), (2, 1)]),   # b11 + b22  (b11=idx0, b22=idx2)
        vec4([(0, 1), (3, 1)]),   # c11 + c22  (c11=idx0, c22=idx3)
    ))
    # M2: (a21+a22)b11 -> contributes to C21 and C22
    # C21 += 1, C22 -= 1
    terms.append((
        vec4([(2, 1), (3, 1)]),   # a21 + a22
        vec4([(0, 1)]),           # b11
        vec4([(2, 1), (3, -1)]),  # c21 - c22  (c21=idx2, c22=idx3)
    ))
    # M3: a11(b21-b22) -> contributes to C12 and C22
    # C12 += 1, C22 += 1
    terms.append((
        vec4([(0, 1)]),           # a11
        vec4([(1, -1), (2, 1)]),  # -b12 + b21  (b12=idx1, b21=idx2)
        vec4([(1, 1), (3, 1)]),   # c12 + c22  (c12=idx1, c22=idx3)
    ))
    # M4: a22(b12-b11) -> contributes to C11 and C21
    # C11 += 1, C21 += 1
    terms.append((
        vec4([(3, 1)]),           # a22
        vec4([(0, -1), (1, 1)]),  # -b11 + b12  (b11=idx0, b12=idx1)
        vec4([(0, 1), (2, 1)]),   # c11 + c21  (c11=idx0, c21=idx2)
    ))
    # M5: (a11+a12)b22 -> contributes to C11 and C12
    # C11 -= 1, C12 += 1
    terms.append((
        vec4([(0, 1), (1, 1)]),   # a11 + a12
        vec4([(2, 1)]),           # b22
        vec4([(0, -1), (1, 1)]),  # -c11 + c12  (c11=idx0, c12=idx1)
    ))
    # M6: (a21-a11)(b11+b12) -> contributes to C22
    # C22 += 1
    terms.append((
        vec4([(0, -1), (2, 1)]),  # -a11 + a21
        vec4([(0, 1), (1, 1)]),   # b11 + b12
        vec4([(3, 1)]),           # c22
    ))
    # M7: (a12-a22)(b21+b22) -> contributes to C11
    # C11 += 1
    terms.append((
        vec4([(1, 1), (3, -1)]),  # a12 - a22
        vec4([(2, 1), (3, 1)]),   # b21 + b22  (b21=idx2, b22=idx3)
        vec4([(0, 1)]),           # c11
    ))
    
    return terms


def _verify(U, V, W, n, m, p):
    """Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _build_strassen_decomposition():
    """Build rank-44 Strassen-based decomposition for 3x4x4.
    
    Split j (middle index, 0..3) into j_low={0,1} and j_high={2,3}.
    For each half, split i (0..2) into i_top={0,1} and i_bot={2},
    and k (0..3) into k_left={0,1} and k_right={2,3}.
    
    For each (j_half, i_part, k_part):
    - i_top (2x2) x k_part (2x2): Strassen rank 7
    - i_bot (1x2) x k_part (2x2): schoolbook rank 4
    """
    n, m, p = 3, 4, 4
    total_rank = 44
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 12 x 44
    V = [[0.0] * total_rank for _ in range(m * p)]  # 16 x 44
    W = [[0.0] * total_rank for _ in range(p * n)]  # 12 x 44
    
    strassen_terms = _strassen_222_terms()
    
    col = 0
    for j_offset in [0, 2]:  # j_low=0, j_high=2
        for i_offset, i_size in [(0, 2), (2, 1)]:  # i_top rows 0-1, i_bot row 2
            for k_offset in [0, 2]:  # k_left cols 0-1, k_right cols 2-3
                if i_size == 2:
                    # 2x2x2 Strassen block: A[i_offset:i_offset+2, j_offset:j_offset+2]
                    #                    B[j_offset:j_offset+2, k_offset:k_offset+2]
                    #                    C[i_offset:i_offset+2, k_offset:k_offset+2]
                    for (u, v, w) in strassen_terms:
                        for a in range(4):
                            i_local = a // 2
                            j_local = a % 2
                            U[(i_offset + i_local) * m + (j_offset + j_local)][col] = u[a]
                        for a in range(4):
                            j_local = a // 2
                            k_local = a % 2
                            V[(j_offset + j_local) * p + (k_offset + k_local)][col] = v[a]
                        for a in range(4):
                            k_local = a // 2
                            i_local = a % 2
                            W[(k_offset + k_local) * n + (i_offset + i_local)][col] = w[a]
                        col += 1
                else:
                    # 1x2x2 schoolbook: A[i_offset, j_offset:j_offset+2]
                    #                    B[j_offset:j_offset+2, k_offset:k_offset+2]
                    #                    C[i_offset, k_offset:k_offset+2]
                    for j_local in range(2):
                        for k_local in range(2):
                            U[i_offset * m + (j_offset + j_local)][col] = 1.0
                            V[(j_offset + j_local) * p + (k_offset + k_local)][col] = 1.0
                            W[(k_offset + k_local) * n + i_offset][col] = 1.0
                            col += 1
    
    assert col == total_rank, f"Expected {total_rank} terms, got {col}"
    
    if not _verify(U, V, W, n, m, p):
        return None
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-START
def construct():
    # Try the rank-44 Strassen-based decomposition with verification.
    try:
        r44 = _build_strassen_decomposition()
        if r44 is not None:
            return r44
    except Exception:
        pass

    # Fallback: guaranteed-correct schoolbook rank-48 decomposition.
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
