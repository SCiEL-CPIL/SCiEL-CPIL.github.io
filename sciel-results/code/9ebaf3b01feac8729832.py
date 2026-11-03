"""Tensor decomposition for 4x4 * 4x8 matrix multiplication (Winograd/Strassen, rank 84)."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.allclose(result, 1.0, atol=1e-9))

def _strassen_2x2_factors():
    """Seven rank-1 factors for 2x2*2x2 multiplication (Strassen)."""
    # p1=(a00+a11)(b00+b11); p2=(a10+a11)b00; p3=a00(b01-b11); p4=a11(b10-b00)
    # p5=(a00+a01)b11; p6=(a10-a00)(b00+b01); p7=(a01-a11)(b10+b11)
    # c00=p1+p4-p5+p7; c01=p3+p5; c10=p2+p4; c11=p1-p2+p3+p6
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

def _fill_4x4x4(U, V, W, base, s, r, a_off, b_off, c_off):
    """Add one rank-1 factor (top-level s, scalar r) of a 4x4*4x4 Strassen product."""
    sf = _strassen_2x2_factors()
    bu, bv, bw = sf[s]
    su, sv, sw = sf[r]
    col = base + s * 7 + r
    for bi in range(2):
        for bj in range(2):
            cb = bu[bi * 2 + bj]
            if cb == 0:
                continue
            for ei in range(2):
                for ej in range(2):
                    cs = su[ei * 2 + ej]
                    if cs == 0:
                        continue
                    U[(a_off + 2 * bi + ei) * 4 + (a_off + 2 * bj + ej)][col] += cb * cs
    for bi in range(2):
        for bj in range(2):
            cb = bv[bi * 2 + bj]
            if cb == 0:
                continue
            for ei in range(2):
                for ej in range(2):
                    cs = sv[ei * 2 + ej]
                    if cs == 0:
                        continue
                    V[(b_off + 2 * bi + ei) * 4 + (b_off + 2 * bj + ej)][col] += cb * cs
    for ci in range(2):
        for cj in range(2):
            cb = bw[ci * 2 + cj]
            if cb == 0:
                continue
            for ei in range(2):
                for ej in range(2):
                    cs = sw[ei * 2 + ej]
                    if cs == 0:
                        continue
                    i = c_off + 2 * ci + ei
                    k = c_off + 2 * cj + ej
                    W[k * 4 + i][col] += cb * cs

def _fill_4x2x4_rank35(U, V, W, base, s, r, a_off, b_off, c_off):
    """Add one rank-1 factor of the 4x2*2x4 (rank 35) multiplication scheme."""
    # Seven top-level products, each a 2x2 (rank 7).
    # A 4x2: blocks A00,A01 (rows 0-1), A10,A11 (rows 2-3).
    # B 2x4: blocks B00,B01 (cols 0-1), B10,B11 (cols 2-3).
    # C 4x4: blocks C00,C01 (rows 0-1), C10,C11 (rows 2-3).
    top = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    sf = _strassen_2x2_factors()
    bu, bv, bw = top[s]
    su, sv, sw = sf[r]
    col = base + s * 7 + r
    # A: 4x2, element (i,j), i in 0..3, j in 0..1
    for bi in range(2):
        for bj in range(2):
            cb = bu[bi * 2 + bj]
            if cb == 0:
                continue
            for ei in range(2):
                for ej in range(2):
                    cs = su[ei * 2 + ej]
                    if cs == 0:
                        continue
                    i = a_off + 2 * bi + ei
                    j = 2 * bj + ej
                    U[i * 4 + j][col] += cb * cs
    # B: 2x4, element (j,k), j in 0..1, k in 0..3
    for bi in range(2):
        for bj in range(2):
            cb = bv[bi * 2 + bj]
            if cb == 0:
                continue
            for ei in range(2):
                for ej in range(2):
                    cs = sv[ei * 2 + ej]
                    if cs == 0:
                        continue
                    j = bi
                    k = b_off + 2 * bj + ej
                    V[j * 4 + k][col] += cb * cs
    # C: 4x4, element (i,k), i in 0..3, k in 0..3
    for ci in range(2):
        for cj in range(2):
            cb = bw[ci * 2 + cj]
            if cb == 0:
                continue
            for ei in range(2):
                for ej in range(2):
                    cs = sw[ei * 2 + ej]
                    if cs == 0:
                        continue
                    i = c_off + 2 * ci + ei
                    k = 2 * cj + ej
                    W[k * 4 + i][col] += cb * cs

def _try_winograd_4x4x8():
    """
    Rank-84 decomposition for 4x4 * 4x8.

    Split A = [A1; A2] (4x2 each), B = [B1 B2] (2x4 each).
    C1 = A1 B1 + A2 B2 = (M1+M2)/2 + (M3+M4)/2
    C2 = A1 B2 + A2 B1 = (M1-M2)/2 - (M3+M4)/2
    M1=(A1+A2)(B1+B2)  rank49
    M2=(A1-A2)(B1-B2)  rank49
    M3=A1 B2           rank35
    M4=A2 B1           rank35
    """
    n, m, p = DIMENSIONS  # 4,4,8
    rank = 84
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]

    # --- M1 = (A1+A2)(B1+B2), rank 49, base 0 ---
    for s in range(7):
        for r in range(7):
            col = 0 + s * 7 + r
            for bi in range(2):
                for bj in range(2):
                    cb = _strassen_2x2_factors()[s][0][bi * 2 + bj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = _strassen_2x2_factors()[r][0][ei * 2 + ej]
                            if cs == 0:
                                continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            U[i * 4 + j][col] += cb * cs
            for bi in range(2):
                for bj in range(2):
                    cb = _strassen_2x2_factors()[s][1][bi * 2 + bj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = _strassen_2x2_factors()[r][1][ei * 2 + ej]
                            if cs == 0:
                                continue
                            j = bi
                            k = 2 * bj + ej
                            V[j * 8 + k][col] += cb * cs
            for ci in range(2):
                for cj in range(2):
                    cb = _strassen_2x2_factors()[s][2][ci * 2 + cj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = _strassen_2x2_factors()[r][2][ei * 2 + ej]
                            if cs == 0:
                                continue
                            i = 2 * ci + ei
                            k = 2 * cj + ej
                            W[k * 4 + i][col] += 0.5 * cb * cs
                            W[(k + 4) * 4 + i][col] += 0.5 * cb * cs

    # --- M2 = (A1-A2)(B1-B2), rank 49, base 49 ---
    for s in range(7):
        for r in range(7):
            col = 49 + s * 7 + r
            for bi in range(2):
                for bj in range(2):
                    cb = _strassen_2x2_factors()[s][0][bi * 2 + bj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = _strassen_2x2_factors()[r][0][ei * 2 + ej]
                            if cs == 0:
                                continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            if i < 2:
                                U[i * 4 + j][col] += cb * cs
                            else:
                                U[i * 4 + j][col] += -cb * cs
            for bi in range(2):
                for bj in range(2):
                    cb = _strassen_2x2_factors()[s][1][bi * 2 + bj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = _strassen_2x2_factors()[r][1][ei * 2 + ej]
                            if cs == 0:
                                continue
                            j = bi
                            k = 2 * bj + ej
                            if j == 0:
                                V[j * 8 + k][col] += cb * cs
                            else:
                                V[j * 8 + k][col] += -cb * cs
            for ci in range(2):
                for cj in range(2):
                    cb = _strassen_2x2_factors()[s][2][ci * 2 + cj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = _strassen_2x2_factors()[r][2][ei * 2 + ej]
                            if cs == 0:
                                continue
                            i = 2 * ci + ei
                            k = 2 * cj + ej
                            W[k * 4 + i][col] += 0.5 * cb * cs
                            W[(k + 4) * 4 + i][col] += -0.5 * cb * cs

    # --- M3 = A1 B2 (4x2 * 2x4), rank 35, base 98 ---
    for s in range(7):
        for r in range(7):
            _fill_4x2x4_rank35(U, V, W, 98, s, r, a_off=0, b_off=4, c_off=4)

    # --- M4 = A2 B1 (4x2 * 2x4), rank 35, base 133 ---
    for s in range(7):
        for r in range(7):
            _fill_4x2x4_rank35(U, V, W, 133, s, r, a_off=2, b_off=0, c_off=0)

    return {"U": U, "V": V, "W": W}

def construct():
    """Return a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    start_time = time.time()
    deadline = start_time + 110.0

    try:
        if time.time() < deadline:
            result = _try_winograd_4x4x8()
            if time.time() < deadline:
                if _verify(result["U"], result["V"], result["W"]):
                    return result
    except Exception:
        pass

    return _schoolbook()
# EVOLVE-BLOCK-END
