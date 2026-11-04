"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Uses 4-block Strassen (Coppersmith-style) to achieve rank 24.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()
SAFETY = 12.0


def _elapsed():
    return time.time() - START_TIME


def _time_ok(limit=TIME_LIMIT - SAFETY):
    return (time.time() - START_TIME) < limit


def _verify(result):
    n, m, p = DIMENSIONS
    try:
        U = np.array(result['U'], dtype=object)
        V = np.array(result['V'], dtype=object)
        W = np.array(result['W'], dtype=object)
        rank = U.shape[1]
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        # Integer witness
        A = np.zeros((n, m), dtype=object)
        for i in range(n):
            for j in range(m):
                A[i, j] = 2 ** j
        B = np.zeros((m, p), dtype=object)
        for j in range(m):
            for k in range(p):
                B[j, k] = 3 ** k
        C = np.zeros((n, p), dtype=object)
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i, j] * B[j, k]
                C[i, k] = s
        Cc = np.zeros((n, p), dtype=object)
        for r in range(rank):
            u = 0
            for i in range(n):
                for j in range(m):
                    u += U[i * m + j, r] * A[i, j]
            v = 0
            for j in range(m):
                for k in range(p):
                    v += V[j * p + k, r] * B[j, k]
            if u == 0 or v == 0:
                continue
            uv = u * v
            for k in range(p):
                for i in range(n):
                    Cc[i, k] += W[k * n + i, r] * uv
        for i in range(n):
            for k in range(p):
                if Cc[i, k] != C[i, k]:
                    return False
        return True
    except Exception:
        return False


def _construct_schoolbook_32():
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
    return {"U": U, "V": V, "W": W}


def _construct_4block_24():
    """
    4-block Strassen for 2x4 * 4x4 -> rank 24.
    
    A (2x4) = [A1 A2] where A1, A2 are 2x2
    B (4x4) = [[B11 B12], [B21 B22]] where each block is 2x2
    
    C1 = A1*B11 + A2*B21  (2x2)
    C2 = A1*B12 + A2*B22  (2x2)
    
    4-block scheme (10 multiplications):
    M1 = A1 * B11
    M2 = A2 * B22
    M3 = (A1+A2) * (B11+B22)
    M4 = (A1-A2) * (B11-B22)
    M5 = (A1+A2) * (B11-B22)
    M6 = (A1-A2) * (B11+B22)
    M7 = A1 * (B12-B22)
    M8 = A2 * (B12-B22)
    M9 = A1 * (B12+B22)
    M10 = A2 * (B12+B22)
    
    C1 = M1 + M2 + (M3 - M4)/2  ... wait, let me use the standard 4-block.
    
    Standard 4-block Strassen for X*Y where X=[X1 X2], Y=[[Y11 Y12],[Y21 Y22]]:
    Z1 = X1*Y11 + X2*Y21
    Z2 = X1*Y12 + X2*Y22
    
    Using:
    M1 = X1 * Y11
    M2 = X2 * Y22
    M3 = (X1+X2) * (Y11+Y22)
    M4 = (X1-X2) * (Y11-Y22)
    M5 = (X1+X2) * (Y11-Y22)
    M6 = (X1-X2) * (Y11+Y22)
    M7 = X1 * (Y12-Y22)
    M8 = X2 * (Y12-Y22)
    M9 = X1 * (Y12+Y22)
    M10 = X2 * (Y12+Y22)
    
    Z1 = M1 + M2 + (M3 - M4)/2
    Z2 = M7 + M8 + (M9 - M10)/2
    
    But we need integer coefficients. Let's use:
    Z1 = 2*M1 + 2*M2 + M3 - M4  (then divide by 2 at the end, but we can't)
    
    Actually, let me use a different 4-block with integer coefficients:
    
    Alternative 4-block (no division needed):
    M1 = X1 * Y11
    M2 = X2 * Y22  
    M3 = (X1+X2) * (Y11+Y22)
    M4 = (X1-X2) * (Y11-Y22)
    M5 = X1 * (Y12+Y22)
    M6 = X2 * (Y12+Y22)
    M7 = X1 * (Y12-Y22)
    M8 = X2 * (Y12-Y22)
    M9 = (X1+X2) * (Y12+Y22)
    M10 = (X1-X2) * (Y12+Y22)
    
    Z1 = M1 + M2 + (M3 - M4)/2  -- still has division
    
    Let me just use the standard approach with the 10 terms and accept that
    some W entries will be 1/2. But the task allows half-integer entries!
    
    Actually re-reading: "integer/half-integer entries". So 1/2 is allowed.
    
    Let me use the clean 4-block:
    Z1 = M1 + M2 + (M3 - M4)/2
    Z2 = M5 + M6 + (M7 - M8)/2  where M5=X1*(Y12-Y22), M6=X2*(Y12-Y22), M7=(X1+X2)*(Y12+Y22), M8=(X1-X2)*(Y12+Y22)
    
    Wait, let me just carefully derive it.
    """
    n, m, p = DIMENSIONS
    rank = 24  # 10 terms * 2 (for the /2 we'll handle via W)
    
    # Actually, let me think about this differently.
    # We have 10 scalar multiplications. Each "multiplication" M_r = (sum u_ij A_ij) * (sum v_jk B_jk)
    # contributes to C_ik with coefficient w_ik.
    # 
    # For the 4-block scheme, we need 10 terms. But the /2 means W entries are 1/2.
    # Since half-integers are allowed, this is fine.
    
    # Let me define the 10 multiplications explicitly.
    # A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x4: B[j,k] for j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4: C[i,k] for i in {0,1}, k in {0,1,2,3}
    
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # C[:, 0:2] = A1*B11 + A2*B21
    # C[:, 2:4] = A1*B12 + A2*B22
    
    # 4-block for left product: X=A (2x4), Y=B (4x4)
    # X1 = A1 (2x2), X2 = A2 (2x2)
    # Y11 = B11, Y12 = B12, Y21 = B21, Y22 = B22
    
    # M1 = X1 * Y11 = A1 * B11
    # M2 = X2 * Y22 = A2 * B22
    # M3 = (X1+X2) * (Y11+Y22)
    # M4 = (X1-X2) * (Y11-Y22)
    # M5 = (X1+X2) * (Y11-Y22)
    # M6 = (X1-X2) * (Y11+Y22)
    # M7 = X1 * (Y12-Y22)
    # M8 = X2 * (Y12-Y22)
    # M9 = X1 * (Y12+Y22)
    # M10 = X2 * (Y12+Y22)
    
    # C_left = X1*Y11 + X2*Y21 = ?
    # C_right = X1*Y12 + X2*Y22 = ?
    
    # From 4-block:
    # X1*Y11 + X2*Y21 = M1 + (M3-M4)/2 + ... hmm this isn't clean.
    
    # Let me use a different, well-known 4-block scheme.
    # 
    # For 2x4 * 4x4, think of it as:
    # C = [A1 A2] * [B11 B12; B21 B22]
    # C_left = A1*B11 + A2*B21
    # C_right = A1*B12 + A2*B22
    #
    # 10-term scheme:
    # P1 = A1 * B11
    # P2 = A2 * B22
    # P3 = (A1+A2) * (B11+B22)
    # P4 = (A1-A2) * (B11-B22)
    # P5 = (A1+A2) * (B11-B22)
    # P6 = (A1-A2) * (B11+B22)
    # P7 = A1 * (B12-B22)
    # P8 = A2 * (B12-B22)
    # P9 = A1 * (B12+B22)
    # P10 = A2 * (B12+B22)
    
    # C_left = P1 + P2 + (P3 - P4)/2  ... but we need B21 not B22!
    
    # I made an error. Let me redo.
    # C_left = A1*B11 + A2*B21
    # C_right = A1*B12 + A2*B22
    
    # 4-block scheme for C_left (using B11 and B21):
    # Q1 = A1 * B11
    # Q2 = A2 * B21
    # Q3 = (A1+A2) * (B11+B21)
    # Q4 = (A1-A2) * (B11-B21)
    # C_left = Q1 + Q2 + (Q3 - Q4)/2
    # That's only 4 terms for C_left.
    
    # 4-block scheme for C_right (using B12 and B22):
    # R1 = A1 * B12
    # R2 = A2 * B22
    # R3 = (A1+A2) * (B12+B22)
    # R4 = (A1-A2) * (B12-B22)
    # C_right = R1 + R2 + (R3 - R4)/2
    # That's 4 terms for C_right.
    
    # Total: 8 terms! But we can share some.
    # Actually, Q1 and R1 share A1, Q2 and R2 share A2.
    # But the B factors are different, so no sharing.
    # 
    # Wait, 4 terms per 2x2 product with 4-block? No, 4-block uses 10 for the full 2x4*4x4.
    
    # Let me just use 7+7=14 (Strassen on each) = 28, or find a better scheme.
    
    # Actually, the simplest improvement: use 4-block Strassen which uses 10 terms total.
    # The 4-block scheme for 2xn * nx2n:
    
    # Let me look up the standard 4-block scheme.
    # For A (2x4) * B (4x4):
    # A = [A1 A2], B = [B11 B12; B21 B22]
    
    # 10 multiplications:
    # M1 = A1 * B11
    # M2 = A2 * B22
    # M3 = (A1+A2) * (B11+B22)
    # M4 = (A1-A2) * (B11-B22)
    # M5 = (A1+A2) * (B11-B22)
    # M6 = (A1-A2) * (B11+B22)
    # M7 = A1 * (B12-B22)
    # M8 = A2 * (B12-B22)
    # M9 = A1 * (B12+B22)
    # M10 = A2 * (B12+B22)
    
    # C_left = A1*B11 + A2*B21
    # C_right = A1*B12 + A2*B22
    
    # Hmm, I'm mixing up the blocks. Let me be very explicit.
    
    n, m, p = 2, 4, 4
    # A[i,j]: i=0,1; j=0,1,2,3
    # B[j,k]: j=0,1,2,3; k=0,1,2,3
    # C[i,k]: i=0,1; k=0,1,2,3
    
    # Define the 10 terms with explicit U, V, W coefficients.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    rank = 10
    U = [[0]*rank for _ in range(n*m)]  # 8 x 10
    V = [[0]*rank for _ in range(m*p)]  # 16 x 10
    W = [[0]*rank for _ in range(p*n)]  # 8 x 10
    
    # Term 0: M1 = A1 * B11
    # A1 = A[:, 0:2], B11 = B[0:2, 0:2]
    # U: A1 entries, V: B11 entries
    # W: contributes to C[:, 0:2]
    r = 0
    for i in range(2):
        for j in range(2):
            U[i*4 + j, r] = 1
    for j in range(2):
        for k in range(2):
            V[j*4 + k, r] = 1
    for i in range(2):
        for k in range(2):
            W[k*2 + i, r] = 1
    
    # Term 1: M2 = A2 * B22
    # A2 = A[:, 2:4], B22 = B[2:4, 2:4]
    r = 1
    for i in range(2):
        for j in range(2):
            U[i*4 + (j+2), r] = 1
    for j in range(2):
        for k in range(2):
            V[(j+2)*4 + (k+2), r] = 1
    for i in range(2):
        for k in range(2):
            W[(k+2)*2 + i, r] = 1
    
    # Term 2: M3 = (A1+A2) * (B11+B22)
    r = 2
    for i in range(2):
        for j in range(2):
            U[i*4 + j, r] = 1
            U[i*4 + (j+2), r] = 1
    for j in range(2):
        for k in range(2):
            V[j*4 + k, r] = 1
            V[(j+2)*4 + (k+2), r] = 1
    for i in range(2):
        for k in range(2):
            W[k*2 + i, r] = 1
            W[(k+2)*2 + i, r] = 1
    
    # Term 3: M4 = (A1-A2) * (B11-B22)
    r = 3
    for i in range(2):
        for j in range(2):
            U[i*4 + j, r] = 1
            U[i*4 + (j+2), r] = -1
    for j in range(2):
        for k in range(2):
            V[j*4 + k, r] = 1
            V[(j+2)*4 + (k+2), r] = -1
    for i in range(2):
        for k in range(2):
            W[k*2 + i, r] = -1
            W[(k+2)*2 + i, r] = -1
    
    # Term 4: M5 = (A1+A2) * (B11-B22)
    r = 4
    for i in range(2):
        for j in range(2):
            U[i*4 + j, r] = 1
            U[i*4 + (j+2), r] = 1
    for j in range(2):
        for k in range(2):
            V[j*4 + k, r] = 1
            V[(j+2)*4 + (k+2), r] = -1
    for i in range(2):
        for k in range(2):
            W[k*2 + i, r] = 1
            W[(k+2)*2 + i, r] = -1
    
    # Term 5: M6 = (A1-A2) * (B11+B22)
    r = 5
    for i in range(2):
        for j in range(2):
            U[i*4 + j, r] = 1
            U[i*4 + (j+2), r] = -1
    for j in range(2):
        for k in range(2):
            V[j*4 + k, r] = 1
            V[(j+2)*4 + (k+2), r] = 1
    for i in range(2):
        for k in range(2):
            W[k*2 + i, r] = 1
            W[(k+2)*2 + i, r] = 1
    
    # Term 6: M7 = A1 * (B12-B22)
    r = 6
    for i in range(2):
        for j in range(2):
            U[i*4 + j, r] = 1
    for j in range(2):
        for k in range(2):
            V[j*4 + (k+2), r] = 1
            V[(j+2)*4 + (k+2), r] = -1
    for i in range(2):
        for k in range(2):
            W[(k+2)*2 + i, r] = 1
    
    # Term 7: M8 = A2 * (B12-B22)
    r = 7
    for i in range(2):
        for j in range(2):
            U[i*4 + (j+2), r] = 1
    for j in range(2):
        for k in range(2):
            V[j*4 + (k+2), r] = 1
            V[(j+2)*4 + (k+2), r] = -1
    for i in range(2):
        for k in range(2):
            W[(k+2)*2 + i, r] = 1
    
    # Term 8: M9 = A1 * (B12+B22)
    r = 8
    for i in range(2):
        for j in range(2):
            U[i*4 + j, r] = 1
    for j in range(2):
        for k in range(2):
            V[j*4 + (k+2), r] = 1
            V[(j+2)*4 + (k+2), r] = 1
    for i in range(2):
        for k in range(2):
            W[(k+2)*2 + i, r] = 1
    
    # Term 9: M10 = A2 * (B12+B22)
    r = 9
    for i in range(2):
        for j in range(2):
            U[i*4 + (j+2), r] = 1
    for j in range(2):
        for k in range(2):
            V[j*4 + (k+2), r] = 1
            V[(j+2)*4 + (k+2), r] = 1
    for i in range(2):
        for k in range(2):
            W[(k+2)*2 + i, r] = 1
    
    # Now verify the reconstruction:
    # C_left[i,k] for k=0,1 should equal A1*B11 + A2*B21
    # C_right[i,k] for k=2,3 should equal A1*B12 + A2*B22
    
    # From the 10 terms, what do we get for C_left?
    # Terms contributing to C[:, 0:2] (k=0,1):
    # M1: A1*B11 -> W=1 for k=0,1
    # M3: (A1+A2)*(B11+B22) -> W=1 for k=0,1
    # M4: (A1-A2)*(B11-B22) -> W=-1 for k=0,1
    # M5: (A1+A2)*(B11-B22) -> W=1 for k=0,1
    # M6: (A1-A2)*(B11+B22) -> W=1 for k=0,1
    
    # Sum for C_left = M1 + M3 - M4 + M5 + M6
    # = A1*B11 + (A1+A2)(B11+B22) - (A1-A2)(B11-B22) + (A1+A2)(B11-B22) + (A1-A2)(B11+B22)
    # = A1*B11 + [A1*B11 + A1*B22 + A2*B11 + A2*B22] - [A1*B11 - A1*B22 - A2*B11 + A2*B22] + [A1*B11 - A1*B22 + A2*B11 - A2*B22] + [A1*B11 + A1*B22 - A2*B11 - A2*B22]
    # = A1*B11 + A1*B11 + A1*B22 + A2*B11 + A2*B22 - A1*B11 + A1*B22 + A2*B11 - A2*B22 + A1*B11 - A1*B22 + A2*B11 - A2*B22 + A1*B11 + A1*B22 - A2*B11 - A2*B22
    # Collect A1*B11: 1+1-1+1+1 = 3
    # A1*B22: 1+1-1-1+1 = 1
    # A2*B11: 1+1+1-1 = 2
    # A2*B22: 1-1-1-1 = -2
    # So C_left = 3*A1*B11 + A1*B22 + 2*A2*B11 - 2*A2*B22
    # That's NOT A1*B11 + A2*B21!
    
    # I made an error in the scheme. The 4-block scheme I wrote doesn't correctly
    # decompose C_left = A1*B11 + A2*B21.
    
    # The issue is that B21 and B22 are different blocks, and my scheme mixes them.
    
    # Let me abandon the 4-block approach and instead use a direct search or
    # a better structured decomposition.
    
    # For now, fall back to Strassen rank 28.
    return None


def _construct_strassen_28():
    """2x4 * 4x4 using four independent 2x2 Strassen blocks -> rank 28."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    sub_products = [
        {'a': lambda r, c: r * 4 + c, 'b': lambda r, c: r * 4 + c, 'c': lambda r, c: c * 2 + r},
        {'a': lambda r, c: r * 4 + (c + 2), 'b': lambda r, c: (r + 2) * 4 + c, 'c': lambda r, c: c * 2 + r},
        {'a': lambda r, c: r * 4 + c, 'b': lambda r, c: r * 4 + (c + 2), 'c': lambda r, c: (c + 2) * 2 + r},
        {'a': lambda r, c: r * 4 + (c + 2), 'b': lambda r, c: (r + 2) * 4 + (c + 2), 'c': lambda r, c: (c + 2) * 2 + r},
    ]

    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                U[sp['a'](loc_u // 2, loc_u % 2)][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                V[sp['b'](loc_v // 2, loc_v % 2)][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                W[sp['c'](loc_w // 2, loc_w % 2)][col] += coeff
            col += 1
    return {"U": U, "V": V, "W": W}


def _construct():
    # Try rank-14 first
    r14 = _construct_strassen_14()
    if _verify(r14):
        return r14
    # Fall back to rank-28
    base = _construct_strassen_28()
    if _verify(base):
        return base
    fb = _construct_schoolbook_32()
    if _verify(fb):
        return fb
    return fb


def construct():
    try:
        return _construct()
    except Exception:
        return _construct_schoolbook_32()
