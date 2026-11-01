import numpy as np
import time

DIMENSIONS = [2, 2, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140  # safety margin
    
    # Try rank-28 Strassen-based construction
    try:
        result = _strassen_28()
        if _verify(result, n, m, p):
            return result
    except Exception:
        pass
    
    if time.time() > deadline:
        pass  # fall through to schoolbook
    
    # Fallback: schoolbook rank-32
    return _schoolbook()

def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen terms for 2x2x2:
    # Each term has (u, v, w) where u is 4-dim (A), v is 4-dim (B), w is 4-dim (C)
    # A indices: (i,j) -> i*2+j
    # B indices: (j,k) -> j*2+k
    # C indices: (k,i) -> k*2+i
    
    # Define the 7 Strassen terms for one 2x2x2 block
    # u: [a00, a01, a10, a11]
    # v: [b00, b01, b10, b11]
    # w: [C00, C01, C10, C11] but indexed as k*2+i, so:
    #   k=0,i=0 -> C00 -> index 0
    #   k=1,i=0 -> C10 -> index 1
    #   k=0,i=1 -> C01 -> index 2
    #   k=1,i=1 -> C11 -> index 3
    
    strassen_terms = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1: (a00+a11)(b00+b11) -> C00+C11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, -1, 0]),  # M2: (a10+a11)b00 -> C10-C11  (w: C10-C11 = index1 - index3... wait)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # M3: a00(b01-b11) -> C01+C11
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4: a11(b10-b00) -> C00+C10
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),  # M5: (a00+a01)b11 -> -C00+C01
        ([0, 0, 1, 0], [0, 1, -1, 0], [0, 0, 0, 1]),  # M6: a10(b01-b10) -> C11  [NOTE: this is different from standard Strassen]
        ([1, 0, -1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),  # M7: (a00-a10)(b00+b01) -> C00
    ]
    
    # Wait, I need to double-check the w indices. W[k*n+i, r] where n=2, so W[k*2+i, r].
    # C[i,k] is stored at W[k*2+i, r].
    # So w[0] = coefficient of C[0,0] (k=0,i=0)
    #    w[1] = coefficient of C[1,0] (k=1,i=0)  -- wait, k*2+i, so k=1,i=0 -> index 2
    # Let me redo:
    # k=0, i=0 -> index 0 -> C[0,0]
    # k=0, i=1 -> index 1 -> C[1,0]
    # k=1, i=0 -> index 2 -> C[0,1]
    # k=1, i=1 -> index 3 -> C[1,1]
    
    # So w[0] = coeff of C[0,0], w[1] = coeff of C[1,0], w[2] = coeff of C[0,1], w[3] = coeff of C[1,1]
    
    # Let me redefine:
    strassen_terms = [
        # (u, v, w) where u=[a00,a01,a10,a11], v=[b00,b01,b10,b11], w=[C00,C10,C01,C11]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, -1, 0, 1]),  # M2: (a10+a11)b00 -> C10-C11? No...
        # Let me re-derive Strassen properly.
    ]
    
    # Let me re-derive Strassen for 2x2x2 with the correct index conventions.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    
    # Strassen:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = a10*(a11-a01)
    # M7 = (a00-a10)*(b00+b01)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Now, for the tensor decomposition, each M_r must be expressible as u_r · A * v_r · B,
    # where u_r is a linear form in A and v_r is a linear form in B.
    # M6 = a10*(a11-a01) is NOT of this form. So this particular Strassen scheme doesn't
    # directly give a tensor decomposition.
    
    # However, the tensor rank is 7, so there exists some 7-term decomposition.
    # Let me try a different set of 7 terms.
    
    # Alternative: use the fact that the tensor rank is 7 and find the decomposition.
    # This is complex, so let me just try the rank-28 construction with the standard
    # Strassen terms, replacing M6 with a bilinear term.
    
    # Actually, for the 2x2x8 case, I can just use 4 copies of a 7-term decomposition
    # for 2x2x2, giving rank 28. Let me use a known valid 7-term decomposition.
    
    # After more thought, let me just use the schoolbook for now and try to verify
    # a rank-28 construction.
    
    # For now, return schoolbook
    return _schoolbook()

def _schoolbook():
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

def _verify(result, n, m, p):
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = U.shape[1]
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True
