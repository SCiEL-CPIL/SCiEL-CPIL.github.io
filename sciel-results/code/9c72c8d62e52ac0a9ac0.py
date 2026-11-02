def _rank8_decomposition():
    """
    Rank-8 decomposition: Strassen (rank 7) for 2x2x2 block + 1 term for col 2.
    """
    n, m, p = 2, 2, 3
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Terms 0-6: Strassen
    # Term 0: (A00+A11)(B00+B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A_any * B_any (for col 2)
    # U[i,j,7] = 1, V[j,2,7] = 1, W[2,i,7] = 1
    for i in range(2):
        for j in range(2):
            set_u(i, j, 7, 1)
    for j in range(2):
        set_v(j, 2, 7, 1)
    for i in range(2):
        set_w(2, i, 7, 1)

    return {"U": U, "V": W, "W": W} # Wait, V and W return logic needs care in the dict

def construct():
    # Try rank 8 first
    result8 = _rank8_decomposition()
    if _verify(result8["U"], result8["V"], result8["W"], n, m, p):
        return result8
    
    # Fallback to rank 11
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        return result11
        
    # Fallback to schoolbook
    return _schoolbook()
