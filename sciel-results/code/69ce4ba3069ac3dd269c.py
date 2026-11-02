def _strassen_2x2x2():
    """Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Indices:
    # A: i*2+j
    # B: j*2+k
    # W: k*2+i (corresponding to C[i,k])
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00(0), A11(3)
    # V: B00(0), B11(3)
    # W: C00(0), C11(3) with +1, +1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    # U: A10(2), A11(3)
    # V: B00(0)
    # W: C10(2) with +1, C11(3) with -1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    # U: A00(0)
    # V: B10(2) with +1, B11(3) with -1
    # W: C00(0) with -1
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = -1
    
    # Term 3: M4 = A11(B01-B00)
    # U: A11(3)
    # V: B01(1) with +1, B00(0) with -1
    # W: C01(1) with +1, C11(3) with +1
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    # U: A00(0), A01(1)
    # V: B11(3)
    # W: C00(0) with +1, C01(1) with +1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    # U: A10(2) with +1, A00(0) with -1
    # V: B00(0), B01(1)
    # W: C11(3) with +1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    # U: A01(1) with +1, A11(3) with -1
    # V: B10(2), B11(3)
    # W: C00(0) with -1, C10(2) with +1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W
