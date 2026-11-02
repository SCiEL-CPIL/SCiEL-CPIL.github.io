def construct():
    deadline = time.time() + 130.0
    n, m, p = 2, 2, 5
    
    # Rank 19 decomposition
    # 7 products for Strassen (cols 0,1)
    # 4 products for each of cols 2,3,4
    
    rank = 19
    Umat = [[0] * rank for _ in range(n * m)]
    Vmat = [[0] * rank for _ in range(m * p)]
    Wmat = [[0] * rank for _ in range(p * n)]
    
    def idxU(i, j): return i * m + j
    def idxV(j, k): return j * p + k
    def idxW(k, i): return k * n + i
    
    col = 0
    
    # Strassen for cols 0, 1
    # M1 = (a+d)(B00+B10)
    Umat[idxU(0,0)][col] += 1
    Umat[idxU(1,1)][col] += 1
    Vmat[idxV(0,0)][col] += 1
    Vmat[idxV(1,0)][col] += 1
    Wmat[idxW(0,0)][col] += 1
    Wmat[idxW(0,1)][col] += 1
    Wmat[idxW(1,0)][col] += -1
    Wmat[idxW(1,1)][col] += 1
    col += 1
    
    # M2 = (c+d)B00
    Umat[idxU(1,0)][col] += 1
    Umat[idxU(1,1)][col] += 1
    Vmat[idxV(0,0)][col] += 1
    Wmat[idxW(0,0)][col] += 1
    Wmat[idxW(1,0)][col] += -1
    Wmat[idxW(1,1)][col] += 1
    col += 1
    
    # M3 = a(B11-B10)
    Umat[idxU(0,0)][col] += 1
    Vmat[idxV(1,1)][col] += 1
    Vmat[idxV(1,0)][col] += -1
    Wmat[idxW(1,0)][col] += 1
    Wmat[idxW(0,1)][col] += 1
    col += 1
    
    # M4 = d(B01-B00)
    Umat[idxU(1,1)][col] += 1
    Vmat[idxV(0,1)][col] += 1
    Vmat[idxV(0,0)][col] += -1
    Wmat[idxW(0,0)][col] += 1
    Wmat[idxW(1,0)][col] += -1
    Wmat[idxW(0,1)][col] += 1
    col += 1
    
    # M5 = (a+b)B11
    Umat[idxU(0,0)][col] += 1
    Umat[idxU(0,1)][col] += 1
    Vmat[idxV(1,1)][col] += 1
    Wmat[idxW(1,0)][col] += -1
    Wmat[idxW(0,1)][col] += 1
    col += 1
    
    # M6 = (c-a)(B00+B01)
    Umat[idxU(1,0)][col] += 1
    Umat[idxU(0,0)][col] += -1
    Vmat[idxV(0,0)][col] += 1
    Vmat[idxV(0,1)][col] += 1
    Wmat[idxW(0,1)][col] += 1
    Wmat[idxW(1,1)][col] += 1
    col += 1
    
    # M7 = (b-d)(B10+B11)
    Umat[idxU(0,1)][col] += 1
    Umat[idxU(1,1)][col] += -1
    Vmat[idxV(1,0)][col] += 1
    Vmat[idxV(1,1)][col] += 1
    Wmat[idxW(0,0)][col] += 1
    Wmat[idxW(0,1)][col] += 1
    Wmat[idxW(1,1)][col] += 1
    col += 1
    
    # Extra columns 2, 3, 4
    for k in range(2, p):
        # P1 = a B0k -> C0k
        Umat[idxU(0,0)][col] += 1
        Vmat[idxV(0,k)][col] += 1
        Wmat[idxW(k,0)][col] += 1
        col += 1
        
        # P2 = d B1k -> C1k
        Umat[idxU(1,1)][col] += 1
        Vmat[idxV(1,k)][col] += 1
        Wmat[idxW(k,1)][col] += 1
        col += 1
        
        # P3 = b B1k -> C0k
        Umat[idxU(0,1)][col] += 1
        Vmat[idxV(1,k)][col] += 1
        Wmat[idxW(k,0)][col] += 1
        col += 1
        
        # P4 = c B0k -> C1k
        Umat[idxU(1,0)][col] += 1
        Vmat[idxV(0,k)][col] += 1
        Wmat[idxW(k,1)][col] += 1
        col += 1
        
    # Convert to int/half
    Uout = [[_as_int_or_half(x) for x in row] for row in Umat]
    Vout = [[_as_int_or_half(x) for x in row] for row in Vmat]
    Wout = [[_as_int_or_half(x) for x in row] for row in Wmat]
    
    result = {"U": Uout, "V": Vout, "W": Wout}
    
    if time.time() < deadline - 5.0:
        if _verify(Uout, Vout, Wout, n, m, p):
            return result
            
    return _schoolbook(n, m, p)
