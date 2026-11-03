def strassen_2x2(A, B):
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]
    e, f = B[0][0], B[0][1]
    g, h = B[1][0], B[1][1]
    
    M1 = (a + d) * (e + h)
    M2 = (c + d) * (e + f)
    M3 = (a - b) * (h - g) # Wait, is it h-g?
    M4 = (d - c) * (g + f)
    M5 = (a + e) * (g + i) # No i.
    M6 = (b - d) * (h + g)
    M7 = (c - a) * (f + e)
