def strassen_mult(A, B):
    """Multiply two 2x2 matrices using Strassen's algorithm."""
    a, b = A[0]
    c, d = A[1]
    e, f = B[0]
    g, h = B[1]
    
    M1 = (a + d) * (e + h)
    M2 = (b + d) * e
    M3 = a * (f - h)
    M4 = d * (g + e)
    M5 = (a + b) * h
    M6 = (c - a) * (e + f)
    M7 = (d - c) * (g - h)
    
    C = [[M1 - M4 + M6 + M7, M3 + M5],
         [M2 + M4, M1 - M3 + M5 + M7]]
    return C
