import numpy as np

def strassen_block(i0, j0, k0, n, m, p):
    """Return 7 (U, V, W) factor vectors for a 2x2x2 Strassen block."""
    terms = []
    # Term 1: (a+d)(e+h) -> C11 += 1, C22 += 1
    U = [0]*8  # 2x2
    U[0] = 1; U[3] = 1  # a, d
    V = [0]*8
    V[0] = 1; V[3] = 1  # e, h
    W = [0]*8  # 2x2 for C
    W[0] = 1; W[3] = 1  # C11, C22
    terms.append((U, V, W))
    
    # Term 2: (c+d)e -> C21 += 1, C22 -= 1
    U = [0]*8
    U[2] = 1; U[3] = 1  # c, d
    V = [0]*8
    V[0] = 1  # e
    W = [0]*8
    W[2] = 1; W[3] = -1  # C21, C22
    terms.append((U, V, W))
    
    # Term 3: a(f-h) -> C12 += 1, C22 += 1
    U = [0]*8
    U[0] = 1  # a
    V = [0]*8
    V[1] = 1; V[2] = -1  # f, -g
    W = [0]*8
    W[1] = 1; W[3] = 1  # C12, C22
    terms.append((U, V, W))
    
    # Term 4: d(g-e) -> C11 += 1, C21 += 1
    U = [0]*8
    U[3] = 1  # d
    V = [0]*8
    V[0] = -1; V[2] = 1  # -e, g
    W = [0]*8
    W[0] = 1; W[2] = 1  # C11, C21
    terms.append((U, V, W))
    
    # Term 5: (a+b)h -> C11 -= 1, C12 += 1
    U = [0]*8
    U[0] = 1; U[1] = 1  # a, b
    V = [0]*8
    V[3] = 1  # h
    W = [0]*8
    W[0] = -1; W[1] = 1  # C11, C12
    terms.append((U, V, W))
    
    # Term 6: (c-a)(e+f) -> C22 += 1
    U = [0]*8
    U[0] = -1; U[2] = 1  # -a, c
    V = [0]*8
    V[0] = 1; V[1] = 1  # e, f
    W = [0]*8
    W[3] = 1  # C22
    terms.append((U, V, W))
    
    # Term 7: (b-d)(g+h) -> C11 += 1, C22 += 1
    U = [0]*8
    U[1] = 1; U[3] = -1  # b, -d
    V = [0]*8
    V[2] = 1; V[3] = 1  # g, h
    W = [0]*8
    W[0] = 1; W[3] = 1  # C11, C22
    terms.append((U, V, W))
    
    return terms
