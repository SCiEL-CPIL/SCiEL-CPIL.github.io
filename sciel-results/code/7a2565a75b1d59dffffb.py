import numpy as np
import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 5 Strassen blocks (2x2x2) + 1 Schoolbook block (2x2x1)
    # 5 * 7 + 4 = 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add to matrices
    def add_U(i, j, val):
        U[i * m + j][r] += val
        
    def add_V(j, k, val):
        V[j * p + k][r] += val
        
    def add_W(k, i, val):
        W[k * n + i][r] += val

    # 5 Strassen blocks for columns 0-9
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Term 1: (a+d)(x+w)
        r = 7 * b + 0
        add_U(0, 0, 1); add_U(1, 1, 1) # a, d
        add_V(0, k0, 1); add_V(0, k1, 1) # x, y (Note: V index is j*p+k. j=0 is row 0 of B)
        # Wait, B is 2x11. j is row of B (0,1). k is col.
        # x is B[0, k0], y is B[0, k1], z is B[1, k0], w is B[1, k1]
        # My previous mapping:
        # V[11*k0] is B[0, k0]? No.
        # V index is j*p + k.
        # B row j, col k.
        # So V[j*p + k].
        # x = B[0, k0] -> V[0*p + k0] = V[k0]
        # y = B[0, k1] -> V[k1]
        # z = B[1, k0] -> V[1*p + k0] = V[11 + k0]
        # w = B[1, k1] -> V[11 + k1]
        
        # Let's re-verify V indices in the helper logic above.
        # In the thought process I used V[11*k0] for x.
        # 11*k0 is 11*0 + k0 if k0<11? No, 11*k0 is row k0?
        # No, V is (m*p, rank). m=2, p=11.
        # Index is j*p + k.
        # j=0, k=k0 -> k0.
        # j=0, k=k1 -> k1.
        # j=1, k=k0 -> 11+k0.
        # j=1, k=k1 -> 11+k1.
        
        # So:
        add_V(0, k0, 1); add_V(0, k1, 1)
        add_W(k0, 0, 1); add_W(k1, 1, 1)
        
        # Term 2: (c+d)x
        r = 7 * b + 1
        add_U(1, 0, 1); add_U(1, 1, 1) # c, d
        add_V(0, k0, 1) # x
        add_W(k0, 1, 1); add_W(k1, 1, -1)
        
        # Term 3: a(y-w)
        r = 7 * b + 2
        add_U(0, 0, 1) # a
        add_V(0, k1, 1); add_V(1, k0, -1) # y, -z
        add_W(k1, 0, 1); add_W(k1, 1, 1)
        
        # Term 4: d(z-x)
        r = 7 * b + 3
        add_U(1, 1, 1) # d
        add_V(1, k0, 1); add_V(0, k0, -1) # z, -x
        add_W(k0, 0, 1); add_W(k0, 1, 1)
        
        # Term 5: (a+b)w
        r = 7 * b + 4
        add_U(0, 0, 1); add_U(0, 1, 1) # a, b
        add_V(1, k1, 1) # w
        add_W(k0, 0, -1); add_W(k1, 0, 1)
        
        # Term 6: (c-a)(x+y)
        r = 7 * b + 5
        add_U(1, 0, 1); add_U(0, 0, -1) # c, -a
        add_V(0, k0, 1); add_V(0, k1, 1) # x, y
        add_W(k1, 1, 1)
        
        # Term 7: (b-d)(z+w)
        r = 7 * b + 6
        add_U(0, 1, 1); add_U(1, 1, -1) # b, -d
        add_V(1, k0, 1); add_V(1, k1, 1) # z, w
        add_W(k0, 0, 1); add_W(k1, 1, 1)

    # 1 Schoolbook block for column 10
    k = 10
    # Term 1: a x
    r = 35 + 0
    add_U(0, 0, 1)
    add_V(0, k, 1)
    add_W(k, 0, 1)
    
    # Term 2: b y
    r = 35 + 1
    add_U(0, 1, 1)
    add_V(1, k, 1)
    add_W(k, 0, 1)
    
    # Term 3: c x
    r = 35 + 2
    add_U(1, 0, 1)
    add_V(0, k, 1)
    add_W(k, 1, 1)
    
    # Term 4: d y
    r = 35 + 3
    add_U(1, 1, 1)
    add_V(1, k, 1)
    add_W(k, 1, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
