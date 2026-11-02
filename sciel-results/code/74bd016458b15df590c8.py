import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen M1: (a+e)(g+i)
        # U: a(0,0), e(1,1)
        # V: g(0, k0), i(1, k0)
        # W: C(0, k0), C(1, k0)
        U[0][col] += 1; U[3][col] += 1
        V[0*p+k0][col] += 1; V[1*p+k0][col] += 1
        W[k0*n+0][col] += 1; W[k0*n+1][col] += 1
        col += 1
        
        # Strassen M2: (d+e)(g+h)
        # U: d(1,0), e(1,1)
        # V: g(0, k0), h(0, k1)
        # W: C(1, k0), C(1, k1)
        U[1*m+0][col] += 1; U[1*m+1][col] += 1
        V[0*p+k0][col] += 1; V[0*p+k1][col] += 1
        W[k0*n+1][col] += 1; W[k1*n+1][col] += 1
        col += 1
        
        # Strassen M3: (a-b)(i+j)
        # U: a(0,0), b(0,1)
        # V: i(1, k0), j(1, k1)
        # W: C(0, k1), C(1, k1)
        U[0*m+0][col] += 1; U[0*m+1][col] -= 1
        V[1*p+k0][col] += 1; V[1*p+k1][col] += 1
        W[k1*n+0][col] += 1; W[k1*n+1][col] += 1
        col += 1
        
        # Strassen M4: e(h+j)
        # U: e(1,1)
        # V: h(0, k1), j(1, k1)
        # W: C(0, k0), C(1, k0)
        U[1*m+1][col] += 1
        V[0*p+k1][col] += 1; V[1*p+k1][col] += 1
        W[k0*n+0][col] += 1; W[k0*n+1][col] += 1
        col += 1
        
        # Strassen M5: a(g+i)
        # U: a(0,0)
        # V: g(0, k0), i(1, k0)
        # W: C(0, k0), C(0, k1)
        U[0*m+0][col] += 1
        V[0*p+k0][col] += 1; V[1*p+k0][col] += 1
        W[k0*n+0][col] += 1; W[k1*n+0][col] += 1
        col += 1
        
        # Strassen M6: (b+d)(j-i)
        # U: b(0,1), d(1,0)
        # V: j(1, k1), i(1, k0)
        # W: C(1, k1)
        U[0*m+1][col] += 1; U[1*m+0][col] += 1
        V[1*p+k1][col] += 1; V[1*p+k0][col] -= 1
        W[k1*n+1][col] += 1
        col += 1
        
        # Strassen M7: (a-d)(g-h)
        # U: a(0,0), d(1,0)
        # V: g(0, k0), h(0, k1)
        # W: C(0, k0), C(1, k0)
        U[0*m+0][col] += 1; U[1*m+0][col] -= 1
        V[0*p+k0][col] += 1; V[0*p+k1][col] -= 1
        W[k0*n+0][col] -= 1; W[k0*n+1][col] -= 1
        col += 1
        
        # Tail M8: c*k
        # U: c(0,2)
        # V: k(2, k0)
        # W: C(0, k0)
        U[0*m+2][col] += 1
        V[2*p+k0][col] += 1
        W[k0*n+0][col] += 1
        col += 1
        
        # Tail M9: c*l
        # U: c(0,2)
        # V: l(2, k1)
        # W: C(0, k1)
        U[0*m+2][col] += 1
        V[2*p+k1][col] += 1
        W[k1*n+0][col] += 1
        col += 1
        
        # Tail M10: f*k
        # U: f(1,2)
        # V: k(2, k0)
        # W: C(1, k0)
        U[1*m+2][col] += 1
        V[2*p+k0][col] += 1
        W[k0*n+1][col] += 1
        col += 1
        
        # Tail M11: f*l
        # U: f(1,2)
        # V: l(2, k1)
        # W: C(1, k1)
        U[1*m+2][col] += 1
        V[2*p+k1][col] += 1
        W[k1*n+1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
