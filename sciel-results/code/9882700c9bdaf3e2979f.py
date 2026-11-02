"""Tensor decomposition for M_{2,2,15} using block Strassen algorithm."""

def construct():
    n, m, p = 2, 2, 15
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 7 blocks of 2 columns using Strassen (rank 7 each)
    for b in range(7):
        k0, k1 = 2 * b, 2 * b + 1
        
        # M1 = (A00+A11)(B0k0+B1k1) -> C0k0 + C1k1
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[15+k1][col] = 1
        W[2*k0][col] = 1; W[2*k1+1][col] = 1
        col += 1
        
        # M2 = (A10+A11)B0k0 -> C1k0 - C1k1
        U[2][col] = 1; U[3][col] = 1
        V[k0][col] = 1
        W[2*k0+1][col] = 1; W[2*k1+1][col] = -1
        col += 1
        
        # M3 = A00(B0k1-B1k1) -> C0k1 + C1k1
        U[0][col] = 1
        V[k1][col] = 1; V[15+k1][col] = -1
        W[2*k1][col] = 1; W[2*k1+1][col] = 1
        col += 1
        
        # M4 = A11(B1k0-B0k0) -> C0k0 + C1k0
        U[3][col] = 1
        V[15+k0][col] = 1; V[k0][col] = -1
        W[2*k0][col] = 1; W[2*k0+1][col] = 1
        col += 1
        
        # M5 = (A00+A01)B1k1 -> -C0k0 + C0k1
        U[0][col] = 1; U[1][col] = 1
        V[15+k1][col] = 1
        W[2*k0][col] = -1; W[2*k1][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B0k0+B0k1) -> C1k1
        U[2][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B1k0+B1k1) -> C0k0
        U[1][col] = 1; U[3][col] = -1
        V[15+k0][col] = 1; V[15+k1][col] = 1
        W[2*k0][col] = 1
        col += 1
    
    # Last single column (k=14) using schoolbook (rank 4)
    k = 14
    # C0k = A00*B0k
    U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
    # C0k += A01*B1k
    U[1][col] = 1; V[15+k][col] = 1; W[2*k][col] = 1; col += 1
    # C1k = A10*B0k
    U[2][col] = 1; V[k][col] = 1; W[2*k+1][col] = 1; col += 1
    # C1k += A11*B1k
    U[3][col] = 1; V[15+k][col] = 1; W[2*k+1][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
