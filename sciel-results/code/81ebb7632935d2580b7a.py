DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(5):
        b0 = 2 * b
        b1 = 2 * b + 1
        
        # M1 = (A00+A11)*(B[0,b0]+B[1,b1])
        U[0][col] += 1
        U[3][col] += 1
        V[0*p + b0][col] += 1
        V[1*p + b1][col] += 1
        W[b0*n + 0][col] += 1
        W[b1*n + 1][col] += 1
        col += 1
        
        # M2 = (A10+A11)*B[0,b0]
        U[2][col] += 1
        U[3][col] += 1
        V[0*p + b0][col] += 1
        W[b0*n + 1][col] += 1
        W[b1*n + 1][col] -= 1
        col += 1
        
        # M3 = A00*(B[1,b0]-B[1,b1])
        U[0][col] += 1
        V[1*p + b0][col] += 1
        V[1*p + b1][col] -= 1
        W[b1*n + 0][col] += 1
        W[b1*n + 1][col] += 1
        col += 1
        
        # M4 = A11*(B[0,b1]-B[0,b0])
        U[3][col] += 1
        V[0*p + b1][col] += 1
        V[0*p + b0][col] -= 1
        W[b0*n + 0][col] += 1
        W[b0*n + 1][col] += 1
        col += 1
        
        # M5 = (A00+A01)*B[1,b1]
        U[0][col] += 1
        U[1][col] += 1
        V[1*p + b1][col] += 1
        W[b0*n + 0][col] -= 1
        W[b1*n + 0][col] += 1
        col += 1
        
        # M6 = (A10-A00)*(B[0,b0]+B[1,b0])
        U[2][col] += 1
        U[0][col] -= 1
        V[0*p + b0][col] += 1
        V[1*p + b0][col] += 1
        W[b1*n + 1][col] += 1
        col += 1
        
        # M7 = (A01-A11)*(B[0,b1]+B[1,b1])
        U[1][col] += 1
        U[3][col] -= 1
        V[0*p + b1][col] += 1
        V[1*p + b1][col] += 1
        W[b0*n + 0][col] += 1
        col += 1
    
    b_last = 10
    U[0][col] += 1
    V[0*p + b_last][col] += 1
    W[b_last*n + 0][col] += 1
    col += 1
    U[1][col] += 1
    V[1*p + b_last][col] += 1
    W[b_last*n + 0][col] += 1
    col += 1
    U[2][col] += 1
    V[0*p + b_last][col] += 1
    W[b_last*n + 1][col] += 1
    col += 1
    U[3][col] += 1
    V[1*p + b_last][col] += 1
    W[b_last*n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
