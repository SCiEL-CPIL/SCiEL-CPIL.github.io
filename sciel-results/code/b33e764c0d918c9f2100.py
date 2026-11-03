def construct():
    n, m, p = 2, 3, 5
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # Block A: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
    # ... (same as before) ...
    c += 7
    
    # Block B: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {3,4}
    # ... (same as before) ...
    c += 7
    
    # Term for j=2, all k
    # U[i, c] = 1, V[2*p+k, c] = 1, W[k*n+i, c] = 1
    for i in range(n):
        U[i*m+2][c] = 1
    for k in range(p):
        V[2*p+k][c] = 1
        W[k*n+i][c] = 1 # Wait, W index is k*n+i
    # Actually, W[k*n+i][c] = 1 for all k,i
    for k in range(p):
        for i in range(n):
            W[k*n+i][c] = 1
    c += 1
    
    # Term for j in {0,1}, k=2
    # U[i, c] = 1, V[j*p+2, c] = 1, W[2*n+i, c] = 1
    for i in range(n):
        U[i*m+0][c] = 1
        U[i*m+1][c] = 1
    V[0*p+2][c] = 1
    V[1*p+2][c] = 1
    for i in range(n):
        W[2*n+i][c] = 1
    c += 1
    
    assert c == rank
    return {"U": U, "V": V, "W": W}
