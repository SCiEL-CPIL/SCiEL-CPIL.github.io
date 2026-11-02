import time

DIMENSIONS = [3, 5, 5]

def _m31p(p):
    """M_{3,1,p} with rank 3*p."""
    n, m = 3, 1
    rank = n * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for k in range(p):
            U[i][col] = 1
            V[k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    return U, V, W

def _m322():
    """M_{3,2,2} with rank 6."""
    n, m, p = 3, 2, 2
    rank = 6
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _m321():
    """M_{3,2,1} with rank 3."""
    n, m, p = 3, 2, 1
    rank = 3
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            U[i*m+j][i] = 1
        V[j][i] = 1
        W[0*n+i][i] = 1
    return U, V, W

def _combine_p_split(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    r1 = len(W1[0])
    r2 = len(W2[0])
    rank = r1 + r2
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*(p1+p2))]
    W = [[0]*rank for _ in range((p1+p2)*n)]
    for i in range(n*m):
        for c in range(r1):
            U[i][c] = U1[i][c]
        for c in range(r2):
            U[i][c+r1] = U2[i][c]
    for i in range(m*p1):
        for c in range(r1):
            V[i][c] = V1[i][c]
    for i in range(m*p2):
        for c in range(r2):
            V[m*p1+i][c+r1] = V2[i][c]
    for i in range(p1*n):
        for c in range(r1):
            W[i][c] = W1[i][c]
    for i in range(p2*n):
        for c in range(r2):
            W[p1*n+i][c+r1] = W2[i][c]
    return U, V, W

def _combine_m_split(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    r1 = len(W1[0])
    r2 = len(W2[0])
    rank = r1 + r2
    U = [[0]*rank for _ in range(n*(m1+m2))]
    V = [[0]*rank for _ in range((m1+m2)*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n*m1):
        for c in range(r1):
            U[i][c] = U1[i][c]
    for i in range(n*m2):
        for c in range(r2):
            U[n*m1+i][c+r1] = U2[i][c]
    for i in range(m1*p):
        for c in range(r1):
            V[i][c] = V1[i][c]
    for i in range(m2*p):
        for c in range(r2):
            V[m1*p+i][c+r1] = V2[i][c]
    for i in range(p*n):
        for c in range(r1):
            W[i][c] = W1[i][c]
        for c in range(r2):
            W[i][c+r1] = W2[i][c]
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Split m=5 as 2+2+1
    # M_{3,5,5} = M_{3,2,5} + M_{3,2,5} + M_{3,1,5}
    
    # Build M_{3,2,5} = M_{3,2,2} + M_{3,2,3}
    # M_{3,2,3} = M_{3,2,2} + M_{3,2,1}
    
    U_322_a, V_322_a, W_322_a = _m322()
    U_321, V_321, W_321 = _m321()
    U_322_b, V_322_b, W_322_b = _m322()
    
    U_323, V_323, W_323 = _combine_p_split(U_322_a, V_322_a, W_322_a,
                                            U_321, V_321, W_321, 3, 2, 2, 1)
    U_325_a, V_325_a, W_325_a = _combine_p_split(U_322_b, V_322_b, W_322_b,
                                                  U_323, V_323, W_323, 3, 2, 2, 3)
    
    # Second M_{3,2,5}
    U_322_c, V_322_c, W_322_c = _m322()
    U_322_d, V_322_d, W_322_d = _m322()
    U_321_b, V_321_b, W_321_b = _m321()
    
    U_323_b, V_323_b, W_323_b = _combine_p_split(U_322_c, V_322_c, W_322_c,
                                                  U_321_b, V_321_b, W_321_b, 3, 2, 2, 1)
    U_325_b, V_325_b, W_325_b = _combine_p_split(U_322_d, V_322_d, W_322_d,
                                                  U_323_b, V_323_b, W_323_b, 3, 2, 2, 3)
    
    # M_{3,1,5}
    U_315, V_315, W_315 = _m31p(5)
    
    # Combine: M_{3,2,5} + M_{3,2,5} + M_{3,1,5} (split m as 2+2+1)
    U_mid, V_mid, W_mid = _combine_m_split(U_325_a, V_325_a, W_325_a,
                                            U_325_b, V_325_b, W_325_b, 3, 2, 2, 5)
    U, V, W = _combine_m_split(U_mid, V_mid, W_mid,
                               U_315, V_315, W_315, 3, 4, 1, 5)
    
    return {"U": U, "V": V, "W": W}
