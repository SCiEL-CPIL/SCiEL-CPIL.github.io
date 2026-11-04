"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 77 decomposition using block structure
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1
        
    # Indices: 0..3 are S, 4 is T
    S = range(4)
    T = 4
    
    # 1. C[S, S] += A[S, S] B[S, S] (16 terms)
    for i in S:
        for k in S:
            for j in S:
                add_term(i * m + k, k * p + j, j * n + i)
                
    # 2. C[S, S] += A[S, T] B[T, S] (16 terms)
    for i in S:
        for j in S:
            add_term(i * m + T, T * p + j, j * n + i)
            
    # 3. C[S, T] += A[S, S] B[S, T] (16 terms)
    for i in S:
        for k in S:
            add_term(i * m + k, k * p + T, T * n + i)
            
    # 4. C[S, T] += A[S, T] B[T, T] (4 terms)
    for i in S:
        add_term(i * m + T, T * p + T, T * n + i)
        
    # 5. C[T, S] += A[T, S] B[S, S] (16 terms)
    for k in S:
        for j in S:
            add_term(T * m + k, k * p + j, j * n + T)
            
    # 6. C[T, S] += A[T, T] B[T, S] (4 terms)
    for j in S:
        add_term(T * m + T, T * p + j, j * n + T)
        
    # 7. C[T, T] += A[T, S] B[S, T] (4 terms)
    for k in S:
        add_term(T * m + k, k * p + T, T * n + T)
        
    # 8. C[T, T] += A[T, T] B[T, T] (1 term)
    add_term(T * m + T, T * p + T, T * n + T)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
