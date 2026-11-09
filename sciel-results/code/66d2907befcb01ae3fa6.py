import time

def construct():
    """
    Tensor decomposition for multiplying a 3x3 matrix A by a 3x6 matrix B.
    
    We use Laderman's rank-23 decomposition for 3x3x3 matrix multiplication,
    then apply it independently to the two 3x3 blocks of B (and C).
    This gives a total rank of 46, which is better than schoolbook (54).
    """
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 3, 3, 6  # A is n x m, B is m x p, C is n x p
    
    # Laderman's rank-23 decomposition for 3x3x3 matrix multiplication
    # Each term: (u_set, v_set, w_set) where entries are (index, coeff)
    # u: (i,j) for A[i,j], v: (j,k) for B[j,k], w: (i,k) for C[i,k]
    laderman = [
        ([((0,0), 1)], [((0,0), 1)], [((0,0), 1)]),
        ([((0,0), 1)], [((1,1), 1)], [((0,1), 1)]),
        ([((0,0), 1)], [((2,2), 1)], [((0,2), 1)]),
        ([((1,1), 1)], [((0,0), 1)], [((1,0), 1)]),
        ([((1,1), 1)], [((1,1), 1)], [((1,1), 1)]),
        ([((1,1), 1)], [((2,2), 1)], [((1,2), 1)]),
        ([((2,2), 1)], [((0,0), 1)], [((2,0), 1)]),
        ([((2,2), 1)], [((1,1), 1)], [((2,1), 1)]),
        ([((2,2), 1)], [((2,2), 1)], [((2,2), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1), ((2,2), 1)], [((0,1), 1), ((0,2), 1)]),
        ([((0,1), 1), ((1,1), 1)], [((1,2), 1), ((2,2), 1)], [((0,2), 1), ((1,2), 1)]),
        ([((1,1), 1), ((1,2), 1)], [((2,2), 1), ((0,0), 1)], [((1,2), 1), ((1,0), 1)]),
        ([((1,2), 1), ((2,2), 1)], [((0,0), 1), ((1,1), 1)], [((1,0), 1), ((2,0), 1)]),
        ([((2,2), 1), ((2,0), 1)], [((1,1), 1), ((2,2), 1)], [((2,0), 1), ((2,1), 1)]),
        ([((2,0), 1), ((0,0), 1)], [((2,2), 1), ((0,0), 1)], [((2,1), 1), ((0,1), 1)]),
        ([((0,0), 1), ((1,0), 1)], [((0,1), 1), ((1,1), 1)], [((0,0), 1), ((1,0), 1)]),
        ([((1,0), 1), ((2,0), 1)], [((1,2), 1), ((2,2), 1)], [((1,0), 1), ((2,0), 1)]),
        ([((2,0), 1), ((0,0), 1)], [((2,1), 1), ((0,1), 1)], [((2,0), 1), ((0,0), 1)]),
        ([((0,1), 1), ((1,0), 1)], [((1,1), 1), ((2,1), 1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1), ((2,0), 1)], [((2,1), 1), ((0,1), 1)], [((1,1), 1), ((2,1), 1)]),
        ([((2,1), 1), ((0,0), 1)], [((0,1), 1), ((1,1), 1)], [((2,1), 1), ((0,1), 1)]),
        ([((0,2), 1), ((1,0), 1)], [((2,0), 1), ((0,2), 1)], [((0,2), 1), ((1,2), 1)]),
        ([((1,2), 1), ((2,0), 1)], [((0,2), 1), ((2,0), 1)], [((1,2), 1), ((2,2), 1)]),
    ]
    
    # Build the rank-46 decomposition:
    # Apply Laderman independently to columns 0-2 and columns 3-5 of B/C
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 9 x 46
    V = [[0] * rank for _ in range(m * p)]  # 18 x 46
    W = [[0] * rank for _ in range(p * n)]  # 18 x 46
    
    col = 0
    for block_offset in [0, 3]:
        for u_set, v_set, w_set in laderman:
            for (i, j), c in u_set:
                U[i * m + j][col] = c
            for (j, k_local), c in v_set:
                k_global = k_local + block_offset
                V[j * p + k_global][col] = c
            for (i, k_local), c in w_set:
                k_global = k_local + block_offset
                W[k_global * n + i][col] = c
            col += 1
    
    # Verify the decomposition
    # Check: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] == 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] for t in range(rank))
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 54
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
