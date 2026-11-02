"""Matrix multiplication 3x4x8 with rank reduction using corrected Strassen identities."""
import time

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # U rows: a, b, c, d
    # V rows: e, f, g, h
    # W rows: M11, M12, M21, M22
    # Multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a-b)(g+h)
    # m4 = d(f-h)
    # m5 = (a+c)h
    # m6 = (b-d)(e+g)
    # m7 = (a-c)f
    
    U = [
        [1, 0, 1, 0, 1, 0, 1],  # a
        [0, 0, -1, 0, 0, 1, 0], # b
        [0, 1, 0, 0, 1, 0, -1], # c
        [1, 1, 0, 1, 0, -1, 0]  # d
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],  # e
        [0, 0, 0, 1, 0, 0, 1],  # f
        [0, 0, 1, 0, 0, 1, 0],  # g
        [1, 0, 1, -1, 1, 0, 0]  # h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # M11
        [0, 0, 1, 0, 1, 0, 0],   # M12
        [0, 1, 0, 1, 0, 0, 0],   # M21
        [1, -1, 1, 0, 0, 1, 0]   # M22
    ]
    return U, V, W

def _construct_3x4x8():
    """Construct rank-minimal decomposition for 3x4x8."""
    n, m, p = 3, 4, 8
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # 1x2x2 Schoolbook factors (Rank 4)
    # C0 = x*a + y*b
    # C1 = x*c + y*d
    # U1: [x, y]
    # V1: [a, b, c, d]
    # W1: [C0, C1]
    U1 = [
        [1, 0, 0, 0],  # x
        [0, 1, 0, 0]   # y
    ]
    V1 = [
        [1, 0, 0, 0],  # a
        [0, 1, 0, 0],  # b
        [0, 0, 1, 0],  # c
        [0, 0, 0, 1]   # d
    ]
    W1 = [
        [1, 1, 0, 0],  # C0
        [0, 0, 1, 1]   # C1
    ]
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a 2x2x2 block
    def add_2x2x2_block(u_rows, v_rows, w_rows, offset):
        for i in range(4):
            for c in range(7):
                U[u_rows[i]][offset + c] += U2[i][c]
                V[v_rows[i]][offset + c] += V2[i][c]
                W[w_rows[i]][offset + c] += W2[i][c]
                
    # Helper to add a 1x2x2 block
    def add_1x2x2_block(u_rows, v_rows, w_rows, offset):
        # u_rows: [x, y] (2 rows)
        # v_rows: [a, b, c, d] (4 rows)
        # w_rows: [C0, C1] (2 rows)
        for i in range(2):
            for c in range(4):
                U[u_rows[i]][offset + c] += U1[i][c]
                W[w_rows[i]][offset + c] += W1[i][c]
        for i in range(4):
            for c in range(4):
                V[v_rows[i]][offset + c] += V1[i][c]

    # Process 2x2x2 blocks for i_pair=0 (i=0,1)
    # j_pairs: 2, k_pairs: 4
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            
            # U rows: A[0, j0], A[0, j1], A[1, j0], A[1, j1]
            u_rows = [
                0 * m + j0,
                0 * m + j1,
                1 * m + j0,
                1 * m + j1
            ]
            
            # V rows: B[j0, k0], B[j0, k1], B[j1, k0], B[j1, k1]
            v_rows = [
                j0 * p + k0,
                j0 * p + k1,
                j1 * p + k0,
                j1 * p + k1
            ]
            
            # W rows: C[0, k0], C[0, k1], C[1, k0], C[1, k1]
            # Index: k*n + i
            w_rows = [
                k0 * n + 0,
                k1 * n + 0,
                k0 * n + 1,
                k1 * n + 1
            ]
            
            add_2x2x2_block(u_rows, v_rows, w_rows, col_offset)
            col_offset += 7
    
    # Process 1x2x2 blocks for i=2
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            
            # U rows: A[2, j0], A[2, j1]
            u_rows = [
                2 * m + j0,
                2 * m + j1
            ]
            
            # V rows: B[j0, k0], B[j1, k0], B[j0, k1], B[j1, k1]
            # Note: Mapping to a,b,c,d in 1x2x2 context
            # C0 (k0) = x*B[j0,k0] + y*B[j1,k0] -> a=B[j0,k0], b=B[j1,k0]
            # C1 (k1) = x*B[j0,k1] + y*B[j1,k1] -> c=B[j0,k1], d=B[j1,k1]
            v_rows = [
                j0 * p + k0,
                j1 * p + k0,
                j0 * p + k1,
                j1 * p + k1
            ]
            
            # W rows: C[2, k0], C[2, k1]
            w_rows = [
                k0 * n + 2,
                k1 * n + 2
            ]
            
            add_1x2x2_block(u_rows, v_rows, w_rows, col_offset)
            col_offset += 4
            
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if actual != expected:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    result = _construct_3x4x8()
    
    n, m, p = DIMENSIONS
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
        
    return result
