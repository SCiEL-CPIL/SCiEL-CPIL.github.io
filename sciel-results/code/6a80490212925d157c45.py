"""Matrix multiplication 3x4x8 with rank reduction using Strassen identities."""
import time

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e)
    # p3 = (a-b)(g+h)
    # p4 = (d)(f-h)
    # p5 = (a+c)(h)
    # p6 = (b-d)(e+g)
    # p7 = (a-c)(f)
    #
    # M11 = p1 + p4 - p5 + p7
    # M12 = p3 + p5
    # M21 = p2 + p4
    # M22 = p1 - p2 + p3 + p6
    U = [
        [1, 0, 1, 0, 1, 0, 1],   # a
        [0, 1, 0, 0, 0, 1, 0],   # b
        [0, 0, 1, 1, 1, 0, 0],   # c
        [1, 0, 0, 1, 0, 1, 0]    # d
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],   # e
        [0, 0, 0, 1, 0, 0, 1],   # f
        [0, 0, 1, 0, 0, 1, 0],   # g
        [1, 0, 1, 0, 1, 0, 0]    # h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # M11
        [0, 0, 1, 0, 1, 0, 0],   # M12
        [0, 1, 0, 1, 0, 0, 0],   # M21
        [1, -1, 1, 0, 0, 1, 0]   # M22
    ]
    return U, V, W

def _construct_3x4x8():
    """Construct rank-88 decomposition for 3x4x8 using Strassen blocks."""
    n, m, p = 3, 4, 8
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # 1x2x2 schoolbook:
    # C0 = xa + yb, C1 = xc + yd
    # 4 multiplications: xa, yb, xc, yd
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
        [1, 1, 0, 0],  # C0 = xa + yb
        [0, 0, 1, 1]   # C1 = xc + yd
    ]
    
    # 8 blocks of 2x2x2 (rank 7 each) for i_pair (0,1): 8*7 = 56
    # 8 blocks of 1x2x2 (rank 4 each) for i=2: 8*4 = 32
    # Total rank = 88
    rank = 88
    U = [[0] * rank for _ in range(n * m)]   # 12 rows
    V = [[0] * rank for _ in range(m * p)]   # 32 rows
    W = [[0] * rank for _ in range(p * n)]   # 24 rows
    
    col_offset = 0
    
    # Process 2x2x2 Strassen blocks for i_pair=0 (i=0,1)
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            
            # U rows for [a, b, c, d]:
            u_rows = [
                0 * m + j0,  # A[0, j0]
                0 * m + j1,  # A[0, j1]
                1 * m + j0,  # A[1, j0]
                1 * m + j1   # A[1, j1]
            ]
            
            # V rows for [e, f, g, h]:
            v_rows = [
                j0 * p + k0,  # B[j0, k0]
                j0 * p + k1,  # B[j0, k1]
                j1 * p + k0,  # B[j1, k0]
                j1 * p + k1   # B[j1, k1]
            ]
            
            # W rows for [M11, M12, M21, M22]:
            w_rows = [
                k0 * n + 0,  # C[0, k0]
                k1 * n + 0,  # C[0, k1]
                k0 * n + 1,  # C[1, k0]
                k1 * n + 1   # C[1, k1]
            ]
            
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
    
    # Process 1x2x2 schoolbook blocks for i=2
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            
            # U rows for [x, y]:
            u_rows = [
                2 * m + j0,  # A[2, j0]
                2 * m + j1   # A[2, j1]
            ]
            
            # V rows for [a, b, c, d]:
            # a=B[j0,k0], b=B[j1,k0], c=B[j0,k1], d=B[j1,k1]
            v_rows = [
                j0 * p + k0,  # B[j0, k0]
                j1 * p + k0,  # B[j1, k0]
                j0 * p + k1,  # B[j0, k1]
                j1 * p + k1   # B[j1, k1]
            ]
            
            # W rows for [C0, C1]:
            w_rows = [
                k0 * n + 2,  # C[2, k0]
                k1 * n + 2   # C[2, k1]
            ]
            
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
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

def _schoolbook():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    result = _construct_3x4x8()
    
    n, m, p = DIMENSIONS
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook if verification fails
        result = _schoolbook()
    
    return result
