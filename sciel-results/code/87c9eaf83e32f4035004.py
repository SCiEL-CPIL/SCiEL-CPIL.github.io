import time

DIMENSIONS = [4, 4, 4]

def _time_check(start, limit=145.0):
    return (time.time() - start) < limit

def _build_strassen_2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    U: shape (4, 7)  -- rows indexed by i*m+j = i*2+j
    V: shape (4, 7)  -- rows indexed by j*p+k = j*2+k
    W: shape (4, 7)  -- rows indexed by k*n+i = k*2+i
    Convention: (C)_{i,k} = sum_j A_{i,j} B_{j,k}
    W index: k*n + i = k*2 + i
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]  # 4 rows
    V = [[0]*rank for _ in range(m*p)]  # 4 rows
    W = [[0]*rank for _ in range(p*n)]  # 4 rows

    # Strassen formulas for 2x2:
    # X1 = A00 + A11
    # X2 = A10 + A11
    # X3 = A00 - A10
    # X4 = A01 - A11
    # X5 = A00 + A01
    # X6 = A10 - A00
    # X7 = A01 - A11
    #
    # Y1 = B00 + B11
    # Y2 = B00
    # Y3 = B11
    # Y4 = B01 - B11
    # Y5 = B00 + B01
    # Y6 = B10 - B00
    # Y7 = B10 + B11
    #
    # M1 = X1 * Y1
    # M2 = X2 * Y2
    # M3 = X3 * Y3
    # M4 = X4 * Y4
    # M5 = X5 * Y5
    # M6 = X6 * Y6
    # M7 = X7 * Y7
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U rows: i*2 + j for i in {0,1}, j in {0,1}
    # U[0] = row for A00 (i=0, j=0)
    # U[1] = row for A01 (i=0, j=1)
    # U[2] = row for A10 (i=1, j=0)
    # U[3] = row for A11 (i=1, j=1)

    # V rows: j*2 + k for j in {0,1}, k in {0,1}
    # V[0] = row for B00 (j=0, k=0)
    # V[1] = row for B01 (j=0, k=1)
    # V[2] = row for B10 (j=1, k=0)
    # V[3] = row for B11 (j=1, k=1)

    # W rows: k*2 + i for k in {0,1}, i in {0,1}
    # W[0] = row for C00 (k=0, i=0)
    # W[1] = row for C01 (k=0, i=1)
    # W[2] = row for C10 (k=1, i=0)
    # W[3] = row for C11 (k=1, i=1)

    # Factor 1: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # Factor 2: M2 = (A10+A11)(B00)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # Factor 3: M3 = (A00-A10)(B11)
    U[0][2] = 1; U[2][2] = -1
    V[3][2] = 1
    W[1][2] = 1; W[3][2] = 1

    # Factor 4: M4 = (A01-A11)(B01-B11)
    U[1][3] = 1; U[3][3] = -1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # Factor 5: M5 = (A00+A01)(B00+B01)
    U[0][4] = 1; U[1][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # Factor 6: M6 = (A10-A00)(B10-B00)
    U[2][5] = 1; U[0][5] = -1
    V[2][5] = 1; V[0][5] = -1
    W[3][5] = 1

    # Factor 7: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W

def _tensor_decompose_recursive(n, m, p):
    """
    Recursive decomposition. For n=m=p=2, use Strassen.
    For larger even dimensions, use block Strassen.
    For odd or non-power-of-2, fall back to schoolbook for that level.
    
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank).
    """
    if n == 2 and m == 2 and p == 2:
        return _build_strassen_2x2()
    
    # Check if we can do block Strassen
    # We need n, m, p all even for clean 2x2 blocking
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        # Block Strassen: 2x2 blocks of size (n/2, m/2, p/2)
        n2, m2, p2 = n // 2, m // 2, p // 2
        sub_U, sub_V, sub_W = _tensor_decompose_recursive(n2, m2, p2)
        sub_rank = len(sub_U[0])
        
        rank = 7 * sub_rank
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # For each of the 7 Strassen factors, place the sub-decomposition
        # into the appropriate blocks of A, B, and C.
        
        # A is n x m, split into 4 blocks of size (n2, m2):
        # A00: rows 0..n2-1, cols 0..m2-1
        # A01: rows 0..n2-1, cols m2..m-1
        # A10: rows n2..n-1, cols 0..m2-1
        # A11: rows n2..n-1, cols m2..m-1
        # A row index: i*m + j
        # A block (bi, bj) with bi,bj in {0,1}: rows bi*n2..(bi+1)*n2-1, cols bj*m2..(bj+1)*m2-1
        # Row index for A at block (bi,bj), local (i,j): (bi*n2 + i) * m + (bj*m2 + j)
        
        # B is m x p, split into 4 blocks of size (m2, p2):
        # B00: rows 0..m2-1, cols 0..p2-1
        # B01: rows 0..m2-1, cols p2..p-1
        # B10: rows m2..m-1, cols 0..p2-1
        # B11: rows m2..m-1, cols p2..p-1
        # B row index: j*p + k
        # B block (bj, bk): rows bj*m2..(bj+1)*m2-1, cols bk*p2..(bk+1)*p2-1
        # Row index for B at block (bj,bk), local (j,k): (bj*m2 + j) * p + (bk*p2 + k)
        
        # C is n x p, split into 4 blocks of size (n2, p2):
        # C00: rows 0..n2-1, cols 0..p2-1
        # C01: rows 0..n2-1, cols p2..p-1
        # C10: rows n2..n-1, cols 0..p2-1
        # C11: rows n2..n-1, cols p2..p-1
        # C row index (W convention): k*n + i
        # C block (ck, ci) with ck,ci in {0,1} (ck for column/k, ci for row/i):
        # C block (ci, ck): rows ci*n2..(ci+1)*n2-1, cols ck*p2..(ck+1)*p2-1
        # W row index for C at block (ci,ck), local (i,k): (ck*p2 + k) * n + (ci*n2 + i)
        
        # Strassen factor definitions:
        # For factor f in 0..6:
        #   X_f = sum of A blocks with coefficients
        #   Y_f = sum of B blocks with coefficients
        #   Z_f = sum of C blocks with coefficients
        # Each sub-decomposition factor s in 0..sub_rank-1:
        #   sub_U gives A contribution, sub_V gives B contribution, sub_W gives C contribution
        
        # A block coefficients for each Strassen factor:
        # Factor 0 (M1): A00 + A11
        # Factor 1 (M2): A10 + A11
        # Factor 2 (M3): A00 - A10
        # Factor 3 (M4): A01 - A11
        # Factor 4 (M5): A00 + A01
        # Factor 5 (M6): A10 - A00
        # Factor 6 (M7): A01 - A11
        A_blocks = [
            [(0,0,1), (1,1,1)],      # f0
            [(1,0,1), (1,1,1)],      # f1
            [(0,0,1), (1,0,-1)],     # f2
            [(0,1,1), (1,1,-1)],     # f3
            [(0,0,1), (0,1,1)],      # f4
            [(1,0,1), (0,0,-1)],     # f5
            [(0,1,1), (1,1,-1)],     # f6
        ]
        
        # B block coefficients for each Strassen factor:
        # Factor 0 (M1): B00 + B11
        # Factor 1 (M2): B00
        # Factor 2 (M3): B11
        # Factor 3 (M4): B01 - B11
        # Factor 4 (M5): B00 + B01
        # Factor 5 (M6): B10 - B00
        # Factor 6 (M7): B10 + B11
        B_blocks = [
            [(0,0,1), (1,1,1)],      # f0
            [(0,0,1)],               # f1
            [(1,1,1)],               # f2
            [(0,1,1), (1,1,-1)],     # f3
            [(0,0,1), (0,1,1)],      # f4
            [(1,0,1), (0,0,-1)],     # f5
            [(1,0,1), (1,1,1)],      # f6
        ]
        
        # C block coefficients for each Strassen factor:
        # Factor 0 (M1): C00 + C11
        # Factor 1 (M2): C10 - C11
        # Factor 2 (M3): C01 + C11
        # Factor 3 (M4): C00 + C10
        # Factor 4 (M5): -C00 + C01
        # Factor 5 (M6): C11
        # Factor 6 (M7): C00
        C_blocks = [
            [(0,0,1), (1,1,1)],      # f0
            [(1,0,1), (1,1,-1)],     # f1
            [(0,1,1), (1,1,1)],      # f2
            [(0,0,1), (1,0,1)],      # f3
            [(0,0,-1), (0,1,1)],     # f4
            [(1,1,1)],               # f5
            [(0,0,1)],               # f6
        ]
        
        for f in range(7):
            for s in range(sub_rank):
                col = f * sub_rank + s
                
                # U (A side)
                for (bi, bj, coeff) in A_blocks[f]:
                    for i in range(n2):
                        for j in range(m2):
                            row_A = (bi * n2 + i) * m + (bj * m2 + j)
                            local_row = i * m2 + j
                            U[row_A][col] += coeff * sub_U[local_row][s]
                
                # V (B side)
                for (bj, bk, coeff) in B_blocks[f]:
                    for j in range(m2):
                        for k in range(p2):
                            row_B = (bj * m2 + j) * p + (bk * p2 + k)
                            local_row = j * p2 + k
                            V[row_B][col] += coeff * sub_V[local_row][s]
                
                # W (C side)
                for (ci, ck, coeff) in C_blocks[f]:
                    for i in range(n2):
                        for k in range(p2):
                            row_W = (ck * p2 + k) * n + (ci * n2 + i)
                            local_row = k * n2 + i
                            W[row_W][col] += coeff * sub_W[local_row][s]
        
        return U, V, W
    else:
        # Schoolbook for odd dimensions
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return U, V, W

def _validate_decomposition(U, V, W, n, m, p):
    """
    Validate that U, V, W correctly decompose matrix multiplication.
    (C)_{i,k} = sum_j A_{i,j} B_{j,k}
    U row: i*m+j
    V row: j*p+k
    W row: k*n+i
    Check: for all i, j, k: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    And: for all (i,j,k) != (i',j',k') where the tensor has 0:
    sum_r U[i*m+j][r] * V[j'*p+k][r] * W[k*n+i'][r] = 0
    """
    rank = len(U[0])
    # Build U as list of lists for fast access
    # Check all tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    # Check zeros: for each pair of (i,j) and (j',k) and (k',i') where not matching
    # Actually, the tensor T_{i,j,k} = 1 if i,j,k form a valid (A_{i,j}, B_{j,k}, C_{i,k}) tuple
    # The decomposition gives: sum_r U[i*m+j][r] V[j'*p+k][r] W[k'*n+i'][r]
    # This should equal T_{i,j,k} if j'=j, k'=k, i'=i, else 0
    # More precisely, the trilinear form is:
    # sum_{i,j,k} a_{i,j} b_{j,k} c_{k,i} = sum_r (sum_{i,j} a_{i,j} U[i*m+j][r]) (sum_{j,k} b_{j,k} V[j*p+k][r]) (sum_{k,i} c_{k,i} W[k*n+i][r])
    # So for the identity to hold for all a, b, c, we need:
    # sum_r U[i*m+j][r] V[j'*p+k][r] W[k'*n+i'][r] = 1 if (i,j,k) matches with j'=j, k'=k, i'=i
    # and 0 otherwise.
    # Let's check all combinations where it should be 0.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 == i and j2 == j and k2 == k:
                                continue
                            s = 0
                            for r in range(rank):
                                s += U[i*m+j][r] * V[j2*p+k2][r] * W[k2*n+i2][r]
                            if s != 0:
                                return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to get a low-rank decomposition via recursive Strassen
    try:
        U, V, W = _tensor_decompose_recursive(n, m, p)
        rank = len(U[0])
        
        # Validate
        if _time_check(start_time, 100.0):
            if _validate_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # If validation failed or took too long, fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    except Exception:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
