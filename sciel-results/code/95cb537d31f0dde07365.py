import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Hardcoded Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    strassen_U = [
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1]
    ]
    strassen_V = [
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ]
    
    # Verify Strassen correctness internally just in case (fast)
    # C = A*B
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+b)(e+f)
    # M2 = (a+b)(h-g)
    # M3 = (c+d)*e
    # M4 = (c+d)*h
    # M5 = (a-d)(e+g)
    # M6 = (b-d)(g+h)
    # M7 = (a+c-2d)(f+h) -- wait, standard Strassen is:
    # M1=(a+b)(e+f), M2=(a+b)(h-g), M3=(c+d)e, M4=(c+d)h, M5=(a-d)(e+g), M6=(b-d)(g+h), M7=(a+c-2d)(f+h) is not standard.
    # Standard:
    # m1=(a+b)(e+f)
    # m2=(a+b)(h-g)
    # m3=(c+d)e
    # m4=(c+d)h
    # m5=(a-d)(e+g)
    # m6=(b-d)(g+h)
    # m7=(a+c-2d)(f+h) -- No, M7 = (a+c-2d)(f+h) is for 2x2?
    # Let's check my hardcoded matrices.
    # U rows:
    # 0: a+b (indices 0,1 -> a,b? No, U[0][2]=1, U[0][3]=1. A is 2x2. A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d.
    # U[0] corresponds to A row 0? No, U is (n*m) x rank.
    # U[i*2+j].
    # Row 0: i=0, j=0 (a). U[0][0]=0, U[0][1]=0, U[0][2]=1, U[0][3]=1, U[0][4]=1.
    # This doesn't look like simple sums.
    # Let's re-verify the hardcoded Strassen matrices against the tensor definition.
    # Tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We need T to be the standard multiplication tensor.
    # Standard tensor has T[i,j,k] = 1 if it represents A[i,j]B[j,k].
    # Wait, the tensor for matrix mult A(n,m) B(m,p) -> C(n,p) is defined on indices (i, j, k) where i in n, j in m, k in p.
    # T[i,j,k] is the coefficient of the product term? No.
    # The tensor M has entries M[i, j, k] = 1 if the product C[i,k] contains A[i,j]B[j,k].
    # Actually, C[i,k] = sum_j A[i,j]B[j,k].
    # So the tensor is T[i, j, k] = 1 for all i,j,k.
    # The decomposition is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # My hardcoded Strassen matrices must satisfy this for n=2, m=2, p=2.
    
    # Let's construct a helper to check a decomposition
    def check_decomp(n, m, p, U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        return False
        return True

    # Check Strassen
    if not check_decomp(2, 2, 2, strassen_U, strassen_V, strassen_W):
        # Fallback to schoolbook for 2x2x2 if hardcoded is wrong (unlikely)
        strassen_U = [[0]*8 for _ in range(4)]
        strassen_V = [[0]*8 for _ in range(4)]
        strassen_W = [[0]*8 for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    strassen_U[i*2+j][col] = 1
                    strassen_V[j*2+k][col] = 1
                    strassen_W[k*2+i][col] = 1
                    col += 1

    # Cache for decompositions
    cache = {}

    def get_decomp(n, m, p):
        if (n, m, p) in cache:
            return cache[(n, m, p)]
        
        # Base case: one dimension is 1
        if n == 1 or m == 1 or p == 1:
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
            result = (U, V, W, rank)
            cache[(n, m, p)] = result
            return result
        
        # Special case: 2x2x2
        if n == 2 and m == 2 and p == 2:
            result = (strassen_U, strassen_V, strassen_W, 7)
            cache[(n, m, p)] = result
            return result

        best_rank = float('inf')
        best_decomp = None

        # Helper to combine two decompositions
        def combine(n1, m1, p1, U1, V1, W1, r1, n2, m2, p2, U2, V2, W2, r2):
            # This is only valid if the splits are aligned with the tensor structure
            # We are splitting along one dimension.
            # If we split n: n1+n2=n, m1=m, p1=p, n2=n, m2=m, p2=p
            # U1 is (n1*m) x r1, U2 is (n2*m) x r2. New U is (n*m) x (r1+r2)
            # V1 is (m*p) x r1, V2 is (m*p) x r2. New V is (m*p) x (r1+r2)
            # W1 is (p*n1) x r1, W2 is (p*n2) x r2. New W is (p*n) x (r1+r2)
            
            total_rank = r1 + r2
            new_U = [[0] * total_rank for _ in range(n * m)]
            new_V = [[0] * total_rank for _ in range(m * p)]
            new_W = [[0] * total_rank for _ in range(p * n)]
            
            # Fill U
            # Rows 0..n1*m-1 come from U1
            for i in range(n1):
                for j in range(m):
                    for r in range(r1):
                        new_U[i * m + j][r] = U1[i * m + j][r]
            # Rows n1*m..n*m-1 come from U2
            for i in range(n2):
                for j in range(m):
                    for r in range(r2):
                        new_U[(n1 + i) * m + j][r1 + r] = U2[i * m + j][r]
                        
            # Fill V
            # V is same size (m*p)
            for j in range(m):
                for k in range(p):
                    for r in range(r1):
                        new_V[j * p + k][r] = V1[j * p + k][r]
                    for r in range(r2):
                        new_V[j * p + k][r1 + r] = V2[j * p + k][r]
                        
            # Fill W
            # Rows 0..p*n1-1 come from W1
            for k in range(p):
                for i in range(n1):
                    for r in range(r1):
                        new_W[k * n + i][r] = W1[k * n1 + i][r]
            # Rows p*n1..p*n-1 come from W2
            for k in range(p):
                for i in range(n2):
                    for r in range(r2):
                        new_W[k * n + (n1 + i)][r1 + r] = W2[k * n2 + i][r]
                        
            return (new_U, new_V, new_W, total_rank)

        def combine_m(n, m1, p, U1, V1, W1, r1, m2, U2, V2, W2, r2):
            # Split m: m1+m2=m. n1=n, p1=p, n2=n, p2=p.
            total_rank = r1 + r2
            new_U = [[0] * total_rank for _ in range(n * m)]
            new_V = [[0] * total_rank for _ in range(m * p)]
            new_W = [[0] * total_rank for _ in range(p * n)]
            
            # Fill U
            # Columns 0..m1-1 come from U1
            for i in range(n):
                for j in range(m1):
                    for r in range(r1):
                        new_U[i * m + j][r] = U1[i * m1 + j][r]
                for j in range(m2):
                    for r in range(r2):
                        new_U[i * m + (m1 + j)][r1 + r] = U2[i * m2 + j][r]
                        
            # Fill V
            # Rows 0..m1*p-1 come from V1
            for j in range(m1):
                for k in range(p):
                    for r in range(r1):
                        new_V[j * p + k][r] = V1[j * p + k][r]
            for j in range(m2):
                for k in range(p):
                    for r in range(r2):
                        new_V[(m1 + j) * p + k][r1 + r] = V2[j * p + k][r]
                        
            # Fill W
            # W is same size (p*n)
            for k in range(p):
                for i in range(n):
                    for r in range(r1):
                        new_W[k * n + i][r] = W1[k * n + i][r]
                    for r in range(r2):
                        new_W[k * n + i][r1 + r] = W2[k * n + i][r]
                        
            return (new_U, new_V, new_W, total_rank)

        def combine_p(n, m, p1, U1, V1, W1, r1, p2, U2, V2, W2, r2):
            # Split p: p1+p2=p. n1=n, m1=m, n2=n, m2=m.
            total_rank = r1 + r2
            new_U = [[0] * total_rank for _ in range(n * m)]
            new_V = [[0] * total_rank for _ in range(m * p)]
            new_W = [[0] * total_rank for _ in range(p * n)]
            
            # Fill U
            # U is same size (n*m)
            for i in range(n):
                for j in range(m):
                    for r in range(r1):
                        new_U[i * m + j][r] = U1[i * m + j][r]
                    for r in range(r2):
                        new_U[i * m + j][r1 + r] = U2[i * m + j][r]
                        
            # Fill V
            # Columns 0..p1-1 come from V1
            for j in range(m):
                for k in range(p1):
                    for r in range(r1):
                        new_V[j * p + k][r] = V1[j * p1 + k][r]
                for k in range(p2):
                    for r in range(r2):
                        new_V[j * p + (p1 + k)][r1 + r] = V2[j * p2 + k][r]
                        
            # Fill W
            # Rows 0..p1*n-1 come from W1
            for k in range(p1):
                for i in range(n):
                    for r in range(r1):
                        new_W[k * n + i][r] = W1[k * n + i][r]
            for k in range(p2):
                for i in range(n):
                    for r in range(r2):
                        new_W[(p1 + k) * n + i][r1 + r] = W2[k * n + i][r]
                        
            return (new_U, new_V, new_W, total_rank)

        # Try splitting n
        for n1 in range(1, n):
            n2 = n - n1
            if time.time() > deadline:
                break
            U1, V1, W1, r1 = get_decomp(n1, m, p)
            U2, V2, W2, r2 = get_decomp(n2, m, p)
            rank = r1 + r2
            if rank < best_rank:
                best_rank = rank
                best_decomp = combine(n1, m, p, U1, V1, W1, r1, n2, m, p, U2, V2, W2, r2)

        # Try splitting m
        for m1 in range(1, m):
            m2 = m - m1
            if time.time() > deadline:
                break
            U1, V1, W1, r1 = get_decomp(n, m1, p)
            U2, V2, W2, r2 = get_decomp(n, m2, p)
            rank = r1 + r2
            if rank < best_rank:
                best_rank = rank
                best_decomp = combine_m(n, m1, p, U1, V1, W1, r1, m2, U2, V2, W2, r2)

        # Try splitting p
        for p1 in range(1, p):
            p2 = p - p1
            if time.time() > deadline:
                break
            U1, V1, W1, r1 = get_decomp(n, m, p1)
            U2, V2, W2, r2 = get_decomp(n, m, p2)
            rank = r1 + r2
            if rank < best_rank:
                best_rank = rank
                best_decomp = combine_p(n, m, p1, U1, V1, W1, r1, p2, U2, V2, W2, r2)

        if best_decomp is None:
            # Fallback to schoolbook
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
            best_decomp = (U, V, W, rank)

        cache[(n, m, p)] = best_decomp
        return best_decomp

    U, V, W, rank = get_decomp(n, m, p)
    
    if time.time() > deadline:
        # Return schoolbook if time runs out (should not happen for 3x4x7 with this fast DP)
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
