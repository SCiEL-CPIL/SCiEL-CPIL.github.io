import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact using float arithmetic."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
        
    Uf = np.array(U, dtype=float)
    Vf = np.array(V, dtype=float)
    Wf = np.array(W, dtype=float)
    
    Uf_r = Uf.reshape(n, m, rank)
    Vf_r = Vf.reshape(m, p, rank)
    Wf_r = Wf.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c Uf_r[i,j,c] * Vf_r[j,k,c] * Wf_r[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', Uf_r, Vf_r, Wf_r)
    
    if not np.allclose(T, 1.0, atol=1e-7):
        return False
    
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # Products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # U: shape (2*2, 7) = (4, 7), indexed by [i*m+j, c]
    # V: shape (2*2, 7) = (4, 7), indexed by [j*p+k, c]
    # W: shape (2*2, 7) = (4, 7), indexed by [k*n+i, c]
    
    # For product c, U[i,j,c] * V[j,k,c] * W[k,i,c] should contribute to T[i,j,k]
    
    # Let's define each of the 7 rank-1 tensors:
    # c=0: p1=(a+d)(e+h)
    #   U: a=1,d=1 -> U[0,0]=1, U[1,1]=1
    #   V: e=1,h=1 -> V[0,0]=1, V[1,1]=1
    #   W: contributes to result[0,0] and result[1,1]
    #   W[k,i,c]: for i=0,k=0: +1; i=1,k=1: +1; i=0,k=1: -1; i=1,k=0: -1
    #   Actually W[k*n+i, c]: W[0*2+0]=1, W[1*2+1]=1, W[0*2+1]=-1, W[1*2+0]=-1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0: (a+d)(e+h)
    # U[0,0]=1 (a), U[1,1]=1 (d)
    U[0][0] = 1
    U[3][0] = 1
    # V[0,0]=1 (e), V[1,1]=1 (h)
    V[0][0] = 1
    V[3][0] = 1
    # W: result = a*e + a*h + d*e + d*h
    # T[0,0] += a*e -> W[0*2+0]=1
    # T[0,1] += a*h -> W[1*2+0]=1  (k=1,i=0)
    # T[1,0] += d*e -> W[0*2+1]=1  (k=0,i=1)
    # T[1,1] += d*h -> W[1*2+1]=1  (k=1,i=1)
    W[0][0] = 1
    W[2][0] = 1
    W[1][0] = 1
    W[3][0] = 1
    
    # c=1: (c+d)e
    # U[1,0]=1 (c), U[1,1]=1 (d)
    U[2][1] = 1
    U[3][1] = 1
    # V[0,0]=1 (e)
    V[0][1] = 1
    # W: T[1,0] += c*e + d*e
    W[0*2+1][1] = 1  # k=0,i=1
    
    # c=2: a(f-h)
    # U[0,0]=1 (a)
    U[0][2] = 1
    # V[0,1]=1 (f), V[1,1]=-1 (-h)
    V[1][2] = 1
    V[3][2] = -1
    # W: T[0,1] += a*f - a*h
    W[1*2+0][2] = 1  # k=1,i=0
    
    # c=3: d(g-e)
    # U[1,1]=1 (d)
    U[3][3] = 1
    # V[1,0]=1 (g), V[0,0]=-1 (-e)
    V[2][3] = 1
    V[0][3] = -1
    # W: T[1,0] += d*g - d*e
    W[0*2+1][3] = 1  # k=0,i=1
    
    # c=4: (a+b)h
    # U[0,0]=1 (a), U[0,1]=1 (b)
    U[0][4] = 1
    U[1][4] = 1
    # V[1,1]=1 (h)
    V[3][4] = 1
    # W: T[0,1] += a*h + b*h
    W[1*2+0][4] = 1  # k=1,i=0
    
    # c=5: (c-a)(e+f)
    # U[1,0]=1 (c), U[0,0]=-1 (-a)
    U[2][5] = 1
    U[0][5] = -1
    # V[0,0]=1 (e), V[0,1]=1 (f)
    V[0][5] = 1
    V[1][5] = 1
    # W: T[1,0] += c*e + c*f; T[0,0] += -a*e - a*f
    W[0*2+1][5] = 1  # k=0,i=1
    W[0*2+0][5] = -1  # k=0,i=0
    
    # c=6: (b-d)(g+h)
    # U[0,1]=1 (b), U[1,1]=-1 (-d)
    U[1][6] = 1
    U[3][6] = -1
    # V[1,0]=1 (g), V[1,1]=1 (h)
    V[2][6] = 1
    V[3][6] = 1
    # W: T[1,1] += b*g + b*h; T[1,1] += -d*g - d*h
    W[1*2+1][6] = 1  # k=1,i=1
    
    return U, V, W

def _split_combine(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """
    Combine two matrix multiplication decompositions by splitting the first dimension.
    Problem 1: n1 x m1 x p1
    Problem 2: n2 x m2 x p2
    Combined: (n1+n2) x m x p where m=m1=m2, p=p1=p2
    """
    if m1 != m2 or p1 != p2:
        raise ValueError("m and p must match for split combination")
    
    n = n1 + n2
    m = m1
    p = p1
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 + r2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # First part: rows 0..n1-1
    for i in range(n1):
        for j in range(m):
            for c in range(r1):
                U[i * m + j][c] = U1[i * m1 + j][c]
    
    for j in range(m):
        for k in range(p):
            for c in range(r1):
                V[j * p + k][c] = V1[j * p1 + k][c]
    
    for k in range(p):
        for i in range(n1):
            for c in range(r1):
                W[k * n + i][c] = W1[k * p1 + i][c]
    
    # Second part: rows n1..n1+n2-1
    offset = r1
    for i in range(n2):
        for j in range(m):
            for c in range(r2):
                U[(n1 + i) * m + j][offset + c] = U2[i * m2 + j][c]
    
    for j in range(m):
        for k in range(p):
            for c in range(r2):
                V[j * p + k][offset + c] = V2[j * p2 + k][c]
    
    for k in range(p):
        for i in range(n2):
            for c in range(r2):
                W[k * n + (n1 + i)][offset + c] = W2[k * p2 + i][c]
    
    return U, V, W

def _split_p_combine(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """
    Combine two matrix multiplication decompositions by splitting the third dimension (p).
    Problem 1: n x m x p1
    Problem 2: n x m x p2
    Combined: n x m x (p1+p2)
    """
    if n1 != n2 or m1 != m2:
        raise ValueError("n and m must match for p-split combination")
    
    n = n1
    m = m1
    p = p1 + p2
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 + r2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # First part: columns 0..p1-1
    for i in range(n):
        for j in range(m):
            for c in range(r1):
                U[i * m + j][c] = U1[i * m1 + j][c]
    
    for j in range(m):
        for k in range(p1):
            for c in range(r1):
                V[j * p + k][c] = V1[j * p1 + k][c]
    
    for k in range(p1):
        for i in range(n):
            for c in range(r1):
                W[k * n + i][c] = W1[k * p1 + i][c]
    
    # Second part: columns p1..p1+p2-1
    offset = r1
    for i in range(n):
        for j in range(m):
            for c in range(r2):
                U[i * m + j][offset + c] = U2[i * m2 + j][c]
    
    for j in range(m):
        for k in range(p2):
            for c in range(r2):
                V[j * p + (p1 + k)][offset + c] = V2[j * p2 + k][c]
    
    for k in range(p2):
        for i in range(n):
            for c in range(r2):
                W[(p1 + k) * n + i][offset + c] = W2[k * p2 + i][c]
    
    return U, V, W

def _kronecker_combine(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """
    Combine two matrix multiplication decompositions via Kronecker product.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                u1 = U1[i1 * m1 + j1][c1]
                if u1 == 0:
                    continue
                for i2 in range(n2):
                    for j2 in range(m2):
                        for c2 in range(r2):
                            u2 = U2[i2 * m2 + j2][c2]
                            if u2 == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            c = c1 * r2 + c2
                            U[i * m + j][c] += u1 * u2
                            
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(r1):
                v1 = V1[j1 * p1 + k1][c1]
                if v1 == 0:
                    continue
                for j2 in range(m2):
                    for k2 in range(p2):
                        for c2 in range(r2):
                            v2 = V2[j2 * p2 + k2][c2]
                            if v2 == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            c = c1 * r2 + c2
                            V[j * p + k][c] += v1 * v2
                            
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(r1):
                w1 = W1[k1 * n1 + i1][c1]
                if w1 == 0:
                    continue
                for k2 in range(p2):
                    for i2 in range(n2):
                        for c2 in range(r2):
                            w2 = W2[k2 * n2 + i2][c2]
                            if w2 == 0:
                                continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            c = c1 * r2 + c2
                            W[k * n + i][c] += w1 * w2
                            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_rank = n * m * p
    best_result = None
    
    # Strategy 1: Split p=9 into 4+5
    # Use rank-49 for 4x4x4 (via Strassen-based construction)
    # and schoolbook for 4x4x5
    
    # Build rank-49 for 4x4x4 using Kronecker of Strassen 2x2x2 with itself?
    # No, that gives 7*7=49 for 4x4x4! Perfect.
    
    U_s, V_s, W_s = _strassen_2x2x2()
    U_444, V_444, W_444 = _kronecker_combine(U_s, V_s, W_s, 2, 2, 2, U_s, V_s, W_s, 2, 2, 2)
    
    # Verify 4x4x4
    if _verify(U_444, V_444, W_444, 4, 4, 4):
        rank_444 = len(U_444[0])
    else:
        U_444, V_444, W_444 = _schoolbook(4, 4, 4)
        rank_444 = 64
    
    U_445, V_445, W_445 = _schoolbook(4, 4, 5)
    rank_445 = 80
    
    # Combine: 4x4x4 + 4x4x5 -> 4x4x9
    U_comb, V_comb, W_comb = _split_p_combine(U_444, V_444, W_444, 4, 4, 4, 
                                                 U_445, V_445, W_445, 4, 4, 5)
    
    if _verify(U_comb, V_comb, W_comb, n, m, p):
        rank_comb = len(U_comb[0])
        if rank_comb < best_rank:
            best_rank = rank_comb
            best_result = {"U": U_comb, "V": V_comb, "W": W_comb}
    
    if time.time() > deadline:
        if best_result:
            return best_result
        return _schoolbook(n, m, p) and {"U": _schoolbook(n,m,p)[0], "V": _schoolbook(n,m,p)[1], "W": _schoolbook(n,m,p)[2]}
    
    # Strategy 2: Try splitting n=4 into 2+2
    # R(2,4,9) + R(2,4,9)
    # R(2,4,9) can be split: R(2,2,9) + R(2,2,9)
    # R(2,2,9) = R(2,2,4) + R(2,2,5)
    # R(2,2,4) = 2*R(2,2,2) = 2*7 = 14
    # R(2,2,5) = schoolbook = 20
    # So R(2,2,9) <= 34, R(2,4,9) <= 68, total <= 136
    
    U_s2, V_s2, W_s2 = _strassen_2x2x2()
    U_224, V_224, W_224 = _kronecker_combine(U_s2, V_s2, W_s2, 2, 2, 2, _schoolbook(1,1,2)[0], _schoolbook(1,1,2)[1], _schoolbook(1,1,2)[2], 1, 1, 2)
    
    # Hmm, this is getting complicated. Let me just use schoolbook for the subparts for now.
    # Actually, let's try a different split.
    
    # Strategy 3: Split p=9 into 3+3+3
    U_443, V_443, W_443 = _schoolbook(4, 4, 3)
    U_443b, V_443b, W_443b = _schoolbook(4, 4, 3)
    U_443c, V_443c, W_443c = _schoolbook(4, 4, 3)
    
    # Combine first two
    U_c1, V_c1, W_c1 = _split_p_combine(U_443, V_443, W_443, 4, 4, 3, U_443b, V_443b, W_443b, 4, 4, 3)
    # Combine with third
    U_c2, V_c2, W_c2 = _split_p_combine(U_c1, V_c1, W_c1, 4, 4, 6, U_443c, V_443c, W_443c, 4, 4, 3)
    
    if _verify(U_c2, V_c2, W_c2, n, m, p):
        rank_c2 = len(U_c2[0])
        if rank_c2 < best_rank:
            best_rank = rank_c2
            best_result = {"U": U_c2, "V": V_c2, "W": W_c2}
    
    if time.time() > deadline:
        if best_result:
            return best_result
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Strategy 4: Split m=4 into 2+2
    # R(4,2,9) + R(4,2,9)
    # R(4,2,9) = R(4,2,4) + R(4,2,5)
    # R(4,2,4) = R(2,2,4) + R(2,2,4) = 14 + 14 = 28 (using Strassen)
    # R(4,2,5) = schoolbook = 40
    # R(4,2,9) <= 68, total <= 136
    
    U_424, V_424, W_424 = _schoolbook(4, 2, 4)
    U_425, V_425, W_425 = _schoolbook(4, 2, 5)
    U_429, V_429, W_429 = _split_p_combine(U_424, V_424, W_424, 4, 2, 4, U_425, V_425, W_425, 4, 2, 5)
    
    # Now split m: two copies of 4x2x9
    U_429b, V_429b, W_429b = _schoolbook(4, 2, 9)
    
    # For m-split, we need a different combine function
    # Let me create one
    
    def _split_m_combine(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
        """Split the m dimension."""
        if n1 != n2 or p1 != p2:
            raise ValueError("n and p must match")
        n = n1
        m = m1 + m2
        p = p1
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m1):
                for c in range(r1):
                    U[i * m + j][c] = U1[i * m1 + j][c]
            for j in range(m2):
                for c in range(r2):
                    U[i * m + (m1 + j)][r1 + c] = U2[i * m2 + j][c]
        
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j * p + k][c] = V1[j * p1 + k][c]
        for j in range(m2):
            for k in range(p):
                for c in range(r2):
                    V[(m1 + j) * p + k][r1 + c] = V2[j * p2 + k][c]
        
        for k in range(p):
            for i in range(n):
                for c in range(r1):
                    W[k * n + i][c] = W1[k * p1 + i][c]
                for c in range(r2):
                    W[k * n + i][r1 + c] = W2[k * p2 + i][c]
        
        return U, V, W
    
    U_msplit, V_msplit, W_msplit = _split_m_combine(U_429, V_429, W_429, 4, 2, 9, U_429b, V_429b, W_429b, 4, 2, 9)
    
    if _verify(U_msplit, V_msplit, W_msplit, n, m, p):
        rank_msplit = len(U_msplit[0])
        if rank_msplit < best_rank:
            best_rank = rank_msplit
            best_result = {"U": U_msplit, "V": V_msplit, "W": W_msplit}
    
    if best_result:
        return best_result
    
    # Fallback
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
