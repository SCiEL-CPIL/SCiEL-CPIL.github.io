import time
from fractions import Fraction

_START = time.time()
DEADLINE = 115.0  # safety margin before 120s wall-clock

def _time_left():
    return DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    
    U: (n*m, rank)
    V: (m*p, rank)  
    W: (p*n, rank)
    
    For each output C[i,k] (indexed as k*n+i in W), we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if j matches, else 0
    
    More precisely: sum_r U[a, r] * V[b, r] * W[c, r] should give the coefficient
    of A[a/m * m + a%m] * B[b//p * p + b%p] in C[c].
    
    The tensor T has T[i,j,k] = 1 if the output C[i,k] includes A[i,j]*B[j,k].
    So for output (i,k), the coefficient of A[i,j]*B[j,k] is 1 if j is the
    connecting index, and 0 otherwise.
    
    Actually the bilinear form for C[i,k] is sum_j A[i,j] * B[j,k].
    So the tensor entry T[i,j,k,l] (where output is (i,l), A index is (i,j), B index is (j,l))
    equals 1 iff the j in A[i,j] matches the j in B[j,l].
    
    In flat indices: A index a = i*m+j, B index b = j*p+l, W index c = l*n+i.
    T[a,b,c] = 1 iff (a//m == i, b%p == l, c//n == l, c%n == i) and (a%m == b//p).
    
    The decomposition says: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if total != 1:
                    return False
                # Also check that cross terms are zero
    # Check cross terms: for (a, b, c) where the indices don't match properly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    total = 0
                    for r in range(rank):
                        total += U[a][r] * V[b][r] * W[c][r]
                    if total != 0:
                        return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook multiplication, rank = n*m*p."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2(n, m, p):
    """Strassen decomposition for 2x2x2. Returns rank-7 decomposition.
    
    This is a helper for building block decompositions.
    The 7 Strassen products for C = A*B where A, B are 2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 - M4 + M6 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M2 - M5
    """
    rank = 7
    U = [[Fraction(0)] * rank for _ in range(4)]  # 2x2 A matrix
    V = [[Fraction(0)] * rank for _ in range(4)]  # 2x2 B matrix
    W = [[Fraction(0)] * rank for _ in range(4)]  # 2x2 C matrix (row-major: c = k*2+i)
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3 (k*n+i with n=2: C00=0, C01=1, C10=2, C11=3)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = Fraction(1); U[3][0] = Fraction(1)
    V[0][0] = Fraction(1); V[3][0] = Fraction(1)
    W[0][0] = Fraction(1); W[3][0] = Fraction(1)  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)*B00
    U[2][1] = Fraction(1); U[3][1] = Fraction(1)
    V[0][1] = Fraction(1)
    W[2][1] = Fraction(1); W[3][1] = Fraction(1)  # C10 += M2, C11 += M2
    
    # M3 = A00*(B01-B11)
    U[0][2] = Fraction(1)
    V[1][2] = Fraction(1); V[3][2] = Fraction(-1)
    W[1][2] = Fraction(1); W[3][2] = Fraction(-1)  # C01 += M3, C11 -= M3
    
    # M4 = A11*(B10-B00)
    U[3][3] = Fraction(1)
    V[2][3] = Fraction(1); V[0][3] = Fraction(-1)
    W[0][3] = Fraction(-1); W[2][3] = Fraction(1)  # C00 -= M4, C10 += M4
    
    # M5 = (A00+A01)*B11
    U[0][4] = Fraction(1); U[1][4] = Fraction(1)
    V[3][4] = Fraction(1)
    W[1][4] = Fraction(1); W[3][4] = Fraction(-1)  # C01 += M5, C11 -= M5
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = Fraction(1); U[0][5] = Fraction(-1)
    V[0][5] = Fraction(1); V[1][5] = Fraction(1)
    W[0][5] = Fraction(1)  # C00 += M6
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = Fraction(1); U[3][6] = Fraction(-1)
    V[2][6] = Fraction(1); V[3][6] = Fraction(1)
    W[0][6] = Fraction(1)  # C00 += M7
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition(n, m, p, splits_n, splits_m, splits_p):
    """Build a block decomposition using Strassen for 2x2x2 blocks.
    
    splits_n, splits_m, splits_p: lists of block sizes that sum to n, m, p respectively.
    For each combination of blocks (ni, mj, pk), if all are 2, use Strassen (rank 7).
    If any is not 2, use schoolbook for that block (rank ni*mj*pk).
    """
    U_list = []
    V_list = []
    W_list = []
    
    for ni in splits_n:
        for mj in splits_m:
            for pk in splits_p:
                if ni == 2 and mj == 2 and pk == 2:
                    block = _strassen_2x2x2(ni, mj, pk)
                    U_list.append(block["U"])
                    V_list.append(block["V"])
                    W_list.append(block["W"])
                else:
                    # Schoolbook for this block
                    brank = ni * mj * pk
                    U_block = [[Fraction(0)] * brank for _ in range(ni * mj)]
                    V_block = [[Fraction(0)] * brank for _ in range(mj * pk)]
                    W_block = [[Fraction(0)] * brank for _ in range(pk * ni)]
                    col = 0
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                U_block[ii * mj + jj][col] = Fraction(1)
                                V_block[jj * pk + kk][col] = Fraction(1)
                                W_block[kk * ni + ii][col] = Fraction(1)
                                col += 1
                    U_list.append(U_block)
                    V_list.append(V_block)
                    W_list.append(W_block)
    
    # Now assemble the full matrices
    total_rank = sum(len(u[0]) for u in U_list)
    U = [[Fraction(0)] * total_rank for _ in range(n * m)]
    V = [[Fraction(0)] * total_rank for _ in range(m * p)]
    W = [[Fraction(0)] * total_rank for _ in range(p * n)]
    
    row_off_u = 0
    row_off_v = 0
    row_off_w = 0
    col_off = 0
    
    for idx, (ni, mj, pk) in enumerate([(splits_n[i], splits_m[j], splits_p[k]) 
                                         for i in range(len(splits_n)) 
                                         for j in range(len(splits_m)) 
                                         for k in range(len(splits_p))]):
        brank = len(U_list[idx][0])
        for ii in range(ni):
            for jj in range(mj):
                u_local = ii * mj + jj
                u_global = row_off_u + u_local
                for r in range(brank):
                    U[u_global][col_off + r] = U_list[idx][u_local][r]
        for jj in range(mj):
            for kk in range(pk):
                v_local = jj * pk + kk
                v_global = row_off_v + v_local
                for r in range(brank):
                    V[v_global][col_off + r] = V_list[idx][v_local][r]
        for kk in range(pk):
            for ii in range(ni):
                w_local = kk * ni + ii
                w_global = row_off_w + w_local
                for r in range(brank):
                    W[w_global][col_off + r] = W_list[idx][w_local][r]
        
        col_off += brank
        row_off_u += ni * mj
        row_off_v += mj * pk
        row_off_w += pk * ni
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 5, 5
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 50
    
    # Try various block decompositions
    # Split 5 as 2+2+1 or 2+1+2 or 1+2+2
    # Split 2 as 2 or 1+1
    
    candidate_splits = [
        # (splits_n, splits_m, splits_p)
        ([2], [2, 2, 1], [2, 2, 1]),
        ([2], [2, 2, 1], [2, 1, 2]),
        ([2], [2, 2, 1], [1, 2, 2]),
        ([2], [2, 1, 2], [2, 2, 1]),
        ([2], [2, 1, 2], [2, 1, 2]),
        ([2], [2, 1, 2], [1, 2, 2]),
        ([2], [1, 2, 2], [2, 2, 1]),
        ([2], [1, 2, 2], [2, 1, 2]),
        ([2], [1, 2, 2], [1, 2, 2]),
        ([1, 1], [2, 2, 1], [2, 2, 1]),
        ([1, 1], [2, 2, 1], [2, 1, 2]),
        ([1, 1], [2, 2, 1], [1, 2, 2]),
        ([1, 1], [2, 1, 2], [2, 2, 1]),
        ([1, 1], [2, 1, 2], [2, 1, 2]),
        ([1, 1], [2, 1, 2], [1, 2, 2]),
        ([1, 1], [1, 2, 2], [2, 2, 1]),
        ([1, 1], [1, 2, 2], [2, 1, 2]),
        ([1, 1], [1, 2, 2], [1, 2, 2]),
        ([2], [2, 2, 1], [1, 1, 1, 1, 1]),  # All p-blocks of size 1
        ([2], [1, 1, 1, 1, 1], [2, 2, 1]),  # All m-blocks of size 1
    ]
    
    for splits_n, splits_m, splits_p in candidate_splits:
        if sum(splits_n) != n or sum(splits_m) != m or sum(splits_p) != p:
            continue
        if _time_left() < 5.0:
            break
        
        try:
            result = _block_decomposition(n, m, p, splits_n, splits_m, splits_p)
            rank = len(result["U"][0])
            
            if rank < best_rank:
                # Verify
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = rank
        except Exception:
            continue
    
    # Convert Fraction to int/float for output
    def convert(mat):
        return [[int(x) if x.denominator == 1 else float(x) for x in row] for row in mat]
    
    return {"U": convert(best["U"]), "V": convert(best["V"]), "W": convert(best["W"])}
