"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _time_ok(deadline, margin=2.0):
    return time.time() < deadline - margin

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    a = U[ui][r]
                    if a != 0:
                        b = V[vj][r]
                        if b != 0:
                            s += a * b * W[wk][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_2x2_2x2(U, V, W, n, m, p, i0, j0, k0, rank_offset):
    """
    Add Strassen 2x2*2x2 decomposition for blocks starting at (i0, j0, k0).
    A block: rows i0..i0+1, cols j0..j0+1 of A (size 2x2)
    B block: rows j0..j0+1, cols k0..k0+1 of B (size 2x2)
    C block: rows k0..k0+1, cols i0..i0+1 of C (size 2x2)
    
    Strassen 7 products:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    
    Returns new rank offset.
    """
    # A indices: (i, j) -> U[i*m+j]
    A11 = i0 * m + j0
    A12 = i0 * m + j0 + 1
    A21 = (i0 + 1) * m + j0
    A22 = (i0 + 1) * m + j0 + 1
    
    # B indices: (j, k) -> V[j*p+k]
    B11 = j0 * p + k0
    B12 = j0 * p + k0 + 1
    B21 = (j0 + 1) * p + k0
    B22 = (j0 + 1) * p + k0 + 1
    
    # C indices: (k, i) -> W[k*n+i]
    C11 = k0 * n + i0
    C12 = k0 * n + i0 + 1
    C21 = (k0 + 1) * n + i0
    C22 = (k0 + 1) * n + i0 + 1
    
    r = rank_offset
    
    # M1 = (A11+A22)*(B11+B22) -> contributes to C11, C22
    U[A11][r] += 1; U[A22][r] += 1
    V[B11][r] += 1; V[B22][r] += 1
    W[C11][r] += 1; W[C22][r] += 1
    r += 1
    
    # M2 = (A21+A22)*B11 -> contributes to C21, C22
    U[A21][r] += 1; U[A22][r] += 1
    V[B11][r] += 1
    W[C21][r] += 1; W[C22][r] -= 1
    r += 1
    
    # M3 = A11*(B12-B22) -> contributes to C12, C22
    U[A11][r] += 1
    V[B12][r] += 1; V[B22][r] -= 1
    W[C12][r] += 1; W[C22][r] += 1
    r += 1
    
    # M4 = A22*(B21-B11) -> contributes to C11, C21
    U[A22][r] += 1
    V[B21][r] += 1; V[B11][r] -= 1
    W[C11][r] += 1; W[C21][r] += 1
    r += 1
    
    # M5 = (A11+A12)*B22 -> contributes to C11, C12
    U[A11][r] += 1; U[A12][r] += 1
    V[B22][r] += 1
    W[C11][r] -= 1; W[C12][r] += 1
    r += 1
    
    # M6 = (A21-A11)*(B11+B12) -> contributes to C22
    U[A21][r] += 1; U[A11][r] -= 1
    V[B11][r] += 1; V[B12][r] += 1
    W[C22][r] += 1
    r += 1
    
    # M7 = (A12-A22)*(B21+B22) -> contributes to C11
    U[A12][r] += 1; U[A22][r] -= 1
    V[B21][r] += 1; V[B22][r] += 1
    W[C11][r] += 1
    r += 1
    
    return r

def _build_4x4_4x4_strassen(n, m, p):
    """
    Build Strassen decomposition for 4x4 * 4x4 part.
    Split into 2x2 super-blocks. For each super-block (I,J) of C (2x2),
    we need C[I,J] = A[I,0]*B[0,J] + A[I,1]*B[1,J]
    Each product is 2x2 * 2x2, using Strassen (rank 7).
    Total: 4 super-blocks * 2 k-terms * 7 = 56
    """
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for I in range(2):  # super-block row of C (i direction)
        for J in range(2):  # super-block col of C (k direction)
            for K in range(2):  # k index (j direction in A/B)
                r = _strassen_2x2_2x2(U, V, W, n, m, p, I, K, J, r)
    
    return {"U": U, "V": V, "W": W}

def _build_4x4_4x5_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for 4x4 * 4x5 part.
    Rank = 4*4*5 = 80
    """
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(5):  # only first 5 columns of B
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _combine_decompositions(decs, n, m, p):
    """
    Combine multiple decompositions by concatenating their rank dimensions.
    Each dec has U, V, W of shape (rows, rank_i).
    Result has rank = sum(rank_i).
    """
    total_rank = sum(len(d["U"][0]) for d in decs)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for d in decs:
        r = len(d["U"][0])
        for i in range(n * m):
            for c in range(r):
                U[i][offset + c] = d["U"][i][c]
        for i in range(m * p):
            for c in range(r):
                V[i][offset + c] = d["V"][i][c]
        for i in range(p * n):
            for c in range(r):
                W[i][offset + c] = d["W"][i][c]
        offset += r
    
    return {"U": U, "V": V, "W": W}

def _try_strassen_hybrid(n, m, p, deadline):
    """
    Try hybrid: 4x4*4x4 (Strassen, rank 56) + 4x4*4x5 (schoolbook, rank 80) = 136
    """
    if not _time_ok(deadline):
        return None
    
    dec1 = _build_4x4_4x4_strassen(n, m, p)
    dec2 = _build_4x4_4x5_schoolbook(n, m, p)
    
    combined = _combine_decompositions([dec1, dec2], n, m, p)
    return combined

def _try_strassen_4x9_direct(n, m, p, deadline):
    """
    Try to apply Strassen-like ideas directly to 4x4 * 4x9.
    
    Idea: Split p=9 into 4+5.
    For the 4x4 part, use Strassen (rank 56).
    For the 4x5 part, try a better decomposition than schoolbook.
    
    Alternative: Split m=4 into 2+2.
    C = A[:,:2]*B[:2,:] + A[:,2:]*B[2:,:]
    Each term is 4x2 * 2x9.
    rank(4x2*2x9): schoolbook = 72.
    Can we do better? 4=2*2, so (2x2 blocks of 2x1) * (2x9).
    This is complex. Let's just use schoolbook for now: 2*72 = 144. No improvement.
    
    Let's try: 4x4*4x9 with p split as 3+3+3.
    Each 4x4*4x3: schoolbook = 48, total = 144. No improvement.
    
    Best bet: 56 + 80 = 136 from the hybrid approach.
    """
    if not _time_ok(deadline):
        return None
    
    # Just use the hybrid for now
    return _try_strassen_hybrid(n, m, p, deadline)

def construct():
    """
    Returns the tensor decomposition for matrix multiplication of dimensions 4x4 * 4x9.
    Tries to find a rank-minimal exact decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # 150s limit, 5s margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen hybrid: 4x4*4x4 (rank 56) + 4x4*4x5 (rank 80) = 136
    if _time_ok(deadline):
        result = _try_strassen_hybrid(n, m, p, deadline)
        if result:
            r = len(result["U"][0])
            if r < best_rank and _validate(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
    
    # Try direct Strassen on 4x4*4x9
    if _time_ok(deadline):
        result = _try_strassen_4x9_direct(n, m, p, deadline)
        if result:
            r = len(result["U"][0])
            if r < best_rank and _validate(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
    
    return best
# EVOLVE-BLOCK-END
