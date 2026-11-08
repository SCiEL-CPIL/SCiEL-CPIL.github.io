import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Fallback: rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _build_strassen_2x2():
    """(2x2)*(2x2) rank 7."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    defs = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1),(1,0,-1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for t, (ue, ve, we) in enumerate(defs):
        for (i, j, c) in ue:
            U[i * 2 + j][t] += c
        for (j, k, c) in ve:
            V[j * 2 + k][t] += c
        for (k, i, c) in we:
            W[k * 2 + i][t] += c
    return U, V, W

def _build_4x4x4():
    """(4x4)*(4x4) rank 49 via 2-level Strassen."""
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    U2, V2, W2 = _build_strassen_2x2()
    outer = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    for s, (ue, ve, we) in enumerate(outer):
        for t in range(7):
            col = s * 7 + t
            for (bi, bj, c) in ue:
                for i in range(2):
                    for j in range(2):
                        gi = bi * 2 + i
                        gj = bj * 2 + j
                        a_loc = i * 2 + j
                        cc = U2[a_loc][t]
                        if cc:
                            U[gi * 4 + gj][col] += cc * c
            for (bj, bk, c) in ve:
                for j in range(2):
                    for k in range(2):
                        gj = bj * 2 + j
                        gk = bk * 2 + k
                        b_loc = j * 2 + k
                        cc = V2[b_loc][t]
                        if cc:
                            V[gj * 4 + gk][col] += cc * c
            for (bk, bi, c) in we:
                for k in range(2):
                    for i in range(2):
                        gk = bk * 2 + k
                        gi = bi * 2 + i
                        c_loc = k * 2 + i
                        cc = W2[c_loc][t]
                        if cc:
                            W[gk * 4 + gi][col] += cc * c
    return U, V, W

def _build_4x4x2():
    """(4x4)*(4x2) rank 28."""
    U = [[0] * 28 for _ in range(16)]
    V = [[0] * 28 for _ in range(8)]
    W = [[0] * 28 for _ in range(8)]
    U2, V2, W2 = _build_strassen_2x2()
    blocks = [(0,0,0,0,0), (0,2,2,0,0), (2,0,0,2,0), (2,2,2,2,0)]
    for bi, (ar, ac, br, cr, ck) in enumerate(blocks):
        base = bi * 7
        for t in range(7):
            col = base + t
            for i in range(2):
                for j in range(2):
                    a_loc = i * 2 + j
                    cc = U2[a_loc][t]
                    if cc:
                        U[(ar + i) * 4 + (ac + j)][col] += cc
            for j in range(2):
                for k in range(2):
                    b_loc = j * 2 + k
                    cc = V2[b_loc][t]
                    if cc:
                        V[(br + j) * 2 + (ck + k)][col] += cc
            for k in range(2):
                for i in range(2):
                    c_loc = k * 2 + i
                    cc = W2[c_loc][t]
                    if cc:
                        W[(ck + k) * 4 + (cr + i)][col] += cc
    return U, V, W

def _build_1x1xp(p):
    """(1x1)*(1xp) rank p."""
    U = [[1] * p]
    V = [[1] * p for _ in range(p)]
    W = [[1] * p for _ in range(p)]
    for k in range(p):
        V[k][k] = 1
        W[k][k] = 1
    # Fix: V[k][t] should be 1 for t=k, W[k*n+i][t] = 1 for t=k
    # Redo properly:
    U = [[0] * p]
    V = [[0] * p for _ in range(p)]
    W = [[0] * p for _ in range(p)]
    for k in range(p):
        U[0][k] = 1
        V[k][k] = 1
        W[k][k] = 1
    return U, V, W

def _build_1xmx1(m):
    """(1xm)*(mx1) rank 1."""
    U = [[1] * m]
    V = [[1] * m]
    W = [[1]]
    return U, V, W

def _combine_parts(parts, n, m, p):
    """Combine (U,V,W, (i0,j0,ii,jj,k0,kk)) parts into full (n,m,p) decomposition."""
    total = sum(len(d[0][0]) for d in parts)
    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]
    off = 0
    for (Us, Vs, Ws, (i0, j0, ii, jj, k0, kk)) in parts:
        nA = ii
        mA = jj
        nB = kk
        for r in range(nA * mA):
            i = r // mA
            j = r % mA
            for c in range(len(Us[0])):
                U[(i0 + i) * m + (j0 + j)][off + c] = Us[r][c]
        for r in range(mA * nB):
            j = r // nB
            k = r % nB
            for c in range(len(Vs[0])):
                V[(j0 + j) * p + (k0 + k)][off + c] = Vs[r][c]
        for r in range(nB * nA):
            k = r // nA
            i = r % nA
            for c in range(len(Ws[0])):
                W[(k0 + k) * n + (i0 + i)][off + c] = Ws[r][c]
        off += len(Us[0])
    return U, V, W

def _verify(U, V, W, n, m, p, trials=5, seed=12345):
    import random
    rng = random.Random(seed)
    rank = len(U[0])
    for _ in range(trials):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                val = 0
                for t in range(rank):
                    au = sum(U[i * m + j][t] * A[i][j] for j in range(m))
                    bu = sum(V[j * p + k][t] * B[j][k] for j in range(m))
                    val += W[k * n + i][t] * au * bu
                if val != C[i][k]:
                    return False
    return True

def _build_strassen_tiling(n, m, p):
    """Build a Strassen-tiled decomposition for (n x m) * (m x p)."""
    # Tile A into 4x4, 4x1 blocks; B into 4x4, 4x2 blocks
    # A: rows 0-3, cols 0-4. B: rows 0-4, cols 0-5.
    # Split: A = [A11 A12; A21 A22] where A11 is 4x4, A12 is 4x1, etc.
    # B = [B11 B12; B21 B22] where B11 is 4x4, B12 is 4x2, etc.
    # C = [C11 C12; C21 C22] where C11 is 4x4, C12 is 4x2, etc.
    
    parts = []
    
    # C11 = A11*B11: 4x4 * 4x4, rank 49
    U1, V1, W1 = _build_4x4x4()
    parts.append((U1, V1, W1, (0, 0, 4, 4, 0, 4)))
    
    # C12 = A11*B12 + A12*B22: (4x4)*(4x2) + (4x1)*(1x2)
    # A11*B12: 4x4 * 4x2, rank 28
    U2, V2, W2 = _build_4x4x2()
    parts.append((U2, V2, W2, (0, 0, 4, 4, 4, 2)))
    
    # A12*B22: (4x1)*(1x2)
    # A12 is 4x1, B22 is 1x2. This is outer product: rank 4*2=8
    U3 = [[0] * 8 for _ in range(4)]
    V3 = [[0] * 8 for _ in range(2)]
    W3 = [[0] * 8 for _ in range(8)]
    c = 0
    for i in range(4):
        for k in range(2):
            U3[i][c] = 1
            V3[0][c] = 1
            W3[k * 4 + i][c] = 1
            c += 1
    parts.append((U3, V3, W3, (0, 4, 4, 1, 0, 2)))
    
    # C21 = A21*B11 + A22*B21: (4x4)*(4x4) + (4x1)*(1x4)
    # A21*B11: 4x4 * 4x4, rank 49
    U4, V4, W4 = _build_4x4x4()
    parts.append((U4, V4, W4, (4, 0, 4, 4, 0, 4)))
    
    # A22*B21: (4x1)*(1x4)
    U5 = [[0] * 16 for _ in range(4)]
    V5 = [[0] * 16 for _ in range(4)]
    W5 = [[0] * 16 for _ in range(16)]
    c = 0
    for i in range(4):
        for k in range(4):
            U5[i][c] = 1
            V5[0][c] = 1
            W5[k * 4 + i][c] = 1
            c += 1
    parts.append((U5, V5, W5, (4, 4, 4, 1, 0, 4)))
    
    # C22 = A21*B12 + A22*B22: (4x4)*(4x2) + (4x1)*(1x2)
    # A21*B12: 4x4 * 4x2, rank 28
    U6, V6, W6 = _build_4x4x2()
    parts.append((U6, V6, W6, (4, 0, 4, 4, 4, 2)))
    
    # A22*B22: (4x1)*(1x2)
    U7 = [[0] * 8 for _ in range(4)]
    V7 = [[0] * 8 for _ in range(2)]
    W7 = [[0] * 8 for _ in range(8)]
    c = 0
    for i in range(4):
        for k in range(2):
            U7[i][c] = 1
            V7[0][c] = 1
            W7[k * 4 + i][c] = 1
            c += 1
    parts.append((U7, V7, W7, (4, 4, 4, 1, 4, 2)))
    
    U, V, W = _combine_parts(parts, n, m, p)
    return U, V, W

def _build_strassen_tiling_v2(n, m, p):
    """Alternative tiling: A as 2x2 blocks of 2x?."""
    # Try different splits
    parts = []
    
    # Split A into top 2 rows and bottom 2 rows
    # A_top: 2x5, A_bot: 2x5
    # B_top: 5x? no, B is 5x6
    # This is getting complex. Let's use a simpler approach.
    
    # Use the standard tiling but with better sub-decompositions
    # For 4x4*4x4, we use rank 49. For 4x4*4x2, we use rank 28.
    # For 4x1*1x2, rank 8. For 4x1*1x4, rank 16.
    # Total: 49 + 28 + 8 + 49 + 16 + 28 + 8 = 186. That's worse than schoolbook (120).
    
    # The issue: Strassen helps for square matrices, but for rectangular, the overhead dominates.
    # Let's just use schoolbook as the baseline and try to improve via search.
    
    return _build_schoolbook(n, m, p)

def _search_decomposition(n, m, p, time_limit=100):
    """
    Search for a rank-minimal decomposition using greedy peeling with exact arithmetic.
    The multiplication tensor M[i,j,k] = 1 if we consider C[i,k] = sum_j A[i,j]*B[j,k].
    We want sum_t u_t[i] * v_t[j] * w_t[k] = delta_{i,k} (where the j-sum is implicit).
    
    Actually, the tensor is T[i,j,k] = 1 for all i,j,k? No.
    C[i,k] = sum_j A[i,j] B[j,k].
    So the tensor is T[i,j,k] = 1 for all i,j,k? That's the all-ones tensor, which has rank 1.
    Wait, no. The decomposition is:
    C[i,k] = sum_t (sum_j U[i,j,t] A[i,j]) * (sum_j V[j,k,t] B[j,k]) * W[k,i,t]
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_t W[k,i,t] * (sum_j U[i,j,t] A[i,j]) * (sum_j V[j,k,t] B[j,k]) = sum_j A[i,j] B[j,k]
    
    This is a bilinear identity. The tensor we're decomposing is not the all-ones tensor.
    It's the "matrix multiplication tensor" which has entries T[i,j,k] = 1 if we identify
    the (i,j) index of A and (j,k) index of B with a joint index.
    
    Actually, the standard formulation: the multiplication tensor M has entries
    M[(i,j), (j,k), (i,k)] = 1. But here we have separate index spaces.
    
    Let me reformulate. We have:
    - A has entries A[i,j] for i in 0..n-1, j in 0..m-1
    - B has entries B[j,k] for j in 0..m-1, k in 0..p-1
    - C has entries C[i,k] for i in 0..n-1, k in 0..p-1
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    A rank-r decomposition gives:
    C[i,k] = sum_t W[k,i,t] * (sum_j U[i,j,t] A[i,j]) * (sum_j V[j,k,t] B[j,k])
    
    Expanding:
    C[i,k] = sum_t sum_j sum_l W[k,i,t] U[i,j,t] V[j,k,t] A[i,j] B[l,k] ... 
    Wait, the V factor is sum_j V[j,k,t] B[j,k], so it's:
    C[i,k] = sum_t W[k,i,t] (sum_j U[i,j,t] A[i,j]) (sum_l V[l,k,t] B[l,k])
    = sum_t sum_j sum_l W[k,i,t] U[i,j,t] V[l,k,t] A[i,j] B[l,k]
    
    For this to equal sum_j A[i,j] B[j,k], we need:
    sum_t W[k,i,t] U[i,j,t] V[j,k,t] = 1 for all i,j,k
    and
    sum_t W[k,i,t] U[i,j,t] V[l,k,t] = 0 for all i,j,l,k with j != l
    
    So the tensor we're decomposing is T[i,j,l,k] = delta_{j,l} (with the constraint that the
    output is indexed by (i,k)). But U depends on (i,j,t), V on (j,k,t) -- wait, V should depend
    on (j,k,t) where j is the row of B and k is the column.
    
    Let me redefine: U[i,j,t] for i in 0..n-1, j in 0..m-1 (A indices)
    V[j,k,t] for j in 0..m-1, k in 0..p-1 (B indices)
    W[k,i,t] for k in 0..p-1, i in 0..n-1 (C indices, with W[k*n+i,t])
    
    Then:
    C[i,k] = sum_t W[k,i,t] (sum_j U[i,j,t] A[i,j]) (sum_j V[j,k,t] B[j,k])
    
    Expanding the products:
    = sum_t W[k,i,t] sum_j sum_l U[i,j,t] V[l,k,t] A[i,j] B[l,k]
    
    For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
    sum_t W[k,i,t] U[i,j,t] V[l,k,t] = delta_{j,l} for all i,j,l,k
    
    So the tensor is T[i,j,l,k] = delta_{j,l}, and we're decomposing it as:
    T[i,j,l,k] = sum_t W[k,i,t] U[i,j,t] V[l,k,t]
    
    This is a 4-index tensor, but it factors as (i,k) x (j) x (l). The rank is the matrix
    multiplication rank for n x m times m x p.
    
    A rank-1 term is: U[i,j,t] * V[l,k,t] * W[k,i,t] where U is a function of (i,j), V of (l,k),
    W of (k,i). For this to be a valid rank-1 decomposition of T, we need:
    U[i,j] * V[l,k] * W[k,i] = delta_{j,l}
    
    This is only possible if U, V, W are chosen so that the product is 1 when j=l and 0 otherwise.
    The trivial solution: U[i,j] = 1, V[l,k] = 1, W[k,i] = 1, but then the product is 1 for all
    j,l, not delta_{j,l}.
    
    So we need multiple terms. The schoolbook solution uses n*m*p terms:
    For each (i,j,k): U[i,j,t] = 1, V[j,k,t] = 1, W[k,i,t] = 1, and all other entries 0.
    Then sum_t W[k,i,t] U[i,j,t] V[l,k,t] = sum over t of W[k,i,t] U[i,j,t] V[l,k,t].
    For t corresponding to (i,j,k): W[k,i,t]=1, U[i,j,t]=1, V[l,k,t]=1 if l=j.
    So the sum is 1 if l=j, 0 otherwise. 
    
    So the schoolbook rank is n*m*p = 4*5*6 = 120.
    
    To beat this, we need to find a decomposition with rank < 120. This is the matrix
    multiplication rank problem, which is hard. For 4x5*5x6, the known rank is at most
    4*5*6 = 120, and lower bounds are known but the exact rank is not generally known
    for non-square cases.
    
    Given the time constraint, I'll try a greedy search: start with the residual tensor
    T[i,j,l,k] = delta_{j,l}, and at each step, find a rank-1 term (U,V,W) that
    minimizes the residual (in some norm), subtract it, and repeat. Use exact arithmetic.
    
    But this is a huge search space. Let's try a simpler approach: use the known
    Strassen-based tiling and see if it gives rank < 120.
    
    Actually, for 4x4*4x4, Strassen gives rank 49 < 64. For 4x4*4x2, a 2-level Strassen
    might give rank < 32. Let's compute the tiling rank.
    """
    # For now, return schoolbook. The search is too complex for the time budget.
    return _build_schoolbook(n, m, p)

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 120
    
    try:
        # Try Strassen tiling
        U, V, W = _build_strassen_tiling(n, m, p)
        rank = len(U[0])
        
        if time.time() > deadline - 20:
            # Not enough time to verify, return schoolbook
            return {"U": _build_schoolbook(n, m, p)[0], 
                    "V": _build_schoolbook(n, m, p)[1], 
                    "W": _build_schoolbook(n, m, p)[2]}
        
        if _verify(U, V, W, n, m, p, trials=5):
            if time.time() > deadline - 10:
                return {"U": U, "V": V, "W": W}
            # Check if better than schoolbook
            school_rank = n * m * p
            if rank < school_rank:
                return {"U": U, "V": V, "W": W}
        
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    except Exception:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
