"""Matrix multiplication 2,5,6 - block decomposition with Strassen."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen. U is (4,7), V is (4,7), W is (4,7)."""
    # U rows: (i,j) -> i*2+j
    # V rows: (j,k) -> j*2+k
    # W rows: (k,i) -> k*2+i
    U = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
    ]
    W = [
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
    ]
    return U, V, W


def _add_block(U, V, W, i0, i1, j0, j1, k0, k1, subU, subV, subW, m, p, n):
    """Add a sub-problem decomposition to global matrices.
    i0,i1: row range of A block (n indices)
    j0,j1: column range of A / row range of B block (m indices)
    k0,k1: column range of B block (p indices)
    subU, subV, subW: (ni*nj, r), (nj*nk, r), (nk*ni, r) for the sub-problem
    """
    ni = i1 - i0
    nj = j1 - j0
    nk = k1 - k0
    r = len(subU[0])
    for c in range(r):
        # U sub-rows: (ii, jj) for ii in range(ni), jj in range(nj)
        # global U index: (i0+ii)*m + (j0+jj)
        for ii in range(ni):
            for jj in range(nj):
                val = subU[ii * nj + jj][c]
                if val != 0:
                    U[(i0 + ii) * m + (j0 + jj)][-1] += val
        # V sub-rows: (jj, kk) for jj in range(nj), kk in range(nk)
        # global V index: (j0+jj)*p + (k0+kk)
        for jj in range(nj):
            for kk in range(nk):
                val = subV[jj * nk + kk][c]
                if val != 0:
                    V[(j0 + jj) * p + (k0 + kk)][-1] += val
        # W sub-rows: (kk, ii) for kk in range(nk), ii in range(ni)
        # global W index: (k0+kk)*n + (i0+ii)
        for kk in range(nk):
            for ii in range(ni):
                val = subW[kk * ni + ii][c]
                if val != 0:
                    W[(k0 + kk) * n + (i0 + ii)][-1] += val


def _block_mult(ni, nj, nk, m, p, n):
    """Return (U, V, W) for a ni x nj x nk block multiplication.
    U is (ni*nj, r), V is (nj*nk, r), W is (nk*ni, r).
    """
    if ni == 1 and nj == 1 and nk == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    if ni == 2 and nj == 2 and nk == 2:
        return _strassen_2x2x2()
    
    # Fallback: schoolbook
    r = ni * nj * nk
    U = [[0] * r for _ in range(ni * nj)]
    V = [[0] * r for _ in range(nj * nk)]
    W = [[0] * r for _ in range(nk * ni)]
    c = 0
    for ii in range(ni):
        for jj in range(nj):
            for kk in range(nk):
                U[ii * nj + jj][c] = 1
                V[jj * nk + kk][c] = 1
                W[kk * ni + ii][c] = 1
                c += 1
    return U, V, W


def construct():
    import time
    start = time.time()
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Split m=5 into 2+3
    m1, m2 = 2, 3
    
    # For block (n=2, m1=2, p=6): split p into 2+2+2, use Strassen for each
    # For block (n=2, m2=3, p=6): split m2 into 1+2
    #   - (n=2, m=1, p=6): schoolbook rank=12
    #   - (n=2, m=2, p=6): split p into 2+2+2, use Strassen for each, rank=3*7=21
    
    # Let's build the full decomposition
    # We'll collect all sub-decompositions and their global positions
    
    sub_blocks = []
    # Each entry: (i0, i1, j0, j1, k0, k1, subU, subV, subW)
    
    # Block 1: i in [0,2), j in [0,2), k in [0,6)
    # Split k into three 2s
    for k_start in range(0, 6, 2):
        subU, subV, subW = _block_mult(2, 2, 2, m, p, n)
        sub_blocks.append((0, 2, 0, 2, k_start, k_start + 2, subU, subV, subW))
    
    # Block 2: i in [0,2), j in [2,5), k in [0,6)
    # Split j into [2,3) and [3,5)
    # Sub-block 2a: i in [0,2), j in [2,3), k in [0,6) -> ni=2, nj=1, nk=6
    #   Schoolbook: rank = 2*1*6 = 12
    subU, subV, subW = _block_mult(2, 1, 6, m, p, n)
    sub_blocks.append((0, 2, 2, 3, 0, 6, subU, subV, subW))
    
    # Sub-block 2b: i in [0,2), j in [3,5), k in [0,6) -> ni=2, nj=2, nk=6
    #   Split k into three 2s, use Strassen
    for k_start in range(0, 6, 2):
        subU, subV, subW = _block_mult(2, 2, 2, m, p, n)
        sub_blocks.append((0, 2, 3, 5, k_start, k_start + 2, subU, subV, subW))
    
    # Compute total rank
    total_rank = sum(len(subU[0]) for _, _, _, _, _, _, subU, _, _ in sub_blocks)
    
    # Initialize global matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in
    col = 0
    for i0, i1, j0, j1, k0, k1, subU, subV, subW in sub_blocks:
        ni = i1 - i0
        nj = j1 - j0
        nk = k1 - k0
        r = len(subU[0])
        for c in range(r):
            # U
            for ii in range(ni):
                for jj in range(nj):
                    val = subU[ii * nj + jj][c]
                    if val != 0:
                        U[(i0 + ii) * m + (j0 + jj)][col + c] += val
            # V
            for jj in range(nj):
                for kk in range(nk):
                    val = subV[jj * nk + kk][c]
                    if val != 0:
                        V[(j0 + jj) * p + (k0 + kk)][col + c] += val
            # W
            for kk in range(nk):
                for ii in range(ni):
                    val = subW[kk * ni + ii][c]
                    if val != 0:
                        W[(k0 + kk) * n + (i0 + ii)][col + c] += val
        col += r
    
    # Verify time
    elapsed = time.time() - start
    if elapsed > 100:
        # Shouldn't happen for this small problem, but safety
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
