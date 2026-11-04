import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """2x2x2 Strassen decomposition. U[i*2+j,c] for A[i,j]; V[j*2+k,c] for B[j,k]; W[k*2+i,c] for C[i,k]."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    # m1=(a+d)(e+h)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # m2=(c+d)e
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # m3=a(f-h)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # m4=d(g-e)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # m5=(a+b)h
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # m6=(c-a)(e+f)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # m7=(b-d)(g+h)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def schoolbook_factors(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _block_rank(ni, mj, pk):
    if ni == 2 and mj == 2 and pk == 2:
        return 7
    return ni * mj * pk

def _find_best_block_partition(n, m, p):
    best_rank = n * m * p
    best_partition = ([1]*n, [1]*m, [1]*p)
    def compositions(total):
        if total == 1:
            yield [1]
            return
        for first in range(1, total + 1):
            for rest in compositions(total - first):
                yield [first] + rest
    n_comps = list(compositions(n))
    m_comps = list(compositions(m))
    p_comps = list(compositions(p))
    for n_sizes in n_comps:
        for m_sizes in m_comps:
            for p_sizes in p_comps:
                rank = 0
                for ni in n_sizes:
                    for mj in m_sizes:
                        for pk in p_sizes:
                            rank += _block_rank(ni, mj, pk)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (list(n_sizes), list(m_sizes), list(p_sizes))
    return best_partition[0], best_partition[1], best_partition[2], best_rank

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    total_rank = 0
    n_offsets = []
    offset = 0
    for s in n_sizes:
        n_offsets.append(offset); offset += s
    m_offsets = []
    offset = 0
    for s in m_sizes:
        m_offsets.append(offset); offset += s
    p_offsets = []
    offset = 0
    for s in p_sizes:
        p_offsets.append(offset); offset += s
    for i_idx, ni in enumerate(n_sizes):
        for j_idx, mj in enumerate(m_sizes):
            for k_idx, pk in enumerate(p_sizes):
                total_rank += _block_rank(ni, mj, pk)
    U_full = np.zeros((n*m, total_rank), dtype=np.float64)
    V_full = np.zeros((m*p, total_rank), dtype=np.float64)
    W_full = np.zeros((p*n, total_rank), dtype=np.float64)
    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                if ni == 2 and mj == 2 and pk == 2:
                    U_block, V_block, W_block = strassen_factors()
                else:
                    U_block, V_block, W_block = schoolbook_factors(ni, mj, pk)
                r = U_block.shape[1]
                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li*mj + lj, :]
                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj*pk + lk, :]
                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk*ni + li, :]
                col_offset += r
    return U_full, V_full, W_full, total_rank

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)
    return bool(np.allclose(vals, 1.0, atol=atol))

def schoolbook_lists(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_rank83_search(deadline):
    """
    Search for a rank-83 decomposition exploiting the 1x5x6 + 2x5x6 split.
    The 1x5x6 block (row i=2) must have rank >= 30 (each of its 30 entries needs a
    term). The 2x5x6 block (rows i=0,1) has 60 entries. A rank-53 block combined
    with rank-30 gives 83. We try to build the 2x5x6 block with 53 terms where the
    shared intermediate dimension allows one term to be dropped via a linear
    dependency. We do a randomized greedy sparse search over linear forms in
    {-1,0,1}.
    """
    n, m, p = 3, 5, 6
    # Build the target tensor T[i,j,k] = 1 for all i,j,k (the multiplication tensor
    # has coefficient 1 at every (i,j,k)).
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    #
    # Strategy: start from a rank-84 candidate (block Strassen on the 2x5x6 part +
    # schoolbook on 1x5x6), then attempt to eliminate one term by finding a
    # linear relation. We do a randomized perturbation + local search.
    #
    # Simpler robust approach: direct randomized search for rank 83.
    # Represent each term c by (u_c in {-1,0,1}^{15}, v_c in {-1,0,1}^{30},
    # w_c in {-1,0,1}^{18}). The constraint for each (i,j,k) is a sum of 83
    # products = 1.
    #
    # This is a hard CSP. Instead we use the block structure: fix the 1x5x6 block
    # (i=2) to schoolbook rank 30 (indices 0..29), and search the 2x5x6 block
    # (i=0,1) for rank 53.
    #
    # For the 2x5x6 block, the entries are:
    #   i=0: C[0,k] = sum_j A[0,j] B[j,k]  -> 30 terms (j,k)
    #   i=1: C[1,k] = sum_j A[1,j] B[j,k]  -> 30 terms (j,k)
    # Total 60 entries. Rank-53 means 7 entries share terms.
    #
    # We attempt a structured search: use Strassen-like 2x2x2 blocks but allow
    # one cross-term elimination. Given the complexity and time budget, we do a
    # limited randomized local search.

    # Build rank-84 baseline for the 2x5x6 + 1x5x6 split.
    # 2x5x6: split m=5 into 2+2+1, p=6 into 2+2+2.
    # Blocks: (2,2,2)x3 = 7*3=21, (2,2,2) already counted... let me just use
    # the general block builder for 2x5x6 with best partition.
    n_sizes, m_sizes, p_sizes, _ = _find_best_block_partition(2, 5, 6)
    U2, V2, W2, r2 = build_block_decomposition(2, 5, 6, n_sizes, m_sizes, p_sizes)
    # 1x5x6 schoolbook
    U1, V1, W1, r1 = build_block_decomposition(1, 5, 6, [1], [1]*5, [1]*6)
    # Assemble rank-84 (or whatever r2+r1 is)
    r_total = r2 + r1
    U = np.zeros((15, r_total), dtype=np.float64)
    V = np.zeros((30, r_total), dtype=np.float64)
    W = np.zeros((18, r_total), dtype=np.float64)
    # i=0,1 rows in U are rows 0..9; i=2 rows are 10..14
    U[:10, :r2] = U2
    U[10:, r2:] = U1
    V[:r2] = V2  # V2 has 30 rows already (m=5,p=6 -> 30)
    V[r2:] = V1
    # W: for i=0,1 (rows k*3+0, k*3+1) and i=2 (row k*3+2)
    # W2 has shape (18, r2) with convention W[k*2+i, c] for the 2-row block
    # We need to map W2[k*2+i] to W_full[k*3+i]
    for k in range(6):
        for i in range(2):
            W[k*3 + i, :r2] = W2[k*2 + i, :]
    for k in range(6):
        W[k*3 + 2, r2:] = W1[k*1 + 0, :]  # W1 has shape (6, 30), W1[k*1+0]

    if not verify_decomposition(U, V, W, 3, 5, 6, atol=1e-9):
        return None

    # Now attempt to reduce rank by 1: try to find a linear dependency among the
    # 84 columns that allows dropping one. We do a randomized local search:
    # pick a term to "absorb", perturb nearby terms, and check.
    # Given the difficulty and time budget, we do a short randomized search.
    rng = np.random.RandomState(42)
    best = (U.copy(), V.copy(), W.copy(), r_total)
    for it in range(3000):
        if time.time() > deadline - 30:
            break
        # Try to drop a random term by redistributing its contribution.
        # This is a heuristic; we attempt small perturbations.
        c_drop = rng.randint(0, r_total)
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        # Zero out column c_drop
        U_new[:, c_drop] = 0
        V_new[:, c_drop] = 0
        W_new[:, c_drop] = 0
        # Try to compensate by adjusting one other column
        c_adj = rng.randint(0, r_total)
        if c_adj == c_drop:
            continue
        # Perturb column c_adj
        for _ in range(3):
            U_new[:, c_adj] += rng.randint(-1, 2, size=15)
            V_new[:, c_adj] += rng.randint(-1, 2, size=30)
            W_new[:, c_adj] += rng.randint(-1, 2, size=18)
        # Check if still valid (rarely will be, but try)
        if verify_decomposition(U_new, V_new, W_new, 3, 5, 6, atol=1e-9):
            # Remove the dropped column
            mask = np.ones(r_total, dtype=bool)
            mask[c_drop] = False
            U_new = U_new[:, mask]
            V_new = V_new[:, mask]
            W_new = W_new[:, mask]
            return (U_new, V_new, W_new, r_total - 1)
    return None

def construct():
    """Constructs factor matrices U, V, W for the 3x5x6 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # First, get the rank-84 block-Strassen solution.
    n_sizes, m_sizes, p_sizes, expected_rank = _find_best_block_partition(n, m, p)
    U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)

    if verify_decomposition(U, V, W, n, m, p, atol=1e-9):
        best = (U, V, W, total_rank)
    else:
        best = None

    # Attempt rank-83 search.
    if time.time() < deadline - 40:
        r83 = _try_rank83_search(deadline)
        if r83 is not None:
            U83, V83, W83, r83_rank = r83
            if verify_decomposition(U83, V83, W83, n, m, p, atol=1e-9):
                best = (U83, V83, W83, r83_rank)

    if best is not None:
        Ub, Vb, Wb, _ = best
        return {"U": Ub.tolist(), "V": Vb.tolist(), "W": Wb.tolist()}

    # Fallback: schoolbook
    U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
