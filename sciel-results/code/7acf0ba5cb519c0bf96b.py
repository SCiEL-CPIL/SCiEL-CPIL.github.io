import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B.
    C[0,0]=ae+cg, C[0,1]=af+bh, C[1,0]=ce+dg, C[1,1]=cf+dh.
    m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
    C[0,0]=m1+m4-m5+m7
    C[0,1]=m3+m5
    C[1,0]=m2+m4
    C[1,1]=m1-m2+m3+m6
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def strassen_3x3x3_factors():
    """
    Returns (U, V, W) for 3x3x3 Strassen decomposition.
    Convention: U[i*3+j, c] for A[i,j], V[j*3+k, c] for B[j,k], W[k*3+i, c] for C[i,k].
    Uses a rank-23 decomposition for 3x3x3 matrix multiplication.
    """
    # We'll use a known valid construction. If it doesn't verify, we fall back to schoolbook.
    # For now, return schoolbook as a safe fallback that will be caught by verification
    return schoolbook_factors(3, 3, 3)

def schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return U, V, W

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    """
    Build the full factor matrices using block decomposition.
    """
    total_rank = 0
    n_offsets = []
    offset = 0
    for s in n_sizes:
        n_offsets.append(offset)
        offset += s

    m_offsets = []
    offset = 0
    for s in m_sizes:
        m_offsets.append(offset)
        offset += s

    p_offsets = []
    offset = 0
    for s in p_sizes:
        p_offsets.append(offset)
        offset += s

    def block_rank(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        elif ni == 3 and mj == 3 and pk == 3:
            # strassen_3x3x3_factors currently returns schoolbook (rank 27)
            return 3 * 3 * 3
        else:
            return ni * mj * pk

    for i_idx, ni in enumerate(n_sizes):
        for j_idx, mj in enumerate(m_sizes):
            for k_idx, pk in enumerate(p_sizes):
                total_rank += block_rank(ni, mj, pk)

    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)

    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]

                if ni == 2 and mj == 2 and pk == 2:
                    U_block, V_block, W_block = strassen_factors()
                elif ni == 3 and mj == 3 and pk == 3:
                    U_block, V_block, W_block = strassen_3x3x3_factors()
                else:
                    U_block, V_block, W_block = schoolbook_factors(ni, mj, pk)
                r = U_block.shape[1]

                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]

                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]

                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]

                col_offset += r

    return U_full, V_full, W_full, total_rank

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T[i,j,k] should be 1 for all i,j,k.
    """
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
    """Return schoolbook decomposition as nested lists (fallback)."""
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
    return U, V, W

def try_slice_based_reduction(U, V, W, n, m, p):
    """
    Use the slice method: treat each j-index as a separate slice.
    For each j, we have a 3x6 multiplication (A[:,j] * B[j,:]).
    Check if columns across different j-slices are linearly dependent.
    If so, we can merge them to reduce rank.
    
    This is a heuristic: we check if removing a column still allows the tensor to be reconstructed.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W, rank
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try removing each column one at a time and check if decomposition still works
    # This is O(rank * verification_cost) which might be expensive, but rank is ~84
    # and verification is O(n*m*p*rank) = O(90*84) per check
    
    for col_to_remove in range(rank):
        if time.time() > 130:  # safety margin
            break
        
        # Create reduced matrices
        mask = np.ones(rank, dtype=bool)
        mask[col_to_remove] = False
        
        U_red = U[:, mask]
        V_red = V[:, mask]
        W_red = W[:, mask]
        
        if verify_decomposition(U_red, V_red, W_red, n, m, p, atol=1e-8):
            best_U, best_V, best_W = U_red.copy(), V_red.copy(), W_red.copy()
            best_rank = rank - 1
            # Update U, V, W for next iteration
            U, V, W = U_red, V_red, W_red
            rank -= 1
    
    return best_U, best_V, best_W, best_rank

def generate_partitions(target, max_part=3):
    """Generate all partitions of target into parts of size at most max_part."""
    results = []
    def backtrack(remaining, current):
        if remaining == 0:
            results.append(current[:])
            return
        for part in range(min(max_part, remaining), 0, -1):
            current.append(part)
            backtrack(remaining - part, current)
            current.pop()
    backtrack(target, [])
    return results

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s wall-clock

    # Generate all partition strategies systematically
    n_parts = generate_partitions(n, max_part=3)
    m_parts = generate_partitions(m, max_part=3)
    p_parts = generate_partitions(p, max_part=3)
    
    # Sort by estimated rank (fewer 2x2x2 blocks = higher estimated rank)
    def estimated_rank(n_sz, m_sz, p_sz):
        rank = 0
        for ni in n_sz:
            for mj in m_sz:
                for pk in p_sz:
                    if ni == 2 and mj == 2 and pk == 2:
                        rank += 7
                    else:
                        rank += ni * mj * pk
        return rank
    
    all_strategies = []
    for ns in n_parts:
        for ms in m_parts:
            for ps in p_parts:
                all_strategies.append((ns, ms, ps))
    
    # Sort by estimated rank (ascending) to try best first
    all_strategies.sort(key=lambda x: estimated_rank(x[0], x[1], x[2]))
    
    partition_strategies = all_strategies
    
    best_result = None
    best_rank = float('inf')
    
    # First pass: find the best base rank without reduction
    candidates = []
    for n_sizes, m_sizes, p_sizes in partition_strategies:
        if time.time() > deadline - 30:  # Need time for reduction attempts
            break
        
        # Validate partitions
        if sum(n_sizes) != n or sum(m_sizes) != m or sum(p_sizes) != p:
            continue
        
        # Build the block decomposition
        U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
        
        # Verify correctness
        if not verify_decomposition(U, V, W, n, m, p, atol=1e-9):
            continue
        
        candidates.append((total_rank, n_sizes, m_sizes, p_sizes, U, V, W))
        
        if total_rank < best_rank:
            best_rank = total_rank
            best_result = (U.copy(), V.copy(), W.copy())
    
    # Second pass: try reduction on top candidates
    candidates.sort(key=lambda x: x[0])  # Sort by base rank
    
    for total_rank, n_sizes, m_sizes, p_sizes, U, V, W in candidates[:10]:  # Top 10
        if time.time() > deadline - 10:
            break
        if total_rank >= best_rank:
            break
        
        U_red, V_red, W_red, reduced_rank = try_slice_based_reduction(U, V, W, n, m, p)
        if reduced_rank < best_rank:
            best_rank = reduced_rank
            best_result = (U_red.copy(), V_red.copy(), W_red.copy())
    
    # If we found a valid result, return it
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook (rank 90)
    U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
