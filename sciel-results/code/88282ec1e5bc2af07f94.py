"""Combined: correct rank-28 Strassen baseline + structured rank-reduction search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # ------------------------------------------------------------------
    # 1. Build the correct rank-28 Strassen solution
    # ------------------------------------------------------------------
    best = _strassen_28(n, m, p)

    # Verify the baseline
    U28 = np.array(best["U"], dtype=float)
    V28 = np.array(best["V"], dtype=float)
    W28 = np.array(best["W"], dtype=float)

    T_target = _build_target_tensor(n, m, p, nm, mp, pn)
    T_check = np.einsum('ar,br,cr->abc', U28, V28, W28)
    if not np.allclose(T_check, T_target, atol=1e-9):
        # Fallback: use the explicitly verified version
        best = _strassen_28_verified(n, m, p)
        U28 = np.array(best["U"], dtype=float)
        V28 = np.array(best["V"], dtype=float)
        W28 = np.array(best["W"], dtype=float)

    # ------------------------------------------------------------------
    # 2. Try to find a lower-rank exact decomposition
    # ------------------------------------------------------------------
    for target_rank in range(27, 23, -1):
        if time.time() > deadline:
            break

        result = _try_rank_reduction(U28, V28, W28, target_rank, n, m, p,
                                     T_target, start_time, deadline)
        if result is not None:
            best = result
            break

    # Ensure return format: list of lists of floats
    U_out = [[float(x) for x in row] for row in np.array(best["U"]).tolist()]
    V_out = [[float(x) for x in row] for row in np.array(best["V"]).tolist()]
    W_out = [[float(x) for x in row] for row in np.array(best["W"]).tolist()]

    return {"U": U_out, "V": V_out, "W": W_out}


def _build_target_tensor(n, m, p, nm, mp, pn):
    """Build the matrix multiplication tensor T[a,b,c] where
    a = i*m + j (A index), b = j*p + k (B index), c = k*n + i (C index)
    T[a,b,c] = 1 iff the triple (i,j,k) contributes to C[i][k].
    """
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _strassen_28(n, m, p):
    """Rank-28 solution: 4 independent 2x2 Strassen blocks.
    
    Uses the trace convention W[k*n + i, r] for C[i][k].
    """
    num_blocks = p // 2
    rank = num_blocks * 7

    # Strassen coefficients for 2x2 x 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0][0] = M1 + M4 - M5 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M4
    # C[1][1] = M1 - M2 + M3 + M6

    # U coefficients for each Strassen term (row = A index 0..3)
    # A indices: 0=(0,0)=a, 1=(0,1)=b, 2=(1,0)=c, 3=(1,1)=d
    U_terms = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]

    # V coefficients for each Strassen term (row = B local index 0..3)
    # B local: 0=(0,0)=e, 1=(0,1)=f, 2=(1,0)=g, 3=(1,1)=h
    V_terms = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]

    # W coefficients for each Strassen term (row = C local index 0..3)
    # C local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    W_terms = [
        [1, 0, 0, 1],   # M1: C[0][0]+C[1][1]
        [0, 0, 1, -1],  # M2: C[1][0]-C[1][1]
        [0, 1, 0, 1],   # M3: C[0][1]+C[1][1]
        [1, 0, 1, 0],   # M4: C[0][0]+C[1][0]
        [-1, 1, 0, 0],  # M5: -C[0][0]+C[0][1]
        [0, 0, 0, 1],   # M6: C[1][1]
        [1, 0, 0, 0],   # M7: C[0][0]
    ]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(num_blocks):
        base = t * 7
        for r in range(7):
            col = base + r

            # Fill U: same for all blocks (A is 2x2)
            for i in range(n * m):
                U[i][col] = U_terms[r][i]

            # Fill V: B is 2x8, local block columns 2t, 2t+1
            # Local B index l maps to global B index:
            #   l=0 (row0,col0) -> global 0*p + 2t = 2t
            #   l=1 (row0,col1) -> global 0*p + 2t+1 = 2t+1
            #   l=2 (row1,col0) -> global 1*p + 2t = p+2t
            #   l=3 (row1,col1) -> global 1*p + 2t+1 = p+2t+1
            v_indices = [2 * t, 2 * t + 1, p + 2 * t, p + 2 * t + 1]
            for l in range(4):
                V[v_indices[l]][col] = V_terms[r][l]

            # Fill W: C is 2x8, local block columns 2t, 2t+1
            # Local C index l maps to global W index (k*n + i):
            #   l=0 (i=0,k=2t)   -> 2t*n + 0 = 4t
            #   l=1 (i=0,k=2t+1) -> (2t+1)*n + 0 = 4t+2
            #   l=2 (i=1,k=2t)   -> 2t*n + 1 = 4t+1
            #   l=3 (i=1,k=2t+1) -> (2t+1)*n + 1 = 4t+3
            w_indices = [4 * t, 4 * t + 2, 4 * t + 1, 4 * t + 3]
            for l in range(4):
                W[w_indices[l]][col] = W_terms[r][l]

    return {"U": U, "V": V, W: W}


def _strassen_28_verified(n, m, p):
    """Explicitly verified rank-28 Strassen solution."""
    num_blocks = p // 2
    rank = num_blocks * 7

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(num_blocks):
        base = b * 7

        # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U[0][base + 0] += 1
        U[3][base + 0] += 1
        V[0 * p + 2 * b][base + 0] += 1
        V[1 * p + 2 * b + 1][base + 0] += 1
        W[2 * b * n + 0][base + 0] += 1      # C[0][2b]
        W[(2 * b + 1) * n + 1][base + 0] += 1  # C[1][2b+1]

        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U[2][base + 1] += 1
        U[3][base + 1] += 1
        V[0 * p + 2 * b][base + 1] += 1
        W[2 * b * n + 1][base + 1] += 1      # C[1][2b]
        W[(2 * b + 1) * n + 1][base + 1] -= 1  # C[1][2b+1]

        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U[0][base + 2] += 1
        V[0 * p + 2 * b + 1][base + 2] += 1
        V[1 * p + 2 * b + 1][base + 2] -= 1
        W[(2 * b + 1) * n + 0][base + 2] += 1  # C[0][2b+1]
        W[(2 * b + 1) * n + 1][base + 2] += 1  # C[1][2b+1]

        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U[3][base + 3] += 1
        V[1 * p + 2 * b][base + 3] += 1
        V[0 * p + 2 * b][base + 3] -= 1
        W[2 * b * n + 0][base + 3] += 1      # C[0][2b]
        W[2 * b * n + 1][base + 3] += 1      # C[1][2b]

        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U[0][base + 4] += 1
        U[1][base + 4] += 1
        V[1 * p + 2 * b + 1][base + 4] += 1
        W[2 * b * n + 0][base + 4] -= 1      # C[0][2b]
        W[(2 * b + 1) * n + 0][base + 4] += 1  # C[0][2b+1]

        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U[2][base + 5] += 1
        U[0][base + 5] -= 1
        V[0 * p + 2 * b][base + 5] += 1
        V[0 * p + 2 * b + 1][base + 5] += 1
        W[(2 * b + 1) * n + 1][base + 5] += 1  # C[1][2b+1]

        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U[1][base + 6] += 1
        U[3][base + 6] -= 1
        V[1 * p + 2 * b][base + 6] += 1
        V[1 * p + 2 * b + 1][base + 6] += 1
        W[2 * b * n + 0][base + 6] += 1      # C[0][2b]

    return {"U": U, "V": V, "W": W}


def _try_rank_reduction(U28, V28, W28, target_rank, n, m, p, T_target,
                        start_time, deadline):
    """Try to find a rank-`target_rank` decomposition by merging Strassen blocks.
    
    Key insight: The 4 Strassen blocks share the same A factors. If we can
    find a way to express the V and W factors of multiple blocks using fewer
    rank-1 terms, we reduce the total rank.
    
    Strategy: For each pair of blocks, try to find a rank-6 (instead of 14)
    decomposition of the combined 2x2x4 sub-tensor.
    """
    nm, mp, pn = n * m, m * p, p * n
    R = 28

    # Try merging pairs of blocks
    # Blocks 0&1, 0&2, 0&3, 1&2, 1&3, 2&3
    block_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    for b1, b2 in block_pairs:
        if time.time() > deadline:
            return None

        # Extract the sub-tensor for blocks b1 and b2
        # Columns b1*7..b1*7+6 and b2*7..b2*7+6
        cols = list(range(b1 * 7, b1 * 7 + 7)) + list(range(b2 * 7, b2 * 7 + 7))
        U_sub = U28[:, cols]
        V_sub = V28[:, cols]
        W_sub = W28[:, cols]

        # The sub-tensor has rank at most 14. Try to find rank-6 or less.
        # For a 2x2x4 tensor, the minimum rank is known to be 8 (for generic).
        # But our specific tensor might have lower rank.
        
        # Try ALS with half-integer constraint
        result = _als_half_integer(n, m, p, nm, mp, pn, 6, T_target,
                                   U28, V28, W28, b1, b2, start_time, deadline)
        if result is not None:
            # Reconstruct full decomposition
            full_U = U28.copy()
            full_V = V28.copy()
            full_W = W28.copy()
            
            # Remove the old columns for blocks b1 and b2
            old_cols = list(range(b1 * 7, b1 * 7 + 7)) + list(range(b2 * 7, b2 * 7 + 7))
            # Keep other columns
            keep_cols = [c for c in range(28) if c not in old_cols]
            
            new_U = np.column_stack([full_U[:, keep_cols], result["U"]])
            new_V = np.column_stack([full_V[:, keep_cols], result["V"]])
            new_W = np.column_stack([full_W[:, keep_cols], result["W"]])
            
            # Verify
            T_new = np.einsum('ar,br,cr->abc', new_U, new_V, new_W)
            if np.allclose(T_new, T_target, atol=1e-9):
                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}

    return None


def _als_half_integer(n, m, p, nm, mp, pn, target_rank, T_target,
                      U28, V28, W28, b1, b2, start_time, deadline):
    """Try to find a rank-`target_rank` decomposition for the sub-tensor of blocks b1,b2."""
    max_attempts = 20
    max_iters = 100

    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None

        rng = np.random.RandomState(attempt * 13 + b1 * 7 + b2)

        # Initialize from the Strassen factors of the two blocks
        cols = list(range(b1 * 7, b1 * 7 + 7)) + list(range(b2 * 7, b2 * 7 + 7))
        U_init = U28[:, cols]
        V_init = V28[:, cols]
        W_init = W28[:, cols]

        # Try to project to lower rank using SVD-like approach
        # For each mode, find the best rank-r approximation
        U = U_init[:, :target_rank].copy() if target_rank <= 14 else U_init.copy()
        V = V_init[:, :target_rank].copy() if target_rank <= 14 else V_init.copy()
        W = W_init[:, :target_rank].copy() if target_rank <= 14 else W_init.copy()

        # ALS refinement
        success = False
        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Update U
            for a in range(nm):
                i = a // m
                j = a % m
                M = np.zeros((p, target_rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                rhs = np.ones(p)
                try:
                    U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                except:
                    pass

            # Update V
            for b in range(mp):
                j = b // p
                k = b % p
                M = np.zeros((n, target_rank))
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    M[i, :] = U[a, :] * W[c, :]
                rhs = np.ones(n)
                try:
                    V[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                except:
                    pass

            # Update W
            for c in range(pn):
                k = c // n
                i = c % n
                M = np.zeros((m, target_rank))
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    M[j, :] = U[a, :] * V[b, :]
                rhs = np.ones(m)
                try:
                    W[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                except:
                    pass

            # Check convergence
            T_cur = np.einsum('ar,br,cr->abc', U, V, W)
            max_err = np.max(np.abs(T_cur - T_target))
            if max_err < 1e-10:
                success = True
                break

        if success:
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2

            T_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_r, T_target, atol=1e-9):
                return {"U": U_r, "V": V_r, "W": W_r}

    return None

# EVOLVE-BLOCK-END

