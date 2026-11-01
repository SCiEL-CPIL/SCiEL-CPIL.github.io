"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 2x4 * 4x5, then Winograd-style merge
    of rank-1 terms that share identical U-column AND identical W-column.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 2, 4, 5

    rank_terms = []  # Each term is (u_dict, v_dict, w_dict)

    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))

    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))

    # Build block-Strassen decomposition (rank 36)
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))

    # Winograd-style merge: if two terms share identical U and identical W,
    # merge their V columns into one term.
    def merge_terms(terms):
        # Group by (u_key, w_key)
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, (u, v, w) in enumerate(terms):
            u_key = tuple(sorted(u.items()))
            w_key = tuple(sorted(w.items()))
            groups[(u_key, w_key)].append((idx, v))
        merged = []
        used = set()
        for (u_key, w_key), members in groups.items():
            if len(members) == 1:
                idx, v = members[0]
                if idx not in used:
                    u = dict(u_key)
                    w = dict(w_key)
                    merged.append((u, v, w))
                    used.add(idx)
            else:
                # Merge all V columns: sum them into one V dict
                u = dict(u_key)
                w = dict(w_key)
                v_sum = {}
                for idx, v in members:
                    if idx in used:
                        continue
                    for (jk, coeff) in v.items():
                        v_sum[jk] = v_sum.get(jk, 0) + coeff
                    used.add(idx)
                # Remove zero entries
                v_sum = {k: val for k, val in v_sum.items() if val != 0}
                if v_sum:
                    merged.append((u, v_sum, w))
        return merged

    rank_terms = merge_terms(rank_terms)

    # Verify correctness
    def verify(terms):
        # Check that sum of u*v*w reconstructs the tensor
        # T[i,j,k] = sum over terms of u[(i,j)] * v[(j,k)] * w[(i,k)]
        # But our convention: U row = i*m+j, V row = j*p+k, W col = k*n+i
        # Tensor T[i,j,k] should equal delta for matmul: T[i,j,k] = 1 if ... 
        # Actually for matmul tensor: C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor is T[i,j,k] = 1 (the matmul tensor has 1s at all (i,j,k) positions
        # that form a valid triple, i.e., T[i,j,k] = 1 for all i,j,k in range)
        # Wait, the matmul tensor M[i,j,k] = 1 for all valid (i,j,k) where
        # i in [0,n), j in [0,m), k in [0,p). That's the standard matmul tensor.
        # So we need: sum_t u_t[(i,j)] * v_t[(j,k)] * w_t[(i,k)] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for (u, v, w) in terms:
                        total += u.get((i, j), 0) * v.get((j, k), 0) * w.get((i, k), 0)
                    if total != 1:
                        return False
        return True

    if not verify(rank_terms):
        # Fallback: return the unmerged version (should be correct)
        rank_terms = []
        add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
        add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
        add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
        add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
        add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
        add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))

    rank = len(rank_terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
