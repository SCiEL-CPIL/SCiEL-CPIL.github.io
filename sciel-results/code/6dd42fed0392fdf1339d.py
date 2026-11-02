"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 2x4 * 4x5 with fast verification and
    systematic search over block decompositions. Exploits n=2 structure.
    """
    import time
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 4, 5

    def strassen_2x2x2_terms(a_offset, b_offset, c_offset, sign=1):
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
        return [(
            {k: val * sign for k, val in u.items()},
            {k: val * sign for k, val in v.items()},
            {k: val * sign for k, val in w.items()}
        ) for u, v, w in terms]

    def schoolbook_2x2x1_terms(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        return [(
            {k: val * sign for k, val in u.items()},
            {k: val * sign for k, val in v.items()},
            {k: val * sign for k, val in w.items()}
        ) for u, v, w in terms]

    def schoolbook_2x1x5_terms(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        terms = []
        for k in range(5):
            terms.append((
                {(i0, j0): 1},
                {(j0, k0 + k): 1},
                {(i0, k0 + k): 1}
            ))
            terms.append((
                {(i0+1, j0): 1},
                {(j0, k0 + k): 1},
                {(i0+1, k0 + k): 1}
            ))
        return [(
            {k: val * sign for k, val in u.items()},
            {k: val * sign for k, val in v.items()},
            {k: val * sign for k, val in w.items()}
        ) for u, v, w in terms]

    def decompose_2x2x5(a_offset, b_offset, c_offset, output_split, sign=1):
        """Decompose 2x2x5 block. output_split is tuple of sizes summing to 5."""
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        terms = []
        k_offset = c_k0
        for size in output_split:
            if size == 2:
                terms.extend(strassen_2x2x2_terms((i0, j0), (j1, k_offset), (c_i0, k_offset), sign))
            elif size == 1:
                terms.extend(schoolbook_2x2x1_terms((i0, j0), (j1, k_offset), (c_i0, k_offset), sign))
            elif size == 3:
                terms.extend(strassen_2x2x2_terms((i0, j0), (j1, k_offset), (c_i0, k_offset), sign))
                terms.extend(schoolbook_2x2x1_terms((i0, j0), (j1, k_offset + 2), (c_i0, k_offset + 2), sign))
            k_offset += size
        return terms

    def try_decomposition(inner_split, output_split_per_block):
        """Try a block decomposition. inner_split: tuple summing to m=4.
        output_split_per_block: for each inner block, a tuple summing to p=5."""
        terms = []
        j_offset = 0
        for idx, inner_size in enumerate(inner_split):
            output_split = output_split_per_block[idx]
            if inner_size == 2:
                terms.extend(decompose_2x2x5((0, j_offset), (j_offset, 0), (0, 0), output_split))
            elif inner_size == 1:
                terms.extend(schoolbook_2x1x5_terms((0, j_offset), (j_offset, 0), (0, 0)))
            elif inner_size == 3:
                terms.extend(decompose_2x2x5((0, j_offset), (j_offset, 0), (0, 0), output_split))
                terms.extend(schoolbook_2x1x5_terms((0, j_offset + 2), (j_offset + 2, 0), (0, 0)))
            j_offset += inner_size
        return terms

    def verify_fast(terms):
        """Fast verification using vectorized contribution accumulation."""
        # Accumulate contributions to each (i,j,k) entry
        contrib = {}
        for (u, v, w) in terms:
            for (i, j), uc in u.items():
                for (j2, k), vc in v.items():
                    if j2 != j:
                        continue
                    for (i2, k2), wc in w.items():
                        if i2 != i or k2 != k:
                            continue
                        key = (i, j, k)
                        contrib[key] = contrib.get(key, 0) + uc * vc * wc
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if contrib.get((i, j, k), 0) != 1:
                        return False
        return True

    def merge_terms_fast(terms):
        """Merge terms with identical (U, W) patterns by summing V."""
        from collections import defaultdict

        groups = defaultdict(list)
        for idx, (u, v, w) in enumerate(terms):
            u_key = tuple(sorted(u.items()))
            w_key = tuple(sorted(w.items()))
            groups[(u_key, w_key)].append(v)

        merged = []
        for (u_key, w_key), v_list in groups.items():
            u = dict(u_key)
            w = dict(w_key)
            v_sum = {}
            for v in v_list:
                for (jk, coeff) in v.items():
                    v_sum[jk] = v_sum.get(jk, 0) + coeff
            v_sum = {k: val for k, val in v_sum.items() if val != 0}
            if v_sum:
                merged.append((u, v_sum, w))
        return merged

    # Generate all valid inner splits of m=4
    def gen_splits(total, max_part=3):
        if total == 0:
            yield ()
            return
        for first in range(1, min(max_part, total) + 1):
            for rest in gen_splits(total - first, max_part):
                yield (first,) + rest

    # Generate all valid output splits of p=5
    def gen_output_splits(total, max_part=3):
        if total == 0:
            yield ()
            return
        for first in range(1, min(max_part, total) + 1):
            for rest in gen_output_splits(total - first, max_part):
                yield (first,) + rest

    candidates = []
    best_rank = 40  # schoolbook

    # Try all inner splits and output splits
    inner_splits = list(gen_splits(4, 3))
    output_splits = list(gen_output_splits(5, 3))

    for inner_split in inner_splits:
        for idx, inner_size in enumerate(inner_split):
            if inner_size == 2:
                # For 2-wide blocks, try various output splits
                pass
            elif inner_size == 1:
                pass
            elif inner_size == 3:
                pass

        # Build output split per block
        for os_combo in _gen_output_combos(inner_split, output_splits):
            terms = try_decomposition(inner_split, os_combo)
            if len(terms) >= best_rank:
                continue
            if verify_fast(terms):
                merged = merge_terms_fast(terms)
                if verify_fast(merged) and len(merged) < best_rank:
                    candidates.append(merged)
                    best_rank = len(merged)

    # Also try the known good decompositions explicitly
    explicit_decomps = [
        ((2, 2), ((2, 2, 1), (2, 2, 1))),
        ((2, 2), ((3, 2), (3, 2))),
        ((2, 2), ((2, 3), (2, 3))),
        ((1, 3), ((5,), (2, 2, 1))),
        ((3, 1), ((2, 2, 1), (5,))),
        ((1, 1, 1, 1), ((5,), (5,), (5,), (5,))),
        ((2, 1, 1), ((2, 2, 1), (5,), (5,))),
        ((1, 2, 1), ((5,), (2, 2, 1), (5,))),
        ((1, 1, 2), ((5,), (5,), (2, 2, 1))),
        ((2, 2), ((1, 1, 1, 1, 1), (1, 1, 1, 1, 1))),
    ]
    for inner_split, os_combo in explicit_decomps:
        if time.time() > deadline:
            break
        terms = try_decomposition(inner_split, os_combo)
        if len(terms) >= best_rank:
            continue
        if verify_fast(terms):
            merged = merge_terms_fast(terms)
            if verify_fast(merged) and len(merged) < best_rank:
                candidates.append(merged)
                best_rank = len(merged)

    if not candidates:
        # Fallback: basic schoolbook
        best = []
        for i in range(2):
            for j in range(4):
                for k in range(5):
                    best.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
    else:
        best = min(candidates, key=len)

    rank = len(best)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(best):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff

    return {"U": U, "V": V, "W": W}

def _gen_output_combos(inner_split, output_splits):
    """Generate all combinations of output splits for each inner block."""
    from itertools import product
    # For each inner block, determine valid output splits
    valid_per_block = []
    for inner_size in inner_split:
        if inner_size == 2:
            valid_per_block.append([os for os in output_splits if sum(os) == 5])
        elif inner_size == 1:
            valid_per_block.append([(5,)])
        elif inner_size == 3:
            valid_per_block.append([os for os in output_splits if sum(os) == 5])
        else:
            valid_per_block.append([(5,)])
    return list(product(*valid_per_block))
# EVOLVE-BLOCK-END
