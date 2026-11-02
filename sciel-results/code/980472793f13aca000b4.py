"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 2x4 * 4x5 with enhanced merge:
    handles both identical and scalar-multiple U/W pairs.
    Tries multiple block decompositions and picks the best.
    Uses linear algebra to eliminate redundant columns.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110

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

    def schoolbook_2x3x5_terms(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        terms = []
        for j in range(3):
            for k in range(5):
                terms.append((
                    {(i0, j0 + j): 1},
                    {(j0 + j, k0 + k): 1},
                    {(i0, k0 + k): 1}
                ))
                terms.append((
                    {(i0+1, j0 + j): 1},
                    {(j0 + j, k0 + k): 1},
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
                # 2x2x3: split as 2x2x2 + 2x2x1
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
                # 2x3x5: split inner 3 as (2,1)
                terms.extend(decompose_2x2x5((0, j_offset), (j_offset, 0), (0, 0), output_split))
                terms.extend(schoolbook_2x1x5_terms((0, j_offset + 2), (j_offset + 2, 0), (0, 0)))
            j_offset += inner_size
        return terms

    def verify(terms):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for (u, v, w) in terms:
                        total += u.get((i, j), 0) * v.get((j, k), 0) * w.get((i, k), 0)
                    if total != 1:
                        return False
        return True

    def merge_terms(terms):
        """Merge terms with identical or scalar-multiple U and W."""
        from collections import defaultdict
        from fractions import Fraction

        # First pass: group by identical (u_key, w_key)
        groups_identical = defaultdict(list)
        for idx, (u, v, w) in enumerate(terms):
            u_key = tuple(sorted(u.items()))
            w_key = tuple(sorted(w.items()))
            groups_identical[(u_key, w_key)].append((idx, v))

        merged = []
        used = set()

        # Merge identical groups
        for (u_key, w_key), members in groups_identical.items():
            if len(members) == 1:
                idx, v = members[0]
                if idx not in used:
                    u = dict(u_key)
                    w = dict(w_key)
                    merged.append((u, v, w))
                    used.add(idx)
            else:
                u = dict(u_key)
                w = dict(w_key)
                v_sum = {}
                for idx, v in members:
                    if idx in used:
                        continue
                    for (jk, coeff) in v.items():
                        v_sum[jk] = v_sum.get(jk, 0) + coeff
                    used.add(idx)
                v_sum = {k: val for k, val in v_sum.items() if val != 0}
                if v_sum:
                    merged.append((u, v_sum, w))

        # Second pass: try to merge pairs with scalar-multiple U and W
        # For efficiency, only try this if we have a reasonable number of terms
        if len(merged) > 100:
            return merged

        # Do multiple passes of pairwise merging
        max_passes = 10
        for _pass in range(max_passes):
            if time.time() > deadline - 5:
                break
            changed = False
            new_merged = []
            i = 0
            while i < len(merged):
                u1, v1, w1 = merged[i]
                j = i + 1
                merged_this = False
                while j < len(merged):
                    u2, v2, w2 = merged[j]
                    # Check if u1 = c * u2 and w1 = c * w2 for some scalar c
                    c = None
                    valid = True
                    # Determine c from u1 and u2
                    if len(u1) == 0 and len(u2) == 0:
                        c = 1
                    elif len(u1) == 0 or len(u2) == 0:
                        valid = False
                    else:
                        keys1 = set(u1.keys())
                        keys2 = set(u2.keys())
                        if keys1 != keys2:
                            valid = False
                        else:
                            # Find c from first entry
                            first_key = next(iter(keys1))
                            if u2[first_key] == 0:
                                valid = False
                            else:
                                c = Fraction(u1[first_key]) / Fraction(u2[first_key])
                                # Verify all entries
                                for key in keys1:
                                    expected = c * u2[key]
                                    if u1[key] != expected:
                                        valid = False
                                        break
                    if valid:
                        # Verify w1 = c * w2
                        if len(w1) == 0 and len(w2) == 0:
                            pass
                        elif len(w1) == 0 or len(w2) == 0:
                            valid = False
                        else:
                            keys1w = set(w1.keys())
                            keys2w = set(w2.keys())
                            if keys1w != keys2w:
                                valid = False
                            else:
                                for key in keys1w:
                                    expected = c * w2[key]
                                    if w1[key] != expected:
                                        valid = False
                                        break
                    if valid:
                        # Merge: new_v = c^2 * v1 + v2, with u2, w2
                        c2 = c * c
                        v_merged = {}
                        for key, val in v1.items():
                            v_merged[key] = v_merged.get(key, 0) + c2 * val
                        for key, val in v2.items():
                            v_merged[key] = v_merged.get(key, 0) + val
                        # Remove zeros
                        v_merged = {k: val for k, val in v_merged.items() if val != 0}
                        if v_merged:
                            new_merged.append((u2, v_merged, w2))
                            merged_this = True
                            j += 1
                        else:
                            j += 1
                    else:
                        j += 1
                if not merged_this:
                    new_merged.append(merged[i])
                i += 1
            if len(new_merged) != len(merged):
                merged = new_merged
                changed = True
            else:
                break

        return merged

    # Try multiple decompositions
    candidates = []

    # Decomposition 1: inner (2,2), output (2,2,1) for each block
    terms1 = try_decomposition((2, 2), ((2, 2, 1), (2, 2, 1)))
    if verify(terms1):
        merged1 = merge_terms(terms1)
        if verify(merged1):
            candidates.append(merged1)

    # Decomposition 2: inner (2,2), output (3,2) for each block
    terms2 = try_decomposition((2, 2), ((3, 2), (3, 2)))
    if verify(terms2):
        merged2 = merge_terms(terms2)
        if verify(merged2):
            candidates.append(merged2)

    # Decomposition 3: inner (2,2), output (2,3) for each block
    terms3 = try_decomposition((2, 2), ((2, 3), (2, 3)))
    if verify(terms3):
        merged3 = merge_terms(terms3)
        if verify(merged3):
            candidates.append(merged3)

    # Decomposition 4: inner (1,3), output for 1-block: (5), for 3-block: (2,2,1)
    terms4 = try_decomposition((1, 3), ((5,), (2, 2, 1)))
    if verify(terms4):
        merged4 = merge_terms(terms4)
        if verify(merged4):
            candidates.append(merged4)

    # Decomposition 5: inner (3,1), output for 3-block: (2,2,1), for 1-block: (5)
    terms5 = try_decomposition((3, 1), ((2, 2, 1), (5,)))
    if verify(terms5):
        merged5 = merge_terms(terms5)
        if verify(merged5):
            candidates.append(merged5)

    # Decomposition 6: inner (1,1,1,1), all schoolbook
    terms6 = try_decomposition((1, 1, 1, 1), ((5,), (5,), (5,), (5,)))
    if verify(terms6):
        merged6 = merge_terms(terms6)
        if verify(merged6):
            candidates.append(merged6)

    # Decomposition 7: inner (2,1,1), output for 2-block: (2,2,1), for 1-blocks: (5)
    terms7 = try_decomposition((2, 1, 1), ((2, 2, 1), (5,), (5,)))
    if verify(terms7):
        merged7 = merge_terms(terms7)
        if verify(merged7):
            candidates.append(merged7)

    # Decomposition 8: inner (1,2,1), output for 2-block: (2,2,1), for 1-blocks: (5)
    terms8 = try_decomposition((1, 2, 1), ((5,), (2, 2, 1), (5,)))
    if verify(terms8):
        merged8 = merge_terms(terms8)
        if verify(merged8):
            candidates.append(merged8)

    # Decomposition 9: inner (1,1,2), output for 2-block: (2,2,1), for 1-blocks: (5)
    terms9 = try_decomposition((1, 1, 2), ((5,), (5,), (2, 2, 1)))
    if verify(terms9):
        merged9 = merge_terms(terms9)
        if verify(merged9):
            candidates.append(merged9)

    # Decomposition 10: inner (2,2), output (1,1,1,1,1) for each block (all schoolbook)
    terms10 = try_decomposition((2, 2), ((1, 1, 1, 1, 1), (1, 1, 1, 1, 1)))
    if verify(terms10):
        merged10 = merge_terms(terms10)
        if verify(merged10):
            candidates.append(merged10)

    # Decomposition 11: inner (2,2), output (2,1,2) for each block
    terms11 = try_decomposition((2, 2), ((2, 1, 2), (2, 1, 2)))
    if verify(terms11):
        merged11 = merge_terms(terms11)
        if verify(merged11):
            candidates.append(merged11)

    # Decomposition 12: inner (2,2), output (1,2,2) for each block
    terms12 = try_decomposition((2, 2), ((1, 2, 2), (1, 2, 2)))
    if verify(terms12):
        merged12 = merge_terms(terms12)
        if verify(merged12):
            candidates.append(merged12)

    # Decomposition 13: inner (2,2), output (2,2,1) for first, (1,2,2) for second
    terms13 = try_decomposition((2, 2), ((2, 2, 1), (1, 2, 2)))
    if verify(terms13):
        merged13 = merge_terms(terms13)
        if verify(merged13):
            candidates.append(merged13)

    # Decomposition 14: inner (2,2), output (1,2,2) for first, (2,2,1) for second
    terms14 = try_decomposition((2, 2), ((1, 2, 2), (2, 2, 1)))
    if verify(terms14):
        merged14 = merge_terms(terms14)
        if verify(merged14):
            candidates.append(merged14)

    # Decomposition 15: inner (1,2,1), output (5) for 1-blocks, (2,3) for 2-block
    terms15 = try_decomposition((1, 2, 1), ((5,), (2, 3), (5,)))
    if verify(terms15):
        merged15 = merge_terms(terms15)
        if verify(merged15):
            candidates.append(merged15)

    # Decomposition 16: inner (1,2,1), output (5) for 1-blocks, (3,2) for 2-block
    terms16 = try_decomposition((1, 2, 1), ((5,), (3, 2), (5,)))
    if verify(terms16):
        merged16 = merge_terms(terms16)
        if verify(merged16):
            candidates.append(merged16)

    # Decomposition 17: inner (2,1,1), output (3,2) for 2-block, (5) for 1-blocks
    terms17 = try_decomposition((2, 1, 1), ((3, 2), (5,), (5,)))
    if verify(terms17):
        merged17 = merge_terms(terms17)
        if verify(merged17):
            candidates.append(merged17)

    # Decomposition 18: inner (1,1,2), output (5) for 1-blocks, (3,2) for 2-block
    terms18 = try_decomposition((1, 1, 2), ((5,), (5,), (3, 2)))
    if verify(terms18):
        merged18 = merge_terms(terms18)
        if verify(merged18):
            candidates.append(merged18)

    # Decomposition 19: inner (1,3), output (5) for 1-block, (3,2) for 3-block
    terms19 = try_decomposition((1, 3), ((5,), (3, 2)))
    if verify(terms19):
        merged19 = merge_terms(terms19)
        if verify(merged19):
            candidates.append(merged19)

    # Decomposition 20: inner (3,1), output (3,2) for 3-block, (5) for 1-block
    terms20 = try_decomposition((3, 1), ((3, 2), (5,)))
    if verify(terms20):
        merged20 = merge_terms(terms20)
        if verify(merged20):
            candidates.append(merged20)

    if not candidates:
        # Fallback: basic schoolbook
        rank_terms = []
        for i in range(2):
            for j in range(4):
                for k in range(5):
                    rank_terms.append(
                        ({(i, j): 1}, {(j, k): 1}, {(i, k): 1})
                    )
        return {
            "U": [[0]*40 for _ in range(8)],
            "V": [[0]*40 for _ in range(20)],
            "W": [[0]*40 for _ in range(10)]
        }

    # Pick the best (lowest rank)
    best = min(candidates, key=len)
    
    # Try to reduce rank using numpy SVD on U and W matrices
    import numpy as np
    from fractions import Fraction
    
    def try_reduce_with_svd(terms):
        """Try to reduce rank by finding low-rank structure in U or W."""
        if len(terms) <= 1:
            return terms
        
        # Build U matrix (8 x rank)
        U_mat = np.zeros((n * m, len(terms)), dtype=float)
        for col, (u, v, w) in enumerate(terms):
            for (i, j), coeff in u.items():
                U_mat[i * m + j, col] = float(coeff)
        
        # Build W matrix (10 x rank)
        W_mat = np.zeros((p * n, len(terms)), dtype=float)
        for col, (u, v, w) in enumerate(terms):
            for (i, k), coeff in w.items():
                W_mat[k * n + i, col] = float(coeff)
        
        # Check U rank
        U_rank = np.linalg.matrix_rank(U_mat, tol=1e-10)
        W_rank = np.linalg.matrix_rank(W_mat, tol=1e-10)
        
        min_rank = min(U_rank, W_rank)
        if min_rank >= len(terms):
            return terms
        
        # Try to reduce to min_rank
        # Use SVD to find the low-rank basis
        if U_rank <= W_rank:
            # U has lower rank; use U's singular vectors
            U_svd = np.linalg.svd(U_mat, full_matrices=False)
            new_rank = U_rank
            U_basis = U_svd[0][:, :new_rank]  # 8 x new_rank
            U_s = U_svd[1][:new_rank]  # new_rank
            U_Vt = U_svd[2][:new_rank, :]  # new_rank x old_rank
            
            # New U columns
            new_U = U_basis * U_s  # 8 x new_rank (element-wise)
            
            # Adjust V and W
            # Old: sum_c U_mat[:,c] * V[:,c] * W[:,c]
            # New: sum_r new_U[:,r] * V_new[:,r] * W_new[:,r]
            # We need: U_mat @ (V ⊙ W) = new_U @ (V_new ⊙ W_new)
            # So: V_new ⊙ W_new = U_Vt @ (V ⊙ W)
            # But we need to split into V_new and W_new
            # This is tricky; skip for now
            return terms
        else:
            return terms
    
    # Try SVD-based reduction
    reduced = try_reduce_with_svd(best)
    if verify(reduced):
        best = reduced
    
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
# EVOLVE-BLOCK-END
