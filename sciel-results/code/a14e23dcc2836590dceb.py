import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a greedy search + local refinement to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    # A block is defined by two i's, two j's, two k's
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            # The set of indices covered by this block
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    def local_refine(packing, used_indices):
        """Try to improve packing by swap and insert moves."""
        packing = list(packing)
        used_indices = set(used_indices)
        improved = True
        while improved:
            improved = False
            # Try swap: replace one block with another that frees up more space
            for idx, block in enumerate(packing):
                if time.time() > deadline - 5.0:
                    return packing, used_indices
                # Remove this block
                # Find the block's indices
                block_indices = None
                for bp, bi in all_blocks:
                    if bp == block:
                        block_indices = bi
                        break
                if block_indices is None:
                    continue
                new_used = used_indices - block_indices
                # Try to add a different block
                for other_block, other_indices in all_blocks:
                    if other_block in packing:
                        continue
                    if not (other_indices & new_used):
                        # Try replacing
                        new_packing = packing[:idx] + packing[idx+1:] + [other_block]
                        new_used2 = new_used | other_indices
                        # Check if we can add more blocks
                        for bp3, bi3 in all_blocks:
                            if bp3 in new_packing:
                                continue
                            if not (bi3 & new_used2):
                                if len(new_packing) + 1 > len(packing):
                                    packing = new_packing + [bp3]
                                    used_indices = new_used2 | bi3
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
            if not improved:
                # Try insert: add a block that fits
                for bp, bi in all_blocks:
                    if bp in packing:
                        continue
                    if not (bi & used_indices):
                        packing.append(bp)
                        used_indices |= bi
                        improved = True
                        break
        return packing, used_indices

    # Greedy search for maximum disjoint packing with local refinement
    best_num_blocks = 0
    best_packing = []
    
    num_trials = 200
    for t in range(num_trials):
        if time.time() > deadline - 5.0:
            break
            
        random.shuffle(all_blocks)
        used_indices = set()
        current_packing = []
        
        for block_params, block_indices in all_blocks:
            # Check if disjoint
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
        
        # Apply local refinement
        current_packing, used_indices = local_refine(current_packing, used_indices)
        
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing
            
            # Early exit if we hit the theoretical max (13 blocks)
            if best_num_blocks == 13:
                break

    # Construct U, V, W based on the best packing found
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add the Strassen blocks from the best packing
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Local refinement: try to find pairs of terms that can be merged
    # by checking if their combined tensor has rank 1
    if time.time() < deadline - 10.0:
        def can_merge_two_terms(term1, term2):
            """Check if two terms can be replaced by a single term.
            term = (U_col, V_col, W_col)
            Two terms (u1,v1,w1) and (u2,v2,w2) can be merged into one term
            (u,v,w) if u1⊗v1⊗w1 + u2⊗v2⊗w2 = u⊗v⊗w.
            This is equivalent to checking if the sum tensor has rank 1.
            """
            u1, v1, w1 = term1
            u2, v2, w2 = term2
            
            # The sum tensor T[i,j,k] = u1[i]*v1[j]*w1[k] + u2[i]*v2[j]*w2[k]
            # T has rank 1 iff it can be written as u[i]*v[j]*w[k]
            
            # Check using matricization: T_mat[i, (j,k)] should have rank 1
            # This means all rows of T_mat are scalar multiples of each other
            
            # Collect non-zero entries of T
            entries = {}
            for i in range(U_rows):
                if u1[i] == 0 and u2[i] == 0:
                    continue
                for j in range(V_rows):
                    if v1[j] == 0 and v2[j] == 0:
                        continue
                    for k in range(W_rows):
                        if w1[k] == 0 and w2[k] == 0:
                            continue
                        val = u1[i]*v1[j]*w1[k] + u2[i]*v2[j]*w2[k]
                        if val != 0:
                            entries[(i, j, k)] = val
            
            if not entries:
                return True, ([0]*U_rows, [0]*V_rows, [0]*W_rows)
            
            # Check if T is rank 1
            # A tensor is rank 1 iff for any two pairs of indices (i1,j1,k1) and (i2,j2,k2),
            # T[i1,j1,k1]*T[i2,j2,k2] == T[i1,j2,k1]*T[i2,j1,k2] (when all are defined)
            # More efficiently: check that the matricization has rank 1
            
            # Get all non-zero i indices
            i_indices = sorted(set(idx[0] for idx in entries))
            j_indices = sorted(set(idx[1] for idx in entries))
            k_indices = sorted(set(idx[2] for idx in entries))
            
            # If any dimension has only one non-zero index, check the remaining 2D structure
            if len(i_indices) == 1:
                i0 = i_indices[0]
                # Check if the 2D slice T[i0, j, k] is rank 1
                # This means all rows are scalar multiples
                if len(j_indices) > 1 and len(k_indices) > 1:
                    # Pick reference row
                    j_ref = j_indices[0]
                    row_ref = {k: entries.get((i0, j_ref, k), 0) for k in k_indices}
                    for j in j_indices[1:]:
                        row = {k: entries.get((i0, j, k), 0) for k in k_indices}
                        # Check if row is scalar multiple of row_ref
                        scale = None
                        for k in k_indices:
                            if row_ref[k] != 0:
                                if scale is None:
                                    scale = row[k] / row_ref[k]
                                elif abs(row[k] - scale * row_ref[k]) > 1e-9:
                                    return False, None
                            else:
                                if abs(row[k]) > 1e-9:
                                    return False, None
                        if scale is not None and scale == 0:
                            # All entries in this row should be 0
                            for k in k_indices:
                                if abs(row[k]) > 1e-9:
                                    return False, None
                # Construct the merged term
                u = [0]*U_rows
                v = [0]*V_rows
                w = [0]*W_rows
                u[i0] = 1
                for j in j_indices:
                    for k in k_indices:
                        if (i0, j, k) in entries:
                            if v[j] == 0:
                                v[j] = 1
                                w[k] = entries[(i0, j, k)]
                            else:
                                if abs(w[k] * entries[(i0, j, k)] - v[j] * entries[(i0, j, k)]) > 1e-9:
                                    return False, None
                return True, (u, v, w)
            elif len(j_indices) == 1:
                j0 = j_indices[0]
                if len(i_indices) > 1 and len(k_indices) > 1:
                    i_ref = i_indices[0]
                    col_ref = {k: entries.get((i_ref, j0, k), 0) for k in k_indices}
                    for i in i_indices[1:]:
                        col = {k: entries.get((i, j0, k), 0) for k in k_indices}
                        scale = None
                        for k in k_indices:
                            if col_ref[k] != 0:
                                if scale is None:
                                    scale = col[k] / col_ref[k]
                                elif abs(col[k] - scale * col_ref[k]) > 1e-9:
                                    return False, None
                            else:
                                if abs(col[k]) > 1e-9:
                                    return False, None
                u = [0]*U_rows
                v = [0]*V_rows
                w = [0]*W_rows
                v[j0] = 1
                for i in i_indices:
                    for k in k_indices:
                        if (i, j0, k) in entries:
                            if u[i] == 0:
                                u[i] = 1
                                w[k] = entries[(i, j0, k)]
                            else:
                                if abs(w[k] * entries[(i, j0, k)] - u[i] * entries[(i, j0, k)]) > 1e-9:
                                    return False, None
                return True, (u, v, w)
            elif len(k_indices) == 1:
                k0 = k_indices[0]
                if len(i_indices) > 1 and len(j_indices) > 1:
                    i_ref = i_indices[0]
                    row_ref = {j: entries.get((i_ref, j, k0), 0) for j in j_indices}
                    for i in i_indices[1:]:
                        row = {j: entries.get((i, j, k0), 0) for j in j_indices}
                        scale = None
                        for j in j_indices:
                            if row_ref[j] != 0:
                                if scale is None:
                                    scale = row[j] / row_ref[j]
                                elif abs(row[j] - scale * row_ref[j]) > 1e-9:
                                    return False, None
                            else:
                                if abs(row[j]) > 1e-9:
                                    return False, None
                u = [0]*U_rows
                v = [0]*V_rows
                w = [0]*W_rows
                w[k0] = 1
                for i in i_indices:
                    for j in j_indices:
                        if (i, j, k0) in entries:
                            if u[i] == 0:
                                u[i] = 1
                                v[j] = entries[(i, j, k0)]
                            else:
                                if abs(v[j] * entries[(i, j, k0)] - u[i] * entries[(i, j, k0)]) > 1e-9:
                                    return False, None
                return True, (u, v, w)
            
            # General case: check rank-1 condition
            # For rank 1: T[i,j,k] = a_i * b_j * c_k
            # Check: T[i1,j1,k1]*T[i2,j2,k2] == T[i1,j2,k1]*T[i2,j1,k2]
            # Sample a few combinations
            import random as _random
            _random.seed(42)
            for _ in range(20):
                i1, i2 = _random.sample(i_indices, 2)
                j1, j2 = _random.sample(j_indices, 2)
                k1, k2 = _random.sample(k_indices, 2)
                t11 = entries.get((i1, j1, k1), 0)
                t22 = entries.get((i2, j2, k2), 0)
                t12 = entries.get((i1, j2, k1), 0)
                t21 = entries.get((i2, j1, k2), 0)
                # For rank 1: t11*t22 == t12*t21
                if abs(t11*t22 - t12*t21) > 1e-9:
                    return False, None
            
            # Try to construct the merged term
            u = [0]*U_rows
            v = [0]*V_rows
            w = [0]*W_rows
            
            # Find a non-zero entry to anchor
            ref_key = None
            for key in entries:
                if entries[key] != 0:
                    ref_key = key
                    break
            if ref_key is None:
                return True, (u, v, w)
            
            ri, rj, rk = ref_key
            ref_val = entries[ref_key]
            
            # Set u[ri] = 1, v[rj] = 1, w[rk] = ref_val
            u[ri] = 1
            v[rj] = 1
            w[rk] = ref_val
            
            # Determine u[i] for other i
            for i in i_indices:
                if i == ri:
                    continue
                # Find a j,k where T[i,j,k] != 0 and v[j], w[k] are determined
                found = False
                for j in j_indices:
                    for k in k_indices:
                        if (i, j, k) in entries and v[j] != 0 and w[k] != 0:
                            u[i] = entries[(i, j, k)] / (v[j] * w[k])
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # Try to determine from existing u
                    for j in j_indices:
                        for k in k_indices:
                            if (i, j, k) in entries and u[i] != 0:
                                expected = u[i] * v[j] * w[k]
                                if abs(expected - entries[(i, j, k)]) > 1e-9:
                                    return False, None
                                break
            
            # Determine v[j] for other j
            for j in j_indices:
                if j == rj:
                    continue
                found = False
                for i in i_indices:
                    for k in k_indices:
                        if (i, j, k) in entries and u[i] != 0 and w[k] != 0:
                            v[j] = entries[(i, j, k)] / (u[i] * w[k])
                            found = True
                            break
                    if found:
                        break
                if not found:
                    for i in i_indices:
                        for k in k_indices:
                            if (i, j, k) in entries and v[j] != 0:
                                expected = u[i] * v[j] * w[k]
                                if abs(expected - entries[(i, j, k)]) > 1e-9:
                                    return False, None
                                break
            
            # Determine w[k] for other k
            for k in k_indices:
                if k == rk:
                    continue
                found = False
                for i in i_indices:
                    for j in j_indices:
                        if (i, j, k) in entries and u[i] != 0 and v[j] != 0:
                            w[k] = entries[(i, j, k)] / (u[i] * v[j])
                            found = True
                            break
                    if found:
                        break
                if not found:
                    for i in i_indices:
                        for j in j_indices:
                            if (i, j, k) in entries and w[k] != 0:
                                expected = u[i] * v[j] * w[k]
                                if abs(expected - entries[(i, j, k)]) > 1e-9:
                                    return False, None
                                break
            
            # Verify the construction
            for (i, j, k), val in entries.items():
                expected = u[i] * v[j] * w[k]
                if abs(expected - val) > 1e-9:
                    return False, None
            
            return True, (u, v, w)
        
        # Try to find mergeable pairs
        merged = [False] * rank
        new_U_cols = []
        new_V_cols = []
        new_W_cols = []
        
        for i in range(rank):
            if merged[i]:
                continue
            merged_i = False
            for j in range(i+1, rank):
                if merged[j]:
                    continue
                # Check if terms i and j can be merged
                term_i = (U_cols[i], V_cols[i], W_cols[i])
                term_j = (U_cols[j], V_cols[j], W_cols[j])
                can_merge, merged_term = can_merge_two_terms(term_i, term_j)
                if can_merge:
                    u, v, w = merged_term
                    new_U_cols.append(u)
                    new_V_cols.append(v)
                    new_W_cols.append(w)
                    merged[i] = True
                    merged[j] = True
                    merged_i = True
                    break
            if not merged_i:
                new_U_cols.append(U_cols[i])
                new_V_cols.append(V_cols[i])
                new_W_cols.append(W_cols[i])
        
        if len(new_U_cols) < rank:
            U_cols = new_U_cols
            V_cols = new_V_cols
            W_cols = new_W_cols
            rank = len(U_cols)

    # Convert to matrix format
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
