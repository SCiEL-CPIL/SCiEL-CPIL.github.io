import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses an optimized local search to find the maximum number of disjoint 2x2x2 Strassen blocks.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    block_indices = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append((i1, i2, j1, j2, k1, k2))
                            block_indices.append(indices)

    num_blocks_total = len(all_blocks)
    
    # Precompute index-to-bit mapping
    all_indices = set()
    for indices in block_indices:
        all_indices |= indices
    index_to_bit = {idx: bit for bit, idx in enumerate(all_indices)}
    num_bits = len(all_indices)
    
    # Convert blocks to bitmasks
    block_bitmasks = []
    for indices in block_indices:
        mask = 0
        for idx in indices:
            mask |= (1 << index_to_bit[idx])
        block_bitmasks.append(mask)
    
    # Build compatibility lists: compatible[i] is a list of j such that i and j do not conflict
    # This is much faster than checking bitmasks repeatedly in local search
    compatible = [[] for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if not (block_bitmasks[i] & block_bitmasks[j]):
                compatible[i].append(j)
                compatible[j].append(i)
                
    # Degrees for heuristic ordering
    degrees = [len(compatible[i]) for i in range(num_blocks_total)]
    # Lower degree means fewer conflicts with others? No, compatible list length is "friends".
    # High "friends" count means it conflicts with FEW others.
    # So we prefer blocks with high compatible count (low conflict count).
    # Conflict count = num_blocks_total - 1 - len(compatible[i])
    
    def greedy_packing(order):
        used_mask = 0
        current = []
        for idx in order:
            if not (block_bitmasks[idx] & used_mask):
                current.append(idx)
                used_mask |= block_bitmasks[idx]
        return current

    def local_improve(packing):
        """
        Improves packing by trying to add blocks, and performing 1-for-2 and 2-for-3 swaps.
        """
        if not packing:
            return packing
        
        best_packing = packing[:]
        best_mask = 0
        for idx in best_packing:
            best_mask |= block_bitmasks[idx]
        best_size = len(best_packing)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            
            # 1. Try to add any compatible block
            for idx in range(num_blocks_total):
                if idx not in best_packing and not (block_bitmasks[idx] & best_mask):
                    best_packing.append(idx)
                    best_mask |= block_bitmasks[idx]
                    best_size += 1
                    improved = True
                    break # Restart loop as mask changed
            
            if improved:
                continue

            # 2. Try 1-for-2 swaps: Remove one block, add two
            # Iterate through blocks in current packing
            for pos in range(len(best_packing)):
                if time.time() > deadline - 5.0:
                    break
                removed = best_packing[pos]
                remaining = best_packing[:pos] + best_packing[pos+1:]
                rem_mask = best_mask & ~block_bitmasks[removed]
                
                # Find candidates compatible with rem_mask
                # We want two candidates c1, c2 such that:
                # 1. c1, c2 not in remaining
                # 2. c1 compatible with rem_mask
                # 3. c2 compatible with rem_mask | mask(c1)
                # 4. c1 compatible with c2
                
                # To speed up, iterate through compatible[removed]? No, candidates must be compatible with REMAINING.
                # So candidates are those compatible with all blocks in remaining.
                # This is the intersection of compatible[x] for x in remaining.
                # Since remaining is small (<= 13), we can compute this intersection.
                
                # Optimization: Just iterate through all blocks and check compatibility with rem_mask
                candidates = []
                for c in range(num_blocks_total):
                    if c in remaining:
                        continue
                    if not (block_bitmasks[c] & rem_mask):
                        candidates.append(c)
                
                # Now find pair in candidates
                found_pair = False
                for i in range(len(candidates)):
                    c1 = candidates[i]
                    mask_c1 = block_bitmasks[c1]
                    # Check if c1 conflicts with anything in remaining? Already checked via rem_mask.
                    # Now check c2
                    for j in range(i + 1, len(candidates)):
                        c2 = candidates[j]
                        if not (block_bitmasks[c2] & (rem_mask | mask_c1)):
                            # Found valid pair
                            new_packing = remaining + [c1, c2]
                            new_mask = rem_mask | mask_c1 | block_bitmasks[c2]
                            best_packing = new_packing
                            best_mask = new_mask
                            best_size = len(new_packing)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if improved:
                    break

            # 3. Try 2-for-3 swaps (if size allows)
            if not improved and len(best_packing) >= 2:
                for pos1 in range(len(best_packing)):
                    if time.time() > deadline - 5.0:
                        break
                    for pos2 in range(pos1 + 1, len(best_packing)):
                        r1 = best_packing[pos1]
                        r2 = best_packing[pos2]
                        remaining = [x for i, x in enumerate(best_packing) if i != pos1 and i != pos2]
                        rem_mask = 0
                        for r in remaining:
                            rem_mask |= block_bitmasks[r]
                        
                        candidates = []
                        for c in range(num_blocks_total):
                            if c in remaining:
                                continue
                            if not (block_bitmasks[c] & rem_mask):
                                candidates.append(c)
                        
                        found_triple = False
                        # Find triple in candidates
                        for i in range(len(candidates)):
                            c1 = candidates[i]
                            mask_c1 = block_bitmasks[c1]
                            for j in range(i + 1, len(candidates)):
                                c2 = candidates[j]
                                if block_bitmasks[c2] & (rem_mask | mask_c1):
                                    continue
                                mask_c2 = block_bitmasks[c2]
                                for k in range(j + 1, len(candidates)):
                                    c3 = candidates[k]
                                    if block_bitmasks[c3] & (rem_mask | mask_c1 | mask_c2):
                                        continue
                                    # Found triple
                                    new_packing = remaining + [c1, c2, c3]
                                    new_mask = rem_mask | mask_c1 | mask_c2 | block_bitmasks[c3]
                                    best_packing = new_packing
                                    best_mask = new_mask
                                    best_size = len(new_packing)
                                    improved = True
                                    found_triple = True
                                    break
                                if found_triple:
                                    break
                            if found_triple:
                                break
                        if improved:
                            break
                    if improved:
                        break

        return best_packing

    best_packing = []
    best_size = 0
    
    # Initial ordering: Prefer blocks with high compatibility (low conflict)
    base_order = sorted(range(num_blocks_total), key=lambda x: -degrees[x])
    
    # Run multiple trials with random perturbations
    num_trials = 2000
    trial = 0
    while trial < num_trials:
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the order
        order = base_order[:]
        # Random swaps
        num_swaps = random.randint(0, 20)
        for _ in range(num_swaps):
            i = random.randint(0, len(order) - 1)
            j = random.randint(0, len(order) - 1)
            order[i], order[j] = order[j], order[i]
        
        packing = greedy_packing(order)
        
        if len(packing) > best_size:
            best_size = len(packing)
            best_packing = packing[:]
            
            # Local improvement
            if time.time() < deadline - 5.0:
                improved = local_improve(best_packing)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_packing = improved
            
            # Check if we reached a very high count (e.g., 10+)
            if best_size >= 10:
                # Try a few more local improvements
                for _ in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    improved = local_improve(best_packing)
                    if len(improved) > best_size:
                        best_size = len(improved)
                        best_packing = improved
                    else:
                        break
                
                if best_size >= 12:
                    break

        trial += 1
        
        # Periodically try a different strategy: Start with a random set of 2-3 blocks and extend
        if trial % 100 == 0 and time.time() < deadline - 10.0:
            # Pick 2 random non-conflicting blocks
            c1 = random.randint(0, num_blocks_total - 1)
            c2 = random.randint(0, num_blocks_total - 1)
            if block_bitmasks[c1] & block_bitmasks[c2]:
                continue
            start_packing = [c1, c2]
            start_mask = block_bitmasks[c1] | block_bitmasks[c2]
            
            # Greedy extend
            for idx in base_order:
                if idx not in start_packing and not (block_bitmasks[idx] & start_mask):
                    start_packing.append(idx)
                    start_mask |= block_bitmasks[idx]
            
            if len(start_packing) > best_size:
                best_size = len(start_packing)
                best_packing = start_packing[:]
                improved = local_improve(best_packing)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_packing = improved

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
    for idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[idx]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

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
