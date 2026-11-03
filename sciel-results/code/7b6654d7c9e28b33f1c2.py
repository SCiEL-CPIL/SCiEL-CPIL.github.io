import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a greedy search to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    
    Key improvements:
    1. Precompute all blocks once with their index sets
    2. Use bitmasks for fast disjointness checking
    3. Use a better greedy heuristic: prioritize blocks that cover "rare" indices
    4. Multiple random restarts with different random orderings
    5. Try both pure random shuffle and priority-based ordering
    6. Improved local search with 2-block swaps and better termination
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Total number of (i,j,k) triples
    total_triples = n * m * p  # 105

    # Precompute all possible 2x2x2 blocks with bitmask representation
    all_blocks = []  # list of (params, bitmask, index_set)
    
    # Index of each (i,j,k) triple to a unique bit position
    def triple_to_idx(i, j, k):
        return i * m * p + j * p + k
    
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            bitmask = 0
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = triple_to_idx(i, j, k)
                                        bitmask |= (1 << idx)
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), bitmask, indices))

    # Precompute a bitmask for all triples
    full_mask = (1 << total_triples) - 1

    # Count how many blocks cover each triple (for rarity heuristic)
    triple_block_count = [0] * total_triples
    for _, _, indices in all_blocks:
        for i, j, k in indices:
            triple_block_count[triple_to_idx(i, j, k)] += 1

    best_num_blocks = 0
    best_packing = []

    def greedy_random(block_list, used_mask, num_blocks_current, packing_current):
        """Greedy selection with random ordering."""
        shuffled = list(block_list)
        random.shuffle(shuffled)
        for params, bm, indices in shuffled:
            if (bm & used_mask) == 0:
                packing_current.append(params)
                used_mask |= bm
                num_blocks_current += 1
        return num_blocks_current, packing_current, used_mask

    def greedy_priority(block_list, used_mask, num_blocks_current, packing_current):
        """Greedy selection prioritizing blocks that cover rare indices."""
        shuffled = list(block_list)
        random.shuffle(shuffled)
        
        for params, bm, indices in shuffled:
            uncovered_indices = [i for i in indices if (triple_to_idx(i[0], i[1], i[2]) & used_mask) == 0]
            priority = sum(1.0 / triple_block_count[triple_to_idx(i[0], i[1], i[2])] for i in uncovered_indices)
            
            if (bm & used_mask) == 0 and priority > 0:
                packing_current.append(params)
                used_mask |= bm
                num_blocks_current += 1
        
        return num_blocks_current, packing_current, used_mask

    def local_search(packing, max_iterations=50):
        """Try to improve packing by swapping blocks."""
        if not packing:
            return packing
        
        current_packing = list(packing)
        used_mask = 0
        for params in current_packing:
            i1, i2, j1, j2, k1, k2 = params
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        idx = triple_to_idx(i, j, k)
                        used_mask |= (1 << idx)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            # Try removing one block and adding a different one
            for remove_idx in range(len(current_packing)):
                # Compute mask without this block
                mask_without = 0
                for idx2, params2 in enumerate(current_packing):
                    if idx2 == remove_idx:
                        continue
                    i1, i2, j1, j2, k1, k2 = params2
                    for i in (i1, i2):
                        for j in (j1, j2):
                            for k in (k1, k2):
                                tidx = triple_to_idx(i, j, k)
                                mask_without |= (1 << tidx)
                
                # Try adding a block that fits
                for params_add, bm_add, _ in all_blocks:
                    if params_add in current_packing:
                        continue
                    if (bm_add & mask_without) == 0:
                        new_packing = current_packing[:remove_idx] + current_packing[remove_idx+1:] + [params_add]
                        if len(new_packing) > len(current_packing):
                            current_packing = new_packing
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try 2-block swap: remove two blocks, add two different blocks
                for r1 in range(len(current_packing)):
                    if time.time() > deadline - 5.0:
                        break
                    for r2 in range(r1 + 1, len(current_packing)):
                        mask_without = 0
                        for idx2, params2 in enumerate(current_packing):
                            if idx2 == r1 or idx2 == r2:
                                continue
                            i1, i2, j1, j2, k1, k2 = params2
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        tidx = triple_to_idx(i, j, k)
                                        mask_without |= (1 << tidx)
                        
                        # Try adding two blocks that fit
                        added = []
                        for params_add, bm_add, _ in all_blocks:
                            if params_add in current_packing:
                                continue
                            if (bm_add & mask_without) == 0:
                                if len(added) < 2:
                                    added.append(params_add)
                                    mask_without |= bm_add
                                else:
                                    break
                        
                        if len(added) == 2:
                            new_packing = []
                            for idx2, params2 in enumerate(current_packing):
                                if idx2 != r1 and idx2 != r2:
                                    new_packing.append(params2)
                            new_packing.extend(added)
                            if len(new_packing) > len(current_packing):
                                current_packing = new_packing
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current_packing

    num_trials = 5000
    trial_count = 0
    for t in range(num_trials):
        if time.time() > deadline - 5.0:
            break
        trial_count += 1
        
        if t % 4 == 0:
            # Random greedy
            num_blocks, packing, _ = greedy_random(all_blocks, 0, 0, [])
        elif t % 4 == 1:
            # Priority-based greedy
            num_blocks, packing, _ = greedy_priority(all_blocks, 0, 0, [])
        elif t % 4 == 2:
            # Random greedy + local search
            num_blocks, packing, _ = greedy_random(all_blocks, 0, 0, [])
            packing = local_search(packing, max_iterations=30)
            num_blocks = len(packing)
        else:
            # Priority-based greedy + local search
            num_blocks, packing, _ = greedy_priority(all_blocks, 0, 0, [])
            packing = local_search(packing, max_iterations=30)
            num_blocks = len(packing)
        
        if num_blocks > best_num_blocks:
            best_num_blocks = num_blocks
            best_packing = packing
            
            # Early exit if we hit a good number
            if best_num_blocks >= 12:
                break

    # Also try a more systematic approach: try to find maximum packing
    # by considering blocks in order of increasing overlap with others
    if time.time() < deadline - 10.0:
        # Compute pairwise overlap
        num_blocks_list = len(all_blocks)
        
        # Sort blocks by number of conflicts (fewer conflicts first)
        block_conflicts = []
        for idx, (params, bm, indices) in enumerate(all_blocks):
            conflicts = 0
            for other_idx in range(idx + 1, num_blocks_list):
                if (bm & all_blocks[other_idx][1]) != 0:
                    conflicts += 1
            block_conflicts.append((conflicts, idx))
        
        block_conflicts.sort()
        sorted_block_indices = [idx for _, idx in block_conflicts]
        
        # Greedy on sorted order
        used_mask = 0
        packing = []
        for idx in sorted_block_indices:
            params, bm, indices = all_blocks[idx]
            if (bm & used_mask) == 0:
                packing.append(params)
                used_mask |= bm
        
        # Apply local search to this systematic packing
        packing = local_search(packing, max_iterations=50)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing

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
