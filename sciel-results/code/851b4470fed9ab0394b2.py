import time
import random
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses greedy search with local improvement to find maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 105 - num_blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    num_blocks_total = len(all_blocks)
    
    # Precompute index-to-bit mapping for faster conflict checking
    all_indices = set()
    for _, indices in all_blocks:
        all_indices |= indices
    index_to_bit = {idx: bit for bit, idx in enumerate(all_indices)}
    num_bits = len(all_indices)
    
    # Convert blocks to bitmasks
    block_bitmasks = []
    for _, indices in all_blocks:
        mask = 0
        for idx in indices:
            mask |= (1 << index_to_bit[idx])
        block_bitmasks.append(mask)
    
    # Build conflict graph using bitmasks - optimized with early termination
    conflicts = [set() for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if block_bitmasks[i] & block_bitmasks[j]:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    degrees = [len(conflicts[i]) for i in range(num_blocks_total)]
    
    # Precompute non-conflict sets for each block
    non_conflicts = [set(range(num_blocks_total)) - conflicts[i] - {i} for i in range(num_blocks_total)]
    
    # Precompute "future potential" for each block: how many other blocks it conflicts with
    # This helps in weighted greedy: prefer blocks that block fewer others
    future_potential = [num_blocks_total - 1 - degrees[i] for i in range(num_blocks_total)]
    
    # Precompute pairwise compatibility for faster checks
    compatible = [[False] * num_blocks_total for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if not (block_bitmasks[i] & block_bitmasks[j]):
                compatible[i][j] = True
                compatible[j][i] = True
    
    def greedy_packing(order):
        """Greedy packing given an order of blocks using bitmask."""
        used_mask = 0
        current = []
        for idx in order:
            if not (block_bitmasks[idx] & used_mask):
                current.append(idx)
                used_mask |= block_bitmasks[idx]
        return current
    
    def greedy_packing_with_lookahead(order, lookahead=2):
        """Greedy packing with lookahead: at each step, pick the block that allows the most future additions."""
        used_mask = 0
        current = []
        remaining = set(order)
        
        while remaining:
            best_idx = None
            best_score = -1
            for idx in remaining:
                if block_bitmasks[idx] & used_mask:
                    continue
                # Score: how many other blocks would remain compatible after adding this one
                new_mask = used_mask | block_bitmasks[idx]
                count = 0
                for other in remaining:
                    if other != idx and not (block_bitmasks[other] & new_mask):
                        count += 1
                if count > best_score:
                    best_score = count
                    best_idx = idx
            
            if best_idx is None:
                break
            
            current.append(best_idx)
            used_mask |= block_bitmasks[best_idx]
            remaining.remove(best_idx)
        
        return current
    
    def greedy_packing_weighted(used_mask, available):
        """Weighted greedy: at each step, pick the available block that blocks the fewest future options."""
        current = []
        current_mask = used_mask
        remaining = list(available)
        
        while remaining:
            best_idx = None
            best_score = -1
            best_conflict = num_blocks_total + 1
            for idx in remaining:
                if block_bitmasks[idx] & current_mask:
                    continue
                # Score: prefer blocks that conflict with fewer remaining blocks
                bm = block_bitmasks[idx]
                conflict_count = 0
                for other in remaining:
                    if other != idx and (bm & block_bitmasks[other]):
                        conflict_count += 1
                score = num_blocks_total - conflict_count
                if score > best_score or (score == best_score and conflict_count < best_conflict):
                    best_score = score
                    best_conflict = conflict_count
                    best_idx = idx
            
            if best_idx is None:
                break
            
            current.append(best_idx)
            current_mask |= block_bitmasks[best_idx]
            remaining = [x for x in remaining if x != best_idx]
        
        return current
    
    def get_used_mask(packing):
        """Get the union bitmask of all blocks in a packing."""
        mask = 0
        for idx in packing:
            mask |= block_bitmasks[idx]
        return mask
    
    def find_compatible_candidates(used_mask, exclude_set):
        """Find all blocks compatible with used_mask, excluding those in exclude_set."""
        candidates = []
        for idx in range(num_blocks_total):
            if idx in exclude_set:
                continue
            if not (block_bitmasks[idx] & used_mask):
                candidates.append(idx)
        return candidates
    
    def find_compatible_pair(candidates, base_mask):
        """Find a pair of compatible candidates from the list."""
        for i in range(len(candidates)):
            c1 = candidates[i]
            used_c1_mask = base_mask | block_bitmasks[c1]
            for j in range(i + 1, len(candidates)):
                c2 = candidates[j]
                if not (block_bitmasks[c2] & used_c1_mask):
                    return (c1, c2)
        return None
    
    def find_compatible_triple(candidates, base_mask):
        """Find a triple of compatible candidates from the list."""
        for i in range(len(candidates)):
            c1 = candidates[i]
            used_c1_mask = base_mask | block_bitmasks[c1]
            for j in range(i + 1, len(candidates)):
                c2 = candidates[j]
                if not (block_bitmasks[c2] & used_c1_mask):
                    used_c2_mask = used_c1_mask | block_bitmasks[c2]
                    for k in range(j + 1, len(candidates)):
                        c3 = candidates[k]
                        if not (block_bitmasks[c3] & used_c2_mask):
                            return (c1, c2, c3)
        return None
    
    def local_improve(packing):
        """Try to improve packing by swapping out blocks for better combinations using bitmasks."""
        if not packing:
            return packing
        
        best_packing = packing[:]
        best_mask = get_used_mask(best_packing)
        best_size = len(best_packing)
        
        improved = True
        max_iter = 50
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 2.0:
                break
            
            # Try removing 1 block and adding 2
            for pos in range(len(best_packing)):
                if time.time() > deadline - 2.0:
                    break
                remaining = best_packing[:pos] + best_packing[pos+1:]
                used_remaining_mask = get_used_mask(remaining)
                
                # Find blocks that could be added
                candidates = find_compatible_candidates(used_remaining_mask, set(remaining))
                
                # Try adding 2 compatible candidates
                best_pair = find_compatible_pair(candidates, used_remaining_mask)
                
                if best_pair:
                    new_packing = remaining + list(best_pair)
                    if len(new_packing) > best_size:
                        best_packing = new_packing
                        best_mask = get_used_mask(best_packing)
                        best_size = len(best_packing)
                        improved = True
                        break
            
            # Try removing 2 blocks and adding 3
            if not improved and len(best_packing) >= 2:
                for pos1 in range(len(best_packing)):
                    if time.time() > deadline - 2.0:
                        break
                    for pos2 in range(pos1 + 1, len(best_packing)):
                        remaining = [x for i, x in enumerate(best_packing) if i != pos1 and i != pos2]
                        used_remaining_mask = get_used_mask(remaining)
                        
                        # Find blocks that could be added
                        candidates = find_compatible_candidates(used_remaining_mask, set(remaining))
                        
                        # Try adding 3 compatible candidates
                        best_triple = find_compatible_triple(candidates, used_remaining_mask)
                        
                        if best_triple:
                            new_packing = remaining + list(best_triple)
                            if len(new_packing) > best_size:
                                best_packing = new_packing
                                best_mask = get_used_mask(best_packing)
                                best_size = len(best_packing)
                                improved = True
                                break
                    if improved:
                        break
            
            # Try removing 2 blocks and adding 2 different blocks (swap pair)
            if not improved and len(best_packing) >= 2:
                for pos1 in range(len(best_packing)):
                    if time.time() > deadline - 2.0:
                        break
                    for pos2 in range(pos1 + 1, len(best_packing)):
                        remaining = [x for i, x in enumerate(best_packing) if i != pos1 and i != pos2]
                        used_remaining_mask = get_used_mask(remaining)
                        removed1 = best_packing[pos1]
                        removed2 = best_packing[pos2]
                        
                        # Find blocks that could be added
                        candidates = find_compatible_candidates(used_remaining_mask, set(remaining))
                        
                        # Try adding 2 compatible candidates that are different from removed
                        best_pair = find_compatible_pair(candidates, used_remaining_mask)
                        
                        if best_pair:
                            c1, c2 = best_pair
                            if c1 not in (removed1, removed2) and c2 not in (removed1, removed2):
                                new_packing = remaining + [c1, c2]
                                new_mask = get_used_mask(new_packing)
                                # Accept if same size but different coverage
                                if len(new_packing) == best_size and new_mask != best_mask:
                                    best_packing = new_packing
                                    best_mask = new_mask
                                    improved = True
                                    break
                    if improved:
                        break
            
            # Try removing 1 block and adding 1 different block (swap)
            if not improved:
                for pos in range(len(best_packing)):
                    if time.time() > deadline - 2.0:
                        break
                    remaining = best_packing[:pos] + best_packing[pos+1:]
                    used_remaining_mask = get_used_mask(remaining)
                    removed_block = best_packing[pos]
                    
                    # Find a block that's compatible with remaining but wasn't in original
                    candidates = find_compatible_candidates(used_remaining_mask, set(remaining))
                    for cand in candidates:
                        if cand == removed_block:
                            continue
                        new_packing = remaining + [cand]
                        new_mask = get_used_mask(new_packing)
                        if len(new_packing) == best_size and new_mask != best_mask:
                            # Accept swap to explore different configurations
                            best_packing = new_packing
                            best_mask = new_mask
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing 3 blocks and adding 4
            if not improved and len(best_packing) >= 3:
                for pos1 in range(len(best_packing)):
                    if time.time() > deadline - 2.0:
                        break
                    for pos2 in range(pos1 + 1, len(best_packing)):
                        if time.time() > deadline - 2.0:
                            break
                        for pos3 in range(pos2 + 1, len(best_packing)):
                            if time.time() > deadline - 2.0:
                                break
                            remaining = [x for i, x in enumerate(best_packing) if i not in (pos1, pos2, pos3)]
                            used_remaining_mask = get_used_mask(remaining)
                            removed = {best_packing[pos1], best_packing[pos2], best_packing[pos3]}
                            
                            candidates = find_compatible_candidates(used_remaining_mask, set(remaining))
                            
                            # Try adding 4 compatible candidates
                            # First find a triple
                            best_triple = find_compatible_triple(candidates, used_remaining_mask)
                            if best_triple:
                                c1, c2, c3 = best_triple
                                used_3_mask = used_remaining_mask | block_bitmasks[c1] | block_bitmasks[c2] | block_bitmasks[c3]
                                # Find a 4th compatible
                                for cand in candidates:
                                    if cand not in (c1, c2, c3) and not (block_bitmasks[cand] & used_3_mask):
                                        if cand not in removed:
                                            new_packing = remaining + [c1, c2, c3, cand]
                                            if len(new_packing) > best_size:
                                                best_packing = new_packing
                                                best_mask = get_used_mask(best_packing)
                                                best_size = len(best_packing)
                                                improved = True
                                                break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return best_packing
    
    def final_polish(packing):
        """Try to add any remaining compatible blocks to the packing."""
        if not packing:
            return packing
        
        result = packing[:]
        result_mask = get_used_mask(result)
        
        # Try to add any block that doesn't conflict
        for idx in range(num_blocks_total):
            if idx not in result and not (block_bitmasks[idx] & result_mask):
                result.append(idx)
                result_mask |= block_bitmasks[idx]
        
        return result
    
    # Initial ordering: sort by degree ascending (fewer conflicts first)
    base_order = sorted(range(num_blocks_total), key=lambda x: degrees[x])
    
    # Also create a weighted order: prefer blocks with higher future potential
    weighted_order = sorted(range(num_blocks_total), key=lambda x: (-future_potential[x], degrees[x]))
    
    # Create a "balanced" order that considers both degree and a structural score
    # Blocks that cover more "central" indices might be better
    # Use a simple heuristic: prefer blocks where the indices are more spread out
    def structural_score(idx):
        i1, i2, j1, j2, k1, k2 = all_blocks[idx][0]
        # Spread: larger gaps mean less overlap potential
        spread = (i2 - i1) + (j2 - j1) + (k2 - k1)
        return spread
    
    structural_order = sorted(range(num_blocks_total), key=lambda x: (degrees[x], -structural_score(x)))
    
    # Create a "central" order: prefer blocks covering central indices
    def centrality_score(idx):
        i1, i2, j1, j2, k1, k2 = all_blocks[idx][0]
        # Central indices are more likely to conflict, so prefer less central
        i_center = (i1 + i2) / 2.0
        j_center = (j1 + j2) / 2.0
        k_center = (k1 + k2) / 2.0
        # Distance from center (0 to 1 scale)
        i_dist = abs(i_center - (n-1)/2.0) / max(1, (n-1)/2.0)
        j_dist = abs(j_center - (m-1)/2.0) / max(1, (m-1)/2.0)
        k_dist = abs(k_center - (p-1)/2.0) / max(1, (p-1)/2.0)
        return i_dist + j_dist + k_dist
    
    central_order = sorted(range(num_blocks_total), key=lambda x: (degrees[x], -centrality_score(x)))
    
    best_packing = []
    best_size = 0
    
    # Phase 1: Greedy with random restarts and simulated annealing style perturbations
    num_trials = 5000
    trial = 0
    while trial < num_trials:
        if time.time() > deadline - 10.0:
            break
        
        # Alternate between different ordering strategies
        if trial % 5 == 0:
            order = base_order[:]
        elif trial % 5 == 1:
            order = weighted_order[:]
        elif trial % 5 == 2:
            order = structural_order[:]
        elif trial % 5 == 3:
            order = central_order[:]
        else:
            order = base_order[:]
        
        # Create a perturbed order with varying perturbation strength
        # Use adaptive perturbation: more swaps when stuck
        if best_size == 0:
            num_swaps = random.randint(0, 10)
        elif trial % 20 == 0:
            num_swaps = random.randint(5, 20)
        else:
            num_swaps = random.randint(0, 12)
        
        for _ in range(num_swaps):
            i = random.randint(0, len(order) - 1)
            j = random.randint(0, len(order) - 1)
            order[i], order[j] = order[j], order[i]
        
        # Use lookahead greedy every 10th trial
        if trial % 10 == 5:
            packing = greedy_packing_with_lookahead(order)
        else:
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
            
            # Final polish
            if time.time() < deadline - 3.0:
                polished = final_polish(best_packing)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_packing = polished
            
            if best_size >= 12:
                break
        
        # Every 50 trials, try weighted greedy from scratch
        if trial % 50 == 25 and time.time() < deadline - 5.0:
            weighted_packing = greedy_packing_weighted(0, set(range(num_blocks_total)))
            if len(weighted_packing) > best_size:
                best_size = len(weighted_packing)
                best_packing = weighted_packing[:]
                improved = local_improve(best_packing)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_packing = improved
                polished = final_polish(best_packing)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_packing = polished
        
        # Every 100 trials, try a completely random order
        if trial % 100 == 0 and trial > 0 and time.time() < deadline - 5.0:
            random_order = list(range(num_blocks_total))
            random.shuffle(random_order)
            random_packing = greedy_packing(random_order)
            if len(random_packing) > best_size:
                best_size = len(random_packing)
                best_packing = random_packing[:]
                improved = local_improve(best_packing)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_packing = improved
                polished = final_polish(best_packing)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_packing = polished
        
        trial += 1
    
    # Phase 1.5: If we found a good packing, try to improve it further with multiple local improvement passes
    if time.time() < deadline - 5.0 and best_size >= 5:
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            improved = local_improve(best_packing)
            if len(improved) > best_size:
                best_size = len(improved)
                best_packing = improved
            else:
                # Try final polish
                polished = final_polish(best_packing)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_packing = polished
                else:
                    # Try a random perturbation and re-improve
                    if time.time() < deadline - 3.0:
                        perturbed = best_packing[:]
                        # Remove 1-2 random blocks and re-add greedily
                        if len(perturbed) >= 2:
                            for _ in range(random.randint(1, 2)):
                                if perturbed:
                                    pos = random.randint(0, len(perturbed) - 1)
                                    perturbed.pop(pos)
                            # Re-add greedily
                            current_mask = get_used_mask(perturbed)
                            for idx in base_order:
                                if idx not in perturbed and not (block_bitmasks[idx] & current_mask):
                                    perturbed.append(idx)
                                    current_mask |= block_bitmasks[idx]
                            improved2 = local_improve(perturbed)
                            if len(improved2) > best_size:
                                best_size = len(improved2)
                                best_packing = improved2
                                break
                            else:
                                break
                    break
    
    # Phase 2: More targeted search if time remains
    if time.time() < deadline - 5.0 and best_size < 12:
        # Try all pairs of blocks and see if they can be extended
        # Focus on pairs that are non-adjacent in the conflict graph
        tried_pairs = set()
        for i in range(num_blocks_total):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, num_blocks_total):
                if time.time() > deadline - 5.0:
                    break
                if block_bitmasks[i] & block_bitmasks[j]:
                    continue
                pair_key = (i, j)
                if pair_key in tried_pairs:
                    continue
                tried_pairs.add(pair_key)
                
                # Start with {i, j} and greedily extend
                used_mask = block_bitmasks[i] | block_bitmasks[j]
                current = [i, j]
                for k in base_order:
                    if k == i or k == j:
                        continue
                    if not (block_bitmasks[k] & used_mask):
                        current.append(k)
                        used_mask |= block_bitmasks[k]
                
                if len(current) > best_size:
                    best_size = len(current)
                    best_packing = current[:]
                    # Apply local improvement
                    if time.time() < deadline - 5.0:
                        improved = local_improve(best_packing)
                        if len(improved) > best_size:
                            best_size = len(improved)
                            best_packing = improved
                    # Final polish
                    if time.time() < deadline - 3.0:
                        polished = final_polish(best_packing)
                        if len(polished) > best_size:
                            best_size = len(polished)
                            best_packing = polished
                    if best_size >= 12:
                        break
            if best_size >= 12:
                break
    
    # Phase 3: Try triplet starts for even better packings
    if time.time() < deadline - 3.0 and best_size < 12:
        # Try all triples of mutually compatible blocks
        # Limit to promising triples: those where each pair has low conflict degree
        for i in range(num_blocks_total):
            if time.time() > deadline - 3.0:
                break
            for j in range(i + 1, num_blocks_total):
                if time.time() > deadline - 3.0:
                    break
                if block_bitmasks[i] & block_bitmasks[j]:
                    continue
                # Only consider pairs where both have relatively low degree
                if degrees[i] > 20 or degrees[j] > 20:
                    continue
                for k in range(j + 1, num_blocks_total):
                    if time.time() > deadline - 3.0:
                        break
                    if (block_bitmasks[i] & block_bitmasks[k]) or (block_bitmasks[j] & block_bitmasks[k]):
                        continue
                    if degrees[k] > 20:
                        continue
                    # Start with {i, j, k} and greedily extend
                    used_mask = block_bitmasks[i] | block_bitmasks[j] | block_bitmasks[k]
                    current = [i, j, k]
                    for l in base_order:
                        if l in (i, j, k):
                            continue
                        if not (block_bitmasks[l] & used_mask):
                            current.append(l)
                            used_mask |= block_bitmasks[l]
                    
                    if len(current) > best_size:
                        best_size = len(current)
                        best_packing = current[:]
                        improved = local_improve(best_packing)
                        if len(improved) > best_size:
                            best_size = len(improved)
                            best_packing = improved
                        polished = final_polish(best_packing)
                        if len(polished) > best_size:
                            best_size = len(polished)
                            best_packing = polished
                        if best_size >= 12:
                            break
                if best_size >= 12:
                    break
            if best_size >= 12:
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
    for idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[idx][0]
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
