import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110

    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # For fast index lookup in construction strategy
    # idx_3d[a][b][c] gives index of sorted triple (a,b,c) or -1
    # This is used in the "add point" strategy
    idx_3d = [[[-1] * v for _ in range(v)] for _ in range(v)]
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a][b][c] = subset_to_idx[(t1, t2, t3)]

    # Pre-compute all k-blocks and their masks for the standard greedy
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)
    
    block_masks = [0] * n_blocks
    t_sub_to_blocks = [[] for _ in range(n_subsets)]
    t_sub_freq = [0] * n_subsets

    # Precompute masks
    for b_idx, block in enumerate(all_k_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
            t_sub_to_blocks[idx].append(b_idx)
            t_sub_freq[idx] += 1
        block_masks[b_idx] = mask

    if time.time() > deadline:
        return greedy_fallback(v, k, t, deadline)

    best_blocks = None
    best_score = float('inf')

    def bit_count(x):
        return x.bit_count()

    # Strategy 1: Standard Bitmask Greedy (Fast)
    def run_greedy_bitmask(seed):
        rng = random.Random(seed)
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        used_blocks = set()
        
        while uncovered_mask:
            if time.time() > deadline:
                return None
            
            # Pick a random uncovered triple
            target_idx = -1
            for _ in range(20):
                pos = rng.randint(0, n_subsets - 1)
                if (uncovered_mask >> pos) & 1:
                    target_idx = pos
                    break
            if target_idx == -1:
                # Fallback to lowest set bit
                temp = uncovered_mask
                target_idx = (temp & -temp).bit_length() - 1

            candidates = t_sub_to_blocks[target_idx]
            best_block_idx = -1
            best_gain = -1
            
            # Sample candidates if list is huge, otherwise check all
            # For v=24, k=7, t=3, each triple is in C(21,4)=5985 blocks.
            # Checking all 5985 is fast enough with bit_count.
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                cand_mask = block_masks[b_idx]
                gain = bit_count(uncovered_mask & cand_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break

            blocks.append(all_k_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
        if uncovered_mask == 0:
            return blocks
        return None

    # Strategy 2: Adaptive Construction (Inspiration-inspired)
    # Build blocks by starting with a triple and adding points one by one
    def run_greedy_constructive(seed):
        rng = random.Random(seed)
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        
        points = list(range(v))
        
        while uncovered_mask:
            if time.time() > deadline:
                return None
            
            # Pick a random uncovered triple to start
            target_idx = -1
            for _ in range(20):
                pos = rng.randint(0, n_subsets - 1)
                if (uncovered_mask >> pos) & 1:
                    target_idx = pos
                    break
            if target_idx == -1:
                temp = uncovered_mask
                target_idx = (temp & -temp).bit_length() - 1
                
            target_triple = all_t_subsets[target_idx]
            current_block = list(target_triple)
            
            # Mark target triple as covered (it will be)
            uncovered_mask &= ~(1 << target_idx)
            
            # Add k - t points
            while len(current_block) < k:
                best_point = -1
                best_gain = -1
                
                # Candidates are points not in current_block
                candidates = [p for p in points if p not in current_block]
                rng.shuffle(candidates) # Random tie-breaking
                
                for p in candidates:
                    # Calculate gain of adding p
                    # Gain = number of new triples (p, u, v) covered
                    # where u, v are in current_block
                    gain = 0
                    for i in range(len(current_block)):
                        for j in range(i + 1, len(current_block)):
                            u = current_block[i]
                            v = current_block[j]
                            idx = idx_3d[p][u][v]
                            if idx != -1 and (uncovered_mask >> idx) & 1:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                        
                if best_point == -1:
                    # No point adds coverage? Should not happen if uncovered_mask != 0
                    # but if it does, we break to avoid infinite loop
                    return None
                
                current_block.append(best_point)
                
                # Update uncovered_mask
                # Mark all triples (best_point, u, v) as covered
                for i in range(len(current_block) - 1): # Exclude best_point itself from pairs
                    for j in range(i + 1, len(current_block) - 1):
                        u = current_block[i]
                        v = current_block[j]
                        idx = idx_3d[best_point][u][v]
                        if idx != -1:
                            uncovered_mask &= ~(1 << idx)
                            
            blocks.append(tuple(sorted(current_block)))
            
        return blocks

    # Post-processing: Remove redundant blocks
    def trim_blocks(blocks):
        # Count frequency of each triple
        coverage_count = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                coverage_count[sub] = coverage_count.get(sub, 0) + 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    if coverage_count[sub] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    changed = True
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] -= 1
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            if time.time() > deadline:
                break
        return blocks

    # Run multiple instances of both strategies
    num_runs = 200
    for i in range(num_runs):
        if time.time() > deadline:
            break
            
        if i % 2 == 0:
            result = run_greedy_bitmask(i)
        else:
            result = run_greedy_constructive(i + 1000)
            
        if result and len(result) < best_score:
            trimmed = trim_blocks(result)
            if len(trimmed) < best_score:
                best_score = len(trimmed)
                best_blocks = trimmed

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        used_blocks = set()
        while uncovered_mask:
            if time.time() > deadline:
                break
            temp = uncovered_mask
            target_idx = (temp & -temp).bit_length() - 1
            candidates = t_sub_to_blocks[target_idx]
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                cand_mask = block_masks[b_idx]
                gain = bit_count(uncovered_mask & cand_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            if best_block_idx == -1:
                break
            blocks.append(all_k_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
        best_blocks = blocks

    return best_blocks

def greedy_fallback(v, k, t, deadline):
    start_time = time.time()
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    n_subsets = len(all_t_subsets)

    def block_mask(block):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << subset_to_idx[sub])
        return mask

    uncovered_mask = (1 << n_subsets) - 1
    blocks = []
    while uncovered_mask:
        if time.time() > deadline:
            break
        temp = uncovered_mask
        target_idx = (temp & -temp).bit_length() - 1
        target = all_t_subsets[target_idx]

        remaining_points = [p for p in range(v) if p not in target]
        best_block = None
        best_gain = -1
        for combo in itertools.combinations(remaining_points, k - t):
            candidate_block = tuple(sorted(target + combo))
            cand_mask = block_mask(candidate_block)
            gain = bin(uncovered_mask & cand_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
        blocks.append(list(best_block))
        best_mask = block_mask(best_block)
        uncovered_mask &= ~best_mask
    return blocks
