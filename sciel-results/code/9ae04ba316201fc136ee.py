import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 150s limit

    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # Precompute block coverage
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # Inverse mapping: t-subset -> list of blocks covering it
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Helper to check if a set of blocks covers all t-subsets
    def is_valid(blocks):
        if not blocks:
            return N_t == 0
        counts = np.zeros(N_t, dtype=np.int8)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
        return np.all(counts > 0)

    # --- PHASE 1: Probabilistic Pre-selection ---
    # We want to choose a random set of blocks such that the expected number of 
    # uncovered t-subsets is low.
    # P(t-subset not covered by one random block) = 1 - C(k,t)/C(v,t)
    # P(t-subset not covered by M random blocks) = (1 - C(k,t)/C(v,t))^M
    # Expected uncovered = N_t * (1 - C(k,t)/C(v,t))^M
    
    p_cover_single = np.comb(k, t) / np.comb(v, t)
    
    # We aim for expected uncovered < 500
    # 500 = N_t * (1 - p)^M => M = log(500/N_t) / log(1-p)
    target_uncovered = 500
    if N_t > target_uncovered:
        M_est = int(np.log(target_uncovered / N_t) / np.log(1 - p_cover_single))
    else:
        M_est = 10 # small case
    
    # We will try a few different M values and random seeds
    best_pre_selected = None
    best_uncovered_count = float('inf')
    
    max_pre_selection_time = 5.0
    rng_pre = random.Random(42)
    
    while time.time() < t0 + max_pre_selection_time:
        # Try a slightly varied M
        M = M_est + rng_pre.randint(-5, 5)
        if M < 10: M = 10
        
        # Select M random blocks
        selected_indices = rng_pre.sample(range(N_b), M)
        
        # Calculate uncovered
        covered_mask = np.zeros(N_t, dtype=bool)
        for bi in selected_indices:
            for ti in block_cover[bi]:
                covered_mask[ti] = True
        
        uncovered_indices = np.where(~covered_mask)[0]
        num_uncovered = len(uncovered_indices)
        
        if num_uncovered < best_uncovered_count:
            best_uncovered_count = num_uncovered
            best_pre_selected = selected_indices
            
            if num_uncovered == 0:
                break

    if best_pre_selected is None:
        # Fallback: Greedy
        best_pre_selected = []
        covered_mask = np.zeros(N_t, dtype=bool)
        while not covered_mask.all():
            uncovered_indices = np.where(~covered_mask)[0]
            if len(uncovered_indices) == 0: break
            target_ti = uncovered_indices[0]
            best_bi = -1
            best_gain = -1
            for bi in t_to_blocks[target_ti]:
                gain = 0
                for ti in block_cover[bi]:
                    if not covered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            best_pre_selected.append(best_bi)
            for ti in block_cover[best_bi]:
                covered_mask[ti] = True
            
    # --- PHASE 2: Solve Residual Problem Exactly ---
    # The residual problem is to cover the remaining uncovered t-subsets 
    # using additional blocks. 
    # We can use a backtracking solver.
    
    # Identify the current uncovered set
    covered_mask = np.zeros(N_t, dtype=bool)
    for bi in best_pre_selected:
        for ti in block_cover[bi]:
            covered_mask[ti] = True
            
    initial_uncovered = set(np.where(~covered_mask)[0].tolist())
    
    # If already covered, we are done with residual (size 0)
    if not initial_uncovered:
        residual_solution = []
    else:
        # Backtracking solver for the residual
        # State: current_uncovered (set of ti indices)
        # We want to find a small set of blocks to cover current_uncovered.
        
        # Heuristic: Pick the uncovered t-subset with the fewest candidate blocks 
        # that are NOT already in best_pre_selected? 
        # Actually, we can use ANY block. But to avoid redundancy, we might prefer 
        # blocks that cover many *other* uncovered t-subsets.
        
        # Let's use a simple DFS with pruning.
        # Pruning: If current_size + lower_bound > best_found, prune.
        
        best_residual = None
        best_residual_size = len(initial_uncovered) # Upper bound: cover each individually (worst case)
        
        # Sort t-subsets in initial_uncovered by difficulty (fewest blocks)
        # But difficulty changes as we cover more? No, static difficulty is len(t_to_blocks[ti])
        # Dynamic difficulty: how many of its covering blocks are "good"?
        
        def solve_residual(uncovered_set, current_blocks, time_limit):
            nonlocal best_residual, best_residual_size
            
            if time.time() > time_limit:
                return
            
            if not uncovered_set:
                if len(current_blocks) < best_residual_size:
                    best_residual_size = len(current_blocks)
                    best_residual = list(current_blocks)
                return
                
            # Lower bound: Max(1, ceil(len(uncovered) / max_new_cov))
            # max_new_cov is 56, but realistically less for residual.
            # Simple LB: 1
            if len(current_blocks) + 1 >= best_residual_size:
                return
                
            # Select the most constrained t-subset
            # Find ti in uncovered_set with min len(t_to_blocks[ti])
            # To speed up, we can just pick the first one, or the one with min candidates.
            # Since N_t is small in residual (hopefully), scanning is ok.
            
            min_len = float('inf')
            target_ti = -1
            for ti in uncovered_set:
                # Count how many blocks in t_to_blocks[ti] are "available"? 
                # All blocks are available. 
                # But we can prune candidates that don't cover *any* new t-subset?
                # No, they must cover target_ti.
                
                # Heuristic: Pick ti with fewest total blocks
                if len(t_to_blocks[ti]) < min_len:
                    min_len = len(t_to_blocks[ti])
                    target_ti = ti
            
            if target_ti == -1:
                return
                
            # Candidates: blocks covering target_ti
            candidates = t_to_blocks[target_ti]
            
            # Sort candidates by "gain" (how many other uncovered t's they cover)
            def score_block(bi):
                gain = 0
                for tj in block_cover[bi]:
                    if tj in uncovered_set:
                        gain += 1
                return gain
            
            # Optimization: Only consider top K candidates to reduce branching
            # But for exactness, we should consider all. 
            # Given the residual is small, let's consider all but sorted.
            candidates.sort(key=score_block, reverse=True)
            
            # Pruning: If the best candidate doesn't cover enough to satisfy LB?
            # We'll just iterate.
            
            for bi in candidates:
                # Apply
                new_uncovered = set(uncovered_set)
                for tj in block_cover[bi]:
                    new_uncovered.discard(tj)
                
                current_blocks.append(bi)
                solve_residual(new_uncovered, current_blocks, time_limit)
                current_blocks.pop()
                
                # If we found a good solution, we might stop early? 
                # No, we want the minimum. But if time is up, we return.
                if time.time() > time_limit:
                    return

        # Run residual solver
        residual_time_limit = time.time() + 10.0 # Give 10s to residual solver
        solve_residual(initial_uncovered, [], residual_time_limit)
        
        if best_residual is None:
            # Fallback if solver failed or timed out without finding optimal
            # Use greedy for residual
            best_residual = []
            current_uncovered = set(initial_uncovered)
            while current_uncovered:
                target_ti = next(iter(current_uncovered))
                best_bi = -1
                best_gain = -1
                for bi in t_to_blocks[target_ti]:
                    gain = 0
                    for tj in block_cover[bi]:
                        if tj in current_uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi
                best_residual.append(best_bi)
                for tj in block_cover[best_bi]:
                    current_uncovered.discard(tj)

    # Combine Pre-selection and Residual
    combined_blocks = list(best_pre_selected) + list(best_residual)
    
    # --- PHASE 3: Global Optimization (Simulated Annealing) ---
    # We have a valid covering. Now we try to reduce its size.
    # Moves:
    # 1. Remove a block if it is redundant.
    # 2. Swap a block with another block that allows removing a different block?
    #    Simpler: Try to replace a block with a different block that covers the same 
    #    "unique" t-subsets, or just random swap and check validity, then prune.
    
    # Let's use a "Greedy Prune + Local Search" approach.
    
    def prune_blocks(blocks):
        if not blocks:
            return []
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
        
        current = list(blocks)
        changed = True
        while changed:
            changed = False
            new_current = []
            for bi in current:
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_current.append(bi)
                else:
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    changed = True
            current = new_current
        return current

    current_best = prune_blocks(combined_blocks)
    best_size = len(current_best)
    
    # Local Search
    # We will try to swap blocks to create redundancy, then prune.
    rng_ls = random.Random(123)
    ls_deadline = deadline - 5.0
    
    # Precompute unique t-subsets for each block in current_best
    # This is expensive to update, so we'll recalculate counts periodically or on the fly.
    
    def get_unique_ts(blocks):
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
        uniques = {}
        for bi in blocks:
            u = []
            for ti in block_cover[bi]:
                if counts[ti] == 1:
                    u.append(ti)
            uniques[bi] = u
        return uniques

    # Initial unique map
    unique_map = get_unique_ts(current_best)
    
    while time.time() < ls_deadline:
        # Pick a random block to modify
        if not current_best:
            break
            
        idx = rng_ls.randrange(len(current_best))
        bi = current_best[idx]
        
        # Get unique t-subsets covered by bi
        # unique_ts = unique_map[bi] # Need to be careful if map is stale
        
        # Let's recalculate uniques for bi only? 
        # Counts are global. 
        # Efficient way: 
        # 1. Tentatively remove bi.
        # 2. Check which t-subsets become uncovered.
        # 3. If none, remove bi.
        # 4. If some, find a block bj that covers them. Swap bi with bj.
        # 5. Prune.
        
        # Step 1: Identify "critical" t-subsets for bi
        # A t-subset is critical for bi if it is covered ONLY by bi in the current set.
        
        # To find this fast, we need counts.
        # Instead of full recount, we can maintain counts.
        
        # Let's maintain counts array.
        # Initialize counts
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_best:
            for ti in block_cover[bi]:
                counts[ti] += 1
        
        # Find unique t-subsets for bi
        unique_ts = []
        for ti in block_cover[bi]:
            if counts[ti] == 1:
                unique_ts.append(ti)
                
        if not unique_ts:
            # bi is redundant
            current_best.pop(idx)
            for ti in block_cover[bi]:
                counts[ti] -= 1
            if len(current_best) < best_size:
                best_size = len(current_best)
            continue
            
        # bi is not redundant. We need to replace it with a block that covers all unique_ts.
        # Find candidate blocks that cover ALL unique_ts.
        
        # Intersection of t_to_blocks for all unique_ts
        candidates = set(t_to_blocks[unique_ts[0]])
        for ti in unique_ts[1:]:
            candidates &= set(t_to_blocks[ti])
            
        candidates.discard(bi)
        
        if not candidates:
            # No direct replacement. Try random swap and hope for indirect improvement?
            # Or just skip.
            # To allow exploration, we can pick a random block and swap, then prune.
            # This is "Simulated Annealing" style.
            # Let's do a random swap with a block that covers at least one unique_ts.
            # Pick first unique_ts, pick random block covering it.
            target_ti = unique_ts[0]
            cands = t_to_blocks[target_ti]
            if cands:
                new_bi = rng_ls.choice(cands)
                if new_bi != bi:
                    # Perform swap
                    current_best[idx] = new_bi
                    # Update counts
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    for ti in block_cover[new_bi]:
                        counts[ti] += 1
                    
                    # Prune
                    changed = True
                    while changed:
                        changed = False
                        new_best = []
                        for b in current_best:
                            is_red = True
                            for ti in block_cover[b]:
                                if counts[ti] == 1:
                                    is_red = False
                                    break
                            if not is_red:
                                new_best.append(b)
                            else:
                                for ti in block_cover[b]:
                                    counts[ti] -= 1
                                changed = True
                        current_best = new_best
                    
                    if len(current_best) < best_size:
                        best_size = len(current_best)
                    else:
                        # Revert if not improved? 
                        # In SA, we might accept worse states. 
                        # But for this problem, "worse" means more blocks. 
                        # We only want to keep improvements or neutral moves that lead to future improvements.
                        # Since we don't have a temperature, let's only keep if size <= best_size.
                        # Actually, if size == best_size, it might be a different configuration.
                        # Let's keep it if size <= best_size.
                        pass
            continue

        # We have candidates. Pick the one that covers the most OTHER t-subsets 
        # (to potentially make other blocks redundant).
        best_cand = -1
        best_score = -1
        
        # Limit candidates to avoid slow loop
        cand_list = list(candidates)
        if len(cand_list) > 20:
            cand_list = rng_ls.sample(cand_list, 20)
            
        for c in cand_list:
            # Score: number of t-subsets covered by c that are currently covered by >1 block
            # (i.e., we are "wasting" coverage there, so swapping might not hurt, 
            # but we want to cover the unique_ts which we already know it does).
            # Better score: How many unique_ts of OTHER blocks does c cover?
            # This is complex. 
            # Simple score: Random.
            score = 0
            for ti in block_cover[c]:
                if counts[ti] > 1:
                    score += 1
            if score > best_score:
                best_score = score
                best_cand = c
                
        if best_cand != -1:
            # Swap
            current_best[idx] = best_cand
            for ti in block_cover[bi]:
                counts[ti] -= 1
            for ti in block_cover[best_cand]:
                counts[ti] += 1
                
            # Prune
            changed = True
            while changed:
                changed = False
                new_best = []
                for b in current_best:
                    is_red = True
                    for ti in block_cover[b]:
                        if counts[ti] == 1:
                            is_red = False
                            break
                    if not is_red:
                        new_best.append(b)
                    else:
                        for ti in block_cover[b]:
                            counts[ti] -= 1
                        changed = True
                current_best = new_best
                
            if len(current_best) < best_size:
                best_size = len(current_best)
            # If not improved, we keep the state as is (neutral move) to explore.
            # But if it got worse (more blocks), we should revert?
            # The swap itself doesn't increase block count. Pruning only decreases.
            # So size is non-increasing.
            # So we always keep the result.

    return [list(all_blocks[bi]) for bi in current_best]
