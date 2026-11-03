"""Simulated annealing for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 114.0  # leave 6s buffer for safety

    # Precompute all t-subsets
    all_tsubs = [frozenset(c) for c in itertools.combinations(range(v), t)]
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {fs: i for i, fs in enumerate(all_tsubs)}

    # Precompute for each point, which t-subsets it participates in
    point_tsubs = [[] for _ in range(v)]
    for i, fs in enumerate(all_tsubs):
        for x in fs:
            point_tsubs[x].append(i)

    # Precompute t-subset indices for a block (cached by tuple)
    block_tsub_cache = {}
    def get_block_tsubs(block):
        """Return list of t-subset indices covered by this block (tuple)."""
        if block in block_tsub_cache:
            return block_tsub_cache[block]
        result = [tsub_to_idx[frozenset(c)] for c in itertools.combinations(block, t)]
        block_tsub_cache[block] = result
        return result

    # Greedy construction
    def greedy_cover():
        uncovered = set(range(n_tsubs))
        blocks = []
        seen = set()
        
        while uncovered:
            # Pick random uncovered t-subset as seed
            seed_idx = rng.choice(list(uncovered))
            seed = all_tsubs[seed_idx]
            block = list(seed)
            rng.shuffle(block)
            
            # Greedily add points to maximize coverage of uncovered subsets
            while len(block) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in block]
                rng.shuffle(cands)
                
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        fs = frozenset(list(rest) + [x])
                        idx = tsub_to_idx.get(fs)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            b_tuple = tuple(block)
            if b_tuple not in seen:
                seen.add(b_tuple)
                blocks.append(b_tuple)
                for c in itertools.combinations(block, t):
                    idx = tsub_to_idx[frozenset(c)]
                    uncovered.discard(idx)
        
        return blocks

    # Initialize state from greedy
    def init_state(blocks):
        counts = [0] * n_tsubs
        for b in blocks:
            for idx in get_block_tsubs(b):
                counts[idx] += 1
        single = [1 if c == 1 else 0 for c in counts]
        n_single = sum(single)
        n_uncovered = sum(1 for c in counts if c == 0)
        block_tsubs_list = [get_block_tsubs(b) for b in blocks]
        return counts, single, n_single, n_uncovered, block_tsubs_list

    # Simulated annealing
    def anneal(initial_blocks, max_time):
        blocks = list(initial_blocks)
        counts, single, n_single, n_uncovered, block_tsubs_list = init_state(blocks)
        
        best_blocks = list(blocks)
        best_count = len(blocks)
        best_time = time.time()
        
        # Temperature schedule
        T0 = 1.0
        T_end = 0.01
        # We'll use a time-based cooling: T decreases over the anneal duration
        # Estimate iterations per second and set cooling rate accordingly
        # For now, use a fixed number of iterations with geometric cooling
        
        iteration = 0
        stuck_count = 0
        no_improve = 0
        max_no_improve = 200
        
        while time.time() - start_time < max_time:
            iteration += 1
            
            # Compute temperature based on elapsed fraction
            elapsed = time.time() - start_time
            # Use a smooth decay
            if elapsed > 0:
                T = T_end + (T0 - T_end) * math.exp(-3.0 * elapsed / 60.0)
            else:
                T = T0
            
            progress = False
            
            # Strategy 1: Try to remove redundant blocks
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            for i in indices:
                redundant = True
                for idx in block_tsubs_list[i]:
                    if counts[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for idx in block_tsubs_list[i]:
                        old_c = counts[idx]
                        counts[idx] = old_c - 1
                        if old_c == 1:
                            n_single -= 1
                            single[idx] = 0
                        elif old_c == 2:
                            n_single += 1
                            single[idx] = 1
                        if counts[idx] == 0:
                            n_uncovered += 1
                    
                    blocks.pop(i)
                    block_tsubs_list.pop(i)
                    
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                        best_time = time.time()
                    progress = True
                    no_improve = 0
                    break
            
            if progress:
                stuck_count = 0
                continue
            
            # Strategy 2: Swap a block
            if len(blocks) > 0:
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                old_tsubs = block_tsubs_list[i]
                
                # Find critical t-subsets (count == 1) in old block
                critical = [idx for idx in old_tsubs if counts[idx] == 1]
                
                if len(critical) > 0:
                    # Collect points needed to cover all critical t-subsets
                    points_needed = set()
                    for idx in critical:
                        points_needed.update(all_tsubs[idx])
                    
                    if len(points_needed) <= k:
                        # Try to form a new block containing points_needed
                        new_block_set = set(points_needed)
                        cands = [x for x in range(v) if x not in new_block_set]
                        rng.shuffle(cands)
                        
                        # Greedily fill to maximize coverage
                        for x in cands:
                            if len(new_block_set) >= k:
                                break
                            gain = 0
                            for rest in itertools.combinations(new_block_set, t-1):
                                fs = frozenset(list(rest) + [x])
                                idx = tsub_to_idx.get(fs)
                                if idx is not None:
                                    if counts[idx] == 0:
                                        gain += 100
                                    elif counts[idx] == 1:
                                        gain += 1
                            new_block_set.add(x)
                        
                        if len(new_block_set) == k:
                            new_block = tuple(sorted(new_block_set))
                            if new_block != old_block:
                                new_tsubs = get_block_tsubs(new_block)
                                
                                # Check if swap is valid: no t-subset becomes uncovered
                                old_set = set(old_tsubs)
                                new_set = set(new_tsubs)
                                valid = True
                                for idx in old_tsubs:
                                    if counts[idx] == 1 and idx not in new_set:
                                        valid = False
                                        break
                                
                                if valid:
                                    # Compute delta in n_single for scoring
                                    # After swap: tsubs in old but not new lose 1 count
                                    # tsubs in new but not old gain 1 count
                                    delta_single = 0
                                    for idx in old_tsubs:
                                        if idx not in new_set:
                                            if counts[idx] == 1:
                                                delta_single -= 1  # was single, becomes 0
                                            elif counts[idx] == 2:
                                                delta_single += 1  # was double, becomes single
                                    for idx in new_tsubs:
                                        if idx not in old_set:
                                            if counts[idx] == 0:
                                                pass  # uncovered, no change to single
                                            elif counts[idx] == 1:
                                                delta_single += 1  # becomes double
                                            elif counts[idx] == 2:
                                                delta_single -= 1  # was double, becomes triple
                                    
                                    # Accept if improves n_single or with probability based on T
                                    if delta_single < 0:
                                        accept = True
                                    elif delta_single == 0:
                                        accept = True
                                    else:
                                        # delta_single > 0, accept with probability exp(-delta/T)
                                        accept = rng.random() < math.exp(-delta_single / max(T, 0.01))
                                    
                                    if accept:
                                        # Apply swap
                                        for idx in old_tsubs:
                                            old_c = counts[idx]
                                            counts[idx] = old_c - 1
                                            if old_c == 1:
                                                n_single -= 1
                                                single[idx] = 0
                                            elif old_c == 2:
                                                n_single += 1
                                                single[idx] = 1
                                            if counts[idx] == 0:
                                                n_uncovered += 1
                                        
                                        for idx in new_tsubs:
                                            old_c = counts[idx]
                                            counts[idx] = old_c + 1
                                            if old_c == 0:
                                                n_uncovered -= 1
                                            elif old_c == 1:
                                                n_single += 1
                                                single[idx] = 1
                                            elif old_c == 2:
                                                n_single -= 1
                                                single[idx] = 0
                                        
                                        blocks[i] = new_block
                                        block_tsubs_list[i] = new_tsubs
                                        stuck_count = 0
                                        no_improve = 0
                                        continue
                                else:
                                    # Invalid swap, try with random perturbation
                                    pass
            
            # Strategy 3: Perturbation if stuck
            stuck_count += 1
            if stuck_count >= 30:
                if len(blocks) > 0:
                    i = rng.randrange(len(blocks))
                    old_block = blocks[i]
                    old_tsubs = block_tsubs_list[i]
                    
                    critical = [idx for idx in old_tsubs if counts[idx] == 1]
                    points_needed = set()
                    for idx in critical:
                        points_needed.update(all_tsubs[idx])
                    
                    if len(points_needed) <= k:
                        new_block_set = set(points_needed)
                        cands = [x for x in range(v) if x not in new_block_set]
                        rng.shuffle(cands)
                        for x in cands:
                            if len(new_block_set) >= k:
                                break
                            new_block_set.add(x)
                        
                        if len(new_block_set) == k:
                            new_block = tuple(sorted(new_block_set))
                            new_tsubs = get_block_tsubs(new_block)
                            
                            # Check validity
                            valid = True
                            new_set = set(new_tsubs)
                            for idx in old_tsubs:
                                if counts[idx] == 1 and idx not in new_set:
                                    valid = False
                                    break
                            
                            if valid:
                                for idx in old_tsubs:
                                    old_c = counts[idx]
                                    counts[idx] = old_c - 1
                                    if old_c == 1:
                                        n_single -= 1
                                        single[idx] = 0
                                    elif old_c == 2:
                                        n_single += 1
                                        single[idx] = 1
                                    if counts[idx] == 0:
                                        n_uncovered += 1
                                
                                for idx in new_tsubs:
                                    old_c = counts[idx]
                                    counts[idx] = old_c + 1
                                    if old_c == 0:
                                        n_uncovered -= 1
                                    elif old_c == 1:
                                        n_single += 1
                                        single[idx] = 1
                                    elif old_c == 2:
                                        n_single -= 1
                                        single[idx] = 0
                                
                                blocks[i] = new_block
                                block_tsubs_list[i] = new_tsubs
                                stuck_count = 0
                                no_improve = 0
                else:
                    # Rebuild
                    new_blocks = greedy_cover()
                    blocks = new_blocks
                    counts, single, n_single, n_uncovered, block_tsubs_list = init_state(blocks)
                    stuck_count = 0
                    no_improve = 0
            
            # Track no-improve for potential restart
            no_improve += 1
            if no_improve > max_no_improve:
                # Restart with fresh greedy if we've been stuck too long
                if time.time() - best_time > 30:  # 30s without improvement
                    break
        
        return best_blocks, best_count

    # Main: run multiple annealing rounds with different initial solutions
    overall_best_blocks = None
    overall_best_count = float('inf')
    
    # First greedy + anneal
    blocks1 = greedy_cover()
    best_b, best_c = anneal(blocks1, time_limit)
    overall_best_blocks = best_b
    overall_best_count = best_c
    
    # Use remaining time for additional restarts with different seeds
    while time.time() - start_time < time_limit - 5:
        # New greedy with different random state
        old_rng_state = rng.getstate()
        rng.seed(rng.randint(0, 2**32 - 1))
        blocks_new = greedy_cover()
        rng.setstate(old_rng_state)
        
        # Quick anneal with shorter time budget
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10:
            break
        anneal_time = min(remaining - 5, 30)  # at most 30s per restart
        best_b, best_c = anneal(blocks_new, start_time + anneal_time)
        
        if best_c < overall_best_count:
            overall_best_count = best_c
            overall_best_blocks = best_b
    
    return [list(b) for b in overall_best_blocks]
# EVOLVE-BLOCK-END
