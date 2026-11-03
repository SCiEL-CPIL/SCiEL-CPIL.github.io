"""Improved local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math


def _greedy_cover(v, k, t, rng=None):
    """Fast greedy construction to get a valid starting covering."""
    if rng is None:
        rng = random.Random(42)
    
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in blocks:
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
    def compute_counts(blk_list):
        counts = {}
        for b in blk_list:
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                counts[f] = counts.get(f, 0) + 1
        return counts

    counts = compute_counts(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)
    current_count = len(blocks)
    
    # Precompute all t-subsets as frozensets for quick lookup
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    
    # Precompute for each block (as frozenset of points) the list of t-subsets it covers
    # This is expensive to precompute for all possible blocks, so we'll compute on the fly
    # but cache for current blocks
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.998
    max_iterations = 1000000
    
    iteration = 0
    stall_count = 0
    stall_threshold = 5000
    
    while time.time() - start_time < time_limit and T > T_min:
        iteration += 1
        
        # 1. Try to remove redundant blocks (always accept if valid)
        if current_count > 0:
            i = rng.randrange(current_count)
            b = blocks[i]
            
            # Check if all t-subsets in b have count > 1
            redundant = True
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block
                for fs in itertools.combinations(b, t):
                    f = frozenset(fs)
                    counts[f] -= 1
                    if counts[f] == 0:
                        del counts[f]
                
                blocks.pop(i)
                current_count -= 1
                stall_count = 0
                
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = list(blocks)
                
                continue
        
        # 2. Swap move: Replace a block with a new one
        if current_count > 0:
            i = rng.randrange(current_count)
            old_block = blocks[i]
            
            # Find critical t-subsets (count == 1) in old_block
            critical = []
            for fs in itertools.combinations(old_block, t):
                f = frozenset(fs)
                if counts.get(f, 0) == 1:
                    critical.append(f)
            
            if not critical:
                # No critical subsets, we can try to replace with a better covering block
                # or just skip to focus on reduction
                continue
            
            # Construct new block that covers critical subsets
            points_needed = set()
            for fs in critical:
                points_needed.update(fs)
            
            if len(points_needed) > k:
                continue
                
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Fill remaining spots to maximize coverage
            while len(new_block_set) < k:
                best_x = None
                best_gain = -1
                sample_cands = cands[:30] if len(cands) > 30 else cands
                for x in sample_cands:
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t - 1):
                        fs = frozenset(list(rest) + [x])
                        c = counts.get(fs, 0)
                        if c == 0:
                            gain += 10
                        elif c == 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                new_block_set.add(best_x)
                if best_x in cands:
                    cands.remove(best_x)
                    
            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))
                if new_block != old_block:
                    # Check validity: lost subsets must be covered by new block
                    lost = set()
                    for fs in itertools.combinations(old_block, t):
                        f = frozenset(fs)
                        if counts.get(f, 0) == 1:
                            lost.add(f)
                    
                    gained = set()
                    for fs in itertools.combinations(new_block, t):
                        f = frozenset(fs)
                        gained.add(f)
                    
                    if lost.issubset(gained):
                        # Apply swap
                        for fs in itertools.combinations(old_block, t):
                            f = frozenset(fs)
                            counts[f] -= 1
                            if counts[f] == 0:
                                del counts[f]
                        for fs in itertools.combinations(new_block, t):
                            f = frozenset(fs)
                            if f in counts:
                                counts[f] += 1
                            else:
                                counts[f] = 1
                        blocks[i] = new_block
                        stall_count = 0
                    else:
                        # Accept with probability based on temperature?
                        # For now, skip invalid swaps
                        pass
        
        # 3. Perturbation: Remove a block and greedily repair
        if iteration % 200 == 0:
            if current_count > 1:
                old_blocks = list(blocks)
                old_counts = dict(counts)
                old_current_count = current_count
                
                # Remove a random block
                i = rng.randrange(current_count)
                b = blocks[i]
                for fs in itertools.combinations(b, t):
                    f = frozenset(fs)
                    counts[f] -= 1
                    if counts[f] == 0:
                        del counts[f]
                blocks.pop(i)
                current_count -= 1
                
                # Find uncovered
                unc_list = []
                for c in all_tsets:
                    if counts.get(c, 0) == 0:
                        unc_list.append(c)
                        
                # Greedily cover uncovered
                while unc_list:
                    seed = rng.choice(unc_list)
                    new_b = list(seed)
                    rng.shuffle(new_b)
                    
                    while len(new_b) < k:
                        best_x = None
                        best_gain = -1
                        cands = [x for x in range(v) if x not in new_b]
                        rng.shuffle(cands)
                        for x in cands:
                            gain = 0
                            for rest in itertools.combinations(new_b, t - 1):
                                fs = frozenset(list(rest) + [x])
                                if counts.get(fs, 0) == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        new_b.append(best_x)
                    
                    new_b.sort()
                    new_b_tuple = tuple(new_b)
                    
                    if new_b_tuple not in blocks:
                        blocks.append(new_b_tuple)
                        for fs in itertools.combinations(new_b_tuple, t):
                            f = frozenset(fs)
                            if f in counts:
                                counts[f] += 1
                            else:
                                counts[f] = 1
                        current_count += 1
                        
                        # Update unc_list
                        unc_list = []
                        for c in all_tsets:
                            if counts.get(c, 0) == 0:
                                unc_list.append(c)
                
                # Check if we improved
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = list(blocks)
                    stall_count = 0
                else:
                    # Revert if worse
                    blocks = old_blocks
                    counts = old_counts
                    current_count = old_current_count
        
        T *= alpha
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
