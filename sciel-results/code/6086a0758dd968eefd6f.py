"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def _fast_greedy_cover(v, k, t, rng, seed_block=None):
    """Fast greedy construction using bitmasks for t-subset tracking."""
    # Precompute all t-subsets as frozensets and index them
    all_tsubsets = []
    subset_to_idx = {}
    for i, c in enumerate(itertools.combinations(range(v), t)):
        fs = frozenset(c)
        all_tsubsets.append(fs)
        subset_to_idx[fs] = i
    
    total_t = len(all_tsubsets)
    # uncovered_mask[i] = True if t-subset i is not yet covered
    uncovered = [True] * total_t
    uncovered_count = total_t
    
    blocks = []
    seen_blocks = set()
    
    while uncovered_count > 0:
        # Pick a random uncovered t-subset as seed
        # Find an uncovered subset
        while True:
            idx = rng.randrange(total_t)
            if uncovered[idx]:
                break
        seed = list(all_tsubsets[idx])
        rng.shuffle(seed)
        
        block = list(seed)
        
        # Greedily add points to maximize coverage of uncovered subsets
        while len(block) < k:
            best_x = None
            best_gain = 0
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_idx.get(fs)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain == 0:
                break
            block.append(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in seen_blocks:
            seen_blocks.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                fs = frozenset(c)
                idx = subset_to_idx.get(fs)
                if idx is not None and uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
    
    return blocks


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer
    
    # Precompute all t-subsets
    all_tsubsets = []
    subset_to_idx = {}
    for i, c in enumerate(itertools.combinations(range(v), t)):
        fs = frozenset(c)
        all_tsubsets.append(fs)
        subset_to_idx[fs] = i
    
    total_t = len(all_tsubsets)
    C_k_t = len(list(itertools.combinations(range(k), t)))
    
    # Build initial valid covering
    blocks = _fast_greedy_cover(v, k, t, rng)
    
    # counts[idx] = number of blocks covering t-subset idx
    counts = [0] * total_t
    
    def update_counts_add(block):
        for c in itertools.combinations(block, t):
            fs = frozenset(c)
            idx = subset_to_idx.get(fs)
            if idx is not None:
                counts[idx] += 1
    
    def update_counts_remove(block):
        for c in itertools.combinations(block, t):
            fs = frozenset(c)
            idx = subset_to_idx.get(fs)
            if idx is not None:
                counts[idx] -= 1
    
    for b in blocks:
        update_counts_add(b)
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    def is_redundant(block):
        """Check if removing block would leave all t-subsets covered."""
        for c in itertools.combinations(block, t):
            fs = frozenset(c)
            idx = subset_to_idx.get(fs)
            if idx is not None and counts[idx] <= 1:
                return False
        return True
    
    def find_replacement_block(old_block, critical_subsets):
        """Find a new block that covers all critical subsets and maximizes coverage."""
        if not critical_subsets:
            return None
        
        points_needed = set()
        for fs in critical_subsets:
            points_needed.update(fs)
        
        if len(points_needed) > k:
            return None
        
        new_block_set = set(points_needed)
        cands = [x for x in range(v) if x not in new_block_set]
        rng.shuffle(cands)
        
        # Greedy fill to maximize coverage of uncovered or single-covered subsets
        for x in cands:
            if len(new_block_set) >= k:
                break
            gain = 0
            for rest in itertools.combinations(new_block_set, t-1):
                fs = frozenset(list(rest) + [x])
                idx = subset_to_idx.get(fs)
                if idx is not None:
                    c = counts[idx]
                    if c == 0:
                        gain += 100
                    elif c == 1:
                        gain += 1
            if gain > 0:
                new_block_set.add(x)
            elif len(new_block_set) < k:
                new_block_set.add(x)
        
        if len(new_block_set) < k:
            return None
        
        new_block = tuple(sorted(new_block_set))
        if new_block == old_block:
            return None
        
        # Validate: after removing old_block and adding new_block, no t-subset becomes uncovered
        # Simulate
        for c in itertools.combinations(old_block, t):
            fs = frozenset(c)
            idx = subset_to_idx.get(fs)
            if idx is not None and counts[idx] == 1:
                # Check if new_block covers this subset
                found = False
                for c2 in itertools.combinations(new_block, t):
                    if frozenset(c2) == fs:
                        found = True
                        break
                if not found:
                    return None
        
        return new_block
    
    # Simulated Annealing Loop
    temperature = 1.0
    cooling = 0.9995
    min_temp = 0.01
    
    iteration = 0
    stuck_counter = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Try to remove a redundant block
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            if is_redundant(blocks[i]):
                # Remove it
                b = blocks[i]
                blocks.pop(i)
                update_counts_remove(b)
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                    stuck_counter = 0
                removed = True
                break
        
        if removed:
            continue
        
        # 2. Try to swap a block
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Find critical subsets (count == 1)
            critical_subsets = []
            for c in itertools.combinations(old_block, t):
                fs = frozenset(c)
                idx = subset_to_idx.get(fs)
                if idx is not None and counts[idx] == 1:
                    critical_subsets.append(fs)
            
            if critical_subsets:
                new_block = find_replacement_block(old_block, critical_subsets)
                if new_block is not None:
                    # Perform swap
                    update_counts_remove(old_block)
                    update_counts_add(new_block)
                    blocks[i] = new_block
                    stuck_counter = 0
            else:
                # No critical subsets - this block should be redundant
                # But is_redundant said no, so maybe counts changed?
                # Just try a random swap
                new_block_set = set(old_block)
                # Remove one random element
                rem = rng.choice(list(new_block_set))
                new_block_set.remove(rem)
                # Add a random new element
                cands = [x for x in range(v) if x not in new_block_set]
                if cands:
                    new_block_set.add(rng.choice(cands))
                    new_block = tuple(sorted(new_block_set))
                    if new_block != old_block:
                        # Validate swap
                        valid = True
                        for c in itertools.combinations(old_block, t):
                            fs = frozenset(c)
                            idx = subset_to_idx.get(fs)
                            if idx is not None and counts[idx] == 1:
                                found = False
                                for c2 in itertools.combinations(new_block, t):
                                    if frozenset(c2) == fs:
                                        found = True
                                        break
                                if not found:
                                    valid = False
                                    break
                        
                        if valid:
                            update_counts_remove(old_block)
                            update_counts_add(new_block)
                            blocks[i] = new_block
                            stuck_counter = 0
        
        # 3. Restart if stuck for too long
        stuck_counter += 1
        if stuck_counter > 500:
            new_cover = _fast_greedy_cover(v, k, t, rng)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                counts = [0] * total_t
                for b in blocks:
                    update_counts_add(b)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                stuck_counter = 0
        
        # Cool down
        temperature = max(min_temp, temperature * cooling)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
