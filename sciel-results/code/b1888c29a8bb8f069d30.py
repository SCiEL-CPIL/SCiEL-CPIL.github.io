"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math


def _precompute_triples(v, k, t):
    """Precompute all C(v,k) k-blocks as sorted tuples and their t-subsets."""
    all_blocks = []
    block_triples = {}
    for combo in itertools.combinations(range(v), k):
        b = tuple(combo)
        all_blocks.append(b)
        block_triples[b] = [frozenset(c) for c in itertools.combinations(b, t)]
    return all_blocks, block_triples


def _greedy_cover(v, k, t, rng, block_triples):
    """Scarcity-weighted greedy construction for a tighter initial covering."""
    uncovered = set(frozenset(c) for c in itertools.combinations(range(v), t))
    blocks = []
    block_set = set()
    
    # Count how many times each point appears in uncovered triples
    # This helps identify "scarce" points that appear in fewer uncovered triples
    point_freq = {}
    for f in uncovered:
        for p in f:
            point_freq[p] = point_freq.get(p, 0) + 1
    
    while uncovered:
        # Weighted selection: prefer triples containing scarce points
        # Assign weight inversely proportional to point frequency
        weights = []
        for f in uncovered:
            w = 1.0
            for p in f:
                w *= 1.0 / (point_freq.get(p, 1) + 1)
            weights.append(w)
        
        # Normalize and sample
        total_w = sum(weights)
        if total_w > 0:
            r = rng.random() * total_w
            cum = 0.0
            selected = None
            for i, w in enumerate(weights):
                cum += w
                if cum >= r:
                    selected = list(uncovered)[i]
                    break
            if selected is None:
                selected = list(uncovered)[0]
        else:
            selected = list(uncovered)[0]
        
        block = list(selected)
        rng.shuffle(block)
        
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
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                fs = frozenset(c)
                if fs in uncovered:
                    uncovered.discard(fs)
                    for p in fs:
                        point_freq[p] = point_freq.get(p, 0) - 1
                        
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute all k-blocks and their t-subsets for O(1) lookup
    all_blocks, block_triples = _precompute_triples(v, k, t)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, block_triples)
    
    def get_triples(blk):
        return block_triples[blk]
    
    # Count how many times each t-subset is covered
    counts = {}
    for b in blocks:
        for fs in get_triples(b):
            counts[fs] = counts.get(fs, 0) + 1
    
    total_t = math.comb(v, t)
    uncovered = total_t - len(counts)
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    def apply_block_change(blk, sign):
        """
        sign = 1 to add, -1 to remove.
        Returns change in uncovered count (delta_uncovered).
        """
        delta_uncovered = 0
        for fs in get_triples(blk):
            old_c = counts.get(fs, 0)
            new_c = old_c + sign
            if new_c == 0:
                if fs in counts:
                    del counts[fs]
                if old_c == 1:
                    delta_uncovered += 1
            elif new_c > 0:
                if old_c == 0:
                    delta_uncovered -= 1
                counts[fs] = new_c
        return delta_uncovered
    
    # Simulated annealing parameters
    T0 = 5.0
    T_min = 0.01
    cooling_rate = 0.995
    temperature = T0
    
    no_improve_iter = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove redundant blocks (always beneficial)
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for fs in get_triples(b):
                if counts.get(fs, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(b, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    no_improve_iter = 0
                    break
                else:
                    apply_block_change(b, 1)
                    uncovered -= delta_u
        
        if removed_any:
            temperature = max(T_min, temperature * cooling_rate)
            continue
        
        # 2. Try swaps with simulated annealing acceptance
        if len(blocks) > 0:
            improved = False
            for attempt in range(5):
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                
                # Find t-subsets only covered by this block
                critical = []
                for fs in get_triples(old_block):
                    if counts.get(fs, 0) == 1:
                        critical.append(fs)
                
                if not critical:
                    continue
                
                points_needed = set()
                for fs in critical:
                    points_needed.update(fs)
                
                if len(points_needed) > k:
                    continue
                
                # Try to complete points_needed to a k-set
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                # Greedily fill remaining spots to maximize new coverage
                while len(new_block_set) < k:
                    best_x = None
                    best_gain = -1
                    for x in cands:
                        if x in new_block_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(new_block_set, t-1):
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
                    
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    if new_block != old_block:
                        delta_u_remove = apply_block_change(old_block, -1)
                        delta_u_add = apply_block_change(new_block, 1)
                        
                        new_uncovered = uncovered + delta_u_remove + delta_u_add
                        # Cost change: 0 for swap (same number of blocks), but we care about uncovered
                        cost_delta = new_uncovered - uncovered
                        
                        if new_uncovered == 0:
                            blocks[i] = new_block
                            uncovered = new_uncovered
                            no_improve_iter = 0
                            improved = True
                            break
                        else:
                            # Accept worse solutions with probability exp(-cost_delta/T)
                            if cost_delta > 0 and temperature > T_min:
                                if rng.random() < math.exp(-cost_delta / temperature):
                                    blocks[i] = new_block
                                    uncovered = new_uncovered
                                    no_improve_iter = 0
                                    improved = True
                                    break
                            else:
                                # Reject
                                apply_block_change(new_block, -1)
                                apply_block_change(old_block, 1)
                                uncovered = uncovered - delta_u_remove - delta_u_add
                        # If we rejected, restore counts
                        if not improved:
                            # We already restored in the else branch above for rejection
                            pass
        
        if improved:
            temperature = max(T_min, temperature * cooling_rate)
            continue
        
        # 3. Perturbation if stuck: remove several blocks and greedily rebuild
        if no_improve_iter > max_no_improve:
            if len(blocks) > 3:
                old_blocks = list(blocks)
                old_counts = dict(counts)
                old_uncovered = uncovered
                
                # Remove 2-4 random blocks
                num_remove = min(rng.randint(2, 4), len(blocks) - 1)
                remove_indices = rng.sample(range(len(blocks)), num_remove)
                remove_indices.sort(reverse=True)
                
                for i in remove_indices:
                    b = blocks[i]
                    delta_u = apply_block_change(b, -1)
                    uncovered += delta_u
                    blocks.pop(i)
                
                # Find all uncovered t-subsets
                unc_list = []
                for c in itertools.combinations(range(v), t):
                    f = frozenset(c)
                    if counts.get(f, 0) == 0:
                        unc_list.append(f)
                
                # Greedily add blocks to cover uncovered
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
                            for rest in itertools.combinations(new_b, t-1):
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
                        delta_u_add = apply_block_change(new_b_tuple, 1)
                        uncovered += delta_u_add
                        
                        # Recompute uncovered list
                        unc_list = []
                        for c in itertools.combinations(range(v), t):
                            f = frozenset(c)
                            if counts.get(f, 0) == 0:
                                unc_list.append(f)
                    
                    if not unc_list:
                        break
                
                if uncovered == 0:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    no_improve_iter = 0
                    temperature = max(T_min, temperature * 0.9)
                else:
                    # Restore old state
                    blocks = old_blocks
                    counts = old_counts
                    uncovered = old_uncovered
                    no_improve_iter += 1
                    temperature = max(T_min, temperature * 0.9)
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1
            temperature = max(T_min, temperature * cooling_rate)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
