import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # 150s kill limit. Use 145s as deadline to be safe.
    deadline = start_time + 145.0

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    def get_id(triplet_tuple):
        return triplet_to_id[triplet_tuple]

    # Helper to compute the set of triplet IDs covered by a block
    def block_triplet_ids(block):
        return set(get_id(triplet) for triplet in itertools.combinations(block, t))

    def greedy_construct(v, k, t, seed=0, bias=0.0, target_start=None):
        """
        Greedy construction.
        bias: 0.0 means pure greedy (max gain). 1.0 means random.
        target_start: if provided, start with this triplet.
        """
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        while uncovered:
            # Pick a starting triplet
            if target_start is not None:
                start_triplet_id = target_start
                target_start = None # Only use once? Or maybe we should just pick random after?
                # Actually, for diversity, let's just pick random usually.
                # But if we want to target specific holes, we'd pass it.
                # For now, let's stick to random start for the initial greedy.
            
            # Pick a random uncovered triplet
            # Optimization: picking a random one is O(n) if we use list(uncovered).
            # For v=24, n=2024. This is fast.
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                best_point = -1
                max_gain = -1
                second_best_gain = -1
                second_best_point = -1
                
                # Pairs in current block (for t=3)
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                gains = {}
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    gains[p] = gain
                    if gain > max_gain:
                        second_best_gain = max_gain
                        second_best_point = best_point
                        max_gain = gain
                        best_point = p
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = p

                if max_gain == 0:
                    # No gain, pick random
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    # Mix greedy and random based on bias
                    r = rng.random()
                    if r < (1.0 - bias):
                        pass # keep best_point
                    elif second_best_point != -1 and r < (1.0 - bias) + bias * 0.5:
                        best_point = second_best_point
                    else:
                        candidates = [p for p in range(v) if p not in block_set]
                        best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
        
        return blocks

    # 1. Initial Construction
    # We want to find a good starting point.
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Pure Greedy with different seeds
    for seed in range(20):
        if time.time() > deadline - 60:
            break
        blocks = greedy_construct(v, k, t, seed, bias=0.0)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    # Phase 2: Randomized Greedy
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        bias = 0.2 + 0.05 * seed
        blocks = greedy_construct(v, k, t, seed, bias=bias)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(v, k, t, 0, bias=0.0)

    # 2. Local Search Optimization
    current_blocks = list(best_blocks)
    
    # Precompute triplet IDs for each block to speed up local search
    def get_block_ids(blocks):
        return [block_triplet_ids(b) for b in blocks]

    # We maintain counts of how many times each triplet is covered
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                count[tid] += delta

    update_counts(current_blocks, 1)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[get_id(triplet)] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = block_triplet_ids(b1)
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = block_triplet_ids(b2)
                
                critical_points = set()
                
                # Check b1's triplets
                for tid in ids_b1:
                    c = count[tid]
                    covered_by_b1 = 1
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        critical_points.update(all_triplets[tid])
                        
                # Check b2's triplets (only those not in b1)
                for tid in ids_b2:
                    if tid in ids_b1:
                        continue
                    c = count[tid]
                    if c == 1:
                        critical_points.update(all_triplets[tid])
                
                if not critical_points:
                    # Both blocks are redundant together
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
                
                if len(critical_points) <= k:
                    # We can form a block containing all critical points
                    new_block = list(critical_points)
                    # Fill with random points
                    # To make it deterministic-ish, use a fixed seed based on i,j?
                    # Or just random.
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()
                    
                    # Verify validity (should be valid by construction of critical points)
                    # But we must ensure new_block doesn't break other constraints?
                    # The condition for validity is that ALL previously covered triplets remain covered.
                    # We checked that the "critical" ones (those with count dropping to 0) are in new_block.
                    # Triplets with count > critical threshold remain covered by other blocks.
                    # So this swap is valid.
                    
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    # Main Optimization Loop
    progress_made = True
    while progress_made and time.time() < deadline - 2:
        progress_made = False
        
        # 1. Remove redundants
        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True
            
        # 2. Swap 2 for 1
        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True

    # Final pass to ensure no redundants remain
    current_blocks, _ = try_remove_redundant(current_blocks)

    return current_blocks

if __name__ == "__main__":
    pass
