"""Direct structural construction for covering design C(24,7,3)."""
import itertools
import random
import time
import math


def _precompute_triples(v, t):
    """Precompute all t-subsets and map them to indices."""
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {frozenset(c): i for i, c in enumerate(triples)}
    return triples, triple_to_idx


def _block_triple_indices(block, t, triple_to_idx):
    """Get the indices of all t-subsets covered by a block."""
    return [triple_to_idx[frozenset(c)] for c in itertools.combinations(block, t)]


def _greedy_cover_bitmask(v, k, t, rng, triples, triple_to_idx, num_triples):
    """
    Bitmask-accelerated greedy covering.
    Uses a set of uncovered triple indices for fast gain computation.
    """
    uncovered = set(range(num_triples))
    blocks = []
    seen_blocks = set()
    
    while uncovered:
        # Pick a random uncovered triple as seed (randomization for diversity)
        seed_triple = next(iter(uncovered))
        seed_set = triples[seed_triple]
        block = list(seed_set)
        
        # Greedily extend: always add the point that covers most new uncovered triples
        while len(block) < k:
            best_x = None
            best_gain = -1
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: how many new uncovered triples does x add?
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    c = tuple(sorted(rest + (x,)))
                    idx = triple_to_idx.get(frozenset(c))
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_x = rng.choice(candidates)
                else:
                    break
            
            if best_x is not None:
                block.append(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in seen_blocks:
            seen_blocks.add(b_tuple)
            blocks.append(b_tuple)
            # Remove covered triples from uncovered
            for c in itertools.combinations(block, t):
                idx = triple_to_idx.get(frozenset(c))
                if idx is not None:
                    uncovered.discard(idx)
                
    return blocks


def _greedy_cover_ordered(v, k, t, rng, triples, triple_to_idx, num_triples):
    """
    Greedy with demand-based ordering: prioritize triples that are "harder" to cover.
    A triple's demand is estimated by how many potential blocks could cover it.
    """
    uncovered = set(range(num_triples))
    blocks = []
    seen_blocks = set()
    
    # Precompute for each triple, the number of k-subsets that contain it
    # C(v-t, k-t) is the same for all triples, so use a different heuristic:
    # Process triples in order of how many times their elements appear in already-placed blocks
    
    # Actually, let's use a simpler approach: randomly shuffle the processing order
    triple_order = list(range(num_triples))
    rng.shuffle(triple_order)
    
    for seed_idx in triple_order:
        if seed_idx not in uncovered:
            continue
        
        seed_set = triples[seed_idx]
        block = list(seed_set)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    c = tuple(sorted(rest + (x,)))
                    idx = triple_to_idx.get(frozenset(c))
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_x = rng.choice(candidates)
                else:
                    break
            
            if best_x is not None:
                block.append(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in seen_blocks:
            seen_blocks.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                idx = triple_to_idx.get(frozenset(c))
                if idx is not None:
                    uncovered.discard(idx)
        
        if not uncovered:
            break
    
    return blocks


def _local_search(blocks, v, k, t, triples, triple_to_idx, num_triples, time_limit):
    """
    Local search to reduce block count.
    Uses bitmask-style counting for fast validity checks.
    """
    start_time = time.time()
    best = list(blocks)
    best_n = len(best)
    rng = random.Random(42)
    
    def compute_counts(blk_list):
        """Compute coverage counts for each triple."""
        counts = [0] * num_triples
        for b in blk_list:
            for c in itertools.combinations(b, t):
                idx = triple_to_idx[frozenset(c)]
                counts[idx] += 1
        return counts
    
    def is_valid(blk_list):
        """Check if blk_list is a valid covering."""
        if len(blk_list) > num_triples:
            return False
        seen = set()
        for b in blk_list:
            bs = tuple(sorted(b))
            if bs in seen:
                return False
            seen.add(bs)
        counts = compute_counts(blk_list)
        return all(c >= 1 for c in counts)
    
    def try_remove(blk_list):
        """Try to remove one redundant block."""
        counts = compute_counts(blk_list)
        
        for i, b in enumerate(blk_list):
            removable = True
            for c in itertools.combinations(b, t):
                idx = triple_to_idx[frozenset(c)]
                if counts[idx] <= 1:
                    removable = False
                    break
            if removable:
                return blk_list[:i] + blk_list[i+1:]
        return None
    
    def try_swap(blk_list):
        """Try to swap one block with a modified version."""
        counts = compute_counts(blk_list)
        
        # Find blocks with high waste (cover many over-covered triples)
        def waste_score(b):
            waste = 0
            for c in itertools.combinations(b, t):
                idx = triple_to_idx[frozenset(c)]
                if counts[idx] > 2:
                    waste += 1
            return -waste  # Lower is worse (more waste)
        
        # Pick a random block to modify
        i = rng.randrange(len(blk_list))
        old_b = list(blk_list[i])
        old_set = set(old_b)
        
        # Create a new block by swapping 1-2 elements
        new_set = set(old_b)
        num_swaps = rng.choice([1, 1, 2])
        
        for _ in range(num_swaps):
            # Remove a random element
            if new_set:
                removed = rng.choice(list(new_set))
                new_set.discard(removed)
                # Add a random new element
                candidates = [x for x in range(v) if x not in new_set]
                if candidates:
                    new_set.add(rng.choice(candidates))
        
        if new_set == old_set:
            return None
        
        new_b = tuple(sorted(new_set))
        if new_b in blk_list:
            return None
        
        new_list = blk_list[:]
        new_list[i] = new_b
        
        if is_valid(new_list):
            return new_list
        return None
    
    def try_merge_replace(blk_list):
        """Try to replace two blocks with one better block (if possible)."""
        if len(blk_list) < 2:
            return None
        
        # Pick two random blocks
        i, j = rng.sample(range(len(blk_list)), 2)
        b1 = set(blk_list[i])
        b2 = set(blk_list[j])
        
        # Try to create a single block that covers the union of their essential coverage
        # This is complex; skip for now
        return None
    
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Try removal first
        removed = try_remove(blocks)
        if removed is not None:
            blocks = removed
            if len(blocks) < best_n:
                best_n = len(blocks)
                best = list(blocks)
            continue
        
        # Try swap
        swapped = try_swap(blocks)
        if swapped is not None:
            blocks = swapped
            if len(blocks) < best_n:
                best_n = len(blocks)
                best = list(blocks)
    
    return best, best_n


def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for 120s limit
    
    # Precompute
    triples, triple_to_idx = _precompute_triples(v, t)
    num_triples = len(triples)
    
    best_blocks = None
    best_n = float('inf')
    
    # Multiple randomized restarts with different strategies
    strategies = [
        ("random", _greedy_cover_bitmask),
        ("ordered", _greedy_cover_ordered),
        ("random", _greedy_cover_bitmask),
        ("ordered", _greedy_cover_ordered),
    ]
    
    for strategy_name, greedy_func in strategies:
        if time.time() - start_time > time_limit - 10:
            break
        
        rng = random.Random(hash((strategy_name, len(strategies))) % (2**32))
        blocks = greedy_func(v, k, t, rng, triples, triple_to_idx, num_triples)
        
        if len(blocks) < best_n:
            best_n = len(blocks)
            best_blocks = blocks
        
        # Local search on this result
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            blocks, n = _local_search(blocks, v, k, t, triples, triple_to_idx, num_triples, remaining_time)
            if n < best_n:
                best_n = n
                best_blocks = blocks
    
    # Final local search on the best
    if best_blocks is not None:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 2:
            best_blocks, best_n = _local_search(best_blocks, v, k, t, triples, triple_to_idx, num_triples, remaining_time)
    
    if best_blocks is None:
        # Fallback: simple greedy
        rng = random.Random(12345)
        best_blocks = _greedy_cover_bitmask(v, k, t, rng, triples, triple_to_idx, num_triples)
        best_n = len(best_blocks)
    
    return [list(b) for b in best_blocks]


if __name__ == "__main__":
    import time
    start = time.time()
    result = construct(24, 7, 3)
    elapsed = time.time() - start
    print(f"Blocks: {len(result)}, Time: {elapsed:.2f}s")
    # Verify
    covered = set()
    for b in result:
        for c in itertools.combinations(b, 3):
            covered.add(frozenset(c))
    total = len(list(itertools.combinations(range(24), 3)))
    print(f"Covered: {len(covered)}/{total}")
    assert len(covered) == total, "Invalid covering!"
