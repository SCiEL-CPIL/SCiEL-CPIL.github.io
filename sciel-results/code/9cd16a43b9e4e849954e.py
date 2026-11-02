"""Direct structural construction for covering design C(24,7,3)."""
import itertools
import random
import time
import math
import numpy as np


def _precompute_triples(v, t):
    """Precompute all t-subsets with fast 3D array lookup."""
    triples = list(itertools.combinations(range(v), t))
    # Build 3D array: triple_idx[a,b,c] for 0<=a<b<c<v
    triple_idx_arr = np.full((v, v, v), -1, dtype=np.int32)
    for i, (a, b, c) in enumerate(triples):
        triple_idx_arr[a, b, c] = i
    return triples, triple_idx_arr


def _block_triple_indices(block, t, triple_idx_arr):
    """Get the indices of all t-subsets covered by a block using fast array lookup."""
    return [int(triple_idx_arr[a, b, c]) for a, b, c in itertools.combinations(block, t)]


def _greedy_cover_bitmask(v, k, t, rng, triples, triple_idx_arr, num_triples):
    """
    Greedy covering with fast array-based triple lookup.
    """
    uncovered = set(range(num_triples))
    blocks = []
    seen_blocks = set()
    
    while uncovered:
        seed_triple = next(iter(uncovered))
        seed_set = triples[seed_triple]
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
                    a, b = rest
                    if x > a and x > b:
                        if a < b:
                            idx = int(triple_idx_arr[a, b, x])
                        else:
                            idx = int(triple_idx_arr[b, a, x])
                    elif x < a and x < b:
                        if a < b:
                            idx = int(triple_idx_arr[x, a, b])
                        else:
                            idx = int(triple_idx_arr[x, b, a])
                    else:
                        if a < x:
                            idx = int(triple_idx_arr[a, x, b])
                        else:
                            idx = int(triple_idx_arr[b, x, a])
                    if idx >= 0 and idx in uncovered:
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
                idx = int(triple_idx_arr[c[0], c[1], c[2]])
                if idx >= 0:
                    uncovered.discard(idx)
                
    return blocks


def _greedy_cover_ordered(v, k, t, rng, triples, triple_idx_arr, num_triples):
    """
    Greedy with shuffled triple ordering and fast array lookup.
    """
    uncovered = set(range(num_triples))
    blocks = []
    seen_blocks = set()
    
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
                    a, b = rest
                    if x > a and x > b:
                        if a < b:
                            idx = int(triple_idx_arr[a, b, x])
                        else:
                            idx = int(triple_idx_arr[b, a, x])
                    elif x < a and x < b:
                        if a < b:
                            idx = int(triple_idx_arr[x, a, b])
                        else:
                            idx = int(triple_idx_arr[x, b, a])
                    else:
                        if a < x:
                            idx = int(triple_idx_arr[a, x, b])
                        else:
                            idx = int(triple_idx_arr[b, x, a])
                    if idx >= 0 and idx in uncovered:
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
                idx = int(triple_idx_arr[c[0], c[1], c[2]])
                if idx >= 0:
                    uncovered.discard(idx)
        
        if not uncovered:
            break
    
    return blocks


def _local_search(blocks, v, k, t, triples, triple_idx_arr, num_triples, time_limit):
    """
    Local search to reduce block count with fast numpy-based counting.
    """
    start_time = time.time()
    best = list(blocks)
    best_n = len(best)
    rng = random.Random(42)
    
    # Precompute triple indices for each possible block (all C(24,7) blocks)
    # This is 346105 blocks, each with 35 triples -> precompute as flat array
    all_blocks = list(itertools.combinations(range(v), k))
    num_all_blocks = len(all_blocks)
    # Map each block tuple to its index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    # Precompute triple indices for each block: shape (num_all_blocks, C(k,t))
    num_triples_per_block = math.comb(k, t)
    block_triples = np.zeros((num_all_blocks, num_triples_per_block), dtype=np.int32)
    for bi, b in enumerate(all_blocks):
        for ti, c in enumerate(itertools.combinations(b, t)):
            block_triples[bi, ti] = int(triple_idx_arr[c[0], c[1], c[2]])
    
    def compute_counts(blk_list):
        """Compute coverage counts using numpy."""
        counts = np.zeros(num_triples, dtype=np.int32)
        for b in blk_list:
            bi = block_to_idx[b]
            counts += np.bincount(block_triples[bi], minlength=num_triples)
        return counts
    
    def is_valid(blk_list):
        """Check if blk_list is a valid covering."""
        if len(blk_list) > num_triples:
            return False
        seen = set()
        for b in blk_list:
            if b in seen:
                return False
            seen.add(b)
        counts = compute_counts(blk_list)
        return bool(np.all(counts >= 1))
    
    def try_remove(blk_list):
        """Try to remove one redundant block."""
        counts = compute_counts(blk_list)
        
        for i, b in enumerate(blk_list):
            bi = block_to_idx[b]
            # Check if all triples in this block have count > 1
            b_counts = counts[block_triples[bi]]
            if np.all(b_counts > 1):
                return blk_list[:i] + blk_list[i+1:]
        return None
    
    def try_swap(blk_list):
        """Try to swap one block with a modified version."""
        counts = compute_counts(blk_list)
        
        i = rng.randrange(len(blk_list))
        old_b = blk_list[i]
        old_set = set(old_b)
        
        # Create a new block by swapping 1-2 elements
        new_set = set(old_b)
        num_swaps = rng.choice([1, 1, 2])
        
        for _ in range(num_swaps):
            if new_set:
                removed = rng.choice(tuple(new_set))
                new_set.discard(removed)
                candidates = [x for x in range(v) if x not in new_set]
                if candidates:
                    new_set.add(rng.choice(candidates))
        
        if new_set == old_set:
            return None
        
        new_b = tuple(sorted(new_set))
        if new_b in blk_list:
            return None
        
        new_list = list(blk_list)
        new_list[i] = new_b
        
        if is_valid(new_list):
            return new_list
        return None
    
    while time.time() - start_time < time_limit:
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
    triples, triple_idx_arr = _precompute_triples(v, t)
    num_triples = len(triples)
    
    best_blocks = None
    best_n = float('inf')
    
    # Multiple randomized restarts with different strategies
    strategies = [
        ("random", _greedy_cover_bitmask, 42),
        ("ordered", _greedy_cover_ordered, 137),
        ("random", _greedy_cover_bitmask, 256),
        ("ordered", _greedy_cover_ordered, 789),
    ]
    
    for strategy_name, greedy_func, seed in strategies:
        if time.time() - start_time > time_limit - 10:
            break
        
        rng = random.Random(seed)
        blocks = greedy_func(v, k, t, rng, triples, triple_idx_arr, num_triples)
        
        if len(blocks) < best_n:
            best_n = len(blocks)
            best_blocks = list(blocks)
        
        # Local search on this result
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            blocks, n = _local_search(blocks, v, k, t, triples, triple_idx_arr, num_triples, remaining_time)
            if n < best_n:
                best_n = n
                best_blocks = list(blocks)
    
    # Final local search on the best
    if best_blocks is not None:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 2:
            best_blocks, best_n = _local_search(best_blocks, v, k, t, triples, triple_idx_arr, num_triples, remaining_time)
    
    if best_blocks is None:
        # Fallback: simple greedy
        rng = random.Random(12345)
        best_blocks = _greedy_cover_bitmask(v, k, t, rng, triples, triple_idx_arr, num_triples)
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
