"""Randomized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _get_block_triples(block, subset_to_block_idx):
    """Compute list of triple indices covered by a block."""
    return [subset_to_block_idx[frozenset(c)] for c in itertools.combinations(block, 3)]

def _greedy_cover(v, k, t, rng, subset_to_block_idx):
    """Fast greedy construction."""
    uncovered = set(subset_to_block_idx.values())
    blocks = []
    
    while uncovered:
        seed_idx = rng.choice(list(uncovered))
        # Recover the tuple from index? No, we need the points.
        # We stored indices. Let's map index -> points for greedy seed.
        # Actually, let's just pick a random uncovered triple points.
        # We need a reverse map or just iterate.
        # For speed, let's keep a list of uncovered triples points.
        # This requires changing state.
        
        # Simplified greedy: Pick a random uncovered triple index, get its points.
        # We need points for the seed.
        # Let's assume subset_to_block_idx maps frozenset -> index.
        # We need index -> frozenset.
        # Let's pass a reverse map or just use the points from the frozenset.
        # Wait, the original code had tsubset_indices. I will recreate that structure in construct.
        
        # For now, this helper assumes we can get points.
        # I will refactor construct to hold `tsubset_points` list.
        
        # Let's rewrite _greedy_cover to take tsubset_points.
        pass

def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 110.0  # Safety margin

    # Precompute all t-subsets
    tsubset_points = []
    subset_to_idx = {}
    for c in itertools.combinations(range(v), t):
        fs = frozenset(c)
        subset_to_idx[fs] = len(tsubset_points)
        tsubset_points.append(c)
    
    total_t = len(tsubset_points)
    
    # Greedy Cover
    def greedy_cover():
        uncovered = set(range(total_t))
        blocks = []
        while uncovered:
            seed_idx = rng.choice(list(uncovered))
            seed = tsubset_points[seed_idx]
            block = list(seed)
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
                        idx = subset_to_idx.get(fs)
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
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(subset_to_idx[frozenset(c)])
        return blocks

    blocks = greedy_cover()
    
    # Initialize counts
    counts = [0] * total_t
    block_triples_cache = {}
    for b in blocks:
        tris = _get_block_triples(b, subset_to_idx)
        block_triples_cache[tuple(b)] = tris
        for idx in tris:
            counts[idx] += 1
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    def remove_redundant():
        removed = 0
        i = 0
        while i < len(blocks):
            b = blocks[i]
            is_red = True
            for idx in block_triples_cache[tuple(b)]:
                if counts[idx] <= 1:
                    is_red = False
                    break
            if is_red:
                for idx in block_triples_cache[tuple(b)]:
                    counts[idx] -= 1
                blocks.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    def try_swap(i):
        old_b = blocks[i]
        old_tris = block_triples_cache[tuple(old_b)]
        
        # Find critical points (from triples with count 1)
        critical_points = set()
        for idx in old_tris:
            if counts[idx] == 1:
                critical_points.update(tsubset_points[idx])
        
        if len(critical_points) > k:
            return False
        
        new_set = set(critical_points)
        cands = [x for x in range(v) if x not in new_set]
        rng.shuffle(cands)
        
        for x in cands:
            if len(new_set) >= k:
                break
            # Greedy fill
            gain = 0
            for rest in itertools.combinations(new_set, t-1):
                fs = frozenset(list(rest) + [x])
                idx = subset_to_idx.get(fs)
                if idx is not None and counts[idx] == 1:
                    gain += 1
            if gain > 0:
                new_set.add(x)
            elif len(new_set) < k:
                new_set.add(x)
        
        if len(new_set) != k:
            return False
        
        new_b = tuple(sorted(new_set))
        if new_b == tuple(old_b):
            return False
        
        new_tris = _get_block_triples(new_b, subset_to_idx)
        new_tris_set = set(new_tris)
        
        # Check validity: all count==1 triples of old_b must be in new_tris
        valid = True
        for idx in old_tris:
            if counts[idx] == 1 and idx not in new_tris_set:
                valid = False
                break
        
        if not valid:
            return False
        
        # Apply
        for idx in old_tris:
            counts[idx] -= 1
        for idx in new_tris:
            counts[idx] += 1
        
        blocks[i] = list(new_b)
        block_triples_cache[new_b] = new_tris
        # Clean up cache for old_b if not used elsewhere? 
        # Since blocks are distinct, we can just leave it or delete.
        # Deleting is safer for memory but not critical.
        return True

    no_improve = 0
    while time.time() - start_time < time_limit:
        # 1. Remove redundants
        if remove_redundant() > 0:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
            no_improve = 0
            continue
        
        # 2. Try swaps
        improved = False
        for _ in range(20): # Try a few swaps
            if not blocks: break
            i = rng.randrange(len(blocks))
            if try_swap(i):
                improved = True
                # After a swap, new redundancies might appear
                if remove_redundant() > 0:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                break
        
        if not improved:
            no_improve += 1
            if no_improve > 100:
                # Restart
                new_blocks = greedy_cover()
                if len(new_blocks) < len(blocks):
                    blocks = [list(b) for b in new_blocks]
                    counts = [0] * total_t
                    block_triples_cache = {}
                    for b in blocks:
                        tris = _get_block_triples(b, subset_to_idx)
                        block_triples_cache[tuple(b)] = tris
                        for idx in tris:
                            counts[idx] += 1
                    remove_redundant()
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                else:
                    no_improve += 50
    
    remove_redundant()
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = [list(b) for b in blocks]
        
    return best_blocks
# EVOLVE-BLOCK-END
