import time
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered_mask = (1 << num_t_subsets) - 1
        blocks = []
        
        while uncovered_mask:
            if time.time() - start_time > deadline:
                return None # Signal timeout
            
            # Pick an uncovered subset
            # To introduce variation, we can pick a random uncovered subset
            # Or the first one. Let's pick a random one from the uncovered set.
            # Generating a random index in the set of set bits of a large int is tricky.
            # Alternative: keep a list of uncovered indices and pop from it?
            # Updating the list is O(N).
            # Let's stick to picking the lowest set bit for speed, 
            # but randomize the *tie-breaking* of points.
            
            low_bit = uncovered_mask & -uncovered_mask
            start_idx = low_bit.bit_length() - 1
            start_subset = all_t_subsets[start_idx]
            
            block = list(start_subset)
            
            while len(block) < k:
                gains = []
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    if len(block) >= t - 1:
                        for rest in itertools.combinations(block, t - 1):
                            new_subset = tuple(sorted(rest + (x,)))
                            idx = subset_to_idx[new_subset]
                            if (uncovered_mask >> idx) & 1:
                                gain += 1
                    gains.append((gain, x))
                
                # Find max gain
                max_gain = max(g[0] for g in gains)
                # Randomly pick one of the points with max gain
                candidates = [x for g, x in gains if g == max_gain]
                best_point = random.choice(candidates)
                
                block.append(best_point)
            
            block.sort()
            
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered_mask &= ~(1 << idx)
            
            blocks.append(block)
            
        return blocks

    best_blocks = None
    # Try a few seeds
    for i in range(10):
        blocks = run_greedy(i)
        if blocks is None:
            break
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback to a very simple greedy if all timed out?
        # Or just return empty? No, must be valid.
        # The first run usually finishes fast.
        # If it times out, it means the greedy is too slow.
        # But with bitsets, it should be fast.
        # Let's ensure we return something valid if we fail.
        # Re-run with seed 0 but with a tighter deadline?
        # Actually, if run_greedy returns None, it means it timed out mid-way.
        # We should probably return the partial result if it's valid?
        # No, partial result is not a valid covering.
        # We must ensure we finish.
        # The bitset approach is O(Blocks * k * v * C(k, t-1)).
        # 76 * 7 * 24 * 15 ~ 190,000 operations. Very fast.
        # So it shouldn't time out.
        
        # If we get here, something went wrong. Return a trivial (but large) covering?
        # Or just the first block? No.
        # Let's just return the result of a single deterministic run without random choice to be safe.
        uncovered_mask = (1 << num_t_subsets) - 1
        blocks = []
        while uncovered_mask:
            low_bit = uncovered_mask & -uncovered_mask
            start_idx = low_bit.bit_length() - 1
            start_subset = all_t_subsets[start_idx]
            block = list(start_subset)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    if len(block) >= t - 1:
                        for rest in itertools.combinations(block, t - 1):
                            new_subset = tuple(sorted(rest + (x,)))
                            idx = subset_to_idx[new_subset]
                            if (uncovered_mask >> idx) & 1:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            block.sort()
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered_mask &= ~(1 << idx)
            blocks.append(block)
        return blocks

    return best_blocks
