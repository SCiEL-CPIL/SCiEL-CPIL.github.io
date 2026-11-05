import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s kill

    # Map every t-subset to an integer id
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # Precompute masks for fast gain calculation
    # masks[x] is a dict: key is a (t-1)-tuple of points, value is a bitmask of t-subsets
    # that contain x and that (t-1)-tuple.
    masks = [dict() for _ in range(v)]
    
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i in range(t):
            others = s[:i] + s[i+1:]
            key = others 
            x = s[i]
            masks[x][key] = masks[x].get(key, 0) | bit

    covered = 0  # Bitset of covered t-subsets
    blocks = []
    block_set = set()  # To avoid duplicate blocks

    # Helper: Count bits in an integer
    def popcount(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    def build_block_from(seed_points):
        """
        Grow a block from initial set (seed_points) to size k by max-gain greedy.
        Uses incremental mask accumulation for speed.
        Returns the block as a list of ints.
        """
        B = list(seed_points)
        B_set = set(B)
        available = [x for x in range(v) if x not in B_set]
        uncov = ~covered  # complement mask for speed
        
        # Precompute (t-1) subsets of current B as a list of tuples
        # For each candidate x, gain_mask[x] accumulates the OR of masks[x][S] for S in C(B, t-1)
        # We'll maintain this incrementally.
        
        # Initialize gain masks for all available candidates
        # gain_mask[x] = OR over (t-1)-subsets of B of masks[x][S]
        # Start with B of size t: C(t, t-1) = t subsets
        gain_masks = {}
        for x in available:
            gm = 0
            for S in itertools.combinations(B, t - 1):
                m = masks[x].get(S)
                if m is not None:
                    gm |= m
            gain_masks[x] = gm
        
        while len(B) < k:
            best_x = -1
            best_gain = -1
            # Evaluate all available points using precomputed gain masks
            for x in available:
                g = popcount(gain_masks[x] & uncov)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x == -1:
                break
            
            B.append(best_x)
            B_set.add(best_x)
            available.remove(best_x)
            
            # Update gain masks for remaining candidates
            # New (t-1) subsets are formed by taking (t-2) subsets of old B and adding best_x
            # Actually, the new (t-1)-subsets of new B that weren't in old B are those containing best_x
            # These are: {best_x} union S where S is a (t-2)-subset of old B
            for x in available:
                gm = gain_masks[x]
                for S in itertools.combinations(B[:-1], t - 2):
                    key = S + (best_x,)
                    m = masks[x].get(key)
                    if m is not None:
                        gm |= m
                gain_masks[x] = gm
            
        return B

    def total_new_coverage(block):
        """
        Count how many t-subsets in the block are newly covered.
        Uses precomputed mask for the block.
        """
        # Build the full mask for this block
        block_mask = 0
        for s in itertools.combinations(sorted(block), t):
            block_mask |= 1 << subset_to_index[s]
        return popcount(block_mask & ~covered)

    def random_seed_block():
        """
        Construct a block by picking random points, then greedily filling.
        This explores the space differently than seeding from uncovered triples.
        """
        B = random.sample(range(v), t)
        return build_block_from(B)

    # Main Loop: Hybrid Strong Greedy with Randomized Seeds
    all_points = list(range(v))
    
    # Adaptive seed count: start high, decrease if time is tight
    # But we want to keep quality high. Let's use a fixed robust number.
    SEEDS_PER_ROUND = 120
    
    while True:
        if time.time() > deadline:
            break

        # Check if all covered
        uncov_mask = ((1 << n_subsets) - 1) & ~covered
        if uncov_mask == 0:
            break

        # Collect uncovered subset indices
        uncovered_indices = []
        m = uncov_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            m ^= lsb
            if len(uncovered_indices) >= 500:  # Cap to avoid long list generation
                break
        
        if not uncovered_indices:
            break

        best_block = None
        best_gain_total = -1
        
        # Strategy: Mix of seeds from uncovered triples and random seeds
        # This helps escape local optima where greedy-from-uncovered might converge to same block structure
        
        # 1. Sample from uncovered triples
        n_uncov_seeds = SEEDS_PER_ROUND // 2
        # 2. Sample random seeds
        n_rand_seeds = SEEDS_PER_ROUND - n_uncov_seeds
        
        # Shuffle uncovered to randomize selection
        random.shuffle(uncovered_indices)
        selected_uncov = uncovered_indices[:n_uncov_seeds]
        
        for _ in range(n_rand_seeds):
            if time.time() > deadline:
                break
            # Random seed: pick t random points
            seed = random.sample(range(v), t)
            B = build_block_from(seed)
            bl = tuple(sorted(B))
            if bl in block_set:
                continue
            total = total_new_coverage(B)
            if total > best_gain_total:
                best_gain_total = total
                best_block = B
        
        for idx in selected_uncov:
            if time.time() > deadline:
                break
            seed = all_subsets[idx]
            B = build_block_from(list(seed))
            bl = tuple(sorted(B))
            if bl in block_set:
                continue
            total = total_new_coverage(B)
            if total > best_gain_total:
                best_gain_total = total
                best_block = B
        
        # Additional pass: if we have time, try more random seeds
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining > 5.0 and best_gain_total >= 0:
            extra_seeds = min(40, int(remaining * 8))
            for _ in range(extra_seeds):
                if time.time() > deadline - 2.0:
                    break
                seed = random.sample(range(v), t)
                B = build_block_from(seed)
                bl = tuple(sorted(B))
                if bl in block_set:
                    continue
                total = total_new_coverage(B)
                if total > best_gain_total:
                    best_gain_total = total
                    best_block = B
        
        if best_block is None:
            # If no new block found with gain > 0, we might be stuck or done.
            # If uncov_mask != 0, we are stuck. 
            # Try a few more random restarts before giving up
            found = False
            for _ in range(30):
                if time.time() > deadline:
                    break
                seed = random.sample(range(v), t)
                B = build_block_from(seed)
                bl = tuple(sorted(B))
                if bl in block_set:
                    continue
                total = total_new_coverage(B)
                if total > 0:
                    best_block = B
                    best_gain_total = total
                    found = True
                    break
            if not found:
                break

        bl = sorted(best_block)
        
        # Check if this block is already in our list
        if tuple(bl) in block_set:
            # This shouldn't happen if we checked above, but safe guard
            # If it does, we need to find another block. 
            # To keep it simple, we break if we can't find a new one.
            # A more robust approach would be to continue searching for a non-duplicate block.
            # Given time constraints, we assume the gain check filters out most duplicates 
            # effectively (gain 0 for duplicate).
            # Wait, if gain > 0, it cannot be a duplicate of a previous block that already covered those subsets?
            # No, a duplicate block covers the same subsets. If those subsets are covered, gain is 0.
            # If gain > 0, it covers new subsets, so it's not a duplicate of a block that only covered old subsets.
            # It *could* be a duplicate of a block that covered *some* new subsets? No, if it's the same block, it covers the same subsets.
            # If it was added before, those subsets are covered. So gain would be 0.
            # So gain > 0 implies it's a new block.
            pass

        # Mark covered
        for s in itertools.combinations(bl, t):
            covered |= 1 << subset_to_index[s]
            
        blocks.append(bl)
        block_set.add(tuple(bl))

    return blocks
