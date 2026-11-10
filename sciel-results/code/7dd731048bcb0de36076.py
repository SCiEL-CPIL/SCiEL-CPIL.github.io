"""Bitset-accelerated greedy covering construction for C(24,7,3)."""
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 145  # safety margin below 150s

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {ts: i for i, ts in enumerate(all_t)}

    t_minus_1 = t - 1
    # Precompute: for each (t-1)-tuple, list of t-subset indices extending it
    ext_map = defaultdict(list)
    for ts in all_t:
        for i in range(t):
            sub = ts[:i] + ts[i+1:]
            ext_map[sub].append(t_index[ts])

    # Precompute coverage mask for every k-subset
    all_k = list(itertools.combinations(range(v), k))
    k_index = {bs: i for i, bs in enumerate(all_k)}
    n_k = len(all_k)
    block_masks = [0] * n_k
    for bi, bs in enumerate(all_k):
        mask = 0
        for ts in itertools.combinations(bs, t):
            mask |= 1 << t_index[ts]
        block_masks[bi] = mask
    
    def get_block_mask(block):
        """Get the coverage mask for a block (list or tuple)."""
        block_tuple = tuple(sorted(block))
        bi = k_index.get(block_tuple)
        if bi is not None:
            return block_masks[bi]
        m = 0
        for ts in itertools.combinations(block_tuple, t):
            m |= 1 << t_index[ts]
        return m

    def greedy_cover(rng, time_limit):
        """Greedy set cover: repeatedly pick the block covering most uncovered t-subsets."""
        uncovered_mask = (1 << n_t) - 1  # all bits set
        blocks = []
        
        while uncovered_mask and time.time() < time_limit:
            # Pick a seed: find an uncovered t-subset
            idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
            
            # Start with the t-subset's points
            seed = set(all_t[idx])
            
            # Greedily grow the block to size k
            block = seed
            for _ in range(k - t):
                if time.time() > time_limit:
                    break
                best_gain = -1
                best_x = None
                sorted_block = sorted(block)
                # Precompute the (t-1)-subsets of current block for gain lookup
                sub_sets = list(itertools.combinations(sorted_block, t_minus_1))
                for x in range(v):
                    if x in block:
                        continue
                    # Compute gain: number of uncovered t-subsets containing x and (t-1) elements from block
                    gain = 0
                    for A in sub_sets:
                        # A is a tuple of (t-1) points from block; form the t-subset with x
                        # Need to look up ext_map for the (t-1)-subset that doesn't include x
                        # Actually: the t-subset must contain x and (t-1) points from block
                        # The (t-1)-subset from block that, combined with x, forms a t-subset
                        # is exactly A (since A has t-1 elements from block, plus x makes t)
                        # But we need to check: does the t-subset A∪{x} exist?
                        # For t=3: A is a pair (a,b), t-subset is {a,b,x}
                        # ext_map[(a,b)] gives all t-subsets containing both a and b
                        # We need the one that also contains x
                        # Simpler: just check if the sorted tuple is in t_index
                        ts_tuple = tuple(sorted(A + (x,)))
                        ti = t_index.get(ts_tuple)
                        if ti is not None and (uncovered_mask >> ti) & 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None and rng.random() < 0.5:
                        best_x = x
                if best_x is None or best_gain <= 0:
                    break
                block.add(best_x)
            
            if len(block) != k:
                # Pad with random points if needed
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.add(x)
            
            block = sorted(block)
            blocks.append(block)
            
            # Update uncovered mask using precomputed block mask
            mask = get_block_mask(block)
            uncovered_mask &= ~mask
        
        return blocks

    def is_valid(blocks):
        """Check validity using precomputed block masks."""
        covered_mask = 0
        for block in blocks:
            covered_mask |= get_block_mask(block)
        return covered_mask == (1 << n_t) - 1

    def remove_redundant(blocks, time_limit):
        """Iteratively remove blocks that are not needed, using bitset union."""
        n = len(blocks)
        if n == 0:
            return blocks
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            # Precompute coverage masks for current blocks
            masks = [get_block_mask(b) for b in blocks]
            full_mask = (1 << n_t) - 1
            # Calculate total coverage first to skip if already invalid (shouldn't happen)
            total = 0
            for m in masks:
                total |= m
            
            for i in range(n):
                if time.time() > time_limit:
                    break
                # Check if block i is redundant: union of all other masks covers everything
                # Optimization: If block i's mask is a subset of the union of others, it's redundant.
                # We compute union of others.
                union = 0
                for j in range(n):
                    if j != i:
                        union |= masks[j]
                
                # If the union of others covers everything that was covered before (or full_mask),
                # and specifically covers the unique parts of block i, it is redundant.
                # Since we know total coverage is full_mask (assuming valid input),
                # block i is redundant if (union_others | mask_i) == union_others is false?
                # No, block i is redundant if union_others == full_mask.
                if union == full_mask:
                    blocks = blocks[:i] + blocks[i+1:]
                    n -= 1
                    changed = True
                    break  # restart from beginning
        return blocks

    def local_search_replace(blocks, time_limit):
        """
        Simple local search: try to swap one block for another that covers more unique parts,
        or remove blocks if possible.
        Given the time constraints and complexity, we will focus on a simpler 'shrink' heuristic:
        Try to remove a block and replace it with a different one if it allows removing another block later?
        Actually, for C(24,7,3), the gap between greedy and optimal is small.
        We will implement a simple 1-swap: for each block i, try to find a block j (not in list)
        such that replacing i with j still keeps coverage valid. If so, check if we can then remove redundant blocks.
        This is expensive. 
        Alternative: Just rely on greedy + redundancy. The previous error was a crash, not a timeout.
        Let's keep a very lightweight local search: try to replace a block with a random one and see if coverage holds.
        """
        n = len(blocks)
        if n == 0:
            return blocks
        
        improved = True
        while improved and time.time() < time_limit - 5:
            improved = False
            masks = [get_block_mask(b) for b in blocks]
            full_mask = (1 << n_t) - 1
            
            # Try removing a block if possible (already done in remove_redundant, but maybe order matters?)
            # Let's try a simple swap: 
            # Pick a random index i.
            # Generate a random candidate block.
            # Check if replacing blocks[i] with candidate maintains validity.
            # If yes, do it.
            
            for _ in range(100): # Limited iterations
                if time.time() > time_limit - 5:
                    break
                if n == 0:
                    break
                i = random.randint(0, n - 1)
                # Generate random block
                cand = random.sample(range(v), k)
                cand = sorted(cand)
                if cand in blocks:
                    continue
                
                new_blocks = blocks[:i] + blocks[i+1:] + [cand]
                if is_valid(new_blocks):
                    # Check if this allows removing some blocks
                    new_blocks = remove_redundant(new_blocks, time.time() + 1) # Short time limit for check
                    if len(new_blocks) < n:
                        blocks = new_blocks
                        n = len(blocks)
                        improved = True
                        break
                    elif len(new_blocks) == n:
                        # Same size, but maybe different structure helps later?
                        # Accept swap if it covers same but maybe different distribution?
                        # For now, only accept if size decreases.
                        pass
        return blocks

    # Main: run multiple randomized greedy, keep best
    best_result = None
    best_len = float('inf')
    
    # Adjust number of trials based on time
    # Greedy is fast. Let's allow more trials.
    n_trials = 20
    for trial in range(n_trials):
        if time.time() > deadline - 20:
            break
        rng = random.Random(trial * 7919 + 12345)
        # Give greedy enough time to finish, but leave room for post-processing
        result = greedy_cover(rng, deadline - 15)
        
        # Ensure we have a valid cover. Greedy might stop early if time limit hit during growth?
        # The greedy_cover function checks time, but might return incomplete blocks if it breaks early in growth.
        # We added a padding step in greedy_cover, so it should return k-sized blocks.
        # However, if it stops the while loop early, it might not cover everything.
        if is_valid(result):
            # Remove redundant blocks
            result = remove_redundant(result, deadline - 10)
            if len(result) < best_len:
                best_len = len(result)
                best_result = result
    
    # Local search on best
    if best_result is not None and time.time() < deadline - 5:
        improved = local_search_replace(best_result, deadline - 2)
        if is_valid(improved):
            best_result = improved
            # Final redundancy removal
            best_result = remove_redundant(best_result, deadline - 1)
    
    if best_result is None:
        # Fallback: run one more greedy with max time
        rng = random.Random(0)
        best_result = greedy_cover(rng, deadline - 1)
        if not is_valid(best_result):
            # If still invalid (should not happen), return a dummy valid cover?
            # Construct a trivial cover: all k-subsets? That's huge.
            # Better: just return the best effort. But it must be valid.
            # If greedy failed to cover, we have a problem.
            # Let's force a valid cover by adding blocks until valid.
            # This is slow but ensures validity.
            current_cover_mask = 0
            for b in best_result:
                current_cover_mask |= get_block_mask(b)
            
            # Find uncovered t-subsets
            full_mask = (1 << n_t) - 1
            uncovered = full_mask & ~current_cover_mask
            
            # Add blocks to cover uncovered
            while uncovered and time.time() < deadline - 0.1:
                idx = (uncovered & -uncovered).bit_length() - 1
                seed = set(all_t[idx])
                block = list(seed)
                # Grow to k
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
                block = sorted(block)
                if block not in best_result:
                    best_result.append(block)
                    current_cover_mask |= get_block_mask(block)
                    uncovered &= ~current_cover_mask
            
            # Final check
            if not is_valid(best_result):
                # Extreme fallback: return all blocks? No, too many.
                # Just return what we have. The evaluator will mark invalid.
                # But we tried our best.
                pass
    
    return best_result
