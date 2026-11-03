"""Structural redesign: bitset-accelerated greedy with local search (swap/drop)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct(v, k, t):
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(12345)

    # Precompute all t-subsets as frozensets and index them
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    # Map each t-subset to an index
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}

    # For each point x, the set of t-subsets containing x (as a set of indices)
    # We'll use bitsets (Python ints) for fast intersection
    # point_mask[x] = bitset of t-subset indices that contain x
    point_mask = [0] * v
    for i, s in enumerate(all_tsubs):
        for x in s:
            point_mask[x] |= (1 << i)

    # For a block B (frozenset of points), the t-subsets covered are the union
    # of point_mask[x] for x in B, but we only care about t-subsets fully contained in B.
    # Actually, a t-subset is covered by block B iff all its elements are in B.
    # So covered_tsubs(B) = {i : all_tsubs[i] subset of B}
    # We can compute this as: intersection of complement masks? 
    # Better: for each t-subset i, it's covered by B iff for all x in all_tsubs[i], x in B.
    # Equivalent: the t-subset's element set is a subset of B.
    # Let's precompute for each t-subset its element bitmask (over v points)
    tsub_point_mask = [0] * n_tsubs
    for i, s in enumerate(all_tsubs):
        m = 0
        for x in s:
            m |= (1 << x)
        tsub_point_mask[i] = m

    def block_covers_tsub_indices(block_frozen):
        """Return the set (as a Python set of indices) of t-subsets covered by this block."""
        bm = 0
        for x in block_frozen:
            bm |= (1 << x)
        covered = set()
        for i in range(n_tsubs):
            if tsub_point_mask[i] & bm == tsub_point_mask[i]:
                covered.add(i)
        return covered

    # Greedy construction with bitset tracking
    def greedy_cover():
        uncovered = set(range(n_tsubs))  # indices of uncovered t-subsets
        blocks = []
        while uncovered:
            # Pick the t-subset with minimum "potential" – just pick min index for determinism
            # Actually, for better greedy, pick the uncovered t-subset that appears in the fewest
            # potential blocks? That's expensive. Let's just pick arbitrarily (min index).
            start_tsub = min(uncovered)
            start_pts = all_tsubs[start_tsub]
            block = set(start_pts)
            
            while len(block) < k:
                # For each candidate point x not in block, compute how many uncovered t-subsets
                # would become covered by adding x to block.
                # A t-subset i becomes covered if: (a) it's currently uncovered, (b) all its elements
                # are in block ∪ {x}. Since block already has some points, the t-subset must contain
                # all points in (tsub ∩ (block ∪ {x})). 
                # More precisely: tsub i is covered by block∪{x} iff tsub_point_mask[i] & ~(bm | (1<<x)) == 0
                # where bm is the current block's point mask.
                bm = 0
                for x in block:
                    bm |= (1 << x)
                
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count uncovered t-subsets that would be covered
                    new_bm = bm | (1 << x)
                    gain = 0
                    for i in uncovered:
                        if tsub_point_mask[i] & new_bm == tsub_point_mask[i]:
                            gain += 1
                            if gain > best_gain:
                                # Early exit not trivial, but we can break inner if gain exceeds best
                                pass
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    if best_gain == len(uncovered):
                        break
                block.add(best_x)
            
            block_frozen = frozenset(block)
            blocks.append(list(block_frozen))
            
            # Update uncovered
            bm = 0
            for x in block_frozen:
                bm |= (1 << x)
            to_remove = set()
            for i in uncovered:
                if tsub_point_mask[i] & bm == tsub_point_mask[i]:
                    to_remove.add(i)
            uncovered -= to_remove
        
        return blocks

    # Local search: try to remove a block if all its t-subsets are still covered by others
    def try_remove_blocks(blocks):
        # For each block, compute the set of t-subsets it covers
        # Then check if removing it leaves all t-subsets covered
        # This is expensive; do it greedily: try each block in random order
        n = len(blocks)
        if n == 0:
            return blocks
        # Precompute covered sets
        covered_sets = []
        for b in blocks:
            bm = 0
            for x in b:
                bm |= (1 << x)
            cs = set()
            for i in range(n_tsubs):
                if tsub_point_mask[i] & bm == tsub_point_mask[i]:
                    cs.add(i)
            covered_sets.append(cs)
        
        # Union of all
        all_covered = set()
        for cs in covered_sets:
            all_covered |= cs
        
        # Try to remove blocks
        order = list(range(n))
        rng.shuffle(order)
        new_blocks = list(blocks)
        new_covered_sets = list(covered_sets)
        current_covered = set(all_covered)
        
        for idx in order:
            if idx >= len(new_blocks):
                continue
            cs = new_covered_sets[idx]
            # Check if all t-subsets in cs are covered by other blocks
            others_covered = set()
            for j in range(len(new_blocks)):
                if j == idx:
                    continue
                others_covered |= new_covered_sets[j]
            if cs <= others_covered:
                # Safe to remove
                new_blocks.pop(idx)
                new_covered_sets.pop(idx)
                # Reindex
                for j in range(idx, len(order)):
                    if order[j] > idx:
                        order[j] -= 1
                # Break to restart with new indices? Or continue carefully.
                # Let's break and restart the removal pass
                return try_remove_blocks(new_blocks)
        
        return new_blocks

    # Local search: swap a point in a block with another point to improve coverage
    def local_search(blocks, time_limit):
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            n = len(blocks)
            # Precompute covered sets
            covered_sets = []
            for b in blocks:
                bm = 0
                for x in b:
                    bm |= (1 << x)
                cs = set()
                for i in range(n_tsubs):
                    if tsub_point_mask[i] & bm == tsub_point_mask[i]:
                        cs.add(i)
                covered_sets.append(cs)
            
            # For each block, try swapping each point with each other point
            for bi in range(n):
                if time.time() > deadline:
                    return blocks
                block = blocks[bi]
                cs = covered_sets[bi]
                for pi, old_x in enumerate(block):
                    for new_x in range(v):
                        if new_x in block:
                            continue
                        # Compute new block
                        new_block = block[:]
                        new_block[pi] = new_x
                        new_block_frozen = frozenset(new_block)
                        
                        # Compute new covered set
                        bm = 0
                        for x in new_block_frozen:
                            bm |= (1 << x)
                        new_cs = set()
                        for i in range(n_tsubs):
                            if tsub_point_mask[i] & bm == tsub_point_mask[i]:
                                new_cs.add(i)
                        
                        # Check if this swap is beneficial: does it cover any t-subset
                        # that was only covered by this block?
                        # i.e., is there i in cs \ new_cs that is not covered by any other block?
                        # This requires knowing other blocks' coverage. Expensive.
                        # Simpler heuristic: accept if |new_cs| > |cs| (covers more t-subsets)
                        if len(new_cs) > len(cs):
                            blocks[bi] = new_block
                            covered_sets[bi] = new_cs
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return blocks

    # Main: greedy then local search
    blocks = greedy_cover()
    
    if time.time() < deadline - 5:
        blocks = try_remove_blocks(blocks)
    
    if time.time() < deadline - 5:
        blocks = local_search(blocks, time_limit=deadline - time.time() - 5)
    
    if time.time() < deadline - 2:
        blocks = try_remove_blocks(blocks)
    
    # Final validation
    # Check that all t-subsets are covered
    all_covered = set()
    for b in blocks:
        bm = 0
        for x in b:
            bm |= (1 << x)
        for i in range(n_tsubs):
            if tsub_point_mask[i] & bm == tsub_point_mask[i]:
                all_covered.add(i)
    
    if all_covered != set(range(n_tsubs)):
        # Fallback: return greedy result (shouldn't happen)
        blocks = greedy_cover()
    
    return blocks
# EVOLVE-BLOCK-END
