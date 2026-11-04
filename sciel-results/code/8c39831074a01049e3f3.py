"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import array
import random


def construct(v, k, t):
    """
    Hybrid covering design:
    1. Fast bitmask/inverse-indexed greedy to get a valid covering.
    2. Local search (block swap / remove-add) to reduce block count.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # --- Precompute t-subsets mapping ---
    tsets = list(itertools.combinations(range(v), t))
    N = len(tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(tsets)}
    
    # coverage[i] = number of blocks covering t-subset i
    coverage = array.array('H', [0]) * N
    
    # gain[x] = number of currently-uncovered t-subsets containing point x
    gain = [0] * v
    for i, combo in enumerate(tsets):
        for x in combo:
            gain[x] += 1

    blocks = []
    # Precompute list of t-subset indices for each possible block? No, too big.
    # Instead, compute on the fly or cache for current block.

    def update_coverage(block_list, delta):
        """Update coverage and gain for all t-subsets in block_list."""
        for tcombo in itertools.combinations(block_list, t):
            i = tset_to_idx[tcombo]
            old_cov = coverage[i]
            new_cov = old_cov + delta
            coverage[i] = new_cov
            if (old_cov == 0) != (new_cov == 0):
                # Transition between covered and uncovered
                sign = -1 if delta > 0 else 1
                for x in tcombo:
                    gain[x] += sign

    # --- Phase 1: Greedy Construction ---
    while True:
        if time.time() > deadline - 10.0:
            break
        # Check if all covered
        # Optimization: keep track of uncovered count?
        # For now, scan. N is ~11628, which is small.
        uncovered_exists = False
        for c in coverage:
            if c == 0:
                uncovered_exists = True
                break
        if not uncovered_exists:
            break

        # Find an uncovered t-subset to anchor
        anchor_idx = -1
        for i in range(N):
            if coverage[i] == 0:
                anchor_idx = i
                break
        if anchor_idx == -1:
            break
        
        anchor = tsets[anchor_idx]
        
        # Build block greedily
        block_set = set(anchor)
        block_list = list(anchor)
        
        while len(block_list) < k:
            best_x = -1
            best_g = -1
            # Find point not in block with highest gain
            for x in range(v):
                if x not in block_set:
                    g = gain[x]
                    if g > best_g:
                        best_g = g
                        best_x = x
            if best_x < 0:
                break
            block_set.add(best_x)
            block_list.append(best_x)
        
        block_list.sort()
        blocks.append(block_list)
        update_coverage(block_list, 1)

    # Ensure validity (just in case loop exited early due to time, though greedy should finish fast)
    # If not valid, we must return a valid set. The greedy loop above only breaks if time runs out.
    # If time runs out, we might have an invalid set.
    # However, for v=19, k=8, t=5, greedy is very fast (< 1s).
    # So we assume we have a valid set here.
    
    if time.time() > deadline - 5.0:
        return blocks

    # --- Phase 2: Local Search / Reduction ---
    # Precompute t-subset indices for each block
    block_tsets = []
    for bl in blocks:
        ts = [tset_to_idx[tc] for tc in itertools.combinations(bl, t)]
        block_tsets.append(ts)

    # Rebuild coverage from scratch to be safe
    coverage = array.array('H', [0]) * N
    for ts in block_tsets:
        for i in ts:
            coverage[i] += 1

    # Helper to check if a block is redundant
    def is_redundant(block_idx):
        for i in block_tsets[block_idx]:
            if coverage[i] <= 1:
                return False
        return True

    # Pass 1: Remove redundant blocks
    removed = set()
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 5.0:
            break
        for bi in range(len(blocks) - 1, -1, -1):
            if bi in removed:
                continue
            if is_redundant(bi):
                removed.add(bi)
                for i in block_tsets[bi]:
                    coverage[i] -= 1
                changed = True
    
    if removed:
        blocks = [b for idx, b in enumerate(blocks) if idx not in removed]
        block_tsets = [ts for idx, ts in enumerate(block_tsets) if idx not in removed]

    # Pass 2: Local Search (Swap)
    # Try to replace a block with a new block that covers more unique t-subsets or allows removal of others.
    # A simpler heuristic: Try to remove a block, then try to cover the newly uncovered t-subsets with a single new block.
    # If we can cover all uncovered t-subsets from one block with a single new block, we have the same count.
    # To improve, we need to remove 1 and add 0 (redundancy) or remove 1 and add 1 but the new block helps remove another?
    # Let's stick to a "Remove and Repair" strategy.
    
    # Sort blocks by "criticality" (number of t-subsets they uniquely cover)
    # Lower criticality blocks are better candidates for removal.
    
    # We will iterate through blocks, try to remove one, and see if we can cover the resulting holes with 1 new block.
    # If yes, we replace the block. This doesn't reduce count, but might set up for future reductions?
    # Actually, if we remove block B, we have holes H. If there is a block B' such that H is subset of tsets(B'),
    # then B' covers all holes. We can replace B with B'. This maintains count.
    # To reduce count, we need B' to also cover t-subsets that were ONLY covered by another block B'', allowing B'' to be removed.
    
    # Let's try a simpler move:
    # 1. Identify a block B to remove.
    # 2. Calculate holes H.
    # 3. Find a block B' that covers all H.
    # 4. If such B' exists, and B' is not already in the set, replace B with B'.
    # 5. Check for new redundancies.
    
    # To find B' quickly, we can look for t-subsets in H that are rare (covered by few blocks).
    # Or just iterate all possible blocks? C(19,8) is 75582. That's small enough to iterate if we do it smartly.
    # But doing this for every block is expensive.
    
    # Let's limit the local search time.
    ls_deadline = deadline - 2.0
    max_ls_steps = 5000
    
    step_count = 0
    while time.time() < ls_deadline and step_count < max_ls_steps:
        step_count += 1
        
        # Pick a random block to try to replace/remove
        bi = random.randint(0, len(blocks) - 1)
        
        # Calculate holes if we remove block bi
        holes = []
        for i in block_tsets[bi]:
            if coverage[i] == 1:
                holes.append(i)
        
        if not holes:
            # Block is redundant, remove it
            for i in block_tsets[bi]:
                coverage[i] -= 1
            blocks.pop(bi)
            block_tsets.pop(bi)
            continue
        
        # We need to cover these holes with a new block.
        # The new block must contain at least one point from each hole? No, it must contain the entire hole t-subset.
        # So the new block must be a superset of each hole? No, a block covers a t-subset if the t-subset is a subset of the block.
        # So the new block B' must satisfy: for all h in holes, h is subset of B'.
        # This implies B' must contain the union of all points in holes?
        # If h1 and h2 are disjoint, B' must contain h1 U h2.
        # If h1 and h2 overlap, B' must contain h1 U h2.
        # So B' must contain U = Union of all points in holes.
        
        U = set()
        for h_idx in holes:
            U.update(tsets[h_idx])
        
        if len(U) > k:
            # Cannot cover all holes with a single block
            continue
        
        # We need to find a block B' such that U is subset of B' and B' covers all holes.
        # Since U is subset of B' implies all holes (subsets of U) are subsets of B', this condition is sufficient.
        
        # How many such blocks are there?
        # We need to choose k - len(U) more points from v - len(U) points.
        # This is C(v - len(U), k - len(U)).
        
        # We also want B' to be "good". A good B' should cover many other t-subsets that are currently covered by only 1 block (to enable further removals).
        # Or simply, we can just pick a random valid B'.
        
        # Let's pick a random valid B' to introduce diversity.
        # Generate a random superset of U of size k.
        remaining_points = [x for x in range(v) if x not in U]
        need = k - len(U)
        if need < 0 or need > len(remaining_points):
            continue
            
        extra = random.sample(remaining_points, need)
        new_block_set = U.union(extra)
        new_block = sorted(new_block_set)
        
        # Check if this block is already in the list
        if new_block in blocks:
            continue
            
        # Check if adding this block actually covers the holes (it should by construction)
        # And check if it creates any new redundancy opportunities?
        # For now, just swap.
        
        # Remove old block
        for i in block_tsets[bi]:
            coverage[i] -= 1
        
        # Add new block
        new_block_tsets = [tset_to_idx[tc] for tc in itertools.combinations(new_block, t)]
        for i in new_block_tsets:
            coverage[i] += 1
            
        blocks[bi] = new_block
        block_tsets[bi] = new_block_tsets
        
        # After swap, check for redundancies again
        # We can do a quick pass for redundancies
        for ri in range(len(blocks) - 1, -1, -1):
            if ri == bi:
                continue
            redundant = True
            for i in block_tsets[ri]:
                if coverage[i] <= 1:
                    redundant = False
                    break
            if redundant:
                for i in block_tsets[ri]:
                    coverage[i] -= 1
                blocks.pop(ri)
                block_tsets.pop(ri)
                # Note: indices shift, so we break out of this inner loop or be careful.
                # Since we are modifying the list, let's just break and restart the redundancy check in the next outer iteration or a dedicated pass.
                break

    return blocks

# EVOLVE-BLOCK-END
