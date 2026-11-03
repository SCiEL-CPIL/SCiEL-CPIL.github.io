import time
import random
import itertools
import math
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0

    # Precompute
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    B = len(all_blocks)
    
    # block_cover[bi] = list of t-subset indices covered by block bi
    block_cover = []
    for b in all_blocks:
        covered = [idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # t_to_blocks[ti] = list of block indices that cover t-subset ti
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    def greedy_cover():
        """Standard greedy set cover to get a valid initial solution."""
        uncovered = set(range(N))
        blocks = []
        while uncovered:
            seed_idx = next(iter(uncovered))
            candidates = t_to_blocks[seed_idx]
            best_b = -1
            max_cov = -1
            for bi in candidates:
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
            if best_b == -1:
                break
            blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
        return blocks

    def prune_redundant(blocks):
        """Remove blocks that are not the sole cover for any t-subset."""
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in blocks:
                essential = False
                for ti in block_cover[bi]:
                    if cnt[ti] == 1:
                        essential = True
                        break
                if not essential:
                    changed = True
                    for ti in block_cover[bi]:
                        cnt[ti] -= 1
                else:
                    new_blocks.append(bi)
            blocks = new_blocks
            if time.time() > deadline - 2.0:
                break
        return blocks

    # 1. Initial Solution
    init_blocks = greedy_cover()
    init_blocks = prune_redundant(init_blocks)
    
    best_blocks = list(init_blocks)
    best_score = len(best_blocks)
    
    rng = random.Random(42)
    
    # 2. Parametric Flow-based Pruning
    # This strategy treats the current set of blocks as a feasible flow.
    # We want to remove blocks that have "zero reduced cost" in the dual sense,
    # i.e., they are not essential for covering any t-subset given the current
    # coverage counts, but we use a flow-like perspective to identify them.
    #
    # Actually, the inspiration suggests using a min-cut/min-flow formulation.
    # Let's implement a "Greedy Pruning with Dual Weights" which is effectively
    # a simplified parametric search.
    #
    # We maintain cnt[ti] = number of blocks covering ti.
    # We maintain in_set[bi] = boolean.
    #
    # The "flow" perspective:
    # Source -> Block (capacity 1, cost lambda)
    # Block -> t-subset (capacity inf, cost 0)
    # t-subset -> Sink (capacity 1, cost 0)
    #
    # A block is "redundant" if we can reroute the flow through it to other blocks
    # without increasing the total cost or violating capacities.
    # In the set cover context, this means: if we remove block b, can we cover
    # the t-subsets that become uncovered by adding other blocks, such that the
    # net change in cost is <= 0?
    #
    # Since we minimize the number of blocks, lambda = 1 for each block.
    # If removing b (cost -1) requires adding R blocks (cost +|R|),
    # the move is beneficial if |R| < 1 (impossible) or |R| == 0 (pure removal).
    #
    # Wait, the current program does exactly this: Drop and Repair.
    # The "Flow" inspiration likely refers to using the residual graph to find
    # augmenting paths that reduce cost.
    #
    # Let's implement a "Residual Graph Swap" approach.
    #
    # State:
    # - current_blocks: list of block indices
    # - cnt: np.array of size N
    # - in_set: np.array of size B
    #
    # Move:
    # 1. Pick a block b to drop.
    # 2. Identify U = {ti | cnt[ti] becomes 0}.
    # 3. If U is empty, accept.
    # 4. Else, find a set of blocks R to cover U.
    #    Instead of greedy, use a BFS/DFS in the "exchange graph"?
    #    Or, more simply, use the flow analogy:
    #    We want to find a set of blocks R such that |R| <= 1 (for pure improvement)
    #    or |R| == 1 (swap, cost neutral, allows exploration).
    #
    # Let's focus on finding SWAPS (1 block out, 1 block in) that maintain coverage
    # but allow us to "rotate" into a better position, and pure deletions.
    #
    # Algorithm:
    # 1. Compute cnt.
    # 2. Identify "weak" blocks: blocks b where there exists a t-subset ti in cover(b)
    #    such that cnt[ti] == 1. These are essential for at least one subset.
    #    Blocks that are not weak (all ti in cover(b) have cnt[ti] > 1) can be removed
    #    immediately? No, removing one might make another essential.
    #    But if a block b has NO essential subsets (all cnt[ti] > 1 for ti in cover(b)),
    #    we can try to remove it.
    #    Wait, if all cnt[ti] > 1, then removing b leaves all cnt[ti] >= 1.
    #    So b is removable!
    #
    # Let's verify:
    # If for all ti in cover(b), cnt[ti] > 1, then after removing b,
    # new_cnt[ti] = cnt[ti] - 1 >= 1.
    # So coverage is maintained.
    #
    # So, any block b such that min(cnt[ti] for ti in cover(b)) > 1 is removable.
    #
    # The current `prune_redundant` only removes blocks where min(cnt) == 1 is false?
    # No, `prune_redundant` removes blocks where NO ti has cnt[ti] == 1.
    # i.e., it removes blocks where min(cnt[ti]) > 1.
    #
    # So `prune_redundant` already does this!
    #
    # Why does the current program still have local search?
    # Because `prune_redundant` is static. It doesn't explore swaps.
    #
    # The "Flow" idea: Use the residual network to find augmenting paths.
    # In set cover, a "swap" is: Remove b1, Add b2.
    # This is valid if cover(b1) \ cover(b2) is covered by other blocks.
    # i.e., for all ti in cover(b1) \ cover(b2), cnt[ti] > 1 (after removing b1, before adding b2? No).
    #
    # Let's define the condition for a swap (remove b_out, add b_in) to be valid:
    # For all ti in cover(b_out):
    #   if ti in cover(b_in):
    #       cnt[ti] remains the same (or increases if we add before removing)
    #   else:
    #       cnt[ti] decreases by 1.
    #       So we need cnt[ti] >= 2 BEFORE the swap.
    #
    # So, a swap (b_out, b_in) is valid if:
    # For all ti in cover(b_out) \ cover(b_in), cnt[ti] >= 2.
    #
    # If we find such a swap, the cost is neutral (0).
    # Why do we want neutral swaps? To escape local minima where no single block
    # can be removed, but a sequence of swaps leads to a removable block.
    #
    # This is exactly what the residual graph / max-flow approach would find:
    # An augmenting path in the exchange graph.
    #
    # Let's implement a "Random Swap Search":
    # 1. Maintain cnt and in_set.
    # 2. First, greedily remove all blocks where min(cnt[ti] for ti in cover(b)) > 1.
    # 3. Then, perform random swaps:
    #    a. Pick a random block b_out in current set.
    #    b. Pick a random block b_in NOT in current set.
    #    c. Check if swap is valid: for all ti in cover(b_out) \ cover(b_in), cnt[ti] >= 2.
    #    d. If valid, perform the swap.
    #    e. After each swap, check for removable blocks (min cnt > 1) and remove them.
    #
    # This is a much more structured search than the current "Drop and Repair"
    # which tries to find a set R of repairs. The swap approach is cheaper to check
    # (O(56) per swap candidate) and maintains feasibility always.
    
    current_blocks = list(best_blocks)
    current_cost = len(current_blocks)
    
    cnt = np.zeros(N, dtype=np.int32)
    in_set = np.zeros(B, dtype=bool)
    for bi in current_blocks:
        in_set[bi] = True
        for ti in block_cover[bi]:
            cnt[ti] += 1
            
    # Precompute cover sets as numpy arrays for fast min operations?
    # Or just use the list.
    
    def try_remove_redundant():
        """Remove any block that is not essential (min cnt > 1)."""
        nonlocal current_blocks, current_cost, cnt, in_set
        removed = False
        new_blocks = []
        for bi in current_blocks:
            # Check if bi is removable
            removable = True
            for ti in block_cover[bi]:
                if cnt[ti] == 1:
                    removable = False
                    break
            if removable:
                # Remove it
                for ti in block_cover[bi]:
                    cnt[ti] -= 1
                in_set[bi] = False
                removed = True
            else:
                new_blocks.append(bi)
        if removed:
            current_blocks = new_blocks
            current_cost = len(current_blocks)
            return True
        return False

    # Initial cleanup
    try_remove_redundant()
    
    # Precompute list of blocks not in set for random selection?
    # It's expensive to maintain. We can just pick random indices and check in_set.
    
    steps = 0
    max_steps = 50000
    swap_attempts = 0
    successful_swaps = 0
    
    # To make swaps more likely to be useful, we can bias b_out towards
    # blocks with lower "essentiality" (number of ti with cnt[ti]==1).
    # But for now, random.
    
    while time.time() < deadline - 2.0 and steps < max_steps:
        steps += 1
        
        # 1. Try to remove redundant blocks
        if try_remove_redundant():
            if current_cost < best_score:
                best_score = current_cost
                best_blocks = list(current_blocks)
            continue # Restart loop to catch any new redundancies
        
        # 2. Try a random swap
        # Pick b_out
        if not current_blocks:
            break
        b_out = current_blocks[rng.randrange(len(current_blocks))]
        
        # Pick b_in
        # We want b_in to cover some of the "essential" subsets of b_out?
        # No, b_in should cover the subsets that b_out covers, to allow removal?
        # Actually, the swap is valid if b_out's unique coverage is backed up.
        # To find a good b_in, we can look at the essential subsets of b_out.
        # Let E = {ti in cover(b_out) | cnt[ti] == 1}.
        # If E is empty, b_out is removable (handled above).
        # So E is not empty.
        # We need b_in to cover E? No, b_in doesn't need to cover E.
        # The condition is: for all ti in cover(b_out) \ cover(b_in), cnt[ti] >= 2.
        # This is equivalent to: for all ti in cover(b_out) where cnt[ti] == 1,
        # we MUST have ti in cover(b_in).
        # So b_in MUST cover all essential subsets of b_out.
        
        essential_sets = []
        for ti in block_cover[b_out]:
            if cnt[ti] == 1:
                essential_sets.append(ti)
                
        if not essential_sets:
            # Should have been removed
            continue
            
        # Find candidates for b_in that cover ALL essential_sets.
        # This is an intersection of candidate lists.
        # Start with candidates of the first essential set.
        candidates = t_to_blocks[essential_sets[0]]
        
        # Filter candidates:
        # 1. Not in_set
        # 2. Cover all other essential sets
        valid_b_in = -1
        # To speed up, we can just pick a few random candidates from the first list
        # and check them.
        
        # Shuffle candidates? Or just iterate?
        # Let's try up to 20 random candidates from the first list.
        # If the list is large, random sampling is good.
        
        # Get a sample of candidates
        # Note: candidates might contain b_out itself?
        # If b_out covers essential_sets[0], it is in t_to_blocks[essential_sets[0]].
        # We must skip b_out.
        
        found_valid = False
        # Try 10 random candidates
        for _ in range(10):
            if not candidates:
                break
            b_in = candidates[rng.randrange(len(candidates))]
            if b_in == b_out:
                continue
            if in_set[b_in]:
                continue
                
            # Check if b_in covers all essential_sets
            valid = True
            for ti in essential_sets:
                # Check if ti is in block_cover[b_in]
                # This is O(56) if we iterate.
                # We can precompute a set for block_cover[b_in]?
                # Or just iterate. 56 * 10 = 560 ops. Acceptable.
                if ti not in block_cover[b_in]:
                    valid = False
                    break
                    
            if valid:
                valid_b_in = b_in
                found_valid = True
                break
                
        if not found_valid:
            continue
            
        # Perform the swap
        # Remove b_out
        for ti in block_cover[b_out]:
            cnt[ti] -= 1
        in_set[b_out] = False
        
        # Add b_in
        for ti in block_cover[b_in]:
            cnt[ti] += 1
        in_set[b_in] = True
        
        # Update current_blocks
        # Find and replace
        try:
            idx_out = current_blocks.index(b_out)
            current_blocks[idx_out] = b_in
        except ValueError:
            # Should not happen
            pass
            
        successful_swaps += 1
        
        # After swap, try to remove redundant blocks again
        # (The swap might have made some other block redundant)
        if try_remove_redundant():
            if current_cost < best_score:
                best_score = current_cost
                best_blocks = list(current_blocks)
                
    # Final Prune
    best_blocks = prune_redundant(best_blocks)
    
    # Verify
    cnt_final = np.zeros(N, dtype=np.int32)
    for bi in best_blocks:
        for ti in block_cover[bi]:
            cnt_final[ti] += 1
            
    if np.min(cnt_final) < 1:
        # Fallback
        return [list(all_blocks[bi]) for bi in greedy_cover()]
        
    return [list(all_blocks[bi]) for bi in best_blocks]
