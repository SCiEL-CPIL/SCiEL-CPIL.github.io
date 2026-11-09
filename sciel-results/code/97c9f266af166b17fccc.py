"""Covering design C(v,k,t) via Greedy Construction + Repair-Based Local Search."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper to check if a block is valid (distinct ints) - assumed by construction
    # Helper to get t-subsets of a block
    def get_tsubs(block):
        return list(itertools.combinations(block, t))
        
    # Helper to compute coverage counts
    def compute_coverage(blocks):
        coverage = [0] * n_tsubs
        for b in blocks:
            for ts in get_tsubs(b):
                coverage[tsub_to_idx[ts]] += 1
        return coverage

    # Helper to check if a block is redundant given coverage
    def is_redundant(block, coverage):
        for ts in get_tsubs(block):
            if coverage[tsub_to_idx[ts]] == 1:
                return False
        return True

    # Helper to remove a redundant block and update coverage
    def try_remove_block(blocks, coverage, block_idx):
        b = blocks[block_idx]
        if is_redundant(b, coverage):
            for ts in get_tsubs(b):
                coverage[tsub_to_idx[ts]] -= 1
            blocks.pop(block_idx)
            return True
        return False

    def greedy_construct(rng):
        uncovered = set(range(n_tsubs))
        blocks = []
        
        while uncovered and time.time() < deadline - 5.0:
            # Pick a random uncovered t-subset
            anchor_idx = rng.choice(list(uncovered))
            anchor = all_tsubs[anchor_idx]
            block = list(anchor)
            
            # Greedily add points
            while len(block) < k:
                best_gain = -1
                best_point = None
                candidates = [x for x in range(v) if x not in block]
                
                # Calculate gain for each candidate
                # Optimization: Precompute t-1 subsets of current block
                current_t_minus_1 = list(itertools.combinations(block, t-1))
                
                for x in candidates:
                    gain = 0
                    for sub in current_t_minus_1:
                        new_sub = list(sub)
                        new_sub.append(x)
                        new_sub.sort()
                        idx = tsub_to_idx.get(tuple(new_sub), -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        # Random tie break
                        if gain == best_gain and rng.random() < 0.5:
                            best_point = x # Actually just keep first max for speed, or randomize ties properly
                            
                if best_point is None:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for ts in get_tsubs(block):
                idx = tsub_to_idx[ts]
                if idx in uncovered:
                    uncovered.discard(idx)
        
        return blocks, uncovered

    def local_search(blocks, coverage):
        """
        Attempts to reduce block count by modifying blocks to create redundancy.
        Returns updated blocks and coverage.
        """
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            n_blocks = len(blocks)
            
            # 1. Try to remove any redundant block
            for i in range(n_blocks - 1, -1, -1):
                if try_remove_block(blocks, coverage, i):
                    improved = True
                    break # Restart loop as indices changed
            
            if not improved and time.time() < deadline - 2.0:
                # 2. Repair-based reduction
                # Pick a random block to try to remove
                if n_blocks > 0:
                    target_idx = random.randint(0, n_blocks - 1)
                    target_block = blocks[target_idx]
                    
                    # Find t-subsets in target_block that are uniquely covered by it
                    unique_tsubs = []
                    for ts in get_tsubs(target_block):
                        idx = tsub_to_idx[ts]
                        if coverage[idx] == 1:
                            unique_tsubs.append(ts)
                    
                    if unique_tsubs:
                        # We need to cover these unique t-subsets with other blocks.
                        # Strategy: Pick one unique t-subset S.
                        # Try to modify an existing block B' to cover S.
                        # B' should be "close" to S.
                        S = random.choice(unique_tsubs)
                        
                        # Find candidates B' that share t-1 elements with S
                        # Or just try to modify any block that shares some elements.
                        # For efficiency, look for blocks that intersect S in t-1 elements.
                        
                        s_set = set(S)
                        candidates_for_modification = []
                        
                        for j, b in enumerate(blocks):
                            if j == target_idx:
                                continue
                            b_set = set(b)
                            # Check if b can be modified to cover S
                            # We need to change b to b' such that S is subset of b'.
                            # This means b' must contain S.
                            # If b already contains S, it would have covered it (coverage > 1).
                            # So b does NOT contain S.
                            # To make b contain S, we need to add elements of S \ b to b,
                            # and remove elements of b \ S to keep size k.
                            
                            # Let diff_in = S \ b (elements in S not in b)
                            # Let diff_out = b \ S (elements in b not in S)
                            # We need to replace |diff_in| elements from diff_out with diff_in.
                            # Since |S|=t and |b|=k, and S not subset of b:
                            # Let common = S & b.
                            # |S \ b| = t - |common|
                            # We need to replace t - |common| elements.
                            
                            common = s_set.intersection(b_set)
                            num_to_add = t - len(common)
                            
                            if num_to_add <= 0:
                                continue # S is already in b, shouldn't happen if unique
                            if num_to_add > k: # Impossible
                                continue
                            
                            # We need to remove num_to_add elements from b \ S
                            # And add all elements of S \ b
                            # This creates a new block b'.
                            # Check if this swap maintains validity for b'.
                            # i.e., does b' lose any unique coverage?
                            
                            # To be safe, we only do this if the "cost" is low.
                            # Actually, let's just try the swap and see if it helps remove target_block.
                            
                            # Generate all possible swaps? Too many.
                            # Pick a random subset of diff_out to remove.
                            diff_out = list(b_set - s_set)
                            if len(diff_out) < num_to_add:
                                continue
                            
                            # Randomly choose which elements to remove
                            removals = random.sample(diff_out, num_to_add)
                            additions = list(s_set - b_set)
                            
                            new_b_set = (b_set - set(removals)) | set(additions)
                            new_b = sorted(new_b_set)
                            
                            # Check if new_b is valid (size k, distinct) - yes by construction
                            
                            # Now, check if this modification allows us to remove target_block.
                            # Temporarily apply the change to coverage.
                            
                            # 1. Remove old b from coverage
                            old_b = b
                            for ts in get_tsubs(old_b):
                                coverage[tsub_to_idx[ts]] -= 1
                            
                            # 2. Add new_b to coverage
                            for ts in get_tsubs(new_b):
                                coverage[tsub_to_idx[ts]] += 1
                            
                            # 3. Check if target_block is now redundant
                            if is_redundant(target_block, coverage):
                                # Great! Remove target_block
                                for ts in get_tsubs(target_block):
                                    coverage[tsub_to_idx[ts]] -= 1
                                blocks.pop(target_idx)
                                blocks[j] = new_b # Update the modified block
                                improved = True
                                break
                            else:
                                # Revert the change
                                for ts in get_tsubs(new_b):
                                    coverage[tsub_to_idx[ts]] -= 1
                                for ts in get_tsubs(old_b):
                                    coverage[tsub_to_idx[ts]] += 1
                                
                                # If we didn't improve, we might still want to keep the change
                                # if it increases redundancy elsewhere?
                                # For now, only keep if it leads to immediate removal.
                                # This is a conservative local search.
            
        return blocks, coverage

    # Main loop
    while time.time() < deadline:
        rng = random.Random()
        blocks, uncovered = greedy_construct(rng)
        
        if not uncovered:
            # Compute initial coverage
            coverage = compute_coverage(blocks)
            
            # Perform local search
            blocks, coverage = local_search(blocks, coverage)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks.copy()
        
        # If time is running low, ensure we have a valid fallback
        if time.time() > deadline - 10.0:
            break

    if best_blocks is None:
        # Fallback: Run a simple deterministic greedy without local search to ensure validity
        # This should be fast
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered:
            anchor_idx = min(uncovered)
            anchor = all_tsubs[anchor_idx]
            block = list(anchor)
            while len(block) < k:
                best_gain = -1
                best_point = None
                current_t_minus_1 = list(itertools.combinations(block, t-1))
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for sub in current_t_minus_1:
                        new_sub = list(sub)
                        new_sub.append(x)
                        new_sub.sort()
                        idx = tsub_to_idx.get(tuple(new_sub), -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None: break
                block.append(best_point)
            block.sort()
            blocks.append(block)
            for ts in get_tsubs(block):
                idx = tsub_to_idx[ts]
                if idx in uncovered:
                    uncovered.discard(idx)
        best_blocks = blocks
        best_count = len(blocks)

    return best_blocks
