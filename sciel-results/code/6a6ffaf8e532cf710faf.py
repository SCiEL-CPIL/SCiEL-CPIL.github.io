import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s to leave margin before 120s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map from t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # We'll maintain a count of how many blocks cover each t-subset
    # count[i] = number of blocks containing all_t_subsets[i]
    
    def greedy_cover(seed):
        """Run greedy covering algorithm with given seed."""
        rng = random.Random(seed)
        
        # Initialize uncovered counts
        # uncovered_count[i] = 1 if subset i is not yet covered, 0 otherwise
        # Actually, we want to track which subsets are still uncovered
        # Use a list of booleans or just work with counts
        
        # For efficiency, track uncovered subsets as a set of indices
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start a new block
            # Or pick the one that appears in fewest potential blocks?
            # For simplicity, pick randomly from uncovered
            target_idx = rng.choice(list(uncovered))
            target = all_t_subsets[target_idx]
            block = list(target)
            
            while len(block) < k:
                # Find the point x not in block that covers the most uncovered t-subsets
                best_x = None
                best_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Count how many uncovered t-subsets would be covered by adding x
                    # A t-subset is covered by block + x if it contains x and t-1 elements from block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in subset_to_idx:
                            idx = subset_to_idx[candidate]
                            if idx in uncovered:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        if gain == 0:
                            break
                
                if best_x is None or best_gain < 0:
                    # Shouldn't happen if there are still uncovered subsets
                    best_x = rng.choice([x for x in range(v) if x not in block])
                
                block.append(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Remove all t-subsets covered by this block from uncovered
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx.get(sub, -1)
                if idx >= 0:
                    uncovered.discard(idx)
        
        return blocks
    
    def remove_redundant(blocks):
        """Remove blocks that are redundant (all their t-subsets are covered by other blocks)."""
        if not blocks:
            return blocks
        
        # Count how many blocks cover each t-subset
        # subset_count[i] = number of blocks that contain all_t_subsets[i]
        subset_count = [0] * n_t
        
        # For each block, increment count for each t-subset it contains
        block_t_subsets = []
        for block in blocks:
            subs = list(itertools.combinations(block, t))
            block_t_subsets.append(subs)
            for sub in subs:
                idx = subset_to_idx[sub]
                subset_count[idx] += 1
        
        # A block is redundant if for every t-subset it contains, there's another block covering it
        # i.e., for every t-subset in the block, subset_count[sub] > 1
        keep = [True] * len(blocks)
        
        for i, block in enumerate(blocks):
            redundant = True
            for sub in block_t_subsets[i]:
                idx = subset_to_idx[sub]
                if subset_count[idx] == 1:
                    # This t-subset is only covered by block i, so block i is not redundant
                    redundant = False
                    break
            if redundant:
                keep[i] = False
        
        new_blocks = [blocks[i] for i in range(len(blocks)) if keep[i]]
        
        # Repeat until no more changes
        if len(new_blocks) < len(blocks):
            return remove_redundant(new_blocks)
        
        return new_blocks
    
    def block_swap(blocks, rng):
        """Try to replace two blocks with one that covers all their t-subsets.
        Returns (new_blocks, improved) where improved is True if blocks were replaced."""
        if len(blocks) < 2:
            return blocks, False
        
        n = len(blocks)
        
        # Try random pairs
        for _ in range(min(100, n * (n-1) // 2)):
            i, j = rng.sample(range(n), 2)
            b1, b2 = blocks[i], blocks[j]
            
            # Union of the two blocks
            union = set(b1) | set(b2)
            
            # Can we form a k-subset from the union that covers all t-subsets of b1 and b2?
            # We need a k-subset S such that every t-subset of b1 is in S and every t-subset of b2 is in S.
            # This means S must contain all points that appear in any t-subset of b1 or b2.
            # Actually, a t-subset of b1 is covered by S iff all its elements are in S.
            # So S must contain every element that appears in some t-subset of b1 or b2.
            # The set of elements in t-subsets of b1 is just b1 (since t < k).
            # Wait, that's not right. A t-subset of b1 is a subset of b1 with exactly t elements.
            # For S to cover all t-subsets of b1, S must contain b1 entirely? No.
            # S covers a t-subset A if A ⊆ S. So S must contain all t-subsets of b1,
            # which means S must contain every element of b1 that appears in some t-subset.
            # Since t ≥ 2 (in our case t=3), every element of b1 appears in some t-subset of b1.
            # So S must contain b1 entirely. Similarly for b2.
            # So S must contain b1 ∪ b2.
            # If |b1 ∪ b2| ≤ k, then we can use S = b1 ∪ b2 (or a superset up to size k).
            
            if len(union) > k:
                continue
            
            # Check if union covers all t-subsets of b1 and b2
            # Since b1 ⊆ union and b2 ⊆ union, all t-subsets of b1 and b2 are in union.
            # But we need to verify that the t-subsets of b1 and b2 that were ONLY covered by
            # b1 or b2 are still covered. Since union contains b1 and b2, all t-subsets of
            # b1 and b2 are subsets of union, so they're covered by the new block.
            
            # But wait, we also need to make sure we're not losing coverage of other t-subsets.
            # The new block covers all t-subsets of union. The old blocks covered t-subsets of
            # b1 and b2. Since b1 ∪ b2 ⊆ union, and union has size ≤ k, we can form a block.
            # But we might be losing coverage if some t-subset was covered ONLY by b1 or b2
            # and is NOT a subset of union. But since b1 ⊆ union and b2 ⊆ union, any t-subset
            # of b1 or b2 is a subset of union. So no loss.
            
            # However, we need to check that the union doesn't introduce issues.
            # Actually, the concern is: are there t-subsets that were covered by b1 or b2
            # but NOT by the new block? Since the new block is a superset of b1 ∪ b2,
            # it covers all t-subsets that b1 or b2 covered. So no loss.
            
            # But we need the new block to have exactly k elements. If |union| < k, we can
            # add arbitrary elements. This might cover more t-subsets, which is fine.
            
            # Let's create the new block
            new_block = list(union)
            while len(new_block) < k:
                # Add a random element not in new_block
                available = [x for x in range(v) if x not in new_block]
                if not available:
                    break
                new_block.append(rng.choice(available))
            new_block.sort()
            new_block = tuple(new_block)
            
            # Verify the new block covers all t-subsets of b1 and b2
            # (This should be guaranteed by construction, but let's be safe)
            for sub in itertools.combinations(b1, t):
                if not all(x in new_block for x in sub):
                    break
            else:
                for sub in itertools.combinations(b2, t):
                    if not all(x in new_block for x in sub):
                        break
                else:
                    # Valid merge
                    new_blocks = [blocks[m] for m in range(n) if m != i and m != j]
                    new_blocks.append(new_block)
                    return new_blocks, True
        
        return blocks, False
    
    # Main search loop
    best_blocks = None
    best_len = float('inf')
    
    # First, run greedy to get an initial solution
    initial = greedy_cover(42)
    initial = remove_redundant(initial)
    best_blocks = initial
    best_len = len(initial)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return [list(b) for b in best_blocks]
    
    # Now do local search with swaps
    rng = random.Random(123)
    blocks = list(best_blocks)
    
    while time.time() < deadline - 5.0:  # Leave 5s margin
        # Try some swaps
        improved = False
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            blocks, imp = block_swap(blocks, rng)
            if imp:
                improved = True
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = list(blocks)
        
        # Try removing redundant blocks
        blocks = remove_redundant(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
        
        # If we're stuck, try a restart with different seed
        if not improved:
            seed = rng.randint(0, 1000000)
            new_cover = greedy_cover(seed)
            new_cover = remove_redundant(new_cover)
            if len(new_cover) < best_len:
                best_len = len(new_cover)
                best_blocks = list(new_cover)
                blocks = list(best_blocks)
    
    return [list(b) for b in best_blocks]
