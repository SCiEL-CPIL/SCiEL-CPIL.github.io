import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We should aim to finish well before.
    # Let's use 140s as the hard stop for the logic.
    deadline = start_time + 140.0

    # Precompute all t-subsets and their bitmasks
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    t_sub_masks = []
    mask_to_idx = {}
    for i, sub in enumerate(all_t_subs):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_sub_masks.append(mask)
        mask_to_idx[mask] = i

    # Precompute (t-1) subsets for fast gain calculation?
    # We need to iterate over (t-1) subsets of the current block B.
    # Since B grows from size t to k, the number of (t-1) subsets is C(s, t-1).
    # For v=19, k=8, t=5, max s=8, C(8,4)=70. This is small.
    # We can just generate them on the fly using itertools.

    def get_gain(block, x, is_uncovered):
        """
        Calculate how many new t-subsets are covered if x is added to block.
        block: list of points
        x: point to add
        is_uncovered: boolean numpy array or list
        """
        gain = 0
        x_bit = 1 << x
        # Generate all (t-1) subsets of the current block
        # Note: block currently has size s. We are adding x.
        # The new t-subsets must include x and (t-1) points from block.
        for rest in itertools.combinations(block, t - 1):
            # Create mask for 'rest'
            m = 0
            for p in rest:
                m |= (1 << p)
            full_mask = m | x_bit
            idx = mask_to_idx.get(full_mask)
            if idx is not None and is_uncovered[idx]:
                gain += 1
        return gain

    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        is_uncovered = np.ones(n_t_subs, dtype=bool)
        uncovered_count = n_t_subs
        blocks = []
        
        # Precompute point bits
        point_bits = [1 << i for i in range(v)]

        while uncovered_count > 0:
            # Find an uncovered t-subset to start the block
            # We pick the first one for determinism in the base case, 
            # but for diversity, we could pick randomly. 
            # Let's pick the first one to ensure progress, 
            # but randomize the growth order.
            
            # Find an index where is_uncovered is True
            # Using np.nonzero is fast
            uncov_indices = np.nonzero(is_uncovered)[0]
            if len(uncov_indices) == 0:
                break
                
            # Pick a random uncovered subset for diversity? 
            # Or just the first one? 
            # Picking the first one (lexicographically smallest) is standard.
            # To get diverse solutions, we can randomize which uncovered subset we pick.
            start_idx = uncov_indices[random.randint(0, len(uncov_indices) - 1)]
            
            # Extract points from the combination
            # We need the actual points, not just the index.
            # We have all_t_subs[start_idx]
            block = list(all_t_subs[start_idx])
            block_set = set(block)
            
            # Mark this subset as covered immediately? 
            # No, we mark them when we finalize the block.
            # But for gain calculation, we should consider it "covered" by this new block 
            # so we don't count it as gain? 
            # Actually, gain is "newly covered". 
            # The starting subset is currently uncovered. 
            # Adding points will cover it.
            # It's better to just calculate gain for *other* subsets.
            
            # Grow the block
            while len(block) < k:
                best_x = -1
                best_gain = -1
                candidates = []
                
                # Calculate gain for each candidate x
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = get_gain(block, x, is_uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        candidates = [x]
                    elif gain == best_gain:
                        candidates.append(x)
                
                if best_x == -1:
                    # No gain, just pick a random unused point
                    available = [x for x in range(v) if x not in block_set]
                    if not available:
                        break
                    best_x = random.choice(available)
                
                # Random tie-breaking
                if len(candidates) > 1:
                    best_x = random.choice(candidates)
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered status
            for t_sub in itertools.combinations(block, t):
                mask = 0
                for p in t_sub:
                    mask |= (1 << p)
                idx = mask_to_idx.get(mask)
                if idx is not None and is_uncovered[idx]:
                    is_uncovered[idx] = False
                    uncovered_count -= 1
                    
        return blocks

    def reduce_covering(blocks):
        """
        Remove redundant blocks.
        A block is redundant if every t-subset it contains is covered by at least one other block.
        """
        if not blocks:
            return blocks
        
        # Precompute the list of t-subset indices for each block
        block_t_indices = []
        for b in blocks:
            idxs = []
            for t_sub in itertools.combinations(b, t):
                mask = 0
                for p in t_sub:
                    mask |= (1 << p)
                idx = mask_to_idx.get(mask)
                if idx is not None:
                    idxs.append(idx)
            block_t_indices.append(idxs)
        
        # Coverage count for each t-subset
        cov = np.zeros(n_t_subs, dtype=np.int32)
        for idxs in block_t_indices:
            for idx in idxs:
                cov[idx] += 1
        
        removed = np.zeros(len(blocks), dtype=bool)
        changed = True
        
        # Iteratively remove redundant blocks
        # We need to be careful: removing a block changes the coverage counts.
        while changed:
            changed = False
            for i in range(len(blocks)):
                if removed[i]:
                    continue
                idxs = block_t_indices[i]
                if not idxs:
                    removed[i] = True
                    changed = True
                    continue
                    
                # Check if any t-subset in this block has coverage count == 1
                # If yes, this block is essential for that subset.
                is_essential = False
                for idx in idxs:
                    if cov[idx] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    # Remove this block
                    removed[i] = True
                    changed = True
                    # Update coverage counts
                    for idx in idxs:
                        cov[idx] -= 1
        
        result = [blocks[i] for i in range(len(blocks)) if not removed[i]]
        return result

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple restarts
    # How many? Depends on how fast one run is.
    # With v=19, k=8, t=5, greedy is quite fast.
    # Let's try to run as many as possible within the time limit.
    
    attempt = 0
    while time.time() < deadline - 5.0: # 5s safety margin for reduction and final checks
        attempt += 1
        # Use different seeds for diversity
        seed = attempt * 12345 + 67890
        blocks = greedy_construct(seed)
        
        if not blocks:
            continue
            
        # Verify coverage (optional, but good for safety)
        # The greedy algorithm ensures coverage by construction if it terminates correctly.
        # But let's assume it's valid.
        
        # Reduce
        reduced_blocks = reduce_covering(blocks)
        
        if len(reduced_blocks) < best_size:
            best_size = len(reduced_blocks)
            best_blocks = reduced_blocks
            
            # If we hit a very good score, we might want to stop early?
            # Theoretical lower bound is 264. Best known is 335.
            # If we get below 335, that's a breakthrough.
            # Let's keep searching until time runs out to be sure.
            
        # If time is running low, break
        if time.time() > deadline - 10.0:
            break

    if best_blocks is None:
        # Fallback: run one deterministic greedy without randomness
        blocks = greedy_construct(None)
        return reduce_covering(blocks)
        
    return best_blocks

