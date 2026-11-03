import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    points = list(range(v))
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute masks for all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if block i covers t_subset j
    block_masks = [0] * n_blocks
    # Also store list of t-subset indices for each block (for faster count updates)
    block_t_indices = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices
    
    # Precompute for each t-subset, which blocks cover it (reverse index)
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_blocks):
        for idx in block_t_indices[i]:
            t_to_blocks[idx].append(i)
    
    # Build sparse matrix A of size n_k x n_t using numpy for faster gain computation
    # A[i, j] = 1 if t_subs[j] is subset of k_subs[i]
    row_indices = []
    col_indices = []
    data = []
    
    for i, block in enumerate(all_blocks):
        for idx in block_t_indices[i]:
            row_indices.append(i)
            col_indices.append(idx)
            data.append(1)
            
    # Use numpy for fast matrix-vector products
    # Create a dense matrix if it's not too large, otherwise use sparse
    # For v=19, k=8, t=5: n_blocks = C(19,8) = 75582, n_t = C(19,5) = 11628
    # Dense matrix would be 75582 * 11628 * 1 byte = ~878 MB, which is too large
    # So we'll use a different approach: precompute gains differently
    
    # Actually, let's use a more efficient approach with numpy
    # We'll represent the coverage as a 2D numpy array if feasible
    # For now, let's stick with the bit-mask approach but optimize the greedy step
    
    def greedy_cover(weights=None, seed_blocks=None, randomized=False, random_factor=0.8):
        """Greedy set cover with optional Lagrangian weights and randomization.
        If weights is None, uses simple max-gain.
        If weights is a list, uses weighted gain: sum of weights for uncovered t-subsets covered.
        If randomized is True, picks from top candidates randomly.
        """
        uncovered_mask = (1 << n_t) - 1
        selected = []
        
        # Add seed blocks if provided
        if seed_blocks:
            for idx in seed_blocks:
                selected.append(idx)
                uncovered_mask &= ~block_masks[idx]
        
        while uncovered_mask:
            if time.time() > deadline:
                break
            
            best_block_idx = -1
            best_score = 0
            candidates = []
            
            for i in range(n_blocks):
                bm = block_masks[i]
                if bm == 0:
                    continue
                # Compute gain
                if weights is None:
                    gain = (uncovered_mask & bm).bit_count()
                else:
                    # Weighted gain: sum of weights for uncovered t-subsets in this block
                    inter = uncovered_mask & bm
                    if inter == 0:
                        continue
                    # For small numbers of bits, iterate
                    w_sum = 0
                    m = inter
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        w_sum += weights[idx]
                        m ^= lsb
                    gain = w_sum
                
                if gain > best_score:
                    best_score = gain
                    best_block_idx = i
                    candidates = [i]
                elif gain == best_score and gain > 0:
                    candidates.append(i)
            
            if best_block_idx == -1:
                break
            
            if randomized and candidates:
                # Pick from top candidates
                if random_factor < 1.0:
                    # Pick randomly from candidates with gain >= random_factor * best_score
                    threshold = best_score * random_factor
                    filtered = [c for c in candidates if weights is None or True]
                    # For simplicity, just pick randomly from all candidates with max gain
                    chosen = random.choice(candidates)
                else:
                    chosen = best_block_idx
            else:
                chosen = best_block_idx
                
            selected.append(chosen)
            uncovered_mask &= ~block_masks[chosen]
            
        return selected
    
    def local_search(selected):
        """Aggressive local search: removal + swap-based improvement."""
        if not selected:
            return selected
        
        # Initialize counts
        counts = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected:
            for idx in block_t_indices[b_idx]:
                counts[idx] += 1
        
        selected_set = set(selected)
        
        # Phase 1: Iterative removal of redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for b_idx in selected:
                mask = block_masks[b_idx]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    selected_set.discard(b_idx)
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected
        
        # Phase 2: Swap-based local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)
            
            # Recompute counts for current selected
            counts = np.zeros(n_t, dtype=np.int32)
            for b_idx in new_selected:
                for idx in block_t_indices[b_idx]:
                    counts[idx] += 1
            
            # For each block in selected, check if we can swap it out
            for pos, b_idx in enumerate(list(new_selected)):
                if time.time() > deadline:
                    break
                # Find t-subsets uniquely covered by b_idx
                unique_indices = []
                m = block_masks[b_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        unique_indices.append(idx)
                    m ^= lsb
                
                if not unique_indices:
                    # This block is redundant, remove it
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    new_selected.remove(b_idx)
                    new_set.discard(b_idx)
                    improved = True
                    break
                
                # Try to find a block B (not in new_set) that covers all unique_indices
                best_swap = -1
                best_gain = 0
                for candidate in t_to_blocks[unique_indices[0]]:
                    if candidate in new_set:
                        continue
                    # Check if candidate covers all unique_indices
                    covers_all = True
                    for ui in unique_indices:
                        if not (block_masks[candidate] >> ui) & 1:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    # Compute additional gain
                    extra = 0
                    cand_mask = block_masks[candidate]
                    orig_mask = block_masks[b_idx]
                    diff = cand_mask & ~orig_mask
                    m = diff
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if counts[idx] == 0:
                            extra += 10
                        elif counts[idx] == 1:
                            extra += 1
                        m ^= lsb
                    
                    if extra > best_gain:
                        best_gain = extra
                        best_swap = candidate
                
                if best_swap != -1 and best_gain > 0:
                    # Perform the swap
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    for idx in block_t_indices[best_swap]:
                        counts[idx] += 1
                    new_selected.remove(b_idx)
                    new_selected.append(best_swap)
                    new_set.discard(b_idx)
                    new_set.add(best_swap)
                    improved = True
                    break
        
            selected = new_selected
        
        return selected
    
    # Multi-restart with Lagrangian relaxation and randomized restarts
    best_solution = None
    best_len = float('inf')
    
    # First: simple greedy
    sol1 = greedy_cover()
    sol1 = local_search(sol1)
    if sol1:
        best_solution = sol1
        best_len = len(sol1)
    
    # Second: Lagrangian-guided greedy
    weights = [1.0] * n_t
    for iteration in range(5):
        if time.time() > deadline - 30:
            break
        
        sol = greedy_cover(weights)
        if not sol:
            break
        
        covered_mask = 0
        for b_idx in sol:
            covered_mask |= block_masks[b_idx]
        
        new_weights = [0.0] * n_t
        m = (1 << n_t) - 1
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if not (covered_mask >> idx) & 1:
                new_weights[idx] = weights[idx] * 2.0
            else:
                new_weights[idx] = weights[idx] * 0.5
            m ^= lsb
        weights = new_weights
        
        sol = local_search(sol)
        if sol and len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Third: Randomized greedy with different random factors
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        
        # Vary the random factor for diversity
        random_factor = 0.6 + 0.1 * trial
        sol = greedy_cover(randomized=True, random_factor=random_factor)
        if sol:
            sol = local_search(sol)
            if sol and len(sol) < best_len:
                best_solution = sol
                best_len = len(sol)
    
    # Fourth: Randomized restarts with random seed blocks
    for trial in range(3):
        if time.time() > deadline - 15:
            break
        
        # Pick a few random blocks as seeds
        n_seeds = random.randint(5, 20)
        seed_blocks = random.sample(range(n_blocks), min(n_seeds, n_blocks))
        sol = greedy_cover(seed_blocks=seed_blocks)
        if sol:
            sol = local_search(sol)
            if sol and len(sol) < best_len:
                best_solution = sol
                best_len = len(sol)
    
    # Final: Try to improve best_solution with more aggressive local search
    if best_solution and time.time() < deadline - 5:
        best_solution = local_search(best_solution)
    
    # Convert indices back to blocks
    if best_solution:
        return [list(all_blocks[i]) for i in best_solution]
    else:
        return []
