import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)
    
    # Compute masks: for each block, which t-subsets does it cover?
    # Use a list of (block_tuple, mask)
    block_data = []
    for block in all_k_blocks:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_data.append((block, mask))
    
    block_masks = [m for _, m in block_data]
    full_mask = (1 << n_t) - 1
    
    def popcount(x):
        return bin(x).count('1')
    
    def greedy_construct(use_log=False, rng=None):
        """Greedy set cover with optional log-gain."""
        uncovered = full_mask
        selected = []
        remaining = list(range(n_k))
        
        while uncovered != 0:
            if time.time() > deadline - 30:
                break
            best_idx = -1
            best_score = -1
            for i in remaining:
                gain = (block_masks[i] & uncovered).bit_count()
                if gain == 0:
                    continue
                if use_log:
                    # Log-gain: prefer blocks that cover many new subsets
                    # relative to how many they already cover (diminishing returns)
                    total_covered = block_masks[i].bit_count()
                    already_covered = total_covered - gain
                    score = gain / (1 + already_covered * 0.1)
                else:
                    score = gain
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            remaining.remove(best_idx)
        
        return selected
    
    def prune_blocks(selected_indices):
        """Remove redundant blocks iteratively using critical-count tracking."""
        if not selected_indices:
            return selected_indices
        
        current = selected_indices[:]
        
        # Build cover counts and per-block critical counts
        cover_count = [0] * n_t
        for idx in current:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb
        
        # Verify full coverage
        for i in range(n_t):
            if cover_count[i] == 0:
                return selected_indices
        
        # critical_count[i] = number of t-subsets that block i uniquely covers
        critical_count = [0] * len(current)
        for i, idx in enumerate(current):
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                if cover_count[k2] == 1:
                    critical_count[i] += 1
                m ^= lsb
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_current = []
            for i, idx in enumerate(current):
                if critical_count[i] == 0:
                    # Safe to remove: update cover_count and critical_count of remaining blocks
                    m = block_masks[idx]
                    while m:
                        lsb = m & -m
                        k2 = lsb.bit_length() - 1
                        cover_count[k2] -= 1
                        if cover_count[k2] == 1:
                            # This t-subset is now uniquely covered; find which block
                            # (we'll handle this lazily by recomputing at end of pass)
                            pass
                        m ^= lsb
                    improved = True
                else:
                    new_current.append(idx)
            
            if not improved:
                break
            
            current = new_current
            # Recompute critical_count from scratch (cheap enough)
            critical_count = [0] * len(current)
            for i, idx in enumerate(current):
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    if cover_count[k2] == 1:
                        critical_count[i] += 1
                    m ^= lsb
        
        return current
    
    def local_search(selected_indices, max_time=30):
        """Local search: try deletions and 1-for-1 swaps using cover-count deltas."""
        if not selected_indices:
            return selected_indices
        
        current = prune_blocks(selected_indices)
        current_set = set(current)
        best = current[:]
        best_len = len(best)
        
        # Precompute cover counts for current solution
        cover_count = [0] * n_t
        for idx in current:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                cover_count[k2] += 1
                m ^= lsb
        
        # Pass 1: Try deletion of each block
        for i in range(len(current)):
            if time.time() > deadline - 15:
                break
            idx = current[i]
            m = block_masks[idx]
            # Check if any t-subset has cover_count == 1 (critical)
            is_critical = False
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                if cover_count[k2] == 1:
                    is_critical = True
                    break
                m ^= lsb
            
            if not is_critical:
                # Remove it
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    cover_count[k2] -= 1
                    m ^= lsb
                current = [x for j, x in enumerate(current) if j != i]
                current_set.discard(idx)
                best_len -= 1
                # Re-prune to catch further removals
                current = prune_blocks(current)
                current_set = set(current)
                best_len = len(current)
                best = current[:]
                # Recompute cover_count
                cover_count = [0] * n_t
                for idx2 in current:
                    m2 = block_masks[idx2]
                    while m2:
                        lsb = m2 & -m2
                        k2 = lsb.bit_length() - 1
                        cover_count[k2] += 1
                        m2 ^= lsb
        
        # Pass 2: Try 1-for-1 swaps
        if time.time() < deadline - 15:
            for i in range(len(current)):
                if time.time() > deadline - 15:
                    break
                removed_idx = current[i]
                # Compute what becomes uncovered if we remove this block
                m = block_masks[removed_idx]
                uncovered_mask = 0
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    if cover_count[k2] == 1:
                        uncovered_mask |= (1 << k2)
                    m ^= lsb
                
                if uncovered_mask == 0:
                    # Already handled in deletion pass; skip
                    continue
                
                # Find best replacement block
                best_add = -1
                best_add_gain = -1
                for j in range(n_k):
                    if j in current_set:
                        continue
                    gain = (block_masks[j] & uncovered_mask).bit_count()
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = j
                    if gain == uncovered_mask.bit_count():
                        break
                
                if best_add >= 0 and best_add_gain > 0:
                    # Try the swap
                    trial = current[:]
                    trial[i] = best_add
                    pruned = prune_blocks(trial)
                    if len(pruned) < best_len:
                        current = pruned
                        current_set = set(current)
                        best_len = len(current)
                        best = current[:]
                        # Recompute cover_count
                        cover_count = [0] * n_t
                        for idx2 in current:
                            m2 = block_masks[idx2]
                            while m2:
                                lsb = m2 & -m2
                                k2 = lsb.bit_length() - 1
                                cover_count[k2] += 1
                                m2 ^= lsb
        
        return best
    
    # Phase 1: Multiple greedy constructions with different randomizations
    best_solution = None
    best_len = float('inf')
    
    # First, a deterministic greedy
    sol = greedy_construct(use_log=False)
    sol = prune_blocks(sol)
    sol = local_search(sol)
    best_solution = sol
    best_len = len(sol)
    
    # Try log-gain greedy
    if time.time() < deadline - 40:
        sol = greedy_construct(use_log=True)
        sol = prune_blocks(sol)
        sol = local_search(sol)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Phase 2: ILS - Iterated Local Search
    rng = random.Random(42)
    ils_iter = 0
    while time.time() < deadline - 25 and ils_iter < 60:
        ils_iter += 1
        
        # Perturbation: randomly modify the current solution
        perturbed = best_solution[:]
        if len(perturbed) > 2:
            # Remove 1-3 random blocks
            n_remove = min(rng.randint(1, 3), len(perturbed) - 1)
            to_remove = rng.sample(range(len(perturbed)), n_remove)
            to_remove_set = set(to_remove)
            perturbed = [x for i, x in enumerate(perturbed) if i not in to_remove_set]
            
            # With some probability, also replace one remaining block with a random other block
            if rng.random() < 0.4 and len(perturbed) > 1:
                pos = rng.randint(0, len(perturbed) - 1)
                # Pick a random block not already in perturbed
                existing = set(perturbed)
                candidates = [j for j in range(n_k) if j not in existing]
                if candidates:
                    perturbed[pos] = rng.choice(candidates)
        
        # Re-optimize: add blocks to cover uncovered, then prune, then local search
        # First check coverage
        cov = [0] * n_t
        for idx in perturbed:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                cov[k2] += 1
                m ^= lsb
        
        # Find uncovered
        uncovered = full_mask
        for k2 in range(n_t):
            if cov[k2] == 0:
                uncovered |= (1 << k2)
        
        # Greedily add blocks to cover uncovered
        perturbed_set = set(perturbed)
        while uncovered != 0 and time.time() < deadline - 15:
            best_idx = -1
            best_gain = -1
            for j in range(n_k):
                if j in perturbed_set:
                    continue
                gain = (block_masks[j] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
            if best_idx == -1 or best_gain == 0:
                break
            perturbed.append(best_idx)
            perturbed_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        # Prune and local search
        perturbed = prune_blocks(perturbed)
        perturbed = local_search(perturbed)
        
        if len(perturbed) < best_len:
            best_solution = perturbed
            best_len = len(perturbed)
    
    # Final validation and return
    # Verify coverage
    final_cov = [0] * n_t
    for idx in best_solution:
        m = block_masks[idx]
        while m:
            lsb = m & -m
            k2 = lsb.bit_length() - 1
            final_cov[k2] += 1
            m ^= lsb
    
    # If not fully covered (shouldn't happen), add blocks
    for k2 in range(n_t):
        if final_cov[k2] == 0:
            # Find a block covering this t-subset
            for j in range(n_k):
                if (block_masks[j] >> k2) & 1:
                    if j not in best_solution:
                        best_solution.append(j)
                    break
    
    blocks = [list(block_data[i][0]) for i in best_solution]
    
    # Ensure distinct blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_blocks.append(b)
    
    return unique_blocks
