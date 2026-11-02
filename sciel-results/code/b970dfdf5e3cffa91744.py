"""Covering design C(19,8,5) - Fast Bitmask Greedy + Local Search.

Uses Python integer bitmasks for efficient coverage tracking.
Precomputes bitmask for each of the C(19,8) blocks.
Runs randomized greedy with tie-breaking and local search (removal).
"""
import itertools
import random
import time


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Fallback for other parameters (simple greedy, not optimized)
    if not (v == 19 and k == 8 and t == 5):
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            target = min(uncovered)
            tset = set(target)
            remaining = [x for x in range(v) if x not in tset]
            need = k - t
            best_block, best_gain = None, -1
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain, best_block = gain, block
            if best_block is None:
                break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                break
        return blocks

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    full_mask = (1 << n_t) - 1
    
    # Map t-subsets to indices
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute bitmask for each block
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_mask[j] is an integer where bit i is 1 if block j covers t-subset i
    block_masks = [0] * n_blocks
    for j, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[j] = mask

    def run_greedy(seed, alpha=1.0):
        """Randomized greedy with bitmask coverage.
        
        alpha: temperature parameter for tie-breaking.
        alpha=0 means always pick max gain (greedy).
        alpha>0 means sample from candidates with gain >= max_gain * (1 - alpha * random())
        """
        rng = random.Random(seed)
        coverage = 0
        selected_indices = []
        selected_set = set()
        
        while coverage != full_mask:
            if time.time() > deadline - 5.0:
                return None
            
            best_gain = -1
            best_cands = []
            
            for j in range(n_blocks):
                if j in selected_set:
                    continue
                new_cov = block_masks[j] & ~coverage
                gain = new_cov.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [j]
                elif gain == best_gain:
                    best_cands.append(j)
            
            if not best_cands:
                break
                
            # Temperature-based selection: with probability alpha, pick a slightly suboptimal candidate
            if alpha > 0 and len(best_cands) > 1:
                # Find all candidates with gain >= best_gain - 1 (slightly suboptimal)
                all_cands = []
                for j in range(n_blocks):
                    if j in selected_set:
                        continue
                    new_cov = block_masks[j] & ~coverage
                    gain = new_cov.bit_count()
                    if gain >= best_gain - 1:
                        all_cands.append(j)
                if all_cands:
                    b = rng.choice(all_cands)
                else:
                    b = rng.choice(best_cands)
            else:
                b = rng.choice(best_cands)
                
            selected_indices.append(b)
            selected_set.add(b)
            coverage |= block_masks[b]
            
        return selected_indices

    def local_search_remove(selected_indices):
        """Remove redundant blocks."""
        if not selected_indices:
            return selected_indices
            
        # Compute current coverage
        coverage = 0
        for b in selected_indices:
            coverage |= block_masks[b]
            
        if coverage != full_mask:
            return selected_indices
            
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                break
                
            for i in range(len(selected_indices)):
                b = selected_indices[i]
                if (coverage & ~block_masks[b]) == full_mask:
                    selected_indices.pop(i)
                    coverage &= ~block_masks[b]
                    changed = True
                    break
                    
        return selected_indices

    def local_search_swap(selected_indices, rng, max_iterations=100):
        """Try 1-swaps to improve the solution."""
        if not selected_indices:
            return selected_indices
            
        coverage = 0
        for b in selected_indices:
            coverage |= block_masks[b]
            
        if coverage != full_mask:
            return selected_indices
            
        sol = list(selected_indices)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline - 2.0:
            improved = False
            iterations += 1
            
            # Phase 1: Remove redundant blocks
            indices = list(range(len(sol)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 2.0:
                    return sol
                b = sol[i]
                if (coverage & ~block_masks[b]) == full_mask:
                    coverage &= ~block_masks[b]
                    sol.pop(i)
                    improved = True
                    break
                    
            if improved:
                continue
                
            # Phase 2: Try 1-swaps
            # For each block, check if removing it leaves some t-subsets uncovered
            # Then try to replace it with a block that covers those uncovered t-subsets
            for i in range(len(sol)):
                if time.time() > deadline - 2.0:
                    return sol
                    
                b = sol[i]
                # What would be uncovered if we remove b?
                new_coverage = coverage & ~block_masks[b]
                if new_coverage == full_mask:
                    # b is redundant, remove it
                    coverage = new_coverage
                    sol.pop(i)
                    improved = True
                    break
                    
                # Find uncovered t-subsets
                uncovered_mask = full_mask & ~new_coverage
                if uncovered_mask == 0:
                    continue
                    
                # Try to find a replacement block that covers all uncovered t-subsets
                # and has minimal overlap with existing coverage
                best_replacement = -1
                best_score = float('inf')
                
                for j in range(n_blocks):
                    if j in sol:
                        continue
                    # Check if j covers all uncovered t-subsets
                    if (block_masks[j] & uncovered_mask) != uncovered_mask:
                        continue
                    
                    # Score: how many new t-subsets does j cover (beyond what's already covered)?
                    # We want to minimize overlap with existing coverage
                    overlap = (block_masks[j] & coverage).bit_count()
                    new_cover = (block_masks[j] & ~coverage).bit_count()
                    score = overlap - new_cover
                    
                    if score < best_score:
                        best_score = score
                        best_replacement = j
                        
                if best_replacement >= 0:
                    # Perform the swap
                    coverage = new_coverage | block_masks[best_replacement]
                    sol[i] = best_replacement
                    improved = True
                    break
                    
        return sol

    best_sol = None
    best_len = float('inf')
    
    # First run with greedy
    sol = run_greedy(0, alpha=0.0)
    if sol is not None:
        sol = local_search_remove(sol)
        best_sol = sol
        best_len = len(sol)
        
    # Randomized restarts with varying temperatures
    seed = 1
    alphas = [0.0, 0.1, 0.2, 0.3]
    
    while time.time() < deadline - 10.0:
        alpha = alphas[seed % len(alphas)]
        sol = run_greedy(seed, alpha=alpha)
        seed += 1
        if sol is None:
            continue
            
        sol = local_search_remove(sol)
        
        # Try swap local search if we have time
        if time.time() < deadline - 5.0:
            rng = random.Random(seed + 1000)
            sol_swapped = local_search_swap(sol, rng, max_iterations=50)
            if len(sol_swapped) < len(sol):
                sol = sol_swapped
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol
            
    if best_sol is None:
        return []
        
    return [list(all_blocks[b]) for b in best_sol]
