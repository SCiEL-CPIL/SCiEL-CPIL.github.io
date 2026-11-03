import itertools
import random
import time

# Precomputation for the specific case v=19, k=7, t=4
# This allows for very fast greedy and local search steps.

def _precompute_covering_data(v, k, t):
    """
    Precomputes bitmasks for blocks and lists of t-subsets for each block.
    Returns:
    - t_subsets: List of all t-subsets (as tuples)
    - t_to_idx: Dict mapping t-subset tuple to index
    - block_combinations: List of all k-block indices (combinations of range(v) of size k)
    - block_masks: List of bitmasks (int) where bit j is set if t_subsets[j] is in the block
    - block_tsub_indices: List of lists of indices of t-subsets contained in the block
    """
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    block_combinations = list(itertools.combinations(range(v), k))
    num_blocks = len(block_combinations)
    
    block_masks = [0] * num_blocks
    block_tsub_indices = [[] for _ in range(num_blocks)]
    
    for i, block in enumerate(block_combinations):
        # Find all t-subsets within this block
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_tsub_indices[i] = indices
        
    return t_subsets, t_to_idx, block_combinations, block_masks, block_tsub_indices, num_t

# Global cache for precomputation to avoid re-doing it if called multiple times
_PRECOMP_CACHE = {}

def construct(v, k, t):
    key = (v, k, t)
    if key not in _PRECOMP_CACHE:
        if v > 20 or k > 10: # Fallback for larger problems to avoid memory blowup
            # Simple greedy fallback
            uncovered = set(itertools.combinations(range(v), t))
            blocks = []
            while uncovered:
                start_tsub = next(iter(uncovered))
                block = list(start_tsub)
                while len(block) < k:
                    best_point = None
                    best_gain = -1
                    for p in range(v):
                        if p in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t-1):
                            ts = tuple(sorted(rest + (p,)))
                            if ts in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = p
                    if best_point is None:
                        break
                    block.append(best_point)
                block.sort()
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
            return blocks
            
        _PRECOMP_CACHE[key] = _precompute_covering_data(v, k, t)
    
    t_subsets, t_to_idx, block_combinations, block_masks, block_tsub_indices, num_t = _PRECOMP_CACHE[key]
    
    # Number of trials for randomized greedy (Primal-Dual with stochastic tie-breaking)
    num_trials = 30
    best_sol = None
    best_len = float('inf')
    
    full_mask = (1 << num_t) - 1
    start_time = time.time()
    time_limit = 115 # Leave 5s for overhead
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(trial)
        uncovered = full_mask
        sol = []
        
        # Greedy Phase (Primal-Dual Step: Raise duals uniformly, select tight block)
        # Equivalent to: Select block covering max uncovered elements, with random tie-breaking
        while uncovered:
            max_covered = 0
            candidates = []
            
            # Iterate through all blocks to find those with max coverage
            for i in range(len(block_combinations)):
                covered = block_masks[i] & uncovered
                if covered:
                    cnt = covered.bit_count()
                    if cnt > max_covered:
                        max_covered = cnt
                        candidates = [i]
                    elif cnt == max_covered:
                        candidates.append(i)
            
            if not candidates:
                break
                
            # Stochastic selection to explore different solution paths
            best_block_idx = random.choice(candidates)
            sol.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
        # Local Search: Remove Redundant Blocks
        changed = True
        while changed:
            changed = False
            freq = [0] * num_t
            for b_idx in sol:
                for t_idx in block_tsub_indices[b_idx]:
                    freq[t_idx] += 1
            
            new_sol = []
            for b_idx in sol:
                is_redundant = True
                for t_idx in block_tsub_indices[b_idx]:
                    if freq[t_idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_sol.append(b_idx)
            
            if len(new_sol) < len(sol):
                sol = new_sol
                changed = True
            else:
                changed = False
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol.copy()
            
    if best_sol is None:
        return []
        
    result_blocks = []
    for b_idx in best_sol:
        result_blocks.append(list(block_combinations[b_idx]))
        
    return result_blocks
