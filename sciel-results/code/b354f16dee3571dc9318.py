import itertools
import random
import numpy as np

_PRECOMP_CACHE = {}

def _precompute(v, k, t):
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    block_combinations = list(itertools.combinations(range(v), k))
    num_blocks = len(block_combinations)
    
    # For each block, list of t-subset indices it covers
    block_tsub_indices = [[] for _ in range(num_blocks)]
    for i, block in enumerate(block_combinations):
        for ts in itertools.combinations(block, t):
            block_tsub_indices[i].append(t_to_idx[ts])
    
    return t_subsets, t_to_idx, block_combinations, block_tsub_indices, num_t, num_blocks

def construct(v, k, t):
    key = (v, k, t)
    if key not in _PRECOMP_CACHE:
        if v > 20 or k > 10:
            # Fallback greedy
            uncovered = set(itertools.combinations(range(v), t))
            blocks = []
            while uncovered:
                best_block = None
                best_cover = 0
                for block in itertools.combinations(range(v), k):
                    cover = len(uncovered & set(itertools.combinations(block, t)))
                    if cover > best_cover:
                        best_cover = cover
                        best_block = block
                if best_block is None:
                    break
                blocks.append(list(best_block))
                for ts in itertools.combinations(best_block, t):
                    uncovered.discard(ts)
            return blocks
        
        _PRECOMP_CACHE[key] = _precompute(v, k, t)
    
    t_subsets, t_to_idx, block_combinations, block_tsub_indices, num_t, num_blocks = _PRECOMP_CACHE[key]
    
    # Primal-dual approach
    num_trials = 30
    best_sol = None
    best_len = float('inf')
    
    for trial in range(num_trials):
        rng = random.Random(trial)
        
        # Dual variables for each t-subset
        dual = np.zeros(num_t, dtype=np.float64)
        covered = np.zeros(num_t, dtype=bool)
        
        selected = []
        
        while not covered.all():
            # Find block with max sum of duals of uncovered t-subsets
            best_block = -1
            best_score = -1
            
            for b_idx in range(num_blocks):
                score = 0.0
                for t_idx in block_tsub_indices[b_idx]:
                    if not covered[t_idx]:
                        score += dual[t_idx]
                if score > best_score:
                    best_score = score
                    best_block = b_idx
            
            if best_block == -1:
                break
            
            # Select this block
            selected.append(best_block)
            
            # Increase duals for uncovered t-subsets in this block
            for t_idx in block_tsub_indices[best_block]:
                if not covered[t_idx]:
                    dual[t_idx] += 1.0
                    covered[t_idx] = True
            
            # Also mark as covered (they're now covered by this block)
            # The dual increase above already marks them
        
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            freq = np.zeros(num_t, dtype=np.int32)
            for b_idx in selected:
                for t_idx in block_tsub_indices[b_idx]:
                    freq[t_idx] += 1
            
            new_selected = []
            for b_idx in selected:
                redundant = True
                for t_idx in block_tsub_indices[b_idx]:
                    if freq[t_idx] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_selected.append(b_idx)
            
            if len(new_selected) < len(selected):
                selected = new_selected
                changed = True
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_sol = selected.copy()
    
    if best_sol is None:
        return []
    
    return [list(block_combinations[b_idx]) for b_idx in best_sol]
