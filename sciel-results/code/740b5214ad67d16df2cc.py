"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    """Construct the Witt design S(5,6,12) with 132 blocks."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    
    # Precompute all 5-subsets and map to indices for bitset operations
    all_5subsets = list(combinations(points, 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # Precompute the 5-subset indices for each 6-block
    block_subsets = []
    for b in all_blocks:
        subs = frozenset(combinations(b, 5))
        idxs = [subset_to_idx[s] for s in subs]
        block_subsets.append(idxs)
        
    best_solution = []
    
    def backtrack(covered_mask, current_blocks, candidate_indices):
        nonlocal best_solution
        if time.time() > deadline:
            return False
        
        if len(current_blocks) == 132:
            best_solution = current_blocks[:]
            return True
            
        # Heuristic: Pick the candidate that covers the most uncovered 5-subsets
        scored_candidates = []
        for i, b_idx in enumerate(candidate_indices):
            subs = block_subsets[b_idx]
            # Check conflict and count uncovered
            uncovered_count = 0
            valid = True
            for s_idx in subs:
                if covered_mask & (1 << s_idx):
                    valid = False
                    break
                else:
                    uncovered_count += 1
            
            if valid:
                scored_candidates.append((uncovered_count, i, b_idx))
                
        if not scored_candidates:
            return False
            
        # Sort by uncovered_count descending (greedy choice)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Try the best candidate first
        for _, i, b_idx in scored_candidates:
            subs = block_subsets[b_idx]
            new_mask = covered_mask
            for s_idx in subs:
                new_mask |= (1 << s_idx)
                
            current_blocks.append(b_idx)
            
            # Create list of remaining candidates
            # Optimization: Since we are doing DFS, we pass the rest of the list
            # excluding the current one.
            remaining = [idx for j, idx in enumerate(candidate_indices) if j != i]
            
            if backtrack(new_mask, current_blocks, remaining):
                return True
            current_blocks.pop()
            
        return False

    candidates = list(range(len(all_blocks)))
    
    if backtrack(0, [], candidates):
        return [list(all_blocks[idx]) for idx in best_solution]
    else:
        # Fallback to simple greedy if backtracking fails or times out
        covered_mask = 0
        blocks = []
        for b_idx in candidates:
            subs = block_subsets[b_idx]
            conflict = False
            for s_idx in subs:
                if covered_mask & (1 << s_idx):
                    conflict = True
                    break
            if not conflict:
                blocks.append(b_idx)
                for s_idx in subs:
                    covered_mask |= (1 << s_idx)
        return [list(all_blocks[idx]) for idx in blocks]
# EVOLVE-BLOCK-END
