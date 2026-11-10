import itertools
import time
import sys
from collections import defaultdict
import math

sys.setrecursionlimit(10000)

def construct(v, k, t):
    """
    Constructs a covering design using iterative deepening backtracking 
    with bottleneck heuristic and lower-bound pruning.
    """
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    
    # Map each t-subset index to the list of blocks covering it
    # And map each block to the list of t-subset indices it covers
    block_t_indices = {}
    t_subset_to_blocks = defaultdict(list)
    
    for b in all_blocks:
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_subset_to_blocks[idx].append(b)
        block_t_indices[b] = covered_indices
        
    # 1. Greedy Initial Solution
    uncovered = set(range(num_t_subsets))
    greedy_blocks = []
    while uncovered:
        best_block = None
        max_cov = 0
        for b in all_blocks:
            cov = 0
            for idx in block_t_indices[b]:
                if idx in uncovered:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_block = b
        if best_block is None or max_cov == 0:
            break
        greedy_blocks.append(best_block)
        for idx in block_t_indices[best_block]:
            uncovered.discard(idx)
            
    best_size = len(greedy_blocks)
    best_solution = list(greedy_blocks)
    
    # If we found a perfect greedy cover (unlikely for open problems), return
    if best_size == 0:
        return [list(b) for b in best_solution]

    # 2. Iterative Deepening Search
    # We try to find a solution of size best_size - 1, then best_size - 2, etc.
    # We limit the number of attempts to avoid timeout.
    
    def search(max_depth, current_depth, current_blocks, used_blocks, uncovered_set):
        """
        DFS to find a solution with exactly max_depth blocks.
        Returns True if found, updating best_solution.
        """
        nonlocal best_size, best_solution
        
        if time.time() - start_time > time_limit:
            return False
            
        if current_depth == max_depth:
            if not uncovered_set:
                best_solution = list(current_blocks)
                best_size = max_depth
                return True
            return False
            
        # Pruning: Not enough depth left to cover remaining sets
        # Lower bound: Each new block covers at most max_cov_per_block new sets.
        # A tighter bound: The "bottleneck" set has 'avail' blocks. 
        # If avail == 0, we are stuck.
        
        # Find bottleneck: uncovered set with fewest available blocks
        min_avail = float('inf')
        bottleneck_idx = -1
        
        # Optimization: Only check a subset of uncovered if large? 
        # For v=19, num_t_subsets ~ 11k. Scanning all is O(11k).
        # We can optimize by maintaining a list of uncovered and checking degrees.
        
        for idx in uncovered_set:
            avail = 0
            for b in t_subset_to_blocks[idx]:
                if b not in used_blocks:
                    avail += 1
            if avail == 0:
                return False # Dead end
            if avail < min_avail:
                min_avail = avail
                bottleneck_idx = idx
                
        if bottleneck_idx == -1:
            return False
            
        # Pruning: Even if we pick the best block for the bottleneck, 
        # can we cover the rest?
        # Simple bound: remaining sets = len(uncovered_set).
        # Depth remaining = max_depth - current_depth - 1.
        # Max coverage per block is roughly C(k, t) = 56.
        # If len(uncovered_set) > (max_depth - current_depth - 1) * 56, prune.
        # This is a weak bound but fast.
        
        # Get candidates for the bottleneck
        candidates = []
        for b in t_subset_to_blocks[bottleneck_idx]:
            if b not in used_blocks:
                # Score: how many *new* sets does this block cover?
                score = 0
                for idx in block_t_indices[b]:
                    if idx in uncovered_set:
                        score += 1
                candidates.append((score, b))
                
        # Sort by score descending
        candidates.sort(reverse=True)
        
        # Limit branching to top K to control explosion
        # K=3 or 4 is reasonable.
        for score, b in candidates[:4]:
            current_blocks.append(b)
            used_blocks.add(b)
            
            newly_covered = []
            for idx in block_t_indices[b]:
                if idx in uncovered_set:
                    uncovered_set.remove(idx)
                    newly_covered.append(idx)
                    
            if search(max_depth, current_depth + 1, current_blocks, used_blocks, uncovered_set):
                return True
                
            current_blocks.pop()
            used_blocks.remove(b)
            for idx in newly_covered:
                uncovered_set.add(idx)
                
        return False

    # Try to improve the solution size
    # Start from best_size - 1 and go down
    # We must be careful with time.
    target_size = best_size - 1
    attempts = 0
    max_attempts = 50 # Safety valve
    
    while target_size > 0 and attempts < max_attempts:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Reset state for search
        used_blocks = set()
        current_blocks = []
        uncovered_set = set(range(num_t_subsets))
        
        # Use a timeout wrapper or check inside
        found = search(target_size, 0, current_blocks, used_blocks, uncovered_set)
        
        if found:
            target_size -= 1
            attempts = 0 # Reset attempts if we found one, allow more
        else:
            # If not found, we can't guarantee finding a smaller one quickly
            # Break to save time for returning the current best
            break
        attempts += 1

    return [list(b) for b in best_solution]
