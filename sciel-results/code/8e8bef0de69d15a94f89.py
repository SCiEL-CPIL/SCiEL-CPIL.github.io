import itertools
import time
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

def construct(v, k, t):
    """
    Constructs a covering design using a constraint-based backtracking search 
    with a bottleneck heuristic.
    """
    start_time = time.time()
    time_limit = 110
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks) and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    
    block_t_indices = {}
    t_subset_to_blocks = defaultdict(list)
    
    for b in all_blocks:
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_subset_to_blocks[idx].append(b)
        block_t_indices[b] = covered_indices
        
    # Greedy Initial Solution for Upper Bound
    uncovered_init = set(range(num_t_subsets))
    greedy_blocks = []
    while uncovered_init:
        best_block = None
        max_cov = 0
        for b in all_blocks:
            cov = 0
            for idx in block_t_indices[b]:
                if idx in uncovered_init:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_block = b
        if best_block is None or max_cov == 0:
            break
        greedy_blocks.append(best_block)
        for idx in block_t_indices[best_block]:
            uncovered_init.discard(idx)
            
    best_size = len(greedy_blocks)
    best_solution = list(greedy_blocks)
    
    # Backtracking Search
    used_blocks = set()
    uncovered = set(range(num_t_subsets))
    current_blocks = []
    
    def search():
        nonlocal best_size, best_solution
        
        if time.time() - start_time > time_limit:
            return
        
        if len(current_blocks) >= best_size:
            return
            
        if not uncovered:
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_solution = list(current_blocks)
            return
        
        # Find bottleneck: t-subset with fewest available blocks
        min_avail = float('inf')
        bottleneck_idx = -1
        
        for idx in uncovered:
            avail = 0
            for b in t_subset_to_blocks[idx]:
                if b not in used_blocks:
                    avail += 1
            if avail == 0:
                return
            if avail < min_avail:
                min_avail = avail
                bottleneck_idx = idx
                
        if bottleneck_idx == -1:
            return
            
        candidates = []
        for b in t_subset_to_blocks[bottleneck_idx]:
            if b not in used_blocks:
                score = 0
                for idx in block_t_indices[b]:
                    if idx in uncovered:
                        score += 1
                candidates.append((score, b))
                
        candidates.sort(reverse=True)
        
        # Limit branching to top 2 to control search space
        for score, b in candidates[:2]: 
            current_blocks.append(b)
            used_blocks.add(b)
            
            newly_covered = []
            for idx in block_t_indices[b]:
                if idx in uncovered:
                    uncovered.remove(idx)
                    newly_covered.append(idx)
                    
            search()
            
            current_blocks.pop()
            used_blocks.remove(b)
            for idx in newly_covered:
                uncovered.add(idx)
                
    search()
    
    return [list(b) for b in best_solution]
