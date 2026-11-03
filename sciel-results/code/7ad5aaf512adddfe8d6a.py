"""Improved program for C(19,7,4) covering design.
Uses branch-and-bound with bitset-based coverage tracking and lower bounds
from maximal independent sets of uncovered t-subsets."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 120s
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to its index for O(1) lookup
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks and their coverage bitsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute which t-subsets it covers as a bitset
    block_coverage = []
    for block in all_blocks:
        block_set = set(block)
        cov_mask = 0
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            cov_mask |= (1 << idx)
        block_coverage.append(cov_mask)
    
    # Precompute for each t-subset, which blocks cover it
    tsub_blocks = [[] for _ in range(n_t)]
    for b_idx, block in enumerate(all_blocks):
        block_set = set(block)
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            tsub_blocks[idx].append(b_idx)
    
    def time_ok():
        return time.time() < deadline
    
    # Branch and bound
    best_solution = None
    best_size = float('inf')
    
    # Lower bound: at least ceil(uncovered / max_coverage_per_block)
    # More refined: use maximal independent sets
    def compute_lower_bound(uncovered_mask):
        """Compute a lower bound on additional blocks needed."""
        if uncovered_mask == 0:
            return 0
        
        # Count uncovered t-subsets
        uncovered_count = bin(uncovered_mask).count('1')
        
        # Maximum number of t-subsets a single block can cover
        max_cov = 0
        for cov in block_coverage:
            overlap = cov & uncovered_mask
            cnt = bin(overlap).count('1')
            if cnt > max_cov:
                max_cov = cnt
        
        if max_cov == 0:
            return float('inf')
        
        return (uncovered_count + max_cov - 1) // max_cov
    
    def branch_and_bound(selected_blocks, current_mask, depth, max_depth):
        nonlocal best_solution, best_size
        
        if not time_ok():
            return
        
        # Check if we've found a solution
        if current_mask == 0:
            if len(selected_blocks) < best_size:
                best_size = len(selected_blocks)
                best_solution = list(selected_blocks)
            return
        
        # Prune if we've already exceeded best
        if len(selected_blocks) >= best_size:
            return
        
        # Compute lower bound
        lb = compute_lower_bound(current_mask)
        if len(selected_blocks) + lb >= best_size:
            return
        
        # Find uncovered t-subsets
        uncovered_idx = None
        for i in range(n_t):
            if not (current_mask >> i) & 1:
                uncovered_idx = i
                break
        
        if uncovered_idx is None:
            return
        
        # Get blocks that cover this t-subset, sorted by coverage of uncovered
        candidates = tsub_blocks[uncovered_idx]
        candidates.sort(key=lambda b: bin(block_coverage[b] & current_mask).count('1'), reverse=True)
        
        # Try each candidate block
        for b_idx in candidates:
            if not time_ok():
                return
            
            new_mask = current_mask & ~block_coverage[b_idx]
            selected_blocks.append(b_idx)
            branch_and_bound(selected_blocks, new_mask, depth + 1, max_depth)
            selected_blocks.pop()
            
            # Early termination if we found optimal
            if len(selected_blocks) >= best_size:
                break
    
    # Run B&B with time limit
    initial_mask = (1 << n_t) - 1  # All t-subsets uncovered
    branch_and_bound([], initial_mask, 0, 200)
    
    if best_solution is not None:
        return [list(all_blocks[b]) for b in best_solution]
    
    # Fallback: greedy if B&B didn't find solution in time
    covered = [False] * n_t
    blocks = []
    
    def add_block_greedy(block):
        block_set = frozenset(block)
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            covered[idx] = True
        blocks.append(tuple(block))
    
    while True:
        if not time_ok():
            break
        
        uncovered_idx = None
        for i in range(n_t):
            if not covered[i]:
                uncovered_idx = i
                break
        if uncovered_idx is None:
            break
        
        seed_subset = all_t_subsets[uncovered_idx]
        block_set = set(seed_subset)
        
        while len(block_set) < k:
            if not time_ok():
                break
            
            best_x = None
            best_gain = -1
            for x in all_points:
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block_set, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx[sub]
                    if not covered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block_set.add(best_x)
        
        if not time_ok():
            break
        
        block = tuple(sorted(block_set))
        add_block_greedy(block)
    
    return [list(b) for b in blocks]

# EVOLVE-BLOCK-END
