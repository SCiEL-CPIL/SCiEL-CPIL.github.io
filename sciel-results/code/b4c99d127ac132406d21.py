import itertools
import time
import math

# Precomputation for the specific task parameters v=19, k=8, t=5
# However, the function signature is construct(v, k, t), so we should be general or at least robust.
# Given the time limit, precomputing for v=19, k=8, t=5 is safe if we check parameters.
# But the task says "evaluator passes this task's parameters". It implies v=19, k=8, t=5.
# To be safe and fast, we will precompute inside the function if not cached, or just assume the specific case.
# Since the prompt specifies the task is C(19,8,5), we can hardcode optimizations for this case or use a general but efficient method.
# A general method for v=19, k=8, t=5 is fast enough if optimized.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_blocks)
    
    # Precompute coverage: for each block, which t-subsets does it cover?
    # block_subsets[i] is a set of indices of t-subsets covered by block i
    block_subsets = [set() for _ in range(num_blocks)]
    
    # Also, for each t-subset, which blocks cover it?
    # subset_to_blocks[j] is a list of block indices
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    # This double loop is O(num_blocks * C(k, t))
    # 75582 * 56 ~ 4.2M operations. Fast in Python? Maybe ~1-2s.
    # We can optimize by iterating blocks and their t-subsets.
    
    for i, block in enumerate(all_k_blocks):
        block_set = set(block)
        for t_sub in itertools.combinations(block, t):
            idx = subset_to_idx[t_sub]
            block_subsets[i].add(idx)
            subset_to_blocks[idx].append(i)
            
    # Initial Greedy Solution to set a good upper bound
    def greedy_solve():
        uncovered = set(range(num_subsets))
        blocks = []
        while uncovered:
            # Pick subset with fewest blocks? Or just first?
            # Standard greedy: pick block with max gain
            best_gain = -1
            best_block_idx = -1
            # To speed up, we can sample or use the MRV heuristic on subsets
            # Let's pick a random uncovered subset to break symmetry? No, deterministic is better for reproducibility, but randomized can escape local optima.
            # Let's try picking the subset with the fewest covering blocks (MRV)
            # All subsets have same number of covering blocks in the full universe.
            # So MRV is constant.
            # Let's just pick the first uncovered subset.
            target_subset_idx = min(uncovered)
            
            candidates = subset_to_blocks[target_subset_idx]
            for b_idx in candidates:
                gain = len(uncovered & block_subsets[b_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                blocks.append(all_k_blocks[best_block_idx])
                uncovered.difference_update(block_subsets[best_block_idx])
        return blocks

    best_blocks = greedy_solve()
    best_count = len(best_blocks)
    
    # B&B Search
    # State: uncovered (set of indices), current_blocks (list of block indices)
    # We use a recursive DFS with beam width
    
    beam_width = 10  # Try top 10 candidates at each step
    
    def search(uncovered, current_blocks):
        nonlocal best_blocks, best_count
        
        if time.time() > deadline:
            return
        
        if not uncovered:
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = [all_k_blocks[i] for i in current_blocks]
            return
        
        # Lower Bound
        # Each block covers at most C(k, t) subsets
        max_cover = math.comb(k, t)
        lb = math.ceil(len(uncovered) / max_cover)
        
        if len(current_blocks) + lb >= best_count:
            return
        
        # Heuristic: Pick the uncovered subset that has the fewest "good" blocks?
        # Or just pick the first one.
        # To improve, we can pick the subset that is covered by the fewest blocks 
        # that have high gain? That's expensive.
        # Let's stick to picking the first uncovered subset (min index) for consistency, 
        # or randomize to explore different parts of the space.
        # Randomization helps in B&B to avoid getting stuck in a specific greedy path.
        target_subset_idx = min(uncovered)
        
        candidates = subset_to_blocks[target_subset_idx]
        
        # Calculate gain for each candidate
        scored_candidates = []
        for b_idx in candidates:
            gain = len(uncovered & block_subsets[b_idx])
            if gain > 0:
                scored_candidates.append((gain, b_idx))
        
        # Sort by gain descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Explore top beam_width candidates
        for gain, b_idx in scored_candidates[:beam_width]:
            new_uncovered = uncovered - block_subsets[b_idx]
            current_blocks.append(b_idx)
            search(new_uncovered, current_blocks)
            current_blocks.pop()
            
            # If we found a better solution, we might prune more effectively
            # But we already check lb at the start of the function.

    # Start search
    search(set(range(num_subsets)), [])
    
    return best_blocks

# EVOLVE-BLOCK-END
