import itertools
import time

def construct(v, k, t):
    # Parameters for the task: v=19, k=7, t=4
    # But the function signature takes v, k, t.
    
    # Precomputation
    # 1. Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # 2. Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # 3. For each k-subset, find which t-subsets it covers
    k_subset_t_indices = []
    # Also, for each t-subset, find which k-subsets cover it
    t_subset_k_indices = [[] for _ in range(num_t)]
    
    for k_idx, k_subset in enumerate(all_k_subsets):
        k_set = set(k_subset)
        covered_t_indices = []
        for t_subset in all_t_subsets:
            if set(t_subset).issubset(k_set):
                t_idx = t_subset_to_idx[t_subset]
                covered_t_indices.append(t_idx)
                t_subset_k_indices[t_idx].append(k_idx)
        k_subset_t_indices.append(covered_t_indices)
        
    # State
    coverage = [0] * num_t
    blocks = []
    
    # Greedy Loop
    while True:
        # Check if all covered
        if all(c > 0 for c in coverage):
            break
            
        # Find an uncovered t-subset
        # To make it deterministic, find the first one with 0 coverage
        target_t_idx = -1
        for i in range(num_t):
            if coverage[i] == 0:
                target_t_idx = i
                break
        
        if target_t_idx == -1:
            break # Should not happen if loop condition is correct
            
        # Get all k-subsets that cover this target t-subset
        candidate_k_indices = t_subset_k_indices[target_t_idx]
        
        best_k_idx = -1
        best_gain = -1
        
        for k_idx in candidate_k_indices:
            # Calculate gain: number of currently uncovered t-subsets covered by this k-subset
            gain = 0
            for t_idx in k_subset_t_indices[k_idx]:
                if coverage[t_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_k_idx = k_idx
                
        # Add the best block
        blocks.append(list(all_k_subsets[best_k_idx]))
        
        # Update coverage
        for t_idx in k_subset_t_indices[best_k_idx]:
            coverage[t_idx] += 1
            
    return blocks
