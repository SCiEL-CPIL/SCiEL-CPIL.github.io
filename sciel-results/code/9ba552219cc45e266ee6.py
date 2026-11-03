import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145.0
    start_time = time.time()
    
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute coverage for all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_to_covered[i] is a set of t-subset indices covered by block i
    block_to_covered = []
    for block in all_blocks:
        covered = set()
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
        block_to_covered.append(covered)
    
    # Reverse index: t_subset_to_blocks[i] is a set of block indices that cover t-subset i
    t_subset_to_blocks = [set() for _ in range(n_t)]
    for b_idx, covered in enumerate(block_to_covered):
        for ts_idx in covered:
            t_subset_to_blocks[ts_idx].add(b_idx)
    
    # Helper: check if a set of block indices covers all t-subsets
    def is_valid_solution(block_indices):
        covered = set()
        for idx in block_indices:
            covered.update(block_to_covered[idx])
        return len(covered) == n_t
    
    # Helper: get the set of uncovered t-subsets for a given solution
    def get_uncovered(block_indices):
        covered = set()
        for idx in block_indices:
            covered.update(block_to_covered[idx])
        return set(range(n_t)) - covered
    
    # Helper: count how many times each t-subset is covered
    def get_coverage_counts(block_indices):
        counts = np.zeros(n_t, dtype=np.int32)
        for idx in block_indices:
            for ts_idx in block_to_covered[idx]:
                counts[ts_idx] += 1
        return counts
    
    # Initial greedy solution
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 30.0:
                break
            
            # Pick a random uncovered t-subset
            start_idx = rng.choice(list(uncovered))
            candidate_blocks = t_subset_to_blocks[start_idx]
            
            # Choose the block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            for b_idx in candidate_blocks:
                gain = len(block_to_covered[b_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            for ts_idx in block_to_covered[best_block]:
                uncovered.discard(ts_idx)
        
        return blocks
    
    # Get initial solution
    initial_indices = greedy_cover(0)
    if not is_valid_solution(initial_indices):
        # Fallback: more aggressive greedy
        initial_indices = greedy_cover(1)
    
    # Local search: simulated annealing with remove/swap moves
    def local_search(block_indices, seed=0):
        rng = random.Random(seed)
        current_indices = list(block_indices)
        best_indices = list(current_indices)
        best_len = len(current_indices)
        
        # Coverage counts for quick update
        counts = get_coverage_counts(current_indices)
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.95
        max_steps = 10000
        
        step = 0
        while time.time() < deadline - 10.0 and step < max_steps:
            step += 1
            
            # Choose a move: remove (50%) or swap (50%)
            move_type = rng.random()
            
            if move_type < 0.5 and len(current_indices) > 1:
                # Try to remove a block
                # Prefer blocks that cover the fewest "critical" triples (count==1)
                removable = []
                for i, idx in enumerate(current_indices):
                    # Check if removing this block leaves all triples covered
                    # A triple is critical if it's only covered by this block
                    # We can approximate by checking if any triple in this block has count==1
                    has_critical = False
                    for ts_idx in block_to_covered[idx]:
                        if counts[ts_idx] == 1:
                            has_critical = True
                            break
                    if not has_critical:
                        removable.append(i)
                
                if not removable:
                    # Try a random block and check validity
                    i = rng.randint(0, len(current_indices) - 1)
                    removable = [i]
                
                i = rng.choice(removable)
                idx_to_remove = current_indices[i]
                
                # Check if removal is valid
                new_counts = counts.copy()
                for ts_idx in block_to_covered[idx_to_remove]:
                    new_counts[ts_idx] -= 1
                
                if np.min(new_counts) >= 1:
                    # Valid removal
                    new_indices = [x for j, x in enumerate(current_indices) if j != i]
                    delta_len = -1  # We reduced size by 1
                    # Always accept improvement
                    current_indices = new_indices
                    counts = new_counts
                    best_indices = list(current_indices)
                    best_len = len(current_indices)
                else:
                    # Invalid, skip
                    pass
                    
            else:
                # Try a swap: remove one block, add another
                if len(current_indices) == 0:
                    continue
                    
                i = rng.randint(0, len(current_indices) - 1)
                idx_to_remove = current_indices[i]
                
                # Find a replacement block
                # Look for blocks that cover triples with low count
                # Sample some low-count triples
                low_count_triples = [ts_idx for ts_idx in range(n_t) if counts[ts_idx] <= 2]
                
                if not low_count_triples:
                    low_count_triples = list(range(n_t))
                
                sample_size = min(5, len(low_count_triples))
                sampled = rng.sample(low_count_triples, sample_size)
                
                candidates = set()
                for ts_idx in sampled:
                    candidates.update(t_subset_to_blocks[ts_idx])
                
                # Exclude the block we're removing
                candidates.discard(idx_to_remove)
                
                if not candidates:
                    continue
                
                # Pick a random candidate
                idx_to_add = rng.choice(list(candidates))
                
                if idx_to_add in current_indices:
                    continue
                
                # Check if swap is valid
                new_counts = counts.copy()
                for ts_idx in block_to_covered[idx_to_remove]:
                    new_counts[ts_idx] -= 1
                for ts_idx in block_to_covered[idx_to_add]:
                    new_counts[ts_idx] += 1
                
                if np.min(new_counts) >= 1:
                    # Valid swap, size unchanged
                    # Accept with probability based on improvement in "robustness"
                    # Measure: sum of min(count, 3) to prefer balanced coverage
                    old_score = np.sum(np.minimum(counts, 3))
                    new_score = np.sum(np.minimum(new_counts, 3))
                    
                    delta_score = new_score - old_score
                    
                    # Accept if improvement, or with probability exp(delta_score/T) if worse
                    if delta_score >= 0:
                        accept = True
                    else:
                        accept = rng.random() < np.exp(delta_score / T)
                    
                    if accept:
                        current_indices[i] = idx_to_add
                        counts = new_counts
                        # Update best if we found a smaller solution (not possible with swap)
                        # But track best for robustness
                        if len(current_indices) < best_len:
                            best_indices = list(current_indices)
                            best_len = len(current_indices)
            
            # Cool down
            T = max(T_min, T * cooling_rate)
            
            # Occasionally reheat
            if step % 1000 == 0 and T == T_min:
                T = 0.5
        
        return best_indices, best_len
    
    # Run local search from initial solution
    best_indices = list(initial_indices)
    best_len = len(initial_indices)
    
    # Run multiple local searches with different seeds
    for seed in range(5):
        if time.time() > deadline - 20.0:
            break
        
        ls_indices, ls_len = local_search(initial_indices, seed)
        if ls_len < best_len:
            best_len = ls_len
            best_indices = ls_indices
    
    # Try to improve further with more aggressive local search on the best solution
    if time.time() < deadline - 15.0:
        ls_indices, ls_len = local_search(best_indices, seed=100)
        if ls_len < best_len:
            best_len = ls_len
            best_indices = ls_indices
    
    # Final validation
    if not is_valid_solution(best_indices):
        # Fallback to greedy
        best_indices = greedy_cover(0)
    
    # Convert to block tuples
    result = [all_blocks[i] for i in best_indices]
    
    # Ensure no duplicates
    result = list(set(result))
    
    # Final validation
    if not is_valid_solution([all_blocks.index(b) for b in result]):
        # This should not happen, but just in case
        result = [all_blocks[i] for i in greedy_cover(0)]
    
    return result
