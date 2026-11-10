import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks) and their coverage
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # coverages[bi] is a set of indices of t-subsets covered by block bi
    coverages = [None] * num_blocks
    for bi, block in enumerate(all_blocks):
        block_set = set(block)
        cov = set()
        # Generate t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            cov.add(t_to_idx[t_sub])
        coverages[bi] = cov
        
    # State for greedy
    # We will run multiple iterations with random seeds
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_solution_indices = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        random.seed(seed)
        
        uncovered = set(range(num_t))
        selected_blocks = set()
        solution_indices = []
        
        # Greedy Phase
        while uncovered:
            max_gain = 0
            best_candidates = []
            
            # Find block with max gain
            for bi in range(num_blocks):
                if bi in selected_blocks:
                    continue
                # Calculate gain
                gain = len(coverages[bi] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [bi]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(bi)
            
            if max_gain == 0:
                # Should not happen if input is valid, but break to avoid infinite loop
                break
                
            # Pick randomly from best candidates to break symmetry
            chosen_bi = random.choice(best_candidates)
            selected_blocks.add(chosen_bi)
            solution_indices.append(chosen_bi)
            uncovered.difference_update(coverages[chosen_bi])
            
        # Reduction Phase
        # Initialize counts
        counts = [0] * num_t
        for bi in solution_indices:
            for s in coverages[bi]:
                counts[s] += 1
                
        current_sol = list(solution_indices)
        changed = True
        while changed:
            changed = False
            new_sol = []
            for bi in current_sol:
                redundant = True
                for s in coverages[bi]:
                    if counts[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in coverages[bi]:
                        counts[s] -= 1
                    changed = True
                else:
                    new_sol.append(bi)
            current_sol = new_sol
            
        # Update best
        if len(current_sol) < best_len:
            best_len = len(current_sol)
            best_solution_indices = current_sol
            
        seed += 1
        
    # Convert indices back to blocks
    if best_solution_indices is None:
        # Fallback: return empty or simple greedy if something went wrong
        # But it should always find a solution
        return []
        
    return [list(all_blocks[bi]) for bi in best_solution_indices]

