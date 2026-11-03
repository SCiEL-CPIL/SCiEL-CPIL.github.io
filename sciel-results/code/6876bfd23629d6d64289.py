import itertools
import random
import time

def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    
    # Helper to get index or just use set
    # Set of tuples is fine for v=19
    
    def greedy_column_generation(seed):
        random.seed(seed)
        uncovered_set = set(all_t_subsets)
        blocks = []
        
        while uncovered_set:
            # 1. Pick an anchor
            anchor = random.choice(list(uncovered_set))
            
            # 2. Initialize block
            current_block = list(anchor)
            
            # 3. Remove anchor from uncovered
            # Note: anchor is a t-subset, so it's in the set
            uncovered_set.remove(anchor)
            
            # 4. Extend block
            while len(current_block) < k:
                best_point = -1
                best_gain = -1
                best_subs_to_remove = []
                
                # Precompute (t-1) subsets of current_block
                # We need these to check gains
                t_minus_1_subs = list(itertools.combinations(current_block, t-1))
                
                for p in points:
                    if p in current_block:
                        continue
                    
                    gain = 0
                    subs_to_remove = []
                    for sub in t_minus_1_subs:
                        new_sub = tuple(sorted(sub + (p,)))
                        if new_sub in uncovered_set:
                            gain += 1
                            subs_to_remove.append(new_sub)
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                        best_subs_to_remove = subs_to_remove
                    elif gain == best_gain and gain > 0:
                        if random.random() < 0.5:
                            best_point = p
                            best_subs_to_remove = subs_to_remove
                            
                if best_point == -1:
                    # No point improves coverage?
                    # This can happen if the remaining uncovered sets are disjoint from current_block extensions?
                    # Or if we are stuck.
                    # Pick a random point to avoid infinite loop, but it might be suboptimal.
                    available = [p for p in points if p not in current_block]
                    if not available:
                         break
                    best_point = random.choice(available)
                    best_subs_to_remove = []
                    
                current_block.append(best_point)
                
                # Remove covered subsets
                for s in best_subs_to_remove:
                    uncovered_set.remove(s)
                
            blocks.append(tuple(sorted(current_block)))
            
        return blocks

    def optimize_solution(blocks):
        # Convert to lists
        blocks = [list(b) for b in blocks]
        
        # Remove duplicates
        unique_blocks = list(set(tuple(b) for b in blocks))
        blocks = [list(b) for b in unique_blocks]
        
        # Calculate coverage counts
        n_t = len(all_t_subsets)
        t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
        
        counts = [0] * n_t
        block_t_indices = []
        for block in blocks:
            idxs = [t_to_idx[ts] for ts in itertools.combinations(block, t)]
            block_t_indices.append(idxs)
            for idx in idxs:
                counts[idx] += 1
                
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                redundant = True
                for idx in block_t_indices[i]:
                    if counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in block_t_indices[i]:
                        counts[idx] -= 1
                    blocks.pop(i)
                    block_t_indices.pop(i)
                    improved = True
                    break
                    
        return blocks

    best_sol = None
    best_len = float('inf')
    start_time = time.time()
    
    seed = 1
    while time.time() - start_time < 110:
        sol = greedy_column_generation(seed)
        sol_opt = optimize_solution(sol)
        if len(sol_opt) < best_len:
            best_len = len(sol_opt)
            best_sol = sol_opt
        seed += 1
        
    if best_sol is None:
        best_sol = greedy_column_generation(1)
        best_sol = optimize_solution(best_sol)
        
    return best_sol
