import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin
    
    # Precomputation
    # Generate all t-subsets and assign IDs
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, store the list of t-subset IDs it covers
    block_t_ids = []
    for block in all_k_subsets:
        ids = [t_subset_to_id[s] for s in itertools.combinations(block, t)]
        block_t_ids.append(ids)
        
    # For each t-subset ID, store the list of k-subset indices that cover it
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, ids in enumerate(block_t_ids):
        for tid in ids:
            t_to_blocks[tid].append(b_idx)
            
    best_blocks = None
    best_size = float('inf')
    
    # Number of random restarts
    max_restarts = 10 # GRASP is more expensive per restart, so fewer restarts
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # 1. GRASP Construction
        uncovered = set(range(num_t_subsets))
        selected_blocks = set()
        
        while uncovered:
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random uncovered subset
            s_id = random.choice(list(uncovered))
            
            # Get candidate blocks that cover s_id
            candidates = t_to_blocks[s_id]
            
            # Score candidates by how many uncovered subsets they cover
            scored_candidates = []
            for b_idx in candidates:
                if b_idx in selected_blocks:
                    continue
                covered = 0
                for tid in block_t_ids[b_idx]:
                    if tid in uncovered:
                        covered += 1
                scored_candidates.append((covered, b_idx))
            
            if not scored_candidates:
                break # Should not happen
                
            # Sort by score descending
            scored_candidates.sort(reverse=True)
            
            # GRASP: Pick from top alpha fraction
            # alpha = 0.1 (top 10%)
            alpha = 0.1
            num_to_consider = max(1, int(len(scored_candidates) * alpha))
            top_candidates = scored_candidates[:num_to_consider]
            
            # Pick randomly from top candidates
            chosen_score, best_candidate = random.choice(top_candidates)
            
            selected_blocks.add(best_candidate)
            # Remove covered subsets from uncovered
            for tid in block_t_ids[best_candidate]:
                if tid in uncovered:
                    uncovered.discard(tid)
        
        if not uncovered:
            current_size = len(selected_blocks)
            if current_size < best_size:
                best_size = current_size
                best_blocks = set(selected_blocks)
                
        # 2. Local Search (Refinement)
        if best_blocks is None or len(selected_blocks) < best_size:
             current_solution = set(selected_blocks)
        elif best_blocks is not None:
             current_solution = set(best_blocks)
        else:
             continue
             
        # Initialize cover counts
        cover_counts = [0] * num_t_subsets
        for b_idx in current_solution:
            for tid in block_t_ids[b_idx]:
                cover_counts[tid] += 1
        
        # Local Search Loop
        improved = True
        ls_iterations = 0
        max_ls_iterations = 100 # Reduced iterations, but each is more thorough
        
        while improved and ls_iterations < max_ls_iterations:
            if time.time() - start_time > time_limit:
                break
                
            improved = False
            ls_iterations += 1
            
            # Randomize order of blocks to try
            blocks_list = list(current_solution)
            random.shuffle(blocks_list)
            
            for b_idx in blocks_list:
                if b_idx not in current_solution:
                    continue
                    
                # 1. Check for Redundancy
                is_redundant = True
                for tid in block_t_ids[b_idx]:
                    if cover_counts[tid] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    current_solution.discard(b_idx)
                    for tid in block_t_ids[b_idx]:
                        cover_counts[tid] -= 1
                    improved = True
                    continue
                
                # 2. Try Swap
                critical_subsets = []
                for tid in block_t_ids[b_idx]:
                    if cover_counts[tid] == 1:
                        critical_subsets.append(tid)
                
                if not critical_subsets:
                    continue
                    
                # Find best replacement
                # We want a block that covers all critical subsets.
                # Among those, pick one that covers the most *other* subsets (potential future redundancy)
                target_tid = critical_subsets[0]
                candidates = t_to_blocks[target_tid]
                
                best_replacement = -1
                best_score = -1
                
                for new_b_idx in candidates:
                    if new_b_idx in current_solution:
                        continue
                    
                    # Check if new_b_idx covers all critical subsets
                    covers_all = True
                    new_b_tids_set = set(block_t_ids[new_b_idx])
                    for c_tid in critical_subsets:
                        if c_tid not in new_b_tids_set:
                            covers_all = False
                            break
                    
                    if covers_all:
                        # Score: Number of subsets covered by new_b_idx that are currently covered by multiple blocks
                        # This indicates potential for further reduction
                        score = 0
                        for tid in block_t_ids[new_b_idx]:
                            if cover_counts[tid] > 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_replacement = new_b_idx
                
                if best_replacement != -1:
                    # Perform swap
                    current_solution.discard(b_idx)
                    for tid in block_t_ids[b_idx]:
                        cover_counts[tid] -= 1
                    
                    current_solution.add(best_replacement)
                    for tid in block_t_ids[best_replacement]:
                        cover_counts[tid] += 1
                    
                    improved = True
        
        # Check if this local search result is better
        if len(current_solution) < best_size:
            best_size = len(current_solution)
            best_blocks = set(current_solution)

    # Convert best_blocks set of indices to list of blocks
    if best_blocks is None:
        # Fallback: Return a trivial covering if failed? 
        # Construct a valid covering by adding blocks until all are covered.
        # This is a safety net.
        current_solution = set()
        uncovered = set(range(num_t_subsets))
        while uncovered:
            s_id = random.choice(list(uncovered))
            candidates = t_to_blocks[s_id]
            # Pick first candidate not in solution
            for b_idx in candidates:
                if b_idx not in current_solution:
                    current_solution.add(b_idx)
                    for tid in block_t_ids[b_idx]:
                        if tid in uncovered:
                            uncovered.discard(tid)
                    break
        best_blocks = current_solution

    result = [list(all_k_subsets[i]) for i in best_blocks]
    return result
