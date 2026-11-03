import itertools
import numpy as np
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Precompute all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute for each block, the list of t-subset indices it covers
    # and for each t-subset, the list of blocks that cover it
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
    
    # Convert to numpy arrays for faster operations where possible
    # But lists are fine for now
    
    best_solution = None
    best_size = float('inf')
    
    def is_valid_solution(sol):
        """Check if solution covers all t-subsets"""
        if sol is None:
            return False
        covered = np.zeros(n_t, dtype=bool)
        for b_idx in sol:
            for s_idx in subsets_of_block[b_idx]:
                covered[s_idx] = True
        return np.all(covered)
    
    def deletion_pass(sol, time_limit=None):
        """Remove redundant blocks from solution"""
        if sol is None or len(sol) == 0:
            return sol, len(sol) if sol else 0
        
        if time_limit is None:
            time_limit = deadline - 2
        
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in sol:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            new_sol = []
            for b_idx in sol:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for s_idx in subs_b:
                        cover_count[s_idx] -= 1
                    changed = True
                else:
                    new_sol.append(b_idx)
            sol = new_sol
        
        return sol, len(sol)
    
    def local_search_swap(sol, time_limit=None):
        """Try to improve solution via swaps that enable deletions"""
        if sol is None or len(sol) == 0:
            return sol, len(sol) if sol else 0
        
        if time_limit is None:
            time_limit = deadline - 3
        
        current_sol = sol.copy()
        current_size = len(current_sol)
        
        improved = True
        max_rounds = 30
        round_num = 0
        
        while improved and round_num < max_rounds and time.time() < time_limit:
            improved = False
            round_num += 1
            
            # Recompute cover counts
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in current_sol:
                for s_idx in subsets_of_block[b_idx]:
                    cover_count[s_idx] += 1
            
            # Try each block for potential swap
            for i, b_idx in enumerate(current_sol):
                if time.time() > time_limit:
                    break
                
                # Find subsets uniquely covered by this block
                unique_subs = [s_idx for s_idx in subsets_of_block[b_idx] if cover_count[s_idx] == 1]
                
                if not unique_subs:
                    # Block is redundant, remove it
                    for s_idx in subsets_of_block[b_idx]:
                        cover_count[s_idx] -= 1
                    current_sol.pop(i)
                    current_size -= 1
                    improved = True
                    break  # Restart outer loop
                
                # Check if we can find a replacement
                # The replacement must cover all unique_subs
                candidate_blocks = set(blocks_of_subset[unique_subs[0]])
                candidate_blocks.discard(b_idx)
                
                for s_idx in unique_subs[1:]:
                    candidate_blocks.intersection_update(blocks_of_subset[s_idx])
                    if not candidate_blocks:
                        break
                
                if not candidate_blocks:
                    continue
                
                # Score candidates: prefer those that cover many other critical subsets
                best_cand = -1
                best_score = -1
                
                cand_list = list(candidate_blocks)
                random.shuffle(cand_list)
                for cand_idx in cand_list[:8]:  # Limit to 8 candidates for speed
                    # Score: number of subsets in cand_idx that are currently critical (cover_count == 1)
                    # plus a bonus for covering subsets not in b_idx
                    score = 0
                    cand_set = set(subsets_of_block[cand_idx])
                    b_set = set(subsets_of_block[b_idx])
                    
                    for s in subsets_of_block[cand_idx]:
                        if cover_count[s] == 1:
                            score += 2  # Covering critical subsets is valuable
                        if s not in b_set:
                            score += 0.5  # Slight bonus for new coverage
                    
                    if score > best_score:
                        best_score = score
                        best_cand = cand_idx
                
                if best_cand != -1:
                    # Perform swap
                    current_sol[i] = best_cand
                    improved = True
                    break  # Restart outer loop to recompute cover counts
        
        # Final deletion pass
        current_sol, current_size = deletion_pass(current_sol, time_limit)
        
        return current_sol, current_size
    
    # Main search: Multiple greedy constructions with different weightings
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Different weighting strategies for each restart
        if restart == 0:
            # Standard greedy: pick block covering most uncovered t-subsets
            weights = None
        elif restart == 1:
            # Weight by inverse of number of blocks covering each t-subset (rarer subsets get higher weight)
            subset_weights = np.zeros(n_t)
            for i in range(n_t):
                subset_weights[i] = 1.0 / max(len(blocks_of_subset[i]), 1)
            weights = subset_weights
        elif restart == 2:
            # Random weights
            weights = np.random.uniform(0.5, 1.5, n_t)
        elif restart == 3:
            # Emphasize subsets with few covering blocks
            subset_weights = np.zeros(n_t)
            for i in range(n_t):
                cnt = len(blocks_of_subset[i])
                if cnt <= 10:
                    subset_weights[i] = 5.0
                elif cnt <= 20:
                    subset_weights[i] = 3.0
                elif cnt <= 50:
                    subset_weights[i] = 2.0
                else:
                    subset_weights[i] = 1.0
            weights = subset_weights
        elif restart == 4:
            # Quadratic weighting
            subset_weights = np.zeros(n_t)
            for i in range(n_t):
                cnt = len(blocks_of_subset[i])
                subset_weights[i] = 1.0 / (cnt * cnt) if cnt > 0 else 1.0
            weights = subset_weights
        elif restart == 5:
            # Log weighting
            subset_weights = np.zeros(n_t)
            for i in range(n_t):
                cnt = max(len(blocks_of_subset[i]), 1)
                subset_weights[i] = 1.0 / np.log2(cnt + 1)
            weights = subset_weights
        elif restart == 6:
            # Exponential weighting favoring rare subsets
            subset_weights = np.zeros(n_t)
            for i in range(n_t):
                cnt = len(blocks_of_subset[i])
                subset_weights[i] = np.exp(-0.05 * cnt)
            weights = subset_weights
        else:
            # Pure random order
            weights = np.random.permutation(n_t) / n_t
        
        # Greedy construction
        uncovered = np.ones(n_t, dtype=bool)
        remaining_count = n_t
        solution = []
        
        # Precompute block weights based on current uncovered set
        # For efficiency, we'll recompute weights periodically
        
        while remaining_count > 0 and time.time() < deadline - 10:
            best_block = -1
            best_score = -1
            
            # Compute score for each block
            # Score = sum of weights of uncovered t-subsets in this block
            for j in range(n_k):
                if not subsets_of_block[j]:
                    continue
                
                score = 0.0
                for s_idx in subsets_of_block[j]:
                    if uncovered[s_idx]:
                        if weights is None:
                            score += 1.0
                        else:
                            score += weights[s_idx]
                
                if score > best_score:
                    best_score = score
                    best_block = j
            
            if best_block == -1 or best_score <= 0:
                break
            
            # Select the block
            solution.append(best_block)
            
            # Update uncovered
            for s_idx in subsets_of_block[best_block]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    remaining_count -= 1
            
            if remaining_count == 0:
                break
        
        if remaining_count == 0:
            # Valid solution found
            # Run deletion pass
            solution, sol_size = deletion_pass(solution, deadline - 5)
            
            if sol_size < best_size:
                best_size = sol_size
                best_solution = solution.copy()
            
            # Run local search
            if time.time() < deadline - 5:
                improved_sol, improved_size = local_search_swap(solution, deadline - 3)
                if improved_size < best_size:
                    best_size = improved_size
                    best_solution = improved_sol.copy()
    
    # If no valid solution found, do a basic greedy as fallback
    if best_solution is None:
        uncovered = set(range(n_t))
        solution = []
        while uncovered and time.time() < deadline - 2:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = sum(1 for s in subsets_of_block[j] if s in uncovered)
                if cnt > max_cov:
                    max_cov = cnt
                    best_b = j
            if best_b == -1 or max_cov == 0:
                break
            solution.append(best_b)
            for s in subsets_of_block[best_b]:
                if s in uncovered:
                    uncovered.discard(s)
        
        if len(uncovered) == 0:
            best_solution = solution
            best_size = len(solution)
    
    # Final cleanup
    if best_solution is not None:
        best_solution, best_size = deletion_pass(best_solution, deadline - 1)
    
    if best_solution is None:
        # Should not happen, but return empty list as fallback
        return []
    
    return [list(k_subsets[idx]) for idx in best_solution]
