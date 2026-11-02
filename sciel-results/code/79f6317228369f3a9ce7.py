import numpy as np
import itertools
import time
import random
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute block-to-triple incidence
    # M is sparse (n_ks x n_ts)
    row_indices = []
    col_indices = []
    data = []
    
    # Also store the list of triple indices for each block for fast update
    block_triple_lists = []
    
    for b_idx, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        triple_idxs = []
        for sub in subs:
            t_idx = idx_map[sub]
            triple_idxs.append(t_idx)
            row_indices.append(b_idx)
            col_indices.append(t_idx)
            data.append(1)
        block_triple_lists.append(triple_idxs)
        
    M = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_ks, n_ts))
    
    # Precompute rarity: how many blocks cover each triple
    # triple_counts_total[i] = number of blocks containing triple i
    triple_counts_total = np.array(M.sum(axis=0)).flatten()
    # Rarity weight: 1 / (count + 1)
    # We want to prefer blocks that cover "hard" (rare) triples
    triple_rarity = 1.0 / (triple_counts_total + 1e-9)
    
    # Precompute a score for each block based on rarity of its triples
    # block_rarity_score[i] = sum(triple_rarity[j] for j in block_triple_lists[i])
    # This helps in tie-breaking.
    # We can compute this by M.T @ triple_rarity
    block_rarity_score = (M.T.dot(triple_rarity)).flatten()
    
    best_solution = None
    best_size = float('inf')
    
    def get_gains(uncovered):
        # uncovered is boolean array of size n_ts
        # Returns array of size n_ks with gain for each block
        return M.dot(uncovered)

    def greedy_construct(uncovered, existing_solution=None):
        """
        Greedily add blocks to cover 'uncovered' triples.
        Returns list of added block indices.
        """
        if existing_solution is None:
            existing_solution = []
        
        added_blocks = []
        # We need to track which blocks are already in the solution to avoid duplicates
        # But for speed, we can just pick the best gain. If a block is already in, its gain is 0?
        # No, if a block is in, its triples are covered, so gain is 0.
        # So we don't need to explicitly exclude them if we calculate gain correctly.
        
        while np.any(uncovered):
            if time.time() - start_time > time_limit - 5.0:
                return added_blocks, False # Timeout
            
            gains = get_gains(uncovered)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                # Should not happen if uncovered is not empty and M is valid
                break
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 0:
                break
                
            # Tie-breaking:
            # 1. Random selection from top K candidates weighted by rarity?
            # Let's pick the one with highest block_rarity_score among candidates
            # If many candidates, sample to keep it fast
            
            if len(candidates) > 20:
                # Sample 20 candidates
                sample_idx = random.sample(range(len(candidates)), 20)
                candidates = candidates[sample_idx]
                
            # Find the candidate with max rarity score
            best_candidate = candidates[0]
            best_score = -1
            for cand in candidates:
                score = block_rarity_score[cand]
                if score > best_score:
                    best_score = score
                    best_candidate = cand
                    
            added_blocks.append(best_candidate)
            
            # Update uncovered
            subs = block_triple_lists[best_candidate]
            uncovered[subs] = False
            
        return added_blocks, True

    def deletion_pass(solution):
        """
        Remove redundant blocks from solution.
        Returns new solution list.
        """
        if not solution:
            return solution
            
        counts = np.zeros(n_ts, dtype=np.int32)
        for b in solution:
            counts[block_triple_lists[b]] += 1
            
        new_solution = []
        for b in solution:
            subs = block_triple_lists[b]
            # If all triples in b are covered by other blocks (count > 1), we can remove b
            if np.min(counts[subs]) >= 2:
                counts[subs] -= 1
            else:
                new_solution.append(b)
                
        return new_solution

    def run_iterated_local_search(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(n_ts, dtype=bool)
        initial_blocks, success = greedy_construct(uncovered)
        if not success:
            return None, float('inf')
            
        current_solution = initial_blocks
        current_size = len(current_solution)
        
        # Initial deletion
        current_solution = deletion_pass(current_solution)
        current_size = len(current_solution)
        
        best_local_solution = current_solution[:]
        best_local_size = current_size
        
        # Iterative Local Search
        # We have a time budget for this specific restart
        # Let's say we try for a few seconds or a fixed number of iterations
        # But we must respect the global time limit
        
        while time.time() - start_time < time_limit - 10.0:
            if current_size < best_local_size:
                best_local_solution = current_solution[:]
                best_local_size = current_size
                
            # Perturbation: Remove 1-3 random blocks
            # If we remove too many, repair might be slow or result in larger size
            # Start with 1, maybe increase if stuck?
            # Let's stick to 1 or 2.
            num_remove = random.randint(1, 2)
            
            # Pick random indices to remove
            # Ensure we don't remove more than available
            num_remove = min(num_remove, len(current_solution))
            
            if num_remove == 0:
                break
                
            remove_indices = random.sample(range(len(current_solution)), num_remove)
            remove_set = set(remove_indices)
            
            # Create a new solution list without these blocks
            perturbed_solution = [b for i, b in enumerate(current_solution) if i not in remove_set]
            
            # Calculate current coverage of perturbed_solution
            # We need to know what is uncovered
            counts = np.zeros(n_ts, dtype=np.int32)
            for b in perturbed_solution:
                counts[block_triple_lists[b]] += 1
                
            uncovered = (counts == 0)
            
            # Greedy repair
            added_blocks, success = greedy_construct(uncovered)
            
            if not success:
                # Timeout during repair, skip this iteration
                continue
                
            new_solution = perturbed_solution + added_blocks
            new_size = len(new_solution)
            
            # Deletion pass on new solution
            new_solution = deletion_pass(new_solution)
            new_size = len(new_solution)
            
            # Accept if better or equal? 
            # Usually accept if better. If equal, maybe accept with some probability to escape plateaus?
            # For now, accept if strictly better.
            if new_size < best_local_size:
                current_solution = new_solution
                current_size = new_size
            elif new_size == best_local_size:
                # Random restart within the same size? 
                # To avoid getting stuck, we can accept equal solutions with low probability
                if random.random() < 0.1:
                    current_solution = new_solution
                    current_size = new_size
            else:
                # Revert to previous best local solution to continue from a good state?
                # Or revert to pre-perturbation state?
                # Usually, ILS reverts to the last accepted good state.
                # Here, we keep `current_solution` as the state we perturb from.
                # If we don't accept, we should probably revert `current_solution` to `best_local_solution` 
                # or keep the pre-perturbation state?
                # Let's keep the pre-perturbation state (`current_solution` was the one we perturbed).
                # So we just don't update `current_solution`.
                pass
                
        return best_local_solution, best_local_size

    # Main loop: Multiple restarts
    restarts = 0
    while time.time() - start_time < time_limit - 15.0:
        seed = restarts
        sol, size = run_iterated_local_search(seed)
        
        if sol is None:
            break
            
        if size < best_size:
            best_size = size
            best_solution = sol[:]
            
        restarts += 1
        
    # Final polish on best solution
    if best_solution is not None:
        # One more deletion pass just in case
        best_solution = deletion_pass(best_solution)
        
    if best_solution is None:
        # Fallback: Return empty or minimal valid? 
        # Should not happen if time limit is respected and greedy works.
        return []
        
    return [list(all_ks[i]) for i in best_solution]
