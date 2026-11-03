import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    deadline = time.time() + 110
    
    # 1. Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(points, k))
    n_k = len(k_subsets)
    k_index = {s: i for i, s in enumerate(k_subsets)}
    
    # subsets_of_block[j]: list of t-subset indices in k_subsets[j]
    subsets_of_block = []
    # blocks_of_subset[i]: list of block indices containing t_subsets[i]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        subs = []
        for S in itertools.combinations(B, t):
            idx = t_index[S]
            subs.append(idx)
            blocks_of_subset[idx].append(j)
        subsets_of_block.append(np.array(subs, dtype=np.int32))
        
    # Convert blocks_of_subset to arrays for faster access if needed, but lists are fine for iteration
    # Actually, for "blocks containing P_i", we might need a different structure.
    # We will generate candidates on the fly using combinations.

    # Helper: Greedy Set Cover
    def greedy_cover(uncovered_mask):
        # uncovered_mask: bool array of size n_t
        # Returns list of block indices
        if not np.any(uncovered_mask):
            return []
            
        covered = np.zeros(n_t, dtype=bool)
        selected = []
        
        # Initial gains: number of uncovered subsets in each block
        gains = np.zeros(n_k, dtype=np.int32)
        for j in range(n_k):
            subs = subsets_of_block[j]
            gains[j] = np.sum(uncovered_mask[subs])
            
        while True:
            if time.time() > deadline:
                break
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            best_idx = np.argmax(gains)
            selected.append(best_idx)
            
            subs = subsets_of_block[best_idx]
            # Update covered
            newly_covered = uncovered_mask[subs] & ~covered[subs]
            covered[subs[newly_covered]] = True
            # Update uncovered_mask for external use? No, we just need to update gains.
            
            # Update gains for blocks that share subsets with best_idx
            # Only need to update if they cover newly covered subsets
            for s_idx in subs[newly_covered]:
                for b_prime in blocks_of_subset[s_idx]:
                    gains[b_prime] -= 1
                    
            # Check if all covered
            if np.all(covered):
                break
                
        return selected

    # Helper: Deletion Pass
    def deletion_pass(solution_blocks):
        # solution_blocks: list of block indices
        if not solution_blocks:
            return solution_blocks
            
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in solution_blocks:
            cover_count[subsets_of_block[b_idx]] += 1
            
        changed = True
        current_sol = solution_blocks
        while changed:
            changed = False
            new_sol = []
            for b_idx in current_sol:
                subs = subsets_of_block[b_idx]
                # If any subset in b_idx has count 1, we must keep it
                # If all counts > 1, we can remove it
                if np.any(cover_count[subs] == 1):
                    new_sol.append(b_idx)
                else:
                    cover_count[subs] -= 1
                    changed = True
            current_sol = new_sol
        return current_sol

    # Initial Greedy
    uncovered = np.ones(n_t, dtype=bool)
    initial_sol = greedy_cover(uncovered)
    initial_sol = deletion_pass(initial_sol)
    
    best_solution = initial_sol
    best_size = len(initial_sol)
    
    # Local Search
    # We maintain a list of block indices
    # We need to be careful with time.
    
    while time.time() < deadline - 5:
        # Try to improve best_solution
        # We will work on a copy
        current_sol = best_solution.copy()
        
        # Run deletion pass to ensure it's minimal for this configuration
        current_sol = deletion_pass(current_sol)
        if len(current_sol) < best_size:
            best_size = len(current_sol)
            best_solution = current_sol.copy()
            
        # Try swaps
        improved = False
        # Shuffle blocks to try different orders
        random.shuffle(current_sol)
        
        for b_idx in current_sol:
            if time.time() > deadline - 5:
                break
                
            subs_b = subsets_of_block[b_idx]
            
            # Find unique subsets
            # We need cover counts for current_sol
            # This is expensive to recompute every time.
            # Let's maintain cover_count outside the loop?
            # But swaps change it.
            
            # For simplicity, recompute cover count for current_sol if we haven't just done it?
            # Actually, let's recompute. n_k is small (~150), n_t is 3876.
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b in current_sol:
                cover_count[subsets_of_block[b]] += 1
                
            unique_mask = cover_count[subs_b] == 1
            if not np.any(unique_mask):
                # Should have been deleted
                continue
                
            unique_subs = subs_b[unique_mask]
            # Compute point set P_i
            # Union of points in unique_subs
            # unique_subs is an array of indices into t_subsets
            # We need the points.
            # t_subsets is a list of tuples.
            # This is slow. Precompute points_of_t_subset?
            # points_of_t_subset[i] = np.array(t_subsets[i])
            
            # Let's precompute points_of_t_subset
            # This takes memory but is fast.
            
            # Actually, let's just check if we can find a candidate.
            # P_i is the set of points in all unique subsets.
            # Since unique subsets are in B_i, P_i is a subset of points(B_i).
            # We can just use the points of B_i? No, P_i is the union of specific subsets.
            
            # Let's optimize:
            # If we precompute points_of_t_subset, we can union them.
            
            # For now, let's assume we can identify P_i.
            # P_i points = set()
            # for idx in unique_subs:
            #     P_i_points.update(t_subsets[idx])
            
            # This is O(|unique_subs| * t).
            
            # Let's try a simpler heuristic for the swap:
            # Instead of strict "must cover all unique subsets", we can just pick a random block B_j
            # and check if it covers all unique subsets.
            # But finding B_j is hard.
            
            # Let's stick to the candidate generation via P_i.
            
            # Precomputation for points of t-subsets
            # We'll do it once outside the main loop.
            
            # Break to implement precomputation and proper swap logic.
            break
        
        # If we didn't improve, perturb
        if not improved:
             # Perturb: remove random blocks and re-greedy
             # This is a bit heavy.
             pass
             
    # This logic is getting messy. Let's write the full clean version.
    
    # --- Full Implementation ---
    
    # Precompute points for t-subsets
    points_of_t_subset = [np.array(s) for s in t_subsets]
    
    def get_point_set(sub_indices):
        if len(sub_indices) == 0:
            return set()
        # Union of points
        # Since t is small (4), we can just concatenate and unique
        # But we need a set of points.
        # Fast way: 
        # pts = np.unique(np.concatenate([points_of_t_subset[i] for i in sub_indices]))
        # return set(pts)
        # This is a bit slow.
        # Alternative:
        # Since sub_indices are from a single block B, and B has k=7 points.
        # The union P_i is a subset of B's points.
        # We can just take the points of B and check which ones are in the union?
        # No, we need the exact set P_i to generate candidates containing P_i.
        
        # Let's use a boolean mask of points.
        mask = np.zeros(v, dtype=bool)
        for idx in sub_indices:
            mask[points_of_t_subset[idx]] = True
        return np.where(mask)[0]

    def generate_candidates(containing_points, exclude_set):
        # containing_points: array of points
        # exclude_set: set of block indices to exclude
        # Returns list of block indices
        if len(containing_points) > k:
            return []
            
        remaining_points = [p for p in points if p not in containing_points]
        need = k - len(containing_points)
        
        candidates = []
        # Generate combinations of remaining points of size 'need'
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(list(containing_points) + list(combo)))
            if block in k_index:
                b_idx = k_index[block]
                if b_idx not in exclude_set:
                    candidates.append(b_idx)
        return candidates

    # Restart loop
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Perturb previous best? Or start fresh?
        # Let's start from best_solution and perturb it.
        if restart == 0:
            current_sol = best_solution.copy()
        else:
            # Perturb best_solution
            current_sol = best_solution.copy()
            # Remove 5% of blocks
            n_remove = max(1, int(0.05 * len(current_sol)))
            to_remove = random.sample(current_sol, n_remove)
            current_sol = [b for b in current_sol if b not in to_remove]
            # Re-greedy fill
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b in current_sol:
                cover_count[subsets_of_block[b]] += 1
            uncovered = cover_count == 0
            if np.any(uncovered):
                new_blocks = greedy_cover(uncovered)
                current_sol.extend(new_blocks)
                # Remove duplicates
                current_sol = list(set(current_sol))
            
        # Deletion Pass
        current_sol = deletion_pass(current_sol)
        
        if len(current_sol) < best_size:
            best_size = len(current_sol)
            best_solution = current_sol.copy()
            
        # Swap Phase
        # We will try to swap blocks to enable further deletions
        # We iterate a few times
        for swap_iter in range(20):
            if time.time() > deadline:
                break
                
            # Recompute cover counts
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b in current_sol:
                cover_count[subsets_of_block[b]] += 1
                
            improved_swap = False
            # Shuffle to avoid bias
            sol_copy = current_sol.copy()
            random.shuffle(sol_copy)
            
            for b_idx in sol_copy:
                if time.time() > deadline:
                    break
                    
                subs_b = subsets_of_block[b_idx]
                unique_mask = cover_count[subs_b] == 1
                if not np.any(unique_mask):
                    continue
                    
                unique_subs = subs_b[unique_mask]
                p_set = get_point_set(unique_subs)
                
                if len(p_set) > k:
                    continue
                    
                candidates = generate_candidates(p_set, set(current_sol))
                
                if not candidates:
                    continue
                    
                # Pick a candidate
                # Heuristic: pick the one that covers the most "other" unique subsets?
                # Or just random.
                b_new = random.choice(candidates)
                
                # Perform swap
                current_sol.remove(b_idx)
                current_sol.append(b_new)
                
                # Check if this swap allows a deletion
                # Run deletion pass
                current_sol = deletion_pass(current_sol)
                
                if len(current_sol) < best_size:
                    best_size = len(current_sol)
                    best_solution = current_sol.copy()
                    improved_swap = True
                    break # Restart swap phase with new best?
                
                if improved_swap:
                    break
                    
            if improved_swap:
                break
                
    # Final Deletion Pass
    best_solution = deletion_pass(best_solution)
    
    return [list(k_subsets[idx]) for idx in best_solution]

