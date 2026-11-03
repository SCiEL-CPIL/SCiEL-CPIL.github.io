import itertools
import random
import time
import copy
import numpy as np

def construct(q, n, radius):
    # Specialized for K_3(7,1) but robust for small q^n
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters if needed, though task is specific
        # Simple greedy fallback
        all_words = list(itertools.product(range(q), repeat=n))
        word_index = {w: i for i, w in enumerate(all_words)}
        N = len(all_words)
        balls = []
        for i, w in enumerate(all_words):
            neighbors = [i]
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    nw = list(w); nw[pos] = val
                    neighbors.append(word_index[tuple(nw)])
            balls.append(neighbors)
        
        uncovered = set(range(N))
        solution = []
        while uncovered:
            best = -1
            best_count = -1
            for i in range(N):
                if i in solution: continue
                count = sum(1 for nb in balls[i] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best = i
            if best == -1: break
            solution.append(best)
            for nb in balls[best]:
                uncovered.discard(nb)
        return [all_words[i] for i in solution]

    # Specific logic for q=3, n=7, radius=1
    all_words = list(itertools.product(range(3), repeat=7))
    word_index = {w: i for i, w in enumerate(all_words)}
    N = len(all_words) # 2187
    
    # Precompute balls as a numpy boolean matrix for speed
    # B[i, j] = 1 if j is within distance 1 of i
    # Memory: 2187 * 2187 * 1 byte (bool) ~ 4.7 MB. Acceptable.
    B = np.zeros((N, N), dtype=bool)
    for i, w in enumerate(all_words):
        B[i, i] = True
        for pos in range(n):
            orig_val = w[pos]
            for val in range(3):
                if val == orig_val: continue
                nw = list(w)
                nw[pos] = val
                j = word_index[tuple(nw)]
                B[i, j] = True
                
    # Note: B is symmetric for Hamming distance. B[i,j] == B[j,i].
    
    def greedy_init():
        uncovered = np.ones(N, dtype=bool)
        sol = []
        while np.any(uncovered):
            # Find word covering most uncovered points
            # coverage_counts[i] = sum(B[i, :] & uncovered)
            # We want max coverage_counts
            # Using matrix multiplication: B @ uncovered_vector
            # But B is boolean. We can use B.dot(uncovered)
            counts = B.dot(uncovered)
            # Exclude already chosen? 
            # Actually, if we pick i, we should mask it out or just ignore if already in sol.
            # Since we add to sol, we can just pick the max.
            # To avoid re-picking, we can set counts for picked ones to -1.
            # But simpler: just pick max. If it's already in sol, it won't reduce uncovered further?
            # Wait, if i is in sol, B[i,:] is already subtracted from uncovered?
            # No, we update uncovered by discarding.
            # So if i is in sol, B[i,:] & uncovered might be 0 or small.
            # Let's just pick the global max.
            
            best_idx = np.argmax(counts)
            if counts[best_idx] == 0:
                break
            sol.append(best_idx)
            # Update uncovered
            uncovered = uncovered & ~B[best_idx]
            
        return optimize_solution(sol)

    def optimize_solution(indices):
        indices = list(indices)
        # Create a coverage count vector
        # cover_count[j] = number of i in indices such that B[i, j] is True
        # We can compute this as B[indices, :].sum(axis=0)
        # But we need to update it dynamically.
        
        # Initial coverage
        cover_count = np.zeros(N, dtype=int)
        for idx in indices:
            cover_count += B[idx]
            
        # Check validity
        # If min(cover_count) > 0, it's valid.
        
        changed = True
        while changed:
            changed = False
            # Try to remove each codeword
            # A codeword i is redundant if cover_count[j] > 1 for all j in B[i, :]
            # i.e., min(cover_count[B[i, :]]) > 1
            
            # To speed up, we can iterate through indices.
            # If we remove i, new_cover_count = cover_count - B[i]
            # We need min(new_cover_count) > 0
            
            for k in range(len(indices) - 1, -1, -1):
                idx = indices[k]
                # Check if removing idx keeps coverage valid
                # new_counts = cover_count - B[idx]
                # We only care about neighbors of idx for the check? No, we need global min > 0.
                # But if cover_count[j] >= 2 for all j, and we subtract B[idx], 
                # the critical points are where cover_count[j] == 2 and B[idx, j] == 1.
                # If there is any j such that cover_count[j] == 1, it's not redundant.
                # So, check: is there any j in B[idx, :] where cover_count[j] == 1?
                # If no, then for j in B[idx, :], cover_count[j] >= 2, so new_count >= 1.
                # For j not in B[idx, :], new_count = cover_count[j] >= 1 (since current is valid).
                
                # So condition: min(cover_count[B[idx, :]]) > 1
                # Wait, B[idx, :] is boolean. 
                # neighbors = B[idx, :]
                # min_val = np.min(cover_count[neighbors])
                
                if np.min(cover_count[B[idx, :]]) > 1:
                    # Remove it
                    indices.pop(k)
                    cover_count -= B[idx]
                    changed = True
                    break # Restart loop after removal
                    
        return indices

    def try_swap(indices):
        # Try to swap one codeword with another to allow removals
        # This is expensive. We will implement a limited version.
        # For each i in indices:
        #   For each j not in indices:
        #     trial = indices - {i} + {j}
        #     if optimize(trial) < len(indices): return optimize(trial)
        
        # Optimization: Only try swaps that seem promising.
        # Or just iterate and check.
        # N=2187, K~180. 180 * 2187 = 393,660 swaps.
        # Each swap requires a check.
        # If we use the matrix, checking validity is fast.
        
        indices = list(indices)
        K = len(indices)
        
        # Precompute cover_count
        cover_count = np.zeros(N, dtype=int)
        for idx in indices:
            cover_count += B[idx]
            
        # Create a set of indices for fast lookup
        idx_set = set(indices)
        
        # Iterate through indices
        # To avoid O(K*N) with full optimize inside, we can do a quick check.
        # If we remove i and add j, new_cover = cover_count - B[i] + B[j]
        # We need min(new_cover) > 0.
        # This is equivalent to:
        # 1. cover_count - B[i] must be >= 1 everywhere? No, because we add B[j].
        # 2. For all v, (cover_count[v] - B[i, v] + B[j, v]) > 0
        
        # Let's just try a subset of swaps to save time.
        # Randomly sample some swaps.
        
        best_solution = indices
        best_size = K
        
        # Sample 500 random swaps
        for _ in range(500):
            i = random.choice(indices)
            # Pick a random j not in indices
            j = random.randint(0, N - 1)
            if j in idx_set:
                continue
                
            # Calculate new cover count
            # new_cover = cover_count - B[i] + B[j]
            # We can check min(new_cover) > 0
            
            # To avoid creating a full array each time, we can check the critical points.
            # But 500 * 2187 is small. Let's just do it.
            
            # Optimization: 
            # The points that might drop to 0 are those where cover_count was 1.
            # If cover_count[v] == 1, then new_cover[v] = 1 - B[i, v] + B[j, v].
            # We need this > 0.
            # If B[i, v] == 1, then 1 - 1 + B[j, v] = B[j, v]. So we need B[j, v] == 1.
            # If B[i, v] == 0, then 1 - 0 + B[j, v] >= 1. Always safe.
            
            # So, condition: For all v where cover_count[v] == 1 and B[i, v] == 1, we must have B[j, v] == 1.
            
            # Let critical_points = indices where cover_count == 1
            # Let i_neighbors = indices where B[i, :] == 1
            # Let intersection = critical_points & i_neighbors
            # We need B[j, intersection] to be all True.
            
            critical = (cover_count == 1)
            i_neigh = B[i]
            critical_i = critical & i_neigh
            
            if np.any(critical_i):
                # Check if j covers all critical_i
                if not np.all(B[j, critical_i]):
                    continue
            # If no critical points for i, then removing i leaves all counts >= 1?
            # Wait, if cover_count[v] >= 2 for all v in B[i, :], then removing i leaves >= 1.
            # If cover_count[v] == 1 for some v NOT in B[i, :], removing i doesn't affect it.
            # So yes, the check above is sufficient for validity.
            
            # If valid, try to optimize (remove redundancies)
            trial = indices[:]
            trial.remove(i)
            trial.append(j)
            
            # Quick optimize
            trial_opt = optimize_solution(trial)
            
            if len(trial_opt) < best_size:
                best_size = len(trial_opt)
                best_solution = trial_opt
                # Update cover_count and idx_set for the new best?
                # It's complex to update incrementally in the loop.
                # We can break and restart the swap loop with the new best?
                # Or just keep searching.
                # Let's update local variables to reflect the new best for subsequent checks?
                # No, subsequent checks are based on the original `indices` and `cover_count`.
                # This is a heuristic. We accept the improvement and continue.
                
        return best_solution

    start_time = time.time()
    max_time = 115.0
    
    # Initial solution
    current_solution = greedy_init()
    best_solution = current_solution
    
    # ILS Loop
    while time.time() - start_time < max_time:
        # Local Search: Swaps
        improved = True
        while improved:
            improved = False
            new_sol = try_swap(current_solution)
            if len(new_sol) < len(current_solution):
                current_solution = new_sol
                improved = True
        
        # Update best
        if len(current_solution) < len(best_solution):
            best_solution = current_solution
            
        # Perturbation
        # Remove k random codewords and repair
        k = random.randint(5, 15)
        perturbed = best_solution[:]
        for _ in range(k):
            if perturbed:
                idx_to_remove = random.choice(perturbed)
                perturbed.remove(idx_to_remove)
                
        # Repair
        current_solution = repair_solution(perturbed)
        
        # Optimize
        current_solution = optimize_solution(current_solution)

    return [all_words[i] for i in best_solution]

    def repair_solution(indices):
        indices = list(indices)
        covered = np.zeros(N, dtype=bool)
        for idx in indices:
            covered |= B[idx]
            
        uncovered = ~covered
        
        while np.any(uncovered):
            # Find best codeword to cover most uncovered
            counts = B.dot(uncovered)
            # Exclude already in indices?
            # If we add an existing one, it's useless.
            # Set counts for existing indices to -1
            for idx in indices:
                counts[idx] = -1
                
            best_idx = np.argmax(counts)
            if counts[best_idx] <= 0:
                break # Should not happen if uncovered exists and N>0
                
            indices.append(best_idx)
            covered |= B[best_idx]
            uncovered = ~covered
            
        return indices
