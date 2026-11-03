import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    limit = 142.0  # Safety margin for 150s limit
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbor matrix as a numpy boolean array
    # neighbor_matrix[i][j] is True if j is within distance radius of i
    # For q=3, n=8, radius=1: each word has 1 + 8*2 = 17 neighbors
    # Matrix size: 6561 x 6561 ~ 43MB, acceptable
    
    # Efficient generation of neighbor matrix
    row_indices = []
    col_indices = []
    
    for idx in range(total_words):
        # Diagonal (self)
        row_indices.append(idx)
        col_indices.append(idx)
        
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    neighbor_idx = base + new_val * powers[i]
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    
    neighbor_matrix = np.zeros((total_words, total_words), dtype=bool)
    neighbor_matrix[np.array(row_indices), np.array(col_indices)] = True
    
    # Also precompute the neighbor lists as integer arrays for faster access
    # neighbor_lists[i] is an array of indices j where neighbor_matrix[i][j] is True
    neighbor_lists = []
    for i in range(total_words):
        neighbor_lists.append(np.where(neighbor_matrix[i])[0])
    
    def greedy_cover(uncovered_mask, solution, max_time=None):
        """Greedy set cover starting from given solution and uncovered mask."""
        sol = list(solution)
        unc = uncovered_mask.copy()
        sol_set = set(sol)
        
        if max_time is None:
            max_time = limit
        
        while unc.any():
            if time.time() - start_time > max_time:
                break
            
            # Find the word that covers the most uncovered words
            # counts[w] = number of j such that unc[j] and neighbor_matrix[w][j]
            # Use matrix-vector product for efficiency
            unc_int = unc.astype(np.int8)
            counts = neighbor_matrix @ unc_int
            
            # Exclude words already in solution
            if sol:
                sol_arr = np.array(sol, dtype=int)
                counts[sol_arr] = -1  # Mark as unavailable
            
            best_word_idx = int(np.argmax(counts))
            
            if counts[best_word_idx] <= 0:
                break
                
            sol.append(best_word_idx)
            sol_set.add(best_word_idx)
            unc[neighbor_matrix[best_word_idx]] = False
            
        return sol

    def is_complete(solution):
        """Check if solution covers all words."""
        if not solution:
            return total_words == 0
        covered = np.any(neighbor_matrix[solution], axis=0)
        return covered.all()

    def remove_redundant(solution, max_time=None):
        """Remove codewords that don't cover any unique word."""
        if not solution:
            return solution
        sol = list(solution)
        changed = True
        
        if max_time is None:
            max_time = limit
        
        while changed and (time.time() - start_time < max_time):
            changed = False
            if not sol:
                break
            # Compute coverage counts
            coverage_counts = np.sum(neighbor_matrix[sol], axis=0)
            
            # Find a redundant codeword
            for i in range(len(sol) - 1, -1, -1):
                if time.time() - start_time > max_time:
                    break
                w = sol[i]
                # Check if w is redundant
                if coverage_counts[w] > 1:
                    min_cov = np.min(coverage_counts[neighbor_matrix[w]])
                    if min_cov > 1:
                        sol.pop(i)
                        changed = True
                        break  # Restart loop to recompute coverage_counts
        return sol

    def local_search_swaps(solution, max_iters=300, max_time=None):
        """Try to improve solution by swapping codewords and removing redundants."""
        if not solution:
            return solution
        sol = list(solution)
        sol_set = set(sol)
        best_sol = list(sol)
        best_len = len(sol)
        
        if max_time is None:
            max_time = limit
        
        # Precompute coverage counts
        coverage_counts = np.sum(neighbor_matrix[sol], axis=0).astype(np.int32)
        
        for iteration in range(max_iters):
            if time.time() - start_time > max_time:
                break
            
            improved = False
            if sol:
                # Pick a random codeword to potentially remove
                remove_idx = random.randint(0, len(sol) - 1)
                w_remove = sol[remove_idx]
                
                # Words only covered by w_remove
                mask_w_remove = neighbor_matrix[w_remove]
                critical_words = np.where((coverage_counts == 1) & mask_w_remove)[0]
                
                if len(critical_words) == 0:
                    # w_remove is redundant, just remove it
                    sol.pop(remove_idx)
                    sol_set.remove(w_remove)
                    coverage_counts -= mask_w_remove
                    improved = True
                    if len(sol) < best_len:
                        best_sol = list(sol)
                        best_len = len(sol)
                    continue
                
                # Find candidates that cover at least one of the critical words
                not_in_sol = np.ones(total_words, dtype=bool)
                sol_arr = np.array(sol, dtype=int)
                not_in_sol[sol_arr] = False
                
                if len(critical_words) > 0:
                    candidate_covers_critical = neighbor_matrix[:, critical_words].any(axis=1)
                    valid_candidates_mask = not_in_sol & candidate_covers_critical
                    valid_candidates = np.where(valid_candidates_mask)[0]
                    
                    if len(valid_candidates) > 0:
                        # Try a random candidate
                        c_add = int(random.choice(valid_candidates))
                        mask_c_add = neighbor_matrix[c_add]
                        
                        # Check validity: losers must have coverage > 1
                        losers = mask_w_remove & ~mask_c_add
                        if not losers.any() or np.min(coverage_counts[losers]) > 1:
                            # Swap is valid
                            sol[remove_idx] = c_add
                            sol_set.remove(w_remove)
                            sol_set.add(c_add)
                            coverage_counts -= mask_w_remove
                            coverage_counts += mask_c_add
                            improved = True
            
            if not improved:
                # Occasionally try a different strategy: pick the worst codeword
                # (one with most critical words) and try to replace it
                if random.random() < 0.3 and sol:
                    # Find codeword with most critical words
                    worst_idx = 0
                    worst_critical = 0
                    for i in range(len(sol)):
                        w = sol[i]
                        mask_w = neighbor_matrix[w]
                        n_critical = np.sum((coverage_counts == 1) & mask_w)
                        if n_critical > worst_critical:
                            worst_critical = n_critical
                            worst_idx = i
                    
                    if worst_critical > 0:
                        w_remove = sol[worst_idx]
                        mask_w_remove = neighbor_matrix[w_remove]
                        critical_words = np.where((coverage_counts == 1) & mask_w_remove)[0]
                        
                        not_in_sol = np.ones(total_words, dtype=bool)
                        sol_arr = np.array(sol, dtype=int)
                        not_in_sol[sol_arr] = False
                        
                        candidate_covers_critical = neighbor_matrix[:, critical_words].any(axis=1)
                        valid_candidates_mask = not_in_sol & candidate_covers_critical
                        valid_candidates = np.where(valid_candidates_mask)[0]
                        
                        if len(valid_candidates) > 0:
                            # Try multiple candidates
                            for c_add in random.sample(valid_candidates.tolist(), min(10, len(valid_candidates))):
                                c_add = int(c_add)
                                mask_c_add = neighbor_matrix[c_add]
                                losers = mask_w_remove & ~mask_c_add
                                if not losers.any() or np.min(coverage_counts[losers]) > 1:
                                    sol[worst_idx] = c_add
                                    sol_set.remove(w_remove)
                                    sol_set.add(c_add)
                                    coverage_counts -= mask_w_remove
                                    coverage_counts += mask_c_add
                                    improved = True
                                    break
        
        # Final removal pass
        sol = remove_redundant(sol, max_time=max_time)
        if len(sol) < best_len:
            best_sol = list(sol)
            best_len = len(sol)
        
        return best_sol

    # Phase 1: Multiple greedy initializations with random tie-breaking
    # We'll run several greedy passes and keep the best result
    best_solution = None
    best_size = total_words + 1
    
    num_greedy_runs = 8
    greedy_deadline = start_time + 40  # Spend ~40s on greedy
    
    for run in range(num_greedy_runs):
        if time.time() - start_time > greedy_deadline:
            break
        
        uncovered = np.ones(total_words, dtype=bool)
        solution = []
        
        # Greedy with random tie-breaking
        sol_set = set()
        while uncovered.any():
            if time.time() - start_time > greedy_deadline:
                break
            
            unc_int = uncovered.astype(np.int8)
            counts = neighbor_matrix @ unc_int
            
            if sol:
                sol_arr = np.array(list(sol_set), dtype=int)
                counts[sol_arr] = -1
            
            max_count = np.max(counts)
            if max_count <= 0:
                break
            
            # Random tie-breaking: pick from top candidates
            top_candidates = np.where(counts == max_count)[0]
            best_word = int(random.choice(top_candidates))
            
            solution.append(best_word)
            sol_set.add(best_word)
            uncovered[neighbor_matrix[best_word]] = False
        
        if not uncovered.any():
            solution = remove_redundant(solution, max_time=greedy_deadline)
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = list(solution)
    
    if best_solution is None:
        # Fallback: single greedy without time pressure
        uncovered = np.ones(total_words, dtype=bool)
        best_solution = greedy_cover(uncovered, [])
    
    # Phase 2: Local search refinement
    # Spend remaining time on local search
    remaining_time = limit - (time.time() - start_time)
    
    if remaining_time > 5:
        # Multiple rounds of local search
        num_local_rounds = min(5, int(remaining_time / 15))
        for round_num in range(num_local_rounds):
            if time.time() - start_time > limit - 5:
                break
            
            round_deadline = min(limit - 5, time.time() - start_time + remaining_time / (num_local_rounds - round_num))
            best_solution = local_search_swaps(best_solution, max_iters=200, max_time=round_deadline)
            
            if len(best_solution) < best_size:
                best_size = len(best_solution)
    
    # Final cleanup: remove any remaining redundants
    if time.time() - start_time < limit - 3:
        best_solution = remove_redundant(best_solution, max_time=limit - 3)
    
    # Final completeness check
    if not is_complete(best_solution):
        # Fallback: complete with greedy
        covered = np.any(neighbor_matrix[best_solution], axis=0) if best_solution else np.zeros(total_words, dtype=bool)
        uncovered = ~covered
        if time.time() - start_time < limit - 2:
            best_solution = greedy_cover(uncovered, best_solution, max_time=limit - 2)
    
    return [get_digits(idx) for idx in best_solution]
