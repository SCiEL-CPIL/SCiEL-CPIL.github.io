import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    limit = 145.0  # Safety margin for 150s limit
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbor matrix as a numpy boolean array
    # neighbor_matrix[i] is a boolean vector of length total_words
    # neighbor_matrix[i][j] is True if j is within distance radius of i
    neighbor_matrix = np.zeros((total_words, total_words), dtype=bool)
    
    # We can optimize this generation, but for 6561x6561 it's manageable
    # However, generating it via loops in Python might be slow.
    # Let's try to generate it efficiently.
    
    # Alternative: Use the fact that distance 1 neighbors are specific.
    # For q=3, n=8, radius=1.
    # Each word has 1 + n*(q-1) neighbors.
    # 1 + 8*2 = 17 neighbors per word.
    # Total True entries: 6561 * 17 ~ 111k.
    # This is sparse. But boolean matrix is 6561*6561 ~ 43M booleans ~ 43MB.
    # This is acceptable memory-wise.
    
    # Efficient generation:
    # Initialize with diagonal True
    np.fill_diagonal(neighbor_matrix, True)
    
    # Add neighbors
    # To speed up, we can construct indices.
    row_indices = []
    col_indices = []
    
    for idx in range(total_words):
        # Diagonal
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
    
    def greedy_cover(uncovered_mask, solution):
        """Greedy set cover starting from given solution and uncovered mask."""
        sol = list(solution)
        unc = uncovered_mask.copy()
        while unc.any():
            if time.time() - start_time > limit:
                break
            
            # Find the word that covers the most uncovered words
            # uncovered_mask is a boolean array
            # neighbor_matrix is (N, N)
            # We want to sum neighbor_matrix[:, unc] along axis 1?
            # No, we want for each candidate w, count of j such that unc[j] and neighbor_matrix[w][j]
            # This is equivalent to neighbor_matrix[:, unc_indices].sum(axis=1)
            # But boolean indexing creates a copy.
            # Better: (neighbor_matrix * unc[None, :]).sum(axis=1)
            # Or just use dot product with 0/1 array?
            
            # Let's use a trick:
            # counts = np.sum(neighbor_matrix[:, unc], axis=1)
            # This is efficient enough for N=6561.
            
            # To avoid copying the large slice, we can do:
            # counts = neighbor_matrix.dot(unc.astype(np.int8))
            # But neighbor_matrix is bool.
            
            # Let's stick to simple sum on boolean mask.
            # It's an O(N^2) operation essentially.
            # 6561 * 6561 ~ 43M operations. In numpy this is fast.
            
            counts = np.sum(neighbor_matrix[:, unc], axis=1)
            
            # Exclude words already in solution
            sol_arr = np.array(sol, dtype=int)
            if len(sol_arr) > 0:
                counts[sol_arr] = 0
                
            best_word_idx = np.argmax(counts)
            
            if counts[best_word_idx] == 0:
                break
                
            sol.append(int(best_word_idx))
            unc[neighbor_matrix[best_word_idx]] = False
            
        return sol

    def is_complete(solution):
        """Check if solution covers all words."""
        if not solution:
            return total_words == 0
        covered = np.any(neighbor_matrix[solution], axis=0)
        return covered.all()

    def remove_redundant(solution):
        """Remove codewords that don't cover any unique word."""
        if not solution:
            return solution
        sol = list(solution)
        changed = True
        while changed and (time.time() - start_time < limit):
            changed = False
            if not sol:
                break
            # Compute coverage counts
            # coverage_counts[j] = number of codewords in sol that cover j
            coverage_counts = np.sum(neighbor_matrix[sol], axis=0)
            
            # Find a redundant codeword
            # A codeword w is redundant if for all j covered by w, coverage_counts[j] > 1
            # i.e., coverage_counts[neighbor_matrix[w]] are all > 1
            
            # We can check this for all w in sol at once?
            # For each w in sol, check if min(coverage_counts[neighbor_matrix[w]]) > 1
            # If yes, it's redundant.
            
            # This is still O(|sol| * N) in Python if looped, but we can vectorize the check?
            # Let's just loop through sol in reverse to remove one at a time, 
            # but use vectorized check for the specific word.
            
            for i in range(len(sol) - 1, -1, -1):
                if time.time() - start_time > limit:
                    break
                w = sol[i]
                # Check if w is redundant
                # Words covered by w: neighbor_matrix[w]
                # If all coverage_counts for those words are > 1, w is redundant.
                # Note: w covers itself, so coverage_counts[w] >= 1.
                # If coverage_counts[w] == 1, it's not redundant (it's the only one covering itself).
                # Actually, if coverage_counts[w] == 1, it might still be redundant if it covers no *other* unique words?
                # No, if it covers itself uniquely, it's not redundant.
                
                # Condition for redundancy:
                # For all j in support(neighbor_matrix[w]), coverage_counts[j] >= 2
                # This is equivalent to: min(coverage_counts[neighbor_matrix[w]]) >= 2
                
                # Optimization: If coverage_counts[w] == 1, it's not redundant.
                if coverage_counts[w] > 1:
                    min_cov = np.min(coverage_counts[neighbor_matrix[w]])
                    if min_cov > 1:
                        sol.pop(i)
                        changed = True
                        break # Restart loop to recompute coverage_counts
        return sol

    def local_search_swaps(solution, max_iters=200):
        """Try to improve solution by swapping codewords."""
        if not solution:
            return solution
        sol = list(solution)
        sol_set = set(sol)
        best_sol = list(sol)
        best_len = len(sol)
        
        # Precompute coverage counts
        coverage_counts = np.sum(neighbor_matrix[sol], axis=0)
        
        for iteration in range(max_iters):
            if time.time() - start_time > limit:
                break
            
            improved = False
            if sol:
                # Pick a random codeword to potentially remove
                remove_idx = random.randint(0, len(sol) - 1)
                w_remove = sol[remove_idx]
                
                # Words only covered by w_remove
                # These are j where coverage_counts[j] == 1 and neighbor_matrix[w_remove][j] is True
                # Let's find them
                mask_w_remove = neighbor_matrix[w_remove]
                critical_words = np.where((coverage_counts == 1) & mask_w_remove)[0]
                
                if len(critical_words) == 0:
                    # w_remove is redundant, just remove it
                    sol.pop(remove_idx)
                    sol_set.remove(w_remove)
                    # Update coverage counts
                    coverage_counts -= mask_w_remove
                    improved = True
                    if len(sol) < best_len:
                        best_sol = list(sol)
                        best_len = len(sol)
                    continue
                
                # Find candidates that cover at least one of the critical words
                # Candidates c such that c not in sol and neighbor_matrix[c][critical_words].any()
                # To speed up, we can check all c not in sol.
                # N=6561, this is manageable.
                
                # Create a boolean mask for words not in sol
                not_in_sol = np.ones(total_words, dtype=bool)
                sol_arr = np.array(sol, dtype=int)
                not_in_sol[sol_arr] = False
                
                # For each candidate c, check if it covers any critical word
                # We can compute: candidate_covers_critical = neighbor_matrix[:, critical_words].any(axis=1)
                # This is a (N, K) matrix where K is len(critical_words).
                # K is small (usually 1 or few).
                
                if len(critical_words) > 0:
                    # candidate_covers_critical[c] is True if c covers any critical word
                    candidate_covers_critical = neighbor_matrix[:, critical_words].any(axis=1)
                    
                    # Combine with not_in_sol
                    valid_candidates_mask = not_in_sol & candidate_covers_critical
                    
                    valid_candidates = np.where(valid_candidates_mask)[0]
                    
                    if len(valid_candidates) > 0:
                        # Try a random candidate
                        c_add = int(random.choice(valid_candidates))
                        
                        # Check if the swap is valid
                        # New coverage counts: coverage_counts - mask_w_remove + mask_c_add
                        # We need new_coverage_counts >= 1 everywhere.
                        # Specifically, for critical words, new_coverage_counts must be >= 1.
                        # Since they were 1, and we removed w_remove (which covered them), they become 0.
                        # So c_add MUST cover them. We already checked this.
                        # What about other words?
                        # Words covered by w_remove but not c_add:
                        # Their count decreases by 1.
                        # If their count was 1, it becomes 0 -> Invalid.
                        # So we need: for all j, if mask_w_remove[j] and not mask_c_add[j], then coverage_counts[j] > 1.
                        
                        mask_c_add = neighbor_matrix[c_add]
                        
                        # Check validity
                        # Losers: mask_w_remove & ~mask_c_add
                        # Gainers: mask_c_add & ~mask_w_remove (don't care for validity, only for future counts)
                        # Unchanged: ~mask_w_remove & ~mask_c_add
                        
                        # Condition: coverage_counts[Losers] must be > 1
                        # Also, words not covered by w_remove but covered by c_add: count increases, safe.
                        # Words covered by both: count stays same, safe.
                        
                        # So we just need to check Losers.
                        losers = mask_w_remove & ~mask_c_add
                        if losers.any():
                            min_losers_cov = np.min(coverage_counts[losers])
                            if min_losers_cov > 1:
                                # Swap is valid
                                sol[remove_idx] = c_add
                                sol_set.remove(w_remove)
                                sol_set.add(c_add)
                                
                                # Update coverage counts
                                coverage_counts -= mask_w_remove
                                coverage_counts += mask_c_add
                                
                                improved = True
                                # Length doesn't change in a swap, so best_len doesn't update unless we removed earlier?
                                # No, swap keeps length same.
                                # But maybe we can remove more later?
                                # For now, just track if we improved structure.
                                # Actually, we want to minimize length. Swaps don't change length.
                                # So best_sol only updates if length decreases (via removal).
                                # However, a swap might enable a future removal.
                                # So we keep best_sol as the shortest found.
            
            if not improved:
                # Try adding a random word and removing redundants?
                # Or just continue.
                pass
        
        return best_sol if len(best_sol) <= len(sol) else sol

    # Phase 1: Greedy to get initial solution
    # Start with empty solution and all words uncovered
    initial_solution = []
    uncovered = np.ones(total_words, dtype=bool)
    
    # To speed up greedy, we can use the vectorized version
    initial_solution = greedy_cover(uncovered, initial_solution)
    
    # Phase 2: Remove redundant codewords
    solution = remove_redundant(initial_solution)
    
    # Phase 3: Local search with swaps
    solution = local_search_swaps(solution, max_iters=500)
    
    # Phase 4: Try a few more rounds of removal + swap
    for _ in range(3):
        if time.time() - start_time > limit:
            break
        solution = remove_redundant(solution)
        solution = local_search_swaps(solution, max_iters=200)
    
    # Final check
    if not is_complete(solution):
        # Fallback: complete the solution with greedy
        covered = np.any(neighbor_matrix[solution], axis=0) if solution else np.zeros(total_words, dtype=bool)
        uncovered = ~covered
        solution = greedy_cover(uncovered, solution)
    
    return [get_digits(idx) for idx in solution]
