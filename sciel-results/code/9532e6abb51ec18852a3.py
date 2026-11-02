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
    
    # Precompute neighbor masks as Python integers (bitmasks)
    neighbor_masks = [0] * total_words
    
    for idx in range(total_words):
        mask = 1 << idx
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    neighbor_idx = base + new_val * powers[i]
                    mask |= (1 << neighbor_idx)
        neighbor_masks[idx] = mask
    
    def greedy_cover(uncovered_mask, solution):
        """Greedy set cover starting from given solution and uncovered mask."""
        sol = list(solution)
        unc = uncovered_mask
        while unc:
            if time.time() - start_time > limit:
                break
            best_word = -1
            max_covered = 0
            for w in range(total_words):
                if w in sol:
                    continue
                count = (neighbor_masks[w] & unc).bit_count()
                if count > max_covered:
                    max_covered = count
                    best_word = w
            if best_word == -1:
                break
            sol.append(best_word)
            unc &= ~neighbor_masks[best_word]
        return sol
    
    def is_complete(solution):
        """Check if solution covers all words."""
        covered = 0
        for w in solution:
            covered |= neighbor_masks[w]
        return covered == (1 << total_words) - 1
    
    def remove_redundant(solution):
        """Remove codewords that don't cover any unique word."""
        if not solution:
            return solution
        sol = list(solution)
        changed = True
        while changed and (time.time() - start_time < limit):
            changed = False
            for i in range(len(sol) - 1, -1, -1):
                if time.time() - start_time > limit:
                    break
                w = sol[i]
                # Check if w covers any word not covered by others
                others_covered = 0
                for j in range(len(sol)):
                    if j != i:
                        others_covered |= neighbor_masks[sol[j]]
                if (neighbor_masks[w] & ~others_covered) == 0:
                    sol.pop(i)
                    changed = True
                    break
        return sol
    
    def local_search_swaps(solution, max_iters=200):
        """Try to improve solution by swapping codewords."""
        if not solution:
            return solution
        sol = list(solution)
        sol_set = set(sol)
        best_sol = list(sol)
        best_len = len(sol)
        
        # Precompute coverage counts for current solution
        def compute_coverage_counts(sol_list):
            counts = [0] * total_words
            for w in sol_list:
                mask = neighbor_masks[w]
                # Count bits set in mask
                # We need to add 1 to each bit position in mask
                m = mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb
            return counts
        
        # For efficiency, track which words are covered by exactly one codeword
        # A swap is valid if no word goes from coverage 1 to 0
        
        for iteration in range(max_iters):
            if time.time() - start_time > limit:
                break
            
            # Compute coverage counts
            cov_counts = [0] * total_words
            for w in sol:
                m = neighbor_masks[w]
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    cov_counts[idx] += 1
                    m ^= lsb
            
            # Find words covered by exactly one codeword (critical words)
            critical = [i for i in range(total_words) if cov_counts[i] == 1]
            
            # Try random swaps
            improved = False
            if sol:
                # Pick a random codeword to potentially remove
                remove_idx = random.randint(0, len(sol) - 1)
                w_remove = sol[remove_idx]
                
                # Words only covered by w_remove
                only_covered_by_remove = [i for i in range(total_words) 
                                          if cov_counts[i] == 1 and (neighbor_masks[w_remove] >> i) & 1]
                
                if not only_covered_by_remove:
                    # w_remove is redundant, just remove it
                    sol.pop(remove_idx)
                    improved = True
                    continue
                
                # Find candidates that cover at least one of the critical words
                candidates = []
                for c in range(total_words):
                    if c in sol_set:
                        continue
                    # Check if c covers any critical word
                    for cw in only_covered_by_remove:
                        if (neighbor_masks[c] >> cw) & 1:
                            candidates.append(c)
                            break
                
                if not candidates:
                    continue
                
                # Try a random candidate
                c_add = random.choice(candidates)
                
                # Check if the swap is valid: after removing w_remove and adding c_add,
                # all words must still be covered
                # Words that lose coverage: those covered by w_remove but not by c_add and not by others
                # This is complex; simpler to just check completeness after swap
                
                new_sol = sol.copy()
                new_sol[remove_idx] = c_add
                new_sol_set = set(new_sol)
                
                # Quick check: is the new solution complete?
                covered = 0
                for w in new_sol:
                    covered |= neighbor_masks[w]
                if covered == (1 << total_words) - 1:
                    sol = new_sol
                    sol_set = set(sol)
                    improved = True
                    if len(sol) < best_len:
                        best_sol = list(sol)
                        best_len = len(sol)
            
            if not improved and iteration > 50:
                # No improvement in a while, try a different strategy
                # Add a random codeword and then remove redundant ones
                if len(sol) < total_words:
                    random_word = random.randint(0, total_words - 1)
                    if random_word not in sol_set:
                        sol.append(random_word)
                        sol_set.add(random_word)
                        sol = remove_redundant(sol)
                        sol_set = set(sol)
        
        return best_sol if len(best_sol) <= len(sol) else sol
    
    # Phase 1: Greedy to get initial solution
    initial_solution = []
    uncovered = (1 << total_words) - 1
    
    while uncovered:
        if time.time() - start_time > limit:
            break
        best_word = -1
        max_covered = 0
        for w in range(total_words):
            count = (neighbor_masks[w] & uncovered).bit_count()
            if count > max_covered:
                max_covered = count
                best_word = w
        if best_word == -1:
            break
        initial_solution.append(best_word)
        uncovered &= ~neighbor_masks[best_word]
    
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
        covered = 0
        for w in solution:
            covered |= neighbor_masks[w]
        uncovered = ((1 << total_words) - 1) & ~covered
        while uncovered:
            if time.time() - start_time > limit:
                break
            best_word = -1
            max_covered = 0
            for w in range(total_words):
                if w in solution:
                    continue
                count = (neighbor_masks[w] & uncovered).bit_count()
                if count > max_covered:
                    max_covered = count
                    best_word = w
            if best_word == -1:
                break
            solution.append(best_word)
            uncovered &= ~neighbor_masks[best_word]
    
    return [get_digits(idx) for idx in solution]
