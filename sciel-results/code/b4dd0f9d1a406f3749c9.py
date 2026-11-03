import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 145s to be safe.
    limit = 145.0
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbor masks as Python integers (bitmasks)
    # This is O(N * n * q) which is fast for N=6561
    neighbor_masks = [0] * total_words
    all_mask = (1 << total_words) - 1
    
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

    def is_complete(solution):
        """Check if solution covers all words."""
        if not solution:
            return False
        covered = 0
        for w in solution:
            covered |= neighbor_masks[w]
        return covered == all_mask

    def remove_redundant(solution):
        """Remove codewords that don't cover any unique word."""
        if not solution:
            return solution
        sol = list(solution)
        changed = True
        while changed and (time.time() - start_time < limit):
            changed = False
            # Iterate in reverse to avoid index issues when popping
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
                    # Restart the loop to check again from the end, as removal might make others redundant
                    break
        return sol

    def greedy_cover(uncovered_mask, solution, randomize_ties=False):
        """Greedy set cover starting from given solution and uncovered mask."""
        sol = list(solution)
        sol_set = set(sol)
        unc = uncovered_mask
        while unc:
            if time.time() - start_time > limit:
                break
            best_word = -1
            max_covered = 0
            candidates = []
            
            for w in range(total_words):
                if w in sol_set:
                    continue
                count = (neighbor_masks[w] & unc).bit_count()
                if count > max_covered:
                    max_covered = count
                    best_word = w
                    candidates = [w]
                elif count == max_covered and count > 0:
                    candidates.append(w)
            
            if best_word == -1:
                break
                
            if randomize_ties and len(candidates) > 1:
                best_word = random.choice(candidates)
            # If no randomization, best_word is already the first one found with max count
            
            sol.append(best_word)
            sol_set.add(best_word)
            unc &= ~neighbor_masks[best_word]
        return sol

    def local_search_swaps(solution, max_iters=100):
        """Try to improve solution by swapping codewords and removing redundant ones."""
        if not solution:
            return solution
        sol = list(solution)
        sol_set = set(sol)
        best_sol = list(sol)
        best_len = len(sol)
        
        for iteration in range(max_iters):
            if time.time() - start_time > limit:
                break
            
            # Compute coverage counts to identify critical words
            cov_counts = [0] * total_words
            for w in sol:
                m = neighbor_masks[w]
                # Efficiently increment counts for bits set in m
                # Since N=6561, iterating bits is okay, but we can also just use bit_count on intersections?
                # Actually, for swaps, we need to know which words are *only* covered by the word we might remove.
                
            # Strategy: Pick a random codeword to try removing.
            if not sol:
                break
                
            remove_idx = random.randint(0, len(sol) - 1)
            w_remove = sol[remove_idx]
            
            # Find words covered *only* by w_remove
            # These are the critical words. If we remove w_remove, we must cover these with a new word.
            others_covered = 0
            for j in range(len(sol)):
                if j != remove_idx:
                    others_covered |= neighbor_masks[sol[j]]
            
            critical_mask = neighbor_masks[w_remove] & ~others_covered
            
            if critical_mask == 0:
                # w_remove is redundant, just remove it
                sol.pop(remove_idx)
                sol_set.discard(w_remove)
                if len(sol) < best_len:
                    best_sol = list(sol)
                    best_len = len(sol)
                continue
            
            # We need to find a new word c_add that covers all critical words.
            # Ideally, c_add should also cover other words to allow future reductions, 
            # but for now, just covering critical words is the requirement for a valid swap.
            
            candidates = []
            for c in range(total_words):
                if c in sol_set:
                    continue
                # Check if c covers all critical words
                if (critical_mask & ~neighbor_masks[c]) == 0:
                    candidates.append(c)
            
            if not candidates:
                continue
                
            # Pick a random candidate to swap
            c_add = random.choice(candidates)
            
            # Perform the swap
            new_sol = sol.copy()
            new_sol[remove_idx] = c_add
            new_sol_set = set(new_sol)
            
            # Check if the new solution is complete (it should be by construction of critical_mask, 
            # but let's be safe or just trust the logic? 
            # Logic: critical_mask was the only part lost. c_add covers critical_mask. 
            # The rest was covered by others. So new_sol is complete.)
            
            # However, the swap might make *another* codeword redundant. 
            # So we try to remove redundant ones from new_sol.
            new_sol = remove_redundant(new_sol)
            
            if len(new_sol) < best_len:
                best_sol = list(new_sol)
                best_len = len(new_sol)
                sol = new_sol
                sol_set = set(sol)
            # If no improvement, we could revert, but we already mutated 'sol' via new_sol assignment?
            # Wait, 'sol' is a list. 'new_sol' is a copy. We only update 'sol' if we found a better best_len?
            # No, we should update 'sol' to the new state to continue searching from there, 
            # even if it's not the global best yet, to explore the neighborhood.
            # But if we don't improve, we might oscillate. 
            # Let's only accept the swap if it leads to a size reduction or equal size with better structure?
            # Actually, standard local search accepts moves that don't increase size.
            # Here, size is constant during swap. Reduction comes from subsequent remove_redundant.
            # If remove_redundant didn't reduce size, we revert to avoid noise?
            # Let's keep it simple: accept if size <= current best_len.
            if len(new_sol) <= best_len:
                sol = new_sol
                sol_set = set(sol)
        
        return best_sol if len(best_sol) <= len(sol) else sol

    def algebraic_initial_solution():
        """
        Construct an initial solution using the product structure of ternary space.
        For n=8, split into two halves of length 4.
        Use a covering code of length 4, radius 1 (or similar) for each half.
        """
        if n != 8 or q != 3:
            return None
        
        half = n // 2
        half_words = q ** half
        
        def greedy_half_cover(half_n, half_q, half_radius, randomize=False):
            hw = half_q ** half_n
            h_powers = [half_q ** i for i in range(half_n)]
            h_neighbor_masks = [0] * hw
            h_all_mask = (1 << hw) - 1
            
            for idx in range(hw):
                mask = 1 << idx
                for i in range(half_n):
                    original = (idx // h_powers[i]) % half_q
                    base = idx - original * h_powers[i]
                    for new_val in range(half_q):
                        if new_val != original:
                            neighbor_idx = base + new_val * h_powers[i]
                            mask |= (1 << neighbor_idx)
                h_neighbor_masks[idx] = mask
            
            uncovered = h_all_mask
            code = []
            while uncovered:
                if time.time() - start_time > limit:
                    return None
                best_word = -1
                max_covered = 0
                candidates = []
                for w in range(hw):
                    count = (h_neighbor_masks[w] & uncovered).bit_count()
                    if count > max_covered:
                        max_covered = count
                        best_word = w
                        candidates = [w]
                    elif count == max_covered and count > 0:
                        candidates.append(w)
                if best_word == -1:
                    break
                if randomize and len(candidates) > 1:
                    best_word = random.choice(candidates)
                code.append(best_word)
                uncovered &= ~h_neighbor_masks[best_word]
            return code
        
        # Get covering code for each half
        # We try a few randomizations here to get different algebraic structures
        code1 = greedy_half_cover(half, q, radius, randomize=True)
        code2 = greedy_half_cover(half, q, radius, randomize=True)
        
        if code1 is None or code2 is None:
            return None
        
        solution = []
        for c1 in code1:
            for c2 in code2:
                c1_digits = [0] * half
                c2_digits = [0] * half
                temp = c1
                for i in range(half):
                    c1_digits[i] = temp % q
                    temp //= q
                temp = c2
                for i in range(half):
                    c2_digits[i] = temp % q
                    temp //= q
                
                full_idx = 0
                for i in range(half):
                    full_idx += c1_digits[i] * powers[i]
                for i in range(half):
                    full_idx += c2_digits[i] * powers[half + i]
                
                solution.append(full_idx)
        
        return solution

    best_solution = None
    best_size = total_words + 1

    # We will run a few iterations of:
    # 1. Generate Initial Solution (Algebraic or Greedy)
    # 2. Remove Redundant
    # 3. Local Search
    
    num_runs = 10 # We will do up to 10 runs, but stop if time is up
    
    for run in range(num_runs):
        if time.time() - start_time > limit - 10: # Leave 10s for final checks
            break
            
        # Choose initial strategy
        # Alternate between Algebraic and Pure Greedy to diversify
        if run % 2 == 0:
            solution = algebraic_initial_solution()
            if solution is None:
                # Fallback to greedy if algebraic failed (e.g., timeout or error)
                solution = greedy_cover(all_mask, [], randomize_ties=True)
        else:
            solution = greedy_cover(all_mask, [], randomize_ties=True)
        
        if not solution:
            continue
            
        # Phase 2: Remove redundant codewords
        solution = remove_redundant(solution)
        
        # Phase 3: Local search with swaps
        # Allocate some time for local search
        # If we have lots of time, do more iterations
        remaining_time = limit - (time.time() - start_time)
        max_iters = min(500, int(remaining_time / 2)) # Rough estimate
        solution = local_search_swaps(solution, max_iters=max_iters)
        
        # Final check
        if is_complete(solution):
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = list(solution)
                
    # If no valid solution found (should not happen), fallback to a known valid one
    if best_solution is None:
        # Construct a trivial valid solution (all words) - this is huge but valid
        # Or use the first greedy run result if it was valid
        # Let's just run one more greedy to be sure
        solution = greedy_cover(all_mask, [], randomize_ties=False)
        solution = remove_redundant(solution)
        if is_complete(solution):
            best_solution = solution
        else:
            # Absolute fallback
            best_solution = list(range(total_words))
            
    return [get_digits(idx) for idx in best_solution]
