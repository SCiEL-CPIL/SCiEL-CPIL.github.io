import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # 19 for q=3, n=6

    # Build the direct-sum construction
    # Tetracode is a perfect 1-error-correcting code of length 4 over GF(3)
    # It has 9 words.
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Construct initial code: direct sum with all tails
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))
            
    m = len(code) # 81 for q=3, n=6

    # Precompute word indices
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute coverage sets for each codeword as sorted lists for faster iteration
    covers = []
    for c in code:
        c_set = set()
        c_set.add(word_to_index(c))
        c_list = list(c)
        for pos in range(n):
            orig_val = c_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                c_list[pos] = new_val
                c_set.add(word_to_index(tuple(c_list)))
            c_list[pos] = orig_val
        covers.append(sorted(c_set))
    
    # Precompute coverage_count as numpy array
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for ci in range(m):
        for wi in covers[ci]:
            coverage_count[wi] += 1

    best_code = code[:]
    best_count = m

    # Convert covers to numpy arrays for vectorized operations
    covers_np = [np.array(lst, dtype=np.int32) for lst in covers]
    
    # Precompute for each codeword, the set of codewords it covers (for fast swap checks)
    # Instead, let's precompute for each ci, a set of wi it covers (already have as list)
    # For fast "wi in covers[ri]" checks, convert to sets
    covers_set = [set(lst) for lst in covers]

    def is_safe_swap_fast(ai, ri, curr_coverage):
        # Vectorized check: find words in covers[ai] with coverage == 1
        ai_covers = covers_np[ai]
        unique_words = ai_covers[curr_coverage[ai_covers] == 1]
        if len(unique_words) == 0:
            return True
        # Check if all unique words are covered by ri
        ri_set = covers_set[ri]
        for wi in unique_words:
            if wi not in ri_set:
                return False
        return True

    def greedy_remove(active_set, curr_coverage):
        # Returns new active_set and updated curr_coverage
        # Greedy removal pass with vectorized redundancy check
        # We need to iterate in a random order to avoid bias
        order = list(active_set)
        random.shuffle(order)
        
        new_active_set = set(active_set)
        for ci in order:
            if ci not in new_active_set:
                continue
            ci_covers = covers_np[ci]
            if np.all(curr_coverage[ci_covers] > 1):
                new_active_set.discard(ci)
                curr_coverage[ci_covers] -= 1
        return new_active_set

    max_restarts = 20000  # Will break based on time

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Reset coverage counts
        curr_coverage = coverage_count.copy()
        active_set = set(range(m))
        
        # 1. Initial Greedy Removal
        active_set = greedy_remove(active_set, curr_coverage)
        
        current_count = len(active_set)
        
        # 2. Local Search: Swap and Add-Remove
        improved = True
        while improved:
            improved = False
            
            # --- Swap Phase ---
            removed_set = set(range(m)) - active_set
            removed_list = list(removed_set)
            random.shuffle(removed_list)
            active_list = list(active_set)
            random.shuffle(active_list)

            for ri in removed_list:
                if ri not in removed_set:
                    continue
                for ai in active_list:
                    if ai not in active_set:
                        continue
                    if is_safe_swap_fast(ai, ri, curr_coverage):
                        # Perform swap
                        active_set.discard(ai)
                        active_set.add(ri)
                        curr_coverage[covers_np[ai]] -= 1
                        curr_coverage[covers_np[ri]] += 1
                        improved = True
                        break
                if improved:
                    break
            
            # --- Add-Remove Phase ---
            # If no swap found, try adding a word to enable more removals
            if not improved:
                # Identify words that are covered only once (critical words)
                # If we add a codeword that covers many critical words, we might be able to remove multiple codewords
                # Heuristic: Pick a random removed word that covers many currently "critical" words
                # Or just pick a random removed word and see if adding it allows removal of >1 active words
                
                # Optimization: Only try this if we are stuck and have time
                if time.time() < deadline - 10.0:
                    # Get list of removed words
                    removed_candidates = list(removed_set)
                    if removed_candidates:
                        # Shuffle to try different ones
                        random.shuffle(removed_candidates)
                        
                        # Try a few candidates
                        tried = 0
                        max_tries = 50
                        while tried < max_tries:
                            ri = removed_candidates[tried]
                            tried += 1
                            
                            # Tentatively add ri
                            # Check if adding ri allows removing more than 1 active codeword
                            # This is complex to check directly. 
                            # Simpler heuristic: Add ri, run greedy remove. If size decreases, keep it.
                            
                            # Save state
                            old_active_size = len(active_set)
                            # We need to simulate the add
                            # Add ri to coverage
                            curr_coverage[covers_np[ri]] += 1
                            new_active_set = active_set | {ri}
                            
                            # Run greedy remove on the new set
                            # We need a version of greedy_remove that works on a specific set and returns the new set
                            # Re-using greedy_remove logic but careful with mutation
                            # Let's create a helper or inline it
                            
                            # Inline greedy remove for simulation
                            sim_coverage = curr_coverage.copy() # Expensive!
                            # Better: just check if any active codeword becomes redundant given ri is present?
                            # Actually, if we add ri, some active codewords might become redundant.
                            # Let's just try adding ri, then calling greedy_remove, and see if size < old_active_size.
                            # But greedy_remove mutates coverage.
                            
                            # To avoid full copy, we can check:
                            # Does adding ri make any active codeword redundant?
                            # A codeword ai is redundant if all words in covers[ai] are covered by others.
                            # With ri added, covers[ri] is available.
                            
                            # Let's stick to the simpler "Swap" first, and if that fails,
                            # try a "2-for-1" or "3-for-1" replacement?
                            # The "Add-Remove" strategy in the prompt knowledge suggests:
                            # "Add a word, then remove redundant words".
                            
                            # Let's implement a lightweight check:
                            # If we add ri, which active codewords become redundant?
                            # An active codeword ai is redundant if for all w in covers[ai], w is covered by (active_set - {ai}) U {ri}.
                            # This is expensive to check for all ai.
                            
                            # Alternative: Just do the full simulation for a few candidates.
                            # Since m=81, copying an array of size 729 is fast.
                            
                            sim_coverage = curr_coverage.copy()
                            sim_active_set = set(active_set)
                            sim_active_set.add(ri)
                            
                            # Run greedy removal on sim_active_set
                            # We need to update sim_coverage as we remove
                            # We can reuse the logic from greedy_remove but on the sim objects
                            
                            # To keep it fast, let's just try 1-2 candidates per loop iteration
                            # and accept if it improves.
                            
                            # Actually, the current code's swap is a 1-for-1.
                            # A 1-for-2 (add 1, remove 2) is a net improvement of -1.
                            
                            # Let's try:
                            # 1. Add ri
                            # 2. Check if any active ai becomes redundant.
                            #    If yes, remove it.
                            #    Repeat until no more removals.
                            # 3. If final size < old size, accept.
                            
                            # This is effectively running a local search step.
                            
                            # Let's just use the existing greedy_remove function but on a copy?
                            # No, greedy_remove takes active_set and curr_coverage and mutates coverage.
                            # We need a pure function or careful copying.
                            
                            # Let's create a helper that returns the new set without mutating the original coverage,
                            # or just copy the coverage.
                            
                            # Given the small size, copying coverage (729 ints) is ~microseconds.
                            # Doing this 50 times is negligible.
                            
                            # Let's implement a helper:
                            def simulate_add_remove(active_set, ri, curr_coverage):
                                # Returns (new_active_set, new_coverage, improvement)
                                new_coverage = curr_coverage.copy()
                                new_active_set = set(active_set)
                                new_active_set.add(ri)
                                new_coverage[covers_np[ri]] += 1
                                
                                # Greedy remove
                                order = list(new_active_set)
                                random.shuffle(order)
                                for ai in order:
                                    if ai not in new_active_set:
                                        continue
                                    ai_covers = covers_np[ai]
                                    if np.all(new_coverage[ai_covers] > 1):
                                        new_active_set.discard(ai)
                                        new_coverage[ai_covers] -= 1
                                        
                                return new_active_set, new_coverage

                            new_active_set, new_coverage = simulate_add_remove(active_set, ri, curr_coverage)
                            
                            if len(new_active_set) < len(active_set):
                                # Accept the change
                                active_set = new_active_set
                                curr_coverage = new_coverage
                                improved = True
                                break
                            
                            # If no improvement from this ri, try next candidate
                            # We don't need to revert anything because we used copies in simulate_add_remove
                            # But we did NOT modify the original active_set or curr_coverage in the simulation
                            # So we can just continue to the next ri

        current_count = len(active_set)
        if current_count < best_count:
            best_count = current_count
            best_code = [code[i] for i in active_set]

    return [list(c) for c in best_code]
