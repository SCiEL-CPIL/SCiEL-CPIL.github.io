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

    # Simulated annealing parameters
    T_initial = 10.0
    T_cooling = 0.995
    T_min = 0.01

    max_restarts = 500  # More restarts with SA

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Reset coverage counts
        curr_coverage = coverage_count.copy()
        active = [True] * m
        active_set = set(range(m))
        removed_set = set()

        # Random order for removal attempts
        order = list(range(m))
        random.shuffle(order)

        # Greedy removal pass with vectorized redundancy check
        for ci in order:
            if not active[ci]:
                continue
            # Vectorized check: all covered words must have coverage > 1
            ci_covers = covers_np[ci]
            if np.all(curr_coverage[ci_covers] > 1):
                active[ci] = False
                active_set.discard(ci)
                removed_set.add(ci)
                curr_coverage[ci_covers] -= 1

        # Simulated annealing-based local search
        T = T_initial
        current_count = len(active_set)
        
        while T > T_min and time.time() < deadline - 5.0:
            improved = False
            removed_list = list(removed_set)
            random.shuffle(removed_list)
            active_list = list(active_set)
            random.shuffle(active_list)
            
            found_move = False
            for ri in removed_list:
                if ri not in removed_set:
                    continue
                for ai in active_list:
                    if ai not in active_set:
                        continue
                    if is_safe_swap_fast(ai, ri, curr_coverage):
                        # Perform swap (equal size move)
                        active[ai] = False
                        active[ri] = True
                        active_set.discard(ai)
                        active_set.add(ri)
                        removed_set.discard(ri)
                        removed_set.add(ai)
                        curr_coverage[covers_np[ai]] -= 1
                        curr_coverage[covers_np[ri]] += 1
                        improved = True
                        found_move = True
                        break
                if found_move:
                    break
            
            # Try adding a codeword if we can remove more later (explore)
            if not found_move and random.random() < 0.1:
                # Try to add a removed codeword temporarily
                if removed_set:
                    ri = random.choice(list(removed_set))
                    # Check if adding ri allows us to remove more active codewords
                    new_coverage = curr_coverage.copy()
                    new_coverage[covers_np[ri]] += 1
                    removable_count = 0
                    for ai in active_set:
                        ai_covers = covers_np[ai]
                        if np.all(new_coverage[ai_covers] > 1):
                            removable_count += 1
                    
                    # Accept if we can remove at least 2 codewords
                    if removable_count >= 2:
                        # Perform the add
                        active[ri] = True
                        active_set.add(ri)
                        removed_set.discard(ri)
                        curr_coverage = new_coverage
                        improved = True
            
            if not improved:
                # No move found, cool down
                T *= T_cooling
            else:
                # If we made a move, keep temperature (or cool slightly)
                T = max(T * T_cooling, T_min)
            
            current_count = len(active_set)
            if current_count < best_count:
                best_count = current_count
                best_code = [code[i] for i in active_set]
            
            # After making moves, do another greedy removal pass
            order2 = list(active_set)
            random.shuffle(order2)
            for ci in order2:
                if ci not in active_set:
                    continue
                ci_covers = covers_np[ci]
                if np.all(curr_coverage[ci_covers] > 1):
                    active[ci] = False
                    active_set.discard(ci)
                    removed_set.add(ci)
                    curr_coverage[ci_covers] -= 1

    return [list(c) for c in best_code]
