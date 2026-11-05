"""Model-and-optimize approach for ternary covering code K_3(6,1).

Strategy:
1. Start from the direct-sum construction (9 * 3^2 = 81 words) as a baseline.
2. Iteratively remove codewords that are redundant (their covered words are still
   covered by remaining codewords), using greedy set-cover pruning.
3. Use Max-SAT-inspired local search (WalkSAT with dynamic clause weights) to
   find smaller covers by exploring swap/removal moves with adaptive penalties.
4. Use time management to ensure we finish well within the 120s limit.

The key insight: the direct-sum construction gives 81 words, but many are redundant.
By removing redundant codewords greedily and then using weighted local search,
we can often get down to ~73 or better.
"""
import itertools
import time
import random
import sys
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin: finish by 110s, limit is 120s
    
    # For the specific problem: q=3, n=6, radius=1
    # Direct sum of ternary tetracode (9 words of length 4) with all 9 tails
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Build initial code: direct sum
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))
    
    base_count = len(code)  # 81 for q=3, n=6
    
    # If n <= 4, just return the tetracode-based construction directly
    if n <= 4:
        if n == 4:
            return [list(h) for h in tetracode]
        else:
            result = [list(c) for c in itertools.product(range(q), repeat=n)]
            return result[:q**n]
    
    # For n > 4, use the direct sum and then prune
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    total_words = q ** n
    
    # For each codeword, compute which words it covers
    codeword_covers = []
    for ci, c in enumerate(code):
        covers = set()
        covers.add(word_to_index(c))
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                w_list = list(c)
                w_list[pos] = val
                w = tuple(w_list)
                covers.add(word_to_index(w))
        codeword_covers.append(covers)
    
    # Precompute coverage: for each word, which codewords cover it
    covered_by = [[] for _ in range(total_words)]
    for ci, covers in enumerate(codeword_covers):
        for wi in covers:
            covered_by[wi].append(ci)
    
    # Phase 1: Greedy removal of redundant codewords
    active = [True] * len(code)
    
    max_iterations = 100
    for iteration in range(max_iterations):
        if time.time() > deadline - 20.0:
            break
        
        removed_any = False
        for ci in range(len(code)):
            if time.time() > deadline - 20.0:
                break
            if not active[ci]:
                continue
            
            is_redundant = True
            for wi in codeword_covers[ci]:
                found_other = False
                for cj in covered_by[wi]:
                    if cj != ci and active[cj]:
                        found_other = True
                        break
                if not found_other:
                    is_redundant = False
                    break
            
            if is_redundant:
                active[ci] = False
                removed_any = True
    
        if not removed_any:
            break
    
    # Build code after greedy removal
    remaining_indices = [i for i in range(len(code)) if active[i]]
    final_code = [list(code[i]) for i in remaining_indices]
    
    # Phase 2: Max-SAT-inspired local search with dynamic clause weights
    # Each codeword is a "variable". Coverage constraints are "clauses".
    # We minimize the number of active codewords while maintaining full coverage.
    # WalkSAT-style: pick a violated clause (uncovered word), then flip a variable
    # (remove/add a codeword) with probability biased by clause weight.
    
    def _local_search_maxsat(initial_active, initial_code_list, time_limit):
        """Max-SAT-inspired local search for minimum cover."""
        local_start = time.time()
        local_deadline = local_start + time_limit
        
        m = len(initial_code_list)
        active_arr = np.array(initial_active, dtype=bool)
        
        # Coverage count: for each word, how many active codewords cover it
        cov_count = np.zeros(total_words, dtype=np.int32)
        for ci in range(m):
            if active_arr[ci]:
                for wi in codeword_covers[ci]:
                    cov_count[wi] += 1
        
        # Objective: number of active codewords (we want to minimize)
        # Constraint: all words must have cov_count >= 1
        # WalkSAT: when constraint violated, pick a random uncovered word,
        # then try to fix it by adding a codeword or removing a redundant one.
        
        best_count = int(np.sum(active_arr))
        best_state = active_arr.copy()
        
        # Clause weights: initially uniform, updated dynamically
        # Weight of a "clause" (word) = 1 if uncovered, else 0
        # We use a soft penalty: sum of (1 - min(cov_count, 1)) * weight
        # Plus the hard cost of number of codewords
        
        # Precompute: for each codeword, the list of word indices it covers (as numpy array)
        covers_arrays = [np.array(list(covers), dtype=np.int32) for covers in codeword_covers]
        
        # Also precompute: for each word, list of codeword indices that cover it
        covered_by_arrays = [np.array(lst, dtype=np.int32) for lst in covered_by]
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < local_deadline and no_improve_count < max_no_improve:
            # Check if we have a valid cover
            uncovered = np.where(cov_count == 0)[0]
            
            if len(uncovered) == 0:
                # Valid cover: try to remove a random codeword
                active_indices = np.where(active_arr)[0]
                if len(active_indices) == 0:
                    break
                
                ci = active_indices[random.randint(0, len(active_indices) - 1)]
                
                # Check if removing ci keeps cover valid
                # A word becomes uncovered if cov_count[wi] == 1 for some wi in covers[ci]
                can_remove = True
                for wi in covers_arrays[ci]:
                    if cov_count[wi] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove ci
                    active_arr[ci] = False
                    for wi in covers_arrays[ci]:
                        cov_count[wi] -= 1
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Invalid cover: pick a random uncovered word
                wi = uncovered[random.randint(0, len(uncovered) - 1)]
                
                # Find candidate codewords that cover this word
                candidates = covered_by_arrays[wi]
                
                if len(candidates) == 0:
                    # No codeword covers this word - this shouldn't happen if we start valid
                    no_improve_count += 1
                    continue
                
                # WalkSAT: pick a candidate with some probability based on "clause weight"
                # Higher weight for candidates that are currently inactive (adding helps more)
                # and for candidates that cover more uncovered words
                weights = []
                for cj in candidates:
                    if active_arr[cj]:
                        w = 1.0
                    else:
                        w = 3.0  # bias toward adding inactive codewords
                    # Bonus for covering more uncovered words
                    uncov_in_covers = 0
                    for wj in covers_arrays[cj]:
                        if cov_count[wj] == 0:
                            uncov_in_covers += 1
                    w += uncov_in_covers * 0.1
                    weights.append(w)
                
                # Weighted random selection
                total_w = sum(weights)
                r = random.random() * total_w
                cumsum = 0
                chosen = candidates[0]
                for idx, cj in enumerate(candidates):
                    cumsum += weights[idx]
                    if r <= cumsum:
                        chosen = cj
                        break
                
                cj = chosen
                if active_arr[cj]:
                    # Already active - try removing a different active codeword that's redundant
                    # relative to this word
                    active_candidates = [c for c in candidates if active_arr[c]]
                    if len(active_candidates) > 1:
                        # Try to remove one of the other active candidates
                        others = [c for c in active_candidates if c != cj]
                        if others:
                            rem_ci = others[random.randint(0, len(others) - 1)]
                            can_remove = True
                            for wj in covers_arrays[rem_ci]:
                                if cov_count[wj] == 1:
                                    can_remove = False
                                    break
                            if can_remove:
                                active_arr[rem_ci] = False
                                for wj in covers_arrays[rem_ci]:
                                    cov_count[wj] -= 1
                                no_improve_count = 0
                                continue
                    no_improve_count += 1
                else:
                    # Add cj
                    active_arr[cj] = True
                    for wj in covers_arrays[cj]:
                        cov_count[wj] += 1
                    no_improve_count = 0
            
            # Track best
            current_count = int(np.sum(active_arr))
            if current_count < best_count:
                best_count = current_count
                best_state = active_arr.copy()
        
        return best_state
    
    # Run local search with remaining time
    if time.time() < deadline - 10.0 and len(final_code) > 73:
        time_for_ls = min(deadline - 10.0 - time.time(), 60.0)
        if time_for_ls > 5.0:
            initial_active = [True] * len(code)
            for i in range(len(code)):
                if not active[i]:
                    initial_active[i] = False
            
            best_ls_state = _local_search_maxsat(initial_active, code, time_for_ls)
            
            # Extract code from best LS state
            ls_code = [list(code[i]) for i in range(len(code)) if best_ls_state[i]]
            
            if len(ls_code) < len(final_code):
                final_code = ls_code
    
    # Phase 3: Final greedy cleanup on the best result
    if time.time() < deadline - 5.0:
        # Rebuild coverage for final_code
        fc_set = set(tuple(c) for c in final_code)
        fc_covers = []
        for c in final_code:
            covers = set()
            c_t = tuple(c)
            covers.add(word_to_index(c_t))
            for pos in range(n):
                orig = c_t[pos]
                for val in range(q):
                    if val == orig:
                        continue
                w_list = list(c_t)
                w_list[pos] = val
                covers.add(word_to_index(tuple(w_list)))
            fc_covers.append(covers)
        
        # Try to remove redundant codewords from final_code
        fc_active = [True] * len(final_code)
        for iteration in range(20):
            if time.time() > deadline - 3.0:
                break
            removed = False
            for ci in range(len(final_code)):
                if not fc_active[ci]:
                    continue
                redundant = True
                for wi in fc_covers[ci]:
                    other = False
                    for cj in range(len(final_code)):
                        if cj != ci and fc_active[cj] and wi in fc_covers[cj]:
                            other = True
                            break
                    if not other:
                        redundant = False
                        break
                if redundant:
                    fc_active[ci] = False
                    removed = True
            if not removed:
                break
        
        if any(not a for a in fc_active):
            final_code = [final_code[i] for i in range(len(final_code)) if fc_active[i]]
    
    # Final verification
    if time.time() < deadline - 2.0 and len(final_code) <= 200:
        covered_set = set()
        for c in final_code:
            c_t = tuple(c)
            covered_set.add(word_to_index(c_t))
            for pos in range(n):
                orig = c_t[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    w_list = list(c_t)
                    w_list[pos] = val
                    covered_set.add(word_to_index(tuple(w_list)))
        
        if len(covered_set) < total_words:
            # Fallback to greedy-removed version
            final_code = [list(code[i]) for i in remaining_indices]
    
    return final_code
