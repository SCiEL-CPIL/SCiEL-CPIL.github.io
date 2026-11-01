"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct(q, n, radius):
    # Fallback for non-specific cases or if optimization fails
    def fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    if q != 3 or n != 7 or radius != 1:
        return fallback()
        
    total_words = q ** n
    # Precompute neighbors for all words
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
        
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return w[::-1]
        
    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Precompute neighbors (distance <= radius)
    # For radius 1, neighbors are the word itself + 1 word change in each of n positions
    neighbors = []
    for i in range(total_words):
        w = all_words[i]
        nbs = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_copy = w[:]
                w_copy[pos] = new_val
                nbs.append(word_to_idx(w_copy))
        neighbors.append(nbs)
        
    neighbors_arr = np.array(neighbors)
    
    def greedy_cover(randomize=False):
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        while uncovered.any():
            # Count how many uncovered words each candidate covers
            # neighbors_arr is (total_words, radius_size)
            # uncovered[neighbors_arr[i]] gives boolean vector of covered status for candidate i
            # sum gives count
            scores = uncovered[neighbors_arr].sum(axis=1)
            
            if randomize:
                # Get max score
                max_score = scores.max()
                # Get indices with max score
                candidates = np.where(scores == max_score)[0]
                # Pick one randomly
                idx = np.random.choice(candidates)
            else:
                idx = np.argmax(scores)
                
            code.append(idx)
            # Update uncovered
            uncovered[neighbors_arr[idx]] = False
            
        return code

    def reduce_code(code):
        code_set = set(code)
        # Count coverage for each word
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
            
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of the list
            for c in list(code_set):
                # Check if all neighbors of c are covered by at least 2 codewords
                # If min cover count > 1, we can remove c
                min_cov = cover_count[neighbors_arr[c]].min()
                if min_cov > 1:
                    cover_count[neighbors_arr[c]] -= 1
                    code_set.remove(c)
                    changed = True
        return list(code_set)
    
    def simulated_annealing(code, max_time=30.0):
        """Simulated annealing to refine the code size.
        
        Moves: remove a codeword (if coverage stays valid) or swap a codeword.
        Objective: minimize code size while maintaining full coverage.
        """
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=int)
        for c in code_set:
            cover_count[neighbors_arr[c]] += 1
        
        # Ensure initial coverage is valid
        if cover_count.min() < 1:
            return list(code_set)
        
        current_size = len(code_set)
        best_size = current_size
        best_set = set(code_set)
        
        start_sa = time.time()
        deadline_sa = start_sa + max_time
        
        # Temperature schedule
        T0 = 1.0
        T_min = 0.01
        cooling = 0.95
        
        T = T0
        rng = np.random.default_rng(42)
        
        while T > T_min and time.time() < deadline_sa:
            # Try a move: either remove a codeword or swap
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try to remove a random codeword
                if len(code_set) == 0:
                    T *= cooling
                    continue
                c = rng.choice(list(code_set))
                nbs = neighbors_arr[c]
                # Check if removing c would leave any word uncovered
                would_uncover = (cover_count[nbs] == 1).any()
                if would_uncover:
                    # Can't remove without breaking coverage
                    # Accept with probability based on how many become uncovered
                    uncovered_count = (cover_count[nbs] == 1).sum()
                    delta = 1  # size increases conceptually if we had to add back
                    # Actually removing breaks coverage, so we reject unless we fix it
                    # For simplicity, reject
                    T *= cooling
                    continue
                else:
                    # Safe to remove
                    cover_count[nbs] -= 1
                    code_set.remove(c)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_set = set(code_set)
                    T *= cooling
            elif move_type < 0.7:
                # Try to swap: remove one codeword, add another
                if len(code_set) < 2:
                    T *= cooling
                    continue
                c_remove = rng.choice(list(code_set))
                # Pick a random candidate to add
                c_add = rng.integers(0, total_words)
                if c_add in code_set:
                    T *= cooling
                    continue
                
                nbs_remove = neighbors_arr[c_remove]
                nbs_add = neighbors_arr[c_add]
                
                # Check if removing c_remove and adding c_add maintains coverage
                # Words that would be uncovered: those in nbs_remove with cover_count == 1 
                # and NOT in nbs_add
                temp_cover = cover_count.copy()
                temp_cover[nbs_remove] -= 1
                temp_cover[nbs_add] += 1
                
                if temp_cover.min() < 1:
                    # This swap breaks coverage
                    uncovered = (temp_cover < 1).sum()
                    # Accept with probability exp(-uncovered / T)
                    if rng.random() < np.exp(-uncovered / max(T, 1e-10)):
                        cover_count = temp_cover
                        code_set.remove(c_remove)
                        code_set.add(c_add)
                        # Size stays the same
                    T *= cooling
                else:
                    # Valid swap, size unchanged
                    cover_count = temp_cover
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    T *= cooling
            else:
                # Try to add a codeword that helps most (then we might be able to remove others)
                # This is less common
                T *= cooling
        
        # Final reduction
        final_set = set(best_set)
        final_cover = np.zeros(total_words, dtype=int)
        for c in final_set:
            final_cover[neighbors_arr[c]] += 1
        
        changed = True
        while changed:
            changed = False
            for c in list(final_set):
                if final_cover[neighbors_arr[c]].min() > 1:
                    final_cover[neighbors_arr[c]] -= 1
                    final_set.remove(c)
                    changed = True
        
        return list(final_set)
    
    best_code = None
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Run multiple rounds of greedy + reduction + simulated annealing
    # Mix of deterministic and randomized
    num_rounds = 0
    while time.time() < deadline:
        num_rounds += 1
        if num_rounds % 5 == 1:
            cand = greedy_cover(randomize=False)
        else:
            cand = greedy_cover(randomize=True)
            
        cand = reduce_code(cand)
        
        # Apply simulated annealing with time limit
        sa_time = min(5.0, max(1.0, deadline - time.time() - 5.0))
        if sa_time > 1.0:
            cand_sa = simulated_annealing(cand, max_time=sa_time)
            if len(cand_sa) < len(cand):
                cand = cand_sa
        
        if best_code is None or len(cand) < len(best_code):
            best_code = cand
            # If we have a good result, do a final polish with more SA time
            if len(cand) <= 190 and time.time() < deadline - 10:
                polish_time = min(15.0, deadline - time.time() - 5.0)
                if polish_time > 5.0:
                    cand_polish = simulated_annealing(cand, max_time=polish_time)
                    if len(cand_polish) < len(best_code):
                        best_code = cand_polish
    
    if best_code is None:
        best_code = greedy_cover(randomize=False)
        best_code = reduce_code(best_code)
        
    # Convert indices back to words
    return [all_words[i] for i in best_code]
# EVOLVE-BLOCK-END
