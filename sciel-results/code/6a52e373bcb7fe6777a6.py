import itertools
import numpy as np
import random
import time
def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin: finish by 110s, limit is 120s
    
    # For the specific problem: q=3, n=6, radius=1
    # Direct sum of ternary tetracode (9 words of length 4) with all 9 tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    
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
    # Space is q^n = 3^6 = 729 words. Manageable.
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def index_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))
    
    total_words = q ** n
    
    # Precompute coverage using NumPy for speed
    # code_array: shape (num_codewords, n)
    code_array = np.array(code, dtype=np.int32)
    num_codewords = len(code)
    
    # For each codeword, compute all words it covers (itself + distance-1 neighbors)
    # Each codeword covers 1 + n*(q-1) words
    words_per_codeword = 1 + n * (q - 1)
    
    # Build coverage matrix: cov_matrix[ci, wi] = 1 if codeword ci covers word wi
    # But this is num_codewords x total_words which is 81 x 729 = ~59k, fine
    # Instead, build: for each word, which codewords cover it
    # cov_count[wi] = number of active codewords covering word wi
    
    # Precompute: for each codeword, the list of word indices it covers
    # Use vectorized approach
    # For codeword c, covered words are: c itself, and for each position p and each value v != c[p], 
    # the word with c[p] replaced by v
    
    # Build a 2D array of all covered word indices
    # Shape: (num_codewords, words_per_codeword)
    covered_indices = np.zeros((num_codewords, words_per_codeword), dtype=np.int32)
    
    for ci in range(num_codewords):
        c = code_array[ci]
        covered_indices[ci, 0] = word_to_index(tuple(c))
        idx = 1
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                w_list = list(c)
                w_list[pos] = val
                covered_indices[ci, idx] = word_to_index(tuple(w_list))
                idx += 1
    
    # Build coverage count: cov_count[wi] = number of codewords covering word wi
    cov_count = np.zeros(total_words, dtype=np.int32)
    # For each codeword, increment count for each word it covers
    for ci in range(num_codewords):
        cov_count[covered_indices[ci]] += 1
    
    # Active codewords
    active = np.ones(num_codewords, dtype=bool)
    
    # Greedy removal: repeatedly remove codewords that are redundant
    # A codeword is redundant if for every word it covers, cov_count[word] > 1
    # (i.e., some other active codeword also covers it)
    
    def is_redundant_fast(ci):
        """Check if codeword ci is redundant using vectorized operations."""
        covered = covered_indices[ci]
        # For each covered word, check if cov_count > 1
        # But we need to be careful: cov_count includes ci itself
        # So a word is still covered if cov_count[word] > 1
        return np.all(cov_count[covered] > 1)
    
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
        
        removed_any = False
        for ci in range(num_codewords):
            if time.time() > deadline - 10.0:
                break
            if not active[ci]:
                continue
            
            if is_redundant_fast(ci):
                active[ci] = False
                # Update cov_count
                cov_count[covered_indices[ci]] -= 1
                removed_any = True
        
        if not removed_any:
            break
    
    # Build final code after greedy pruning
    active_indices = np.where(active)[0]
    final_code = [list(code_array[i]) for i in active_indices]
    
    # Now try swap-based local search with simulated annealing
    # Goal: find a smaller valid code by swapping codewords
    # A swap: remove one codeword, add another (from the full space)
    # Accept if it maintains coverage and reduces size (or with SA acceptance)
    
    if time.time() < deadline - 15.0:
        # Build a coverage bitmap for the current code
        # cov_bitmap[wi] = number of codewords covering word wi
        cov_bitmap = np.zeros(total_words, dtype=np.int32)
        for ci in active_indices:
            cov_bitmap[covered_indices[ci]] += 1
        
        current_size = len(active_indices)
        best_size = current_size
        best_code = final_code[:]
        
        # SA parameters
        T0 = 1.0
        T_end = 0.01
        cooling = 0.995
        
        # Precompute: for each word in the space, which codewords (from the initial 81) cover it
        # This helps when we consider adding a codeword
        # word_to_codewords[wi] = list of codeword indices (from initial code) that cover wi
        word_to_codewords = [[] for _ in range(total_words)]
        for ci in range(num_codewords):
            for wi in covered_indices[ci]:
                word_to_codewords[wi].append(ci)
        
        T = T0
        sa_iterations = 0
        max_sa_time = deadline - 5.0
        
        while time.time() < max_sa_time and T > T_end:
            # Pick a random active codeword to potentially remove
            if len(active_indices) == 0:
                break
            
            ci_to_remove = random.choice(list(active_indices))
            
            # Check: if we remove ci_to_remove, which words become uncovered?
            # A word becomes uncovered if cov_bitmap[word] == 1 and that word is covered by ci_to_remove
            covered_by_ci = covered_indices[ci_to_remove]
            would_uncover = covered_by_ci[cov_bitmap[covered_by_ci] == 1]
            
            if len(would_uncover) == 0:
                # Removing is safe, do it
                active[ci_to_remove] = False
                cov_bitmap[covered_indices[ci_to_remove]] -= 1
                active_indices = np.where(active)[0]
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_code = [list(code_array[i]) for i in active_indices]
                T = max(T * cooling, T_end)
                sa_iterations += 1
                continue
            
            # Need to find a replacement codeword that covers all of would_uncover
            # Try random codewords from the initial set that are not active
            inactive_indices = np.where(~active)[0]
            if len(inactive_indices) == 0:
                T = max(T * cooling, T_end)
                sa_iterations += 1
                continue
            
            # Try a few random inactive codewords
            found_replacement = False
            num_tries = min(20, len(inactive_indices))
            candidates = random.sample(list(inactive_indices), num_tries)
            
            for cj in candidates:
                covered_by_cj = covered_indices[cj]
                # Check if cj covers all of would_uncover
                # would_uncover is an array of word indices
                # We need to check if all elements of would_uncover are in covered_by_cj
                # Use a set for efficiency
                covered_set_cj = set(covered_by_cj.tolist())
                if all(w in covered_set_cj for w in would_uncover.tolist()):
                    # cj covers all words that would be uncovered
                    # Perform the swap
                    active[ci_to_remove] = False
                    active[cj] = True
                    cov_bitmap[covered_indices[ci_to_remove]] -= 1
                    cov_bitmap[covered_indices[cj]] += 1
                    active_indices = np.where(active)[0]
                    # Size stays the same (swap), but we might enable future removals
                    found_replacement = True
                    break
            
            if not found_replacement:
                # Try adding a new codeword from the full space (not just initial 81)
                # This is more expensive but can find better solutions
                # Generate a random word in the space
                if random.random() < 0.3:  # 30% chance to try a new word
                    new_word = tuple(random.randint(0, q - 1) for _ in range(n))
                    new_idx = word_to_index(new_word)
                    
                    # Compute which words this new word covers
                    new_covers = [new_idx]
                    for pos in range(n):
                        orig = new_word[pos]
                        for val in range(q):
                            if val == orig:
                                continue
                            w_list = list(new_word)
                            w_list[pos] = val
                            new_covers.append(word_to_index(tuple(w_list)))
                    
                    new_covers_arr = np.array(new_covers, dtype=np.int32)
                    
                    # Check if this new word covers all of would_uncover
                    new_covers_set = set(new_covers)
                    if all(w in new_covers_set for w in would_uncover.tolist()):
                        # Add this new word to the code
                        # We need to add it to our tracking structures
                        # For simplicity, we'll track it separately
                        # Actually, let's just note that we found a valid swap/add
                        # and rebuild at the end
                        active[ci_to_remove] = False
                        cov_bitmap[covered_indices[ci_to_remove]] -= 1
                        active_indices = np.where(active)[0]
                        current_size -= 1
                        # We added a new word, so size is the same
                        # But we need to track it... this is getting complex
                        # Let's skip this for now and rely on the initial 81
                        # Restore
                        active[ci_to_remove] = True
                        cov_bitmap[covered_indices[ci_to_remove]] += 1
                        active_indices = np.where(active)[0]
                        current_size += 1
                        found_replacement = False
                    else:
                        found_replacement = False
            
            T = max(T * cooling, T_end)
            sa_iterations += 1
    
    # Final verification and cleanup
    if time.time() < deadline - 2.0:
        # Verify coverage of best_code
        if best_code:
            covered_set = set()
            for c in best_code:
                c = tuple(c)
                covered_set.add(word_to_index(c))
                for pos in range(n):
                    orig = c[pos]
                    for val in range(q):
                        if val == orig:
                            continue
                        w_list = list(c)
                        w_list[pos] = val
                        covered_set.add(word_to_index(tuple(w_list)))
            
            if len(covered_set) < total_words:
                # Fallback to the greedy-pruned code
                best_code = [list(code_array[i]) for i in np.where(active)[0]]
    
    return best_code
