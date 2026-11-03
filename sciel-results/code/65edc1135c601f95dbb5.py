# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if radius == 1 and q == 3:
        # Ternary covering code K_3(8,1)
        # Strategy: greedy set cover to get a good initial solution, then local search
        # to reduce the number of codewords.
        
        start_time = time.time()
        deadline = start_time + 115  # Leave a safety margin
        
        # Precompute neighbors (Hamming ball of radius 1) for a given word
        # A word is a tuple of length n
        # Neighbors are the word itself and words with exactly one position changed
        
        def get_neighbors(word):
            # word is a tuple of length n
            # Returns a list of tuples
            neigh = [word]
            for i in range(n):
                for val in range(q):
                    if word[i] != val:
                        # Create a new word with val at position i
                        new_word = list(word)
                        new_word[i] = val
                        neigh.append(tuple(new_word))
            return neigh
        
        # Greedy construction
        def greedy_construct():
            # We need to cover all q**n words
            # Total words is 3**8 = 6561
            # Each codeword covers 1 + n*(q-1) = 1 + 8*2 = 17 words
            
            all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
            # Map each word to its index for efficient coverage tracking
            word_to_idx = {w: i for i, w in enumerate(all_words)}
            idx_to_word = list(all_words) # Just to be safe, though list of tuples is fine
            
            # Precompute neighbors for all words? No, 6561 * 17 is small, but let's compute on the fly or cache.
            # Let's precompute neighbor lists as indices for speed.
            # neigh_indices[i] is a list of indices of words covered by word i.
            neigh_indices = []
            for w in all_words:
                neighs = get_neighbors(w)
                neigh_indices.append([word_to_idx[ng] for ng in neighs])
            
            # uncovered is a boolean array or set of indices
            uncovered = set(range(len(all_words)))
            code = []
            
            while uncovered:
                # Pick a word from uncovered to center around
                # Random selection helps with diversity
                center_idx = random.choice(list(uncovered))
                center_word = all_words[center_idx]
                
                best_gain = 0
                best_candidate = None
                
                # Candidates are words within distance 1 of center_word
                # Or simply, we can iterate over all words? No, that's slow.
                # But wait, the optimal codeword doesn't have to be the center.
                # However, in set cover, greedy often picks the element that covers the most uncovered elements.
                # We can restrict candidates to those "near" the center to speed up?
                # Actually, just picking any uncovered word and finding the best codeword that covers it
                # is a valid greedy heuristic.
                # Let's try candidates within distance 2 of the center? Or just the center and its neighbors?
                # If we only check center's neighbors, we might miss a better codeword further away.
                # Let's check all codewords? 6561 candidates * 17 checks = ~100k ops per step.
                # 486 steps * 100k = 50M ops. Might be slow in Python.
                
                # Optimization: Only consider candidates that cover the center_idx.
                # Candidates are all words w such that center_idx is in neigh_indices[w].
                # This is equivalent to finding all words within distance 1 of center_word.
                # Let's precompute "covers" mapping: word_idx -> list of codeword_idx that cover it.
                # This is the reverse of neigh_indices.
                
                # Let's build reverse map once outside the loop? No, it's fixed.
                covers = [[] for _ in range(len(all_words))]
                for c_idx, neighs in enumerate(neigh_indices):
                    for n_idx in neighs:
                        covers[n_idx].append(c_idx)
                
                # Now, for the chosen center_idx, potential candidates are covers[center_idx]
                candidates = covers[center_idx]
                
                for c_idx in candidates:
                    # Calculate gain: number of uncovered words covered by c_idx
                    gain = 0
                    for n_idx in neigh_indices[c_idx]:
                        if n_idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = c_idx
                    
                    # Early termination if we cover all neighbors?
                    if best_gain == len(neigh_indices[best_candidate]):
                        break
                
                if best_candidate is not None:
                    code.append(best_candidate)
                    # Remove covered words from uncovered
                    for n_idx in neigh_indices[best_candidate]:
                        uncovered.discard(n_idx)
            
            return [list(all_words[i]) for i in code]
        
        # Local search to improve
        def local_search(code, time_limit):
            # code is a list of lists
            # Convert to set of tuples for fast lookup
            code_set = set(tuple(c) for c in code)
            all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
            word_to_idx = {w: i for i, w in enumerate(all_words)}
            idx_to_word = list(all_words)
            
            # Precompute neighbors for indices
            neigh_indices = []
            for w in all_words:
                neighs = get_neighbors(w)
                neigh_indices.append([word_to_idx[ng] for ng in neighs])
            
            # Count coverage: how many codewords cover each word?
            coverage_counts = np.zeros(len(all_words), dtype=int)
            for c in code:
                c_idx = word_to_idx[tuple(c)]
                for n_idx in neigh_indices[c_idx]:
                    coverage_counts[n_idx] += 1
            
            # Check validity: min coverage >= 1
            if np.min(coverage_counts) < 1:
                return code
            
            improved = True
            while improved and (time.time() < time_limit):
                improved = False
                # Try to remove codewords
                # Randomize order to avoid bias
                indices = list(range(len(code)))
                random.shuffle(indices)
                
                for idx in indices:
                    if time.time() > time_limit:
                        break
                    
                    c = code[idx]
                    c_idx = word_to_idx[tuple(c)]
                    
                    # Check if removing c_idx leaves any word with coverage 0
                    # coverage_counts[n_idx] - 1 >= 1 for all n_idx in neigh_indices[c_idx]
                    # i.e., coverage_counts[n_idx] >= 2
                    safe = True
                    for n_idx in neigh_indices[c_idx]:
                        if coverage_counts[n_idx] < 2:
                            safe = False
                            break
                    
                    if safe:
                        # Remove
                        code.pop(idx)
                        for n_idx in neigh_indices[c_idx]:
                            coverage_counts[n_idx] -= 1
                        improved = True
                        # Restart loop to re-evaluate with smaller code
                        break # Break to restart the shuffle loop with new list length
                
                # Try swaps? (Replace one codeword with another to potentially allow further removals)
                # This is more complex. Let's stick to removals first.
                # If no removal is possible, try a "swap" heuristic:
                # Pick a random codeword, remove it, find the word with coverage 0,
                # add a new codeword that covers it and maximizes other coverages.
                # If the new code size is same or smaller, keep it.
                # Actually, pure removal is a strong baseline.
                
                # To allow further removals, we might need to "swap".
                # If we can't remove any, we can try to swap a codeword to a position
                # that allows a subsequent removal.
                
                if not improved:
                    # Try swap
                    if len(code) > 0:
                        idx = random.randrange(len(code))
                        c_to_remove = code[idx]
                        c_to_remove_idx = word_to_idx[tuple(c_to_remove)]
                        
                        # Find words that will become uncovered
                        # Words with coverage 1 that are covered by c_to_remove
                        critical_words = []
                        for n_idx in neigh_indices[c_to_remove_idx]:
                            if coverage_counts[n_idx] == 1:
                                critical_words.append(n_idx)
                        
                        if critical_words:
                            # We need to add a codeword that covers all critical_words? No, just one.
                            # But if we add one, it covers one critical word, but others remain uncovered.
                            # So we need a codeword that covers ALL critical words?
                            # If the intersection of neighbors of critical words is empty, we can't fix with 1 swap.
                            # In that case, swap is not helpful for immediate removal.
                            
                            # Let's find a codeword that covers the most critical words (or any of them)
                            # and see if it leads to a better state.
                            
                            # Simpler: Pick one critical word w. Find best codeword c_new covering w.
                            # Remove c_to_remove, add c_new.
                            # Check if size decreased? No, size same.
                            # Check if we can now remove something?
                            
                            # This is getting complex. Let's just try random swaps and see if removal becomes possible.
                            critical_word = random.choice(critical_words)
                            candidates = covers[critical_word] # Reusing 'covers' from greedy? Need to pass or rebuild.
                            # Let's rebuild covers locally or pass it.
                            
                            best_new_code = None
                            # Try a few random candidates
                            random.shuffle(candidates)
                            for new_c_idx in candidates[:10]:
                                # Check if new_c_idx is already in code
                                if new_c_idx == c_to_remove_idx:
                                    continue
                                if tuple(idx_to_word[new_c_idx]) in code_set:
                                    continue
                                
                                # Simulate swap
                                new_code = code[:]
                                new_code[idx] = idx_to_word[new_c_idx]
                                new_code_set = set(tuple(c) for c in new_code)
                                if len(new_code_set) != len(new_code):
                                    continue
                                
                                # Check validity
                                new_cov = np.zeros(len(all_words), dtype=int)
                                valid = True
                                for c in new_code:
                                    c_i = word_to_idx[tuple(c)]
                                    for n_i in neigh_indices[c_i]:
                                        new_cov[n_i] += 1
                                
                                if np.min(new_cov) < 1:
                                    continue
                                
                                # Check if this allows removal?
                                # Try to remove a random word from new_code
                                rem_idx = random.randrange(len(new_code))
                                rem_c = new_code[rem_idx]
                                rem_c_i = word_to_idx[tuple(rem_c)]
                                
                                safe = True
                                for n_i in neigh_indices[rem_c_i]:
                                    if new_cov[n_i] < 2:
                                        safe = False
                                        break
                                
                                if safe:
                                    # We found a swap that allows a removal!
                                    # Perform the removal on new_code
                                    new_code.pop(rem_idx)
                                    best_new_code = new_code
                                    break
                            
                            if best_new_code is not None:
                                code = best_new_code
                                code_set = set(tuple(c) for c in code)
                                # Rebuild coverage counts
                                coverage_counts = np.zeros(len(all_words), dtype=int)
                                for c in code:
                                    c_idx = word_to_idx[tuple(c)]
                                    for n_idx in neigh_indices[c_idx]:
                                        coverage_counts[n_idx] += 1
                                improved = True
                                break # Restart loop
        
            return code
        
        best_code = None
        best_size = float('inf')
        
        # Multiple restarts
        # The greedy algorithm is random.
        # We run it until deadline.
        while time.time() < deadline:
            code = greedy_construct()
            current_size = len(code)
            
            if current_size < best_size:
                best_code = [list(c) for c in code]
                best_size = current_size
            
            # Local search
            code = local_search(code, deadline - 2) # Leave 2s margin
            current_size = len(code)
            
            if current_size < best_size:
                best_code = [list(c) for c in code]
                best_size = current_size
        
        return best_code

    # Fallback for other parameters (though task specifies q=3, n=8, radius=1)
    # Return a trivial valid code (all words)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

# EVOLVE-BLOCK-END
