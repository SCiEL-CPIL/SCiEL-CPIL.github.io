import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # The specific case is q=3, n=7, radius=1
    # But we should handle general inputs if possible, though the task is specific.
    # The seed code assumes n>=4.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to seed logic for other parameters if needed, 
        # but the task is specific to 3,7,1.
        # We'll just return the seed logic for non-matching cases to be safe.
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(tail) for tail in itertools.product(range(3), repeat=n)]

    # Total words in space
    total_words = q ** n
    
    # Helper to generate neighbors (within distance radius)
    # For radius 1, neighbors are the word itself and words with 1 position changed.
    def get_neighborhood(word):
        neighbors = [word]
        for i in range(n):
            orig = word[i]
            for val in range(q):
                if val != orig:
                    new_word = word.copy()
                    new_word[i] = val
                    neighbors.append(new_word)
        return neighbors
    
    # Precompute neighborhoods for all words? 
    # 2187 words * 22 neighbors = ~50k pairs. 
    # Storing as sets of tuples might be memory heavy but manageable.
    # Or just compute on the fly.
    
    # Initial Code: Direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert code to set for fast lookup
    code_set = set(code)
    
    # Calculate coverage: For each word in space, which codewords cover it?
    # Or better: For each codeword, which words does it cover?
    # Let's maintain a count of how many codewords cover each word in the space.
    # count[word] = number of codewords c such that dist(word, c) <= radius
    
    # Initialize counts
    # We can do this by iterating through code and adding to counts for all neighbors.
    # But we need to map word to index for array storage.
    # Index = word[0] * 3^6 + word[1] * 3^5 + ...
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)
        
    # Precompute neighbors for all indices?
    # Let's precompute the list of neighbor indices for each index.
    # This speeds up the counting and removal phases significantly.
    neighbor_indices = [None] * total_words
    for i in range(total_words):
        w = idx_to_word(i)
        neigh = get_neighborhood(w)
        neighbor_indices[i] = [word_to_idx(nw) for nw in neigh]
        
    # Initialize coverage counts
    coverage_counts = [0] * total_words
    
    for c in code:
        ci = word_to_idx(c)
        for ni in neighbor_indices[ci]:
            coverage_counts[ni] += 1
            
    # Check if all covered
    # min(coverage_counts) should be > 0
    
    # Phase 1: Remove redundant codewords
    # A codeword c is redundant if for every word x in N(c), coverage_counts[x] > 1.
    # i.e., no word is covered uniquely by c.
    
    current_code = set(code)
    
    improved = True
    while improved and (time.time() < deadline - 10):
        improved = False
        # We need to check each codeword.
        # To avoid modifying the set while iterating, we collect candidates for removal.
        to_remove = []
        
        for c in current_code:
            ci = word_to_idx(c)
            is_redundant = True
            for ni in neighbor_indices[ci]:
                if coverage_counts[ni] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                to_remove.append(c)
                
        if to_remove:
            # Remove them
            for c in to_remove:
                current_code.remove(c)
                ci = word_to_idx(c)
                for ni in neighbor_indices[ci]:
                    coverage_counts[ni] -= 1
            improved = True
            
    # Phase 2: Try to replace pairs with single words (optional, might be too slow)
    # Given time constraints, Phase 1 is usually sufficient to get close to optimal for small n.
    # For K_3(7,1), the 243-word code often reduces to ~190-200 words with simple redundancy removal.
    # To beat 186, we might need more sophisticated moves.
    
    # Let's try a simple local search: 
    # Try to replace a codeword c with a new word w that allows removing another codeword?
    # Or just try to find a better initial configuration via randomization?
    
    # Actually, the "redundancy removal" is deterministic given the initial set.
    # To explore different local minima, we can shuffle the order of removal or the initial set.
    # But the initial set is fixed by the construction.
    
    # Alternative: Start with random valid code? No, hard to generate.
    
    # Let's try to add a "swap" phase:
    # For each c in current_code, try to replace it with a word w not in current_code
    # such that the new set is valid. This doesn't reduce size.
    
    # To reduce size, we need to find c1, c2 such that N(c1) U N(c2) is covered by N(w) U (N(others)).
    # This is equivalent to: (N(c1) \ N(others)) U (N(c2) \ N(others)) subset N(w).
    # Let U1 = N(c1) \ N(others) and U2 = N(c2) \ N(others).
    # If U1 U U2 is empty, we could remove both? No, we already did that.
    # If there exists w such that U1 U U2 subset N(w), we can replace c1, c2 with w.
    
    # This is expensive. Let's see if Phase 1 is enough.
    # If not, we can try random restarts of the reduction process with different tie-breaking?
    # The reduction process here doesn't have tie-breaking, it removes all redundants at once.
    
    # Let's try a different strategy:
    # 1. Start with 243 words.
    # 2. Randomly remove a word.
    # 3. Check if still valid. If not, put it back.
    # 4. Repeat.
    # This is a "random descent" on the size.
    
    # Let's implement this if Phase 1 leaves us with > 186 words.
    
    if len(current_code) > 186 and time.time() < deadline - 5:
        # Randomized local search
        # We need a fast validity check.
        # If we remove c, we need to check if all words in N(c) are still covered by others.
        # i.e., for all x in N(c), coverage_counts[x] > 1.
        # This is the same check as redundancy!
        # So random removal is just picking a random redundant word and removing it?
        # No, if it's redundant, we already removed it in Phase 1.
        # So after Phase 1, NO word is redundant.
        # Thus, we cannot remove any single word without breaking validity.
        
        # We must use moves that change the set structure, e.g., swapping.
        # Move: Replace c1, c2 with w.
        # Condition: For all x in N(c1) U N(c2), x is covered by w or some c in current_code \ {c1, c2}.
        
        # Let's try a few random swaps.
        # Pick random c1, c2 from current_code.
        # Calculate U12 = (N(c1) U N(c2)) \ N(current_code \ {c1, c2}).
        # We need to find w such that U12 subset N(w).
        # Note: w can be c1 or c2 (no change) or new.
        # If we find such w, replace {c1, c2} with {w}.
        
        # To find w, we can check all words? 2187 checks.
        # For each candidate w, check if U12 subset N(w).
        # U12 size is small.
        
        # Let's try this for a limited number of iterations.
        
        current_code_list = list(current_code)
        found_swap = True
        while found_swap and time.time() < deadline - 2:
            found_swap = False
            # Shuffle to randomize
            random.shuffle(current_code_list)
            
            # Try pairs
            # To save time, try first few pairs
            for i in range(min(50, len(current_code_list))):
                for j in range(i+1, min(50, len(current_code_list))):
                    c1 = current_code_list[i]
                    c2 = current_code_list[j]
                    
                    # Calculate U12
                    # Words covered by c1 or c2 but not by any other codeword.
                    # We can compute this using coverage_counts.
                    # If we remove c1 and c2, coverage_counts[x] decreases by 1 if x in N(c1) and 1 if x in N(c2).
                    # If the new count is 0, x is uncovered.
                    
                    # Let's simulate removal.
                    # We need to know if there's a w that covers all newly uncovered words.
                    
                    # Let U be the set of words that would become uncovered.
                    U = set()
                    
                    # Words in N(c1)
                    for x in neighbor_indices[word_to_idx(c1)]:
                        # How many codewords cover x?
                        # coverage_counts[x]
                        # If we remove c1 and c2:
                        # count_new = coverage_counts[x]
                        # if x in N(c1): count_new -= 1
                        # if x in N(c2): count_new -= 1
                        # if count_new == 0: x is uncovered.
                        
                        # We need to be careful if x is in both N(c1) and N(c2).
                        pass
                    
                    # This simulation is O(size of N(c1) U N(c2)) which is ~40.
                    # But we need to do it for many pairs.
                    
                    # Let's optimize:
                    # Precompute for each c, the list of words it covers.
                    # We have neighbor_indices.
                    
                    # Let's just do the simulation.
                    # Create a temporary count delta?
                    # No, just iterate.
                    
                    # Get set of words covered by c1
                    n_c1 = set(neighbor_indices[word_to_idx(c1)])
                    # Get set of words covered by c2
                    n_c2 = set(neighbor_indices[word_to_idx(c2)])
                    
                    union_n = n_c1.union(n_c2)
                    
                    uncovered_if_removed = []
                    for x in union_n:
                        # Count how many codewords cover x, excluding c1 and c2
                        # We can use coverage_counts[x]
                        # If x in n_c1, coverage_counts[x] includes c1.
                        # If x in n_c2, coverage_counts[x] includes c2.
                        
                        cnt = coverage_counts[x]
                        if x in n_c1:
                            cnt -= 1
                        if x in n_c2:
                            cnt -= 1
                            
                        if cnt == 0:
                            uncovered_if_removed.append(x)
                            
                    if not uncovered_if_removed:
                        # We can remove c1 and c2?
                        # No, we need to replace them with something?
                        # If we remove them, we lose coverage of words that are ONLY covered by c1 or c2.
                        # If uncovered_if_removed is empty, it means every word in N(c1) U N(c2) is covered by at least one other codeword.
                        # So we can just remove c1 and c2!
                        # This is a stronger move than swapping.
                        
                        # Remove c1 and c2
                        current_code.remove(c1)
                        current_code.remove(c2)
                        for x in n_c1:
                            coverage_counts[x] -= 1
                        for x in n_c2:
                            coverage_counts[x] -= 1
                        found_swap = True
                        break
                    
                    # If we have uncovered words, can we cover them with a single new word w?
                    if uncovered_if_removed:
                        # Try to find w such that uncovered_if_removed subset N(w)
                        # We can iterate through all possible w?
                        # Or just check if the intersection of N(w) for w in some set...
                        # Let's just try a few random w's or the center of the uncovered set?
                        
                        # For now, skip if not empty.
                        pass
                        
                if found_swap:
                    break
            
    return [list(c) for c in current_code]
