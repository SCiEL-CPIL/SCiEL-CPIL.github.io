import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Special case for the specific task: q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        # 1. Generate the seed code (486 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # 2. Precompute neighbors for all words in {0,1,2}^8
        # Map word to index: w[0]*3^7 + w[1]*3^6 + ... + w[7]
        # Precompute neighbors for all 6561 words
        total_words = 3 ** n
        neighbors = []
        for i in range(total_words):
            w = np.array([0]*n)
            # Convert i to base 3
            temp = i
            for j in range(n-1, -1, -1):
                w[j] = temp % 3
                temp //= 3
            neigh = [i]
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    # Convert back to index
                    idx = 0
                    for j in range(n):
                        idx = idx * 3 + w[j]
                    neigh.append(idx)
                w[pos] = orig
            neighbors.append(neigh)
            
        # Function to convert list of lists to set of indices
        def to_indices(code):
            return set(tuple(w) for w in code) # This is slow, use index mapping
        
        # Better: work with indices directly
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx

        def idx_to_word(idx):
            w = [0]*n
            temp = idx
            for j in range(n-1, -1, -1):
                w[j] = temp % q
                temp //= q
            return w

        # Convert seed code to indices
        seed_indices = [word_to_idx(w) for w in seed_code]
        
        # Greedy Removal Function
        def greedy_remove(indices, max_time=None):
            if max_time is None:
                max_time = time.time() + 5
            code_set = set(indices)
            cover_count = np.zeros(total_words, dtype=int)
            
            for c in code_set:
                for w in neighbors[c]:
                    cover_count[w] += 1
                    
            improved = True
            while improved:
                if time.time() > max_time:
                    break
                improved = False
                # Shuffle to avoid order bias
                list_code = list(code_set)
                random.shuffle(list_code)
                for c in list_code:
                    # Check if removable
                    removable = True
                    for w in neighbors[c]:
                        if cover_count[w] == 1:
                            removable = False
                            break
                    if removable:
                        for w in neighbors[c]:
                            cover_count[w] -= 1
                        code_set.remove(c)
                        improved = True
            
            return [idx_to_word(c) for c in code_set]

        # Greedy Construction Function
        def greedy_construct(seed_idx=None):
            # seed_idx is for randomizing the start
            uncovered = np.ones(total_words, dtype=bool)
            code_indices = []
            
            # To speed up, we can just pick the first uncovered word
            # But to vary, we can pick random uncovered words
            # We'll use a list of uncovered indices and pop from it? 
            # Efficiently finding uncovered:
            # Maintain a list of uncovered indices.
            uncovered_list = list(range(total_words))
            # This is O(N) to init.
            
            # To allow randomization of the "pick w" step, we can shuffle the initial list 
            # or pick a random index from the uncovered set.
            # Using a set is O(1) for lookup but O(N) for iteration.
            # Let's use a list and a boolean array for "is_uncovered".
            
            # Actually, the "pick w" step in greedy covering is often done by picking 
            # the word that is "most central" or just any uncovered word.
            # Picking any uncovered word works.
            # To get diversity, we shuffle the order in which we consider uncovered words.
            
            # Let's try a simple greedy:
            # 1. Start with all words uncovered.
            # 2. While uncovered:
            #    Pick a random uncovered word w.
            #    Find best c in neighbors[w] that covers most uncovered words.
            #    Add c to code.
            #    Mark neighbors[c] as covered.
            
            # To make it fast, we need to efficiently find uncovered neighbors of a candidate c.
            # We can just iterate through neighbors[c] and check `uncovered[u]`.
            
            random.shuffle(uncovered_list) # Initial shuffle for diversity? 
            # Actually, if we always pick the FIRST uncovered word, the order of the list matters.
            # Let's maintain a list of uncovered words.
            
            # Optimization: 
            # Instead of scanning the whole list for an uncovered word, 
            # we can just keep a pointer or use a set.
            # Given 6561 words, scanning is fine.
            
            while True:
                if time.time() > deadline:
                    break
                
                # Find an uncovered word
                w_idx = -1
                for u in uncovered_list:
                    if uncovered[u]:
                        w_idx = u
                        break
                if w_idx == -1:
                    break
                
                # Candidates are neighbors[w_idx]
                best_c = -1
                max_covered = -1
                
                for c in neighbors[w_idx]:
                    count = 0
                    for u in neighbors[c]:
                        if uncovered[u]:
                            count += 1
                    if count > max_covered:
                        max_covered = count
                        best_c = c
                
                if best_c == -1:
                    break
                    
                code_indices.append(best_c)
                
                # Update uncovered
                for u in neighbors[best_c]:
                    uncovered[u] = False
            
            return [idx_to_word(c) for c in code_indices]

        best_code = seed_code
        best_size = len(seed_code)
        
        # Run removal on seed
        removed_seed = greedy_remove(seed_indices)
        if len(removed_seed) < best_size:
            best_code = removed_seed
            best_size = len(removed_seed)
            
        # Run multiple greedy constructions
        # We can vary the randomness by shuffling the "uncovered_list" logic 
        # or just relying on the random selection of w if we modify greedy_construct.
        # The current greedy_construct picks the FIRST uncovered word in the list.
        # To randomize, we should shuffle the list of all words initially, 
        # but since we remove from the list logically (via bool array), 
        # the "first uncovered" depends on the initial order of `uncovered_list`.
        # So, shuffling `uncovered_list` before the loop provides different runs.
        
        iterations = 0
        while time.time() < deadline - 5:
            iterations += 1
            # Create a shuffled list for this run
            # We don't want to re-init the bool array every time if we can avoid it, 
            # but greedy_construct creates a new one.
            # To make greedy_construct use a specific order, we pass the shuffled list?
            # Let's modify greedy_construct to accept a seed or shuffle internally.
            
            # Simpler: Just call greedy_construct which shuffles internally?
            # I'll modify greedy_construct to shuffle the initial `uncovered_list` 
            # based on a random seed or just use `random.shuffle` inside.
            
            # Let's refine greedy_construct to be self-contained with randomization.
            def greedy_construct_rand():
                uncovered = np.ones(total_words, dtype=bool)
                code_indices = []
                # Create a list of all indices, shuffle it
                order = list(range(total_words))
                random.shuffle(order)
                
                # We need to pick an uncovered word. 
                # If we iterate through `order`, we find the first uncovered one.
                # This effectively picks a random uncovered word (biased by order).
                
                while True:
                    if time.time() > deadline:
                        break
                    w_idx = -1
                    for u in order:
                        if uncovered[u]:
                            w_idx = u
                            break
                    if w_idx == -1:
                        break
                    
                    best_c = -1
                    max_covered = -1
                    for c in neighbors[w_idx]:
                        count = 0
                        for u in neighbors[c]:
                            if uncovered[u]:
                                count += 1
                        if count > max_covered:
                            max_covered = count
                            best_c = c
                    
                    if best_c == -1:
                        break
                        
                    code_indices.append(best_c)
                    for u in neighbors[best_c]:
                        uncovered[u] = False
                
                return [idx_to_word(c) for c in code_indices]

            code_g = greedy_construct_rand()
            if len(code_g) < best_size:
                best_code = code_g
                best_size = len(code_g)
            
            # Also try removing from this newly constructed code
            if time.time() < deadline - 5:
                indices_g = [word_to_idx(w) for w in code_g]
                removed_g = greedy_remove(indices_g)
                if len(removed_g) < best_size:
                    best_code = removed_g
                    best_size = len(removed_g)

        return best_code

    else:
        # Fallback for other parameters
        # Simple greedy construction
        total_words = q ** n
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx

        def idx_to_word(idx):
            w = [0]*n
            temp = idx
            for j in range(n-1, -1, -1):
                w[j] = temp % q
                temp //= q
            return w

        # Precompute neighbors
        neighbors = []
        for i in range(total_words):
            w = idx_to_word(i)
            neigh = [i]
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    idx = word_to_idx(w)
                    neigh.append(idx)
                w[pos] = orig
            neighbors.append(neigh)

        uncovered = np.ones(total_words, dtype=bool)
        code_indices = []
        order = list(range(total_words))
        random.shuffle(order)
        
        while True:
            if time.time() > deadline:
                break
            w_idx = -1
            for u in order:
                if uncovered[u]:
                    w_idx = u
                    break
            if w_idx == -1:
                break
            
            best_c = -1
            max_covered = -1
            for c in neighbors[w_idx]:
                count = 0
                for u in neighbors[c]:
                    if uncovered[u]:
                        count += 1
                if count > max_covered:
                    max_covered = count
                    best_c = c
            
            if best_c == -1:
                break
                
            code_indices.append(best_c)
            for u in neighbors[best_c]:
                uncovered[u] = False
                
        return [idx_to_word(c) for c in code_indices]
