import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    # Handle specific case q=3, n=8, radius=1
    if q == 3 and n == 8 and radius == 1:
        total_words = 3**n
        # Precompute neighbors for all words in the space
        # Word index i -> tuple of digits
        # We need a fast way to get neighbors of index i
        
        # Convert index to word
        def idx_to_word(idx):
            w = [0]*n
            for j in range(n):
                w[j] = idx % 3
                idx //= 3
            return w

        # Precompute neighbor indices for all 6561 words
        # neighbors[i] is a list of indices
        all_neighbors = []
        for i in range(total_words):
            w = idx_to_word(i)
            neigh = [i]
            for j in range(n):
                for delta in [-1, 1]:
                    val = w[j] + delta
                    if 0 <= val < 3:
                        new_w = w.copy()
                        new_w[j] = val
                        # Calculate new index
                        new_idx = 0
                        base = 1
                        for k in range(n):
                            new_idx += new_w[k] * base
                            base *= 3
                        neigh.append(new_idx)
            all_neighbors.append(neigh)

        # Greedy Construction
        covered = np.zeros(total_words, dtype=np.int8)
        code_indices = []
        
        # To speed up greedy, we can just pick the word with max uncovered neighbors.
        # Naive greedy:
        # For each step:
        #   best_gain = 0
        #   best_idx = -1
        #   for i in range(total_words):
        #       if covered[i] == 1: continue # Optimization: only consider if it adds something? 
        #       # Actually, we should consider all, but gain is 0 if all neighbors covered.
        #       gain = 0
        #       for nb in all_neighbors[i]:
        #           if covered[nb] == 0:
        #               gain += 1
        #       if gain > best_gain:
        #           best_gain = gain
        #           best_idx = i
        #   if best_gain == 0: break
        #   code_indices.append(best_idx)
        #   for nb in all_neighbors[best_idx]:
        #       covered[nb] = 1

        # This naive loop is O(Steps * TotalWords * NeighSize) = 500 * 6561 * 17 ~ 55M ops.
        # In Python, this might be too slow (10-20s).
        # We can optimize by maintaining a list of "uncovered" words and checking candidates?
        # Or just use the Direct Sum start and prune, which was fast in seed.
        
        # Let's stick to the Seed's approach but improve the Pruning.
        # The Seed's pruning was O(N^2) with N=729.
        # 729^2 * 17 ~ 9M ops. This is very fast (<1s).
        # The problem with Seed is it starts at 729.
        # If we can start at a better code, we win.
        
        # Let's try to construct a code of size ~500 using a "Random Greedy" that is faster.
        # Instead of checking all 6561 words, we can maintain a set of "candidates".
        # But implementing a fast max-gain selection is complex.
        
        # Alternative: Use the Kerdock construction if I can code it.
        # Since I can't guarantee the Kerdock code implementation is correct from memory, 
        # I will use a **Multi-Start Local Search** starting from the Direct Sum code.
        
        # 1. Generate Direct Sum code (729 words).
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Convert to indices
        word_to_idx = {}
        for i in range(total_words):
            word_to_idx[tuple(idx_to_word(i))] = i
            
        init_indices = [word_to_idx[tuple(cw)] for cw in initial_codewords]
        
        # Prune function
        def prune(indices):
            counts = np.zeros(total_words, dtype=np.int32)
            for idx in indices:
                counts[all_neighbors[idx]] += 1
            active = [True] * len(indices)
            changed = True
            while changed:
                changed = False
                for i in range(len(indices)):
                    if not active[i]:
                        continue
                    if np.min(counts[all_neighbors[indices[i]]]) > 1:
                        active[i] = False
                        counts[all_neighbors[indices[i]]] -= 1
                        changed = True
            return [indices[i] for i in range(len(indices)) if active[i]]
        
        # Initial Prune
        current_code = prune(init_indices)
        best_code = current_code
        best_size = len(current_code)
        
        # Local Search: Try to swap or remove/replace to reduce size further?
        # If we are at 486, we can't reduce.
        # If we are at 500, we might be able to get to 486.
        
        # Strategy: Randomly remove a codeword. If it breaks coverage, 
        # find a replacement that fixes it and maybe allows another removal?
        # Simpler: Randomly select a codeword to remove. 
        # If coverage is lost, try to add a random word that covers the lost words.
        # This is effectively a random walk in the space of covering codes.
        
        start_time = time.time()
        timeout = 110 # Leave 10s buffer
        
        # We will do a simple random restart of the pruning order first.
        # The seed did 5 random shuffles. Let's do 50.
        for _ in range(50):
            if time.time() - start_time > timeout:
                break
            shuffled = init_indices.copy()
            random.shuffle(shuffled)
            pruned = prune(shuffled)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned
                if best_size == 486:
                    break # Found optimal known bound
        
        # If we didn't reach 486, try a simple local search on the best code found.
        # Try to remove pairs of codewords and replace with one?
        # This is complex.
        
        # Convert back to words
        return [idx_to_word(i) for i in best_code]

    else:
        # Fallback
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(x) for x in itertools.product(range(q), repeat=n)]
