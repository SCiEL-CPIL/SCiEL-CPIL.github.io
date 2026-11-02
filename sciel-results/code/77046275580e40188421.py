"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin
    
    # Specific to the task: q=3, n=6, radius=1
    # But we should handle general q, n if possible, though the task is specific.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # However, the space size q**n must be manageable. 3**6=729 is small.
    # If q**n is too large, we fall back to the seed method.
    
    if q**n > 10000:
        # Fallback to seed logic for large spaces
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Generic fallback: random or simple construction
            # For now, just return a trivial cover if not the specific case
            # But the task is specifically K_3(6,1)
            pass

    # Optimized for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        N = 3**6
        # Generate all words
        all_words = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int8)
        
        # Precompute balls (indices of words within distance 1)
        # Distance 1 means same word or differ in exactly 1 position
        # We can compute this efficiently
        # For each word, its neighbors are obtained by changing one digit to the other 2 values
        
        # Let's create a list of indices for each ball
        # ball_indices[i] is a list of indices j such that dist(i, j) <= 1
        # We can use a lookup table or just compute on the fly for the matrix M
        
        # M is a boolean matrix (N, N) where M[i, j] is True if j is in ball(i)
        # To save memory and time, we can store the list of neighbors for each i
        # But for vectorized greedy, a dense matrix (729x729) is small (530k bytes)
        
        M = np.zeros((N, N), dtype=bool)
        
        # Helper to convert index to word and vice versa?
        # all_words already exists.
        # For each word i, find neighbors.
        # Since N is small, we can do this with loops or vectorized ops on chunks.
        
        # Vectorized approach to build M:
        # For each dimension d, and for each change delta in {-1, 1} (mod 3 logic is tricky)
        # Actually, simpler:
        # For each word w in all_words:
        #   neighbors = [w]
        #   for pos in range(6):
        #      for new_val in range(3):
        #         if new_val != w[pos]:
        #            nw = w.copy(); nw[pos] = new_val; neighbors.append(nw)
        #   indices = [np.where((all_words == nb).all(axis=1))[0][0] for nb in neighbors]
        #   M[i, indices] = True
        
        # This loop is 729 * (1 + 6*2) = 729 * 13 lookups.
        # To speed up lookup, create a dict mapping tuple(word) -> index
        word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
        
        for i, w in enumerate(all_words):
            M[i, i] = True
            for pos in range(6):
                for new_val in range(3):
                    if new_val != w[pos]:
                        nw = w.copy()
                        nw[pos] = new_val
                        idx = word_to_idx[tuple(nw)]
                        M[i, idx] = True
        
        best_code = None
        best_size = N # Upper bound
        
        # Greedy Set Cover
        # We will run multiple restarts with different random tie-breaking or initializations?
        # Standard greedy is deterministic if ties are broken by index.
        # We can try different tie-breaking strategies or random shuffles of candidates.
        
        for restart in range(50): # 50 restarts is safe within time
            if time.time() > deadline:
                break
                
            uncovered = np.ones(N, dtype=bool)
            code = []
            
            # To add randomness, we can shuffle the order in which we consider ties
            # Or just run standard greedy.
            # Let's try standard greedy first.
            
            while np.any(uncovered):
                # counts[i] = number of uncovered words covered by candidate i
                # M @ uncovered gives the count for each row i
                counts = M @ uncovered
                # Pick max
                # If ties, pick the one with smallest index (deterministic)
                # Or random among max to explore different solutions
                max_val = np.max(counts)
                candidates = np.where(counts == max_val)[0]
                
                if len(candidates) > 1 and random.random() < 0.5:
                    # Random tie break
                    i = candidates[random.randint(0, len(candidates)-1)]
                else:
                    i = candidates[0]
                
                code.append(i)
                # Update uncovered
                uncovered[M[i, :]] = False
                
            # Local Search: Remove redundant codewords
            # Convert code to set for faster lookup? Or just list.
            # We need to check if removing c leaves all covered.
            # This is equivalent to: Is there any word w such that c is the ONLY codeword in its ball?
            # We can maintain a count of how many codewords cover each word.
            
            # Let's implement a more efficient local search
            # coverage_count[w] = number of codewords in code that cover w
            # If min(coverage_count) >= 1, code is valid.
            # If coverage_count[w] == 1 for some w, and c is the only one covering w, c cannot be removed.
            
            # Rebuild coverage counts for the current code
            coverage_counts = np.zeros(N, dtype=int)
            for c in code:
                coverage_counts[M[c, :]] += 1
            
            # Try to remove each codeword
            changed = True
            while changed:
                changed = False
                for idx, c in enumerate(code):
                    # Check if c is essential
                    # c is essential if there exists w such that coverage_counts[w] == 1 and M[c, w] is True
                    # i.e., if min(coverage_counts[M[c, :]]) == 1, then c is essential?
                    # No, if coverage_counts[w] == 1 for any w in ball(c), then c is the unique cover for w.
                    # So if min(coverage_counts[M[c, :]]) == 1, c cannot be removed.
                    
                    # We want to remove c if min(coverage_counts[M[c, :]]) > 1.
                    if np.min(coverage_counts[M[c, :]]) > 1:
                        # Safe to remove
                        coverage_counts[M[c, :]] -= 1
                        code.pop(idx)
                        changed = True
                        break # Restart loop as indices changed
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
                
            # Check time
            if time.time() > deadline:
                break

        if best_code is not None:
            return [list(all_words[i]) for i in best_code]
        
        # If no valid code found (should not happen), fall back
        return [list(w) for w in all_words]
        
    else:
        # Fallback for other parameters
        # For now, use the seed method if close, else random
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Generic greedy for small spaces
            N = q**n
            if N <= 10000:
                all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
                word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
                M = np.zeros((N, N), dtype=bool)
                for i, w in enumerate(all_words):
                    M[i, i] = True
                    # Generate neighbors within radius
                    # This is complex for general radius/q/n. 
                    # For now, just return all words if not optimized
                return [list(w) for w in all_words]
            else:
                return []

# EVOLVE-BLOCK-END
