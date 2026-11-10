import numpy as np
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Precompute index <-> word conversion
    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    def word_to_index(word):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + word[i]
        return idx

    # Precompute neighbors
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    neighbors = [[] for _ in range(total_words)]
    
    # For radius 1, we can generate neighbors efficiently
    # For general radius, we might need a BFS or combinatorial generation, but task is radius 1.
    # We assume radius=1 for the specific optimization path, but handle general case by brute force if needed.
    # Given the contract, q=3, n=7, radius=1.
    
    if radius == 1:
        for i in range(total_words):
            word = index_to_word(i)
            neighbors[i].append(i)
            for pos in range(n):
                orig = word[pos]
                for delta in range(1, q):
                    new_word = word[:]
                    new_word[pos] = (orig + delta) % q
                    new_idx = word_to_index(new_word)
                    neighbors[i].append(new_idx)
    else:
        # Fallback for other radii (slower)
        for i in range(total_words):
            w1 = index_to_word(i)
            for j in range(total_words):
                w2 = index_to_word(j)
                dist = sum(1 for a, b in zip(w1, w2) if a != b)
                if dist <= radius:
                    neighbors[i].append(j)

    # Convert neighbors to numpy arrays for faster access if needed, 
    # but lists are often faster for small sizes in Python loops due to overhead of numpy indexing.
    # Let's stick to lists.

    def run_greedy(seed_val):
        random.seed(seed_val)
        ref_counts = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)
        codewords = []
        
        # gain[i] = number of uncovered words in B(i)
        gain = np.full(total_words, len(neighbors[0]), dtype=np.int32)
        
        while True:
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            max_indices = np.where(gain == max_gain)[0]
            k = random.choice(max_indices)
            
            in_code[k] = True
            codewords.append(k)
            
            for u in neighbors[k]:
                if ref_counts[u] == 0:
                    # u becomes covered
                    # Decrease gain for all v such that u in B(v)
                    # i.e., v in neighbors[u]
                    for v in neighbors[u]:
                        gain[v] -= 1
                ref_counts[u] += 1
        
        return codewords, in_code, ref_counts

    def reduce_set(codewords, in_code, ref_counts):
        changed = True
        while changed:
            changed = False
            to_remove = []
            for idx in codewords:
                if in_code[idx]:
                    # Check if redundant
                    redundant = True
                    for u in neighbors[idx]:
                        if ref_counts[u] <= 1:
                            redundant = False
                            break
                    if redundant:
                        to_remove.append(idx)
            
            for idx in to_remove:
                in_code[idx] = False
                codewords.remove(idx)
                for u in neighbors[idx]:
                    ref_counts[u] -= 1
                changed = True

    def local_search(codewords, in_code, ref_counts, max_iters=2000, time_limit=10.0):
        start_time = time.time()
        # We will try to find swaps that allow removal.
        # Move: Remove c, Add c' (if c' covers all words that c uniquely covers).
        # Then reduce.
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # 1. Try simple removal
            removed = False
            for idx in codewords[:]:
                if in_code[idx]:
                    redundant = True
                    for u in neighbors[idx]:
                        if ref_counts[u] <= 1:
                            redundant = False
                            break
                    if redundant:
                        in_code[idx] = False
                        codewords.remove(idx)
                        for u in neighbors[idx]:
                            ref_counts[u] -= 1
                        removed = True
                        break # Restart loop
            
            if removed:
                continue
            
            # 2. Try Swap
            # Pick a random codeword c to remove
            if not codewords:
                break
            c = random.choice(codewords)
            
            # Find words uniquely covered by c
            # unique_words = {u in B(c) | ref_counts[u] == 1}
            unique_words = []
            for u in neighbors[c]:
                if ref_counts[u] == 1:
                    unique_words.append(u)
            
            if not unique_words:
                # Should have been removed in step 1, but just in case
                in_code[c] = False
                codewords.remove(c)
                for u in neighbors[c]:
                    ref_counts[u] -= 1
                continue
            
            # We need a c' not in code that covers ALL unique_words
            # Optimization: Instead of checking all c', we can check candidates
            # that cover at least one unique word.
            # Let's just iterate all c' not in code.
            # To speed up, we can precompute a "cover map" or just iterate.
            # N=2187, |C|~186. Iterating 2000 candidates is fine.
            
            found_swap = False
            # Shuffle candidates? No, just iterate.
            # We can break early if we find one.
            
            # To avoid bias, we can randomize the order of c' search occasionally?
            # For now, deterministic scan is fine, or random sampling.
            # Let's try random sampling of c' to diversify.
            
            for _ in range(50): # Try 50 random candidates for c'
                c_prime = random.randint(0, total_words - 1)
                if in_code[c_prime]:
                    continue
                
                # Check if c' covers all unique_words
                covers_all = True
                # Optimization: If c' doesn't cover the first unique word, skip
                # But we need to check all.
                # We can precompute a bitmask for unique_words?
                # unique_mask = sum(1 << u for u in unique_words)
                # ball_mask[c_prime] & unique_mask == unique_mask
                
                # Let's use sets for speed in this small loop?
                # No, list membership is O(N). 
                # Let's create a set of unique_words for O(1) lookup?
                # unique_set = set(unique_words)
                # for u in neighbors[c_prime]:
                #    if u in unique_set: unique_set.remove(u)
                # if unique_set: continue
                
                # This is O(|B(c')|) which is ~19. Very fast.
                
                remaining = set(unique_words)
                for u in neighbors[c_prime]:
                    if u in remaining:
                        remaining.discard(u)
                if remaining:
                    continue
                
                # c' covers all unique words.
                # Perform swap
                # Remove c
                in_code[c] = False
                codewords.remove(c)
                for u in neighbors[c]:
                    ref_counts[u] -= 1
                
                # Add c'
                in_code[c_prime] = True
                codewords.append(c_prime)
                for u in neighbors[c_prime]:
                    ref_counts[u] += 1
                
                found_swap = True
                break
            
            if not found_swap:
                # Try a few more specific candidates? 
                # Or just give up for this c.
                pass

    best_codewords = None
    best_size = float('inf')
    
    num_restarts = 5 # Keep low to allow time for local search
    time_start = time.time()
    
    for r in range(num_restarts):
        if time.time() - time_start > 100:
            break
            
        cw, ic, cc = run_greedy(seed_offset=r * 1000 + 42)
        
        reduce_set(cw, ic, cc)
        
        # Local Search with time limit per restart
        local_search(cw, ic, cc, max_iters=1000, time_limit=15.0)
        
        # Final reduction
        reduce_set(cw, ic, cc)
        
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw[:]
            
    if best_codewords is None:
        # Fallback
        cw, ic, cc = run_greedy(0)
        reduce_set(cw, ic, cc)
        best_codewords = cw

    result = [index_to_word(idx) for idx in best_codewords]
    return result
