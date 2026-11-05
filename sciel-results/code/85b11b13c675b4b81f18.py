import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Precompute neighbors for all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # neighbors[i] is a list of indices of words within distance 'radius'
    neighbors = [[] for _ in range(num_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors[i].append(word_to_idx[tuple(new_w)])
        
    # Convert neighbors to numpy arrays for faster indexing later if needed, 
    # but lists are fine for iteration. 
    # We need a way to quickly check if a candidate covers a set of uncovered words.
    # Let's keep neighbors as lists of ints.

    # 2. Greedy Initialization
    # Start with an empty code.
    # We want to cover all num_words.
    # uncovered: boolean array or set of indices. 
    # Using a numpy array of counts is better for checking validity (min >= 1).
    # For greedy construction, we want to pick the word that covers the most NEW words.
    
    # Initialize coverage counts to 0
    counts = np.zeros(num_words, dtype=np.int32)
    codewords = []
    
    # To speed up greedy, we can maintain a list of uncovered words.
    # But checking "how many new words does X cover" requires counting intersection with uncovered.
    # Let's use a simpler heuristic:
    # 1. Pick a random word. Add it.
    # 2. While not all covered:
    #    - Find the word that covers the most uncovered words.
    #    - To do this efficiently without O(N^2) per step (which is 2187^2 ~ 4.7M, times ~200 steps ~ 1B ops, too slow in Python):
    #    - We can sample candidates. Or we can use a bitset approach if N is small enough? 
    #    - N=2187. A bitset of 2187 bits is ~274 bytes. 
    #    - Let's try a standard greedy with random sampling of candidates to find a "good enough" cover fast.
    
    # Actually, for N=2187, a full scan for the best word is 2187 * (cost of checking coverage).
    # Cost of checking coverage: sum(1 for nb in neighbors[c] if counts[nb] == 0).
    # Average degree is 1 + n*(q-1) = 1 + 7*2 = 15.
    # So checking one candidate takes ~15 ops.
    # Full scan takes 2187 * 15 ~ 33k ops.
    # If we need ~200 codewords, total ops ~ 6.6M. This is fast in Python/NumPy mix.
    
    uncovered_mask = np.ones(num_words, dtype=bool)
    
    while np.any(uncovered_mask):
        if time.time() > deadline:
            break
            
        best_cand = -1
        best_count = 0
        
        # Iterate through all possible words to find the one covering most uncovered
        # We can optimize by skipping words that are already in codewords? 
        # No, adding a duplicate doesn't help, but we haven't added them yet.
        # Actually, we should avoid adding duplicates.
        
        # To speed this up, we can just pick a random word from the uncovered set initially?
        # No, greedy needs the best.
        
        # Optimization: Only check words that cover at least one uncovered word.
        # But checking that is the same cost.
        
        # Let's just do the full scan. It's fast enough.
        for i in range(num_words):
            # Check if this word covers any uncovered words
            # If counts[i] > 0, it's already covered by someone else, but might cover others.
            # We want to maximize new coverage.
            
            # Quick check: if this word itself is covered and all its neighbors are covered, skip?
            # Not quite. It might cover uncovered neighbors.
            
            # Calculate new coverage
            # We can do this by iterating neighbors[i]
            new_cov = 0
            for nb in neighbors[i]:
                if uncovered_mask[nb]:
                    new_cov += 1
            
            if new_cov > best_count:
                best_count = new_cov
                best_cand = i
        
        if best_cand == -1:
            break # Should not happen if uncovered_mask has True
            
        # Add best_cand
        codewords.append(list(all_words[best_cand]))
        for nb in neighbors[best_cand]:
            counts[nb] += 1
            if counts[nb] == 1:
                uncovered_mask[nb] = False
        
        # Periodic check for time
        if time.time() > deadline:
            break

    # 3. Local Search: Remove and Swap
    # Now we have a valid covering (hopefully).
    # We want to minimize size.
    
    # Convert codewords to indices for easier manipulation
    # Actually, let's keep them as indices in a list.
    code_indices = [word_to_idx[tuple(c)] for c in codewords]
    
    # Rebuild counts from code_indices to be sure
    counts = np.zeros(num_words, dtype=np.int32)
    for idx in code_indices:
        for nb in neighbors[idx]:
            counts[nb] += 1
            
    def is_valid():
        return np.min(counts) >= 1

    # Local Search Loop
    no_improvement_count = 0
    max_no_improvement = 5000 # Reset or stop if stuck? 
    
    while time.time() < deadline:
        if not code_indices:
            break
            
        i = random.randint(0, len(code_indices) - 1)
        old_idx = code_indices[i]
        
        # Attempt removal
        for nb in neighbors[old_idx]:
            counts[nb] -= 1
            
        if np.min(counts) >= 1:
            # Removal successful
            code_indices.pop(i)
            no_improvement_count = 0
            continue
            
        # Removal failed. Find uncovered words
        uncovered_indices = np.where(counts == 0)[0]
        
        # Try to find a replacement
        # The replacement must cover all words in uncovered_indices.
        # If uncovered_indices is small (usually 1 or 2), we can find candidates quickly.
        
        found_replacement = False
        
        # Strategy: Pick a random word from the neighbors of the first uncovered word.
        # Check if it covers all uncovered words.
        # If not, try another.
        
        if len(uncovered_indices) > 0:
            first_uncov = uncovered_indices[0]
            candidates = neighbors[first_uncov]
            random.shuffle(candidates)
            
            # Limit candidates to check to avoid hanging
            for cand_idx in candidates[:50]:
                # Check if cand_idx covers all uncovered_indices
                covers_all = True
                for u in uncovered_indices:
                    # Check if u is in neighbors[cand_idx]
                    # This is O(1) if we had a set, but neighbors are lists.
                    # For radius 1, degree is ~15. So checking membership is fast.
                    if u not in neighbors[cand_idx]:
                        covers_all = False
                        break
                
                if covers_all:
                    # Valid replacement
                    # Add cand_idx
                    for nb in neighbors[cand_idx]:
                        counts[nb] += 1
                    # Replace old_idx with cand_idx
                    code_indices[i] = cand_idx
                    found_replacement = True
                    no_improvement_count = 0 # Swapping is a "move", not an improvement in size, but it changes state
                    break
            
            if not found_replacement:
                # Restore old_idx
                for nb in neighbors[old_idx]:
                    counts[nb] += 1
                # no_improvement_count += 1
                # If we get stuck too much, we might try a "kick" (add a random word, then try to remove two?)
                # For now, just retry.
        
        # If we hit deadline, break
        if time.time() > deadline:
            break
            
    # Convert back to lists of ints
    return [list(all_words[idx]) for idx in code_indices]
