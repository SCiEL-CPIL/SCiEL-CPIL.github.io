import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    
    # Precompute coverage masks using NumPy for speed
    # masks[i][j] is True if word j is within radius of word i
    masks = np.zeros((total_words, total_words), dtype=bool)
    
    # Optimization: If the space is too large, this might be slow/memory heavy.
    # For q=3, n=6, total_words=729. 729x729 is fine.
    if total_words > 5000:
        # Fallback to a simple greedy if space is too large for bitmask precomputation
        # This handles generic cases, though the task is specific to small n.
        return _simple_greedy(q, n, radius)
        
    for i, w1 in enumerate(all_words):
        for j, w2 in enumerate(all_words):
            dist = sum(1 for a, b in zip(w1, w2) if a != b)
            if dist <= radius:
                masks[i][j] = True

    def greedy_construct(seed):
        np.random.seed(seed)
        counts = np.zeros(total_words, dtype=int)
        codewords = []
        
        while True:
            if np.all(counts > 0):
                break
            
            uncovered_mask = (counts == 0)
            if not np.any(uncovered_mask):
                break
                
            # Compute scores: number of uncovered words covered by each candidate
            scores = np.sum(masks & uncovered_mask, axis=1)
            
            max_score = np.max(scores)
            if max_score == 0:
                break
            
            candidates = np.where(scores == max_score)[0]
            chosen_idx = candidates[np.random.randint(len(candidates))]
            
            codewords.append(chosen_idx)
            counts[masks[chosen_idx]] += 1
            
        return codewords

    def local_search(codewords, seed):
        np.random.seed(seed)
        # Initialize counts
        counts = np.zeros(total_words, dtype=int)
        for idx in codewords:
            counts[masks[idx]] += 1
            
        current_size = len(codewords)
        max_iter = 500
        iter_count = 0
        
        while current_size > 70 and iter_count < max_iter:
            iter_count += 1
            improved = False
            
            # 1. Try to remove codewords
            order = list(range(len(codewords)))
            np.random.shuffle(order)
            
            for pos in order:
                if pos >= len(codewords):
                    continue
                idx = codewords[pos]
                
                # Tentatively remove
                counts[masks[idx]] -= 1
                
                if np.all(counts > 0):
                    # Can remove
                    codewords.pop(pos)
                    current_size -= 1
                    improved = True
                else:
                    # Restore
                    counts[masks[idx]] += 1
            
            # 2. Try swaps if no reduction happened
            if not improved:
                for _ in range(20):
                    if len(codewords) < 2:
                        break
                    pos1 = np.random.randint(len(codewords))
                    idx1 = codewords[pos1]
                    
                    # Try swapping idx1 with a random new word
                    new_idx = np.random.randint(total_words)
                    if new_idx == idx1:
                        continue
                    
                    # Remove idx1, add new_idx
                    counts[masks[idx1]] -= 1
                    counts[masks[new_idx]] += 1
                    
                    if np.all(counts > 0):
                        # Valid swap
                        codewords[pos1] = new_idx
                        improved = True
                        break # Restart reduction phase
                    else:
                        # Revert
                        counts[masks[idx1]] += 1
                        counts[masks[new_idx]] -= 1
                        
        return codewords

    def _simple_greedy(q, n, radius):
        # Fallback for large spaces
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        codewords = []
        
        def get_coverage(word):
            covered = set()
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word = list(word)
                    new_word[pos] = val
                    covered.add(tuple(new_word))
            covered.add(tuple(word))
            return covered
            
        coverages = {w: get_coverage(w) for w in all_words}
        
        while uncovered:
            best_word = None
            best_count = -1
            for w in all_words:
                count = len(coverages[w] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = w
            if best_count == 0:
                break
            codewords.append(list(best_word))
            uncovered -= coverages[best_word]
        return codewords

    best_codewords = None
    best_size = float('inf')
    
    start_time = time.time()
    timeout = 110 # Leave margin for evaluation overhead
    
    for seed in range(20):
        cw_indices = greedy_construct(seed)
        cw_indices = local_search(cw_indices, seed)
        
        if len(cw_indices) < best_size:
            best_size = len(cw_indices)
            best_codewords = cw_indices
            
        if time.time() - start_time > timeout:
            break
            
    if best_codewords is None:
        return all_words
        
    return [all_words[i] for i in best_codewords]
