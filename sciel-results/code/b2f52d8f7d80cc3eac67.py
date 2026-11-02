"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct(q, n, radius):
    # For the specific case of q=3, n=6, radius=1, we use a greedy set cover algorithm.
    # The total number of words is 3^6 = 729, which is small enough for efficient search.
    
    # If parameters differ from the known hard case, fall back to a simple construction
    # to ensure validity, though the task specifies this specific instance.
    if q != 3 or n != 6 or radius != 1:
        # Fallback: Direct sum of tetracode if n>=4, else brute force small cases
        # This ensures the function doesn't crash on other inputs if tested, 
        # though the contract implies the specific 6-match problem.
        if n < 4:
            # Brute force for small n
            all_words = list(itertools.product(range(q), repeat=n))
            # Greedy set cover for small space
            uncovered = set(all_words)
            code = []
            while uncovered:
                best_word = None
                best_count = -1
                for w in all_words:
                    # Count how many uncovered words are within distance radius
                    count = 0
                    for u in uncovered:
                        dist = sum(1 for a, b in zip(w, u) if a != b)
                        if dist <= radius:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_word = w
                code.append(list(best_word))
                # Remove covered words
                to_remove = set()
                for u in uncovered:
                    dist = sum(1 for a, b in zip(best_word, u) if a != b)
                    if dist <= radius:
                        to_remove.add(u)
                uncovered -= to_remove
            return code
        else:
            # Use the tetracode construction as a fallback for larger n
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1
    # 1. Generate all 729 words and assign them integer indices 0..728
    # 2. Precompute the neighborhood (words within distance 1) for each word
    # 3. Greedy Set Cover: repeatedly pick the word that covers the most uncovered words
    
    # Map from word tuple to index
    # Index is simply the base-3 number value
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
        return word

    total_words = q ** n
    indices = list(range(total_words))
    
    # Precompute neighborhoods for all words
    # neighborhoods[i] is a list of indices of words within distance <= radius of word i
    neighborhoods = []
    for i in indices:
        w = idx_to_word(i)
        neigh = [i] # Include self
        # Generate all words at distance 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                neigh.append(word_to_idx(new_w))
        neighborhoods.append(neigh)

    # Greedy Set Cover
    uncovered = set(indices)
    code = []
    
    # To speed up, we can track which words are still uncovered
    # and for each candidate, count how many of its neighbors are in uncovered.
    # Since the space is small (729), we can just iterate.
    
    while uncovered:
        best_idx = -1
        best_count = -1
        
        for i in indices:
            # Count how many neighbors of i are in uncovered
            count = 0
            for nb in neighborhoods[i]:
                if nb in uncovered:
                    count += 1
            
            if count > best_count:
                best_count = count
                best_idx = i
                
        # Add best_idx to code
        code.append(idx_to_word(best_idx))
        
        # Remove all neighbors of best_idx from uncovered
        for nb in neighborhoods[best_idx]:
            if nb in uncovered:
                uncovered.remove(nb)
                
        if not uncovered:
            break

    return code

# EVOLVE-BLOCK-END
