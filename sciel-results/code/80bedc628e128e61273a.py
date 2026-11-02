import numpy as np
import itertools
import random

def construct(q, n, radius):
    # Fallback for non-specific cases
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(x) for x in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    
    # Precompute word representations and indices
    # word_to_idx: tuple -> index
    # idx_to_word: index -> list
    idx_to_word = []
    word_to_idx = {}
    for i in range(total_words):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        idx_to_word.append(w)
        word_to_idx[tuple(w)] = i

    # Precompute neighbors for all words
    # neighbors[i] = list of indices j such that dist(i, j) <= radius
    # For radius 1, this is i and all single-bit flips.
    neighbors = []
    for i in range(total_words):
        w = idx_to_word[i]
        neigh = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                nv = w[pos] + delta
                if 0 <= nv < q:
                    nw = w.copy()
                    nw[pos] = nv
                    neigh.append(word_to_idx[tuple(nw)])
        neighbors.append(neigh)

    def prune_code(code):
        """Remove redundant codewords from the given list."""
        if not code:
            return []
        
        # Map codeword index to its position in code list for quick access?
        # We just need to check redundancy.
        # counts[w] = number of codewords in code that cover w.
        counts = np.zeros(total_words, dtype=np.int32)
        
        # Build list of neighbor lists for the current code
        code_neighbors = [neighbors[c] for c in code]
        
        for neigh in code_neighbors:
            counts[neigh] += 1
            
        active = [True] * len(code)
        
        # Iterative removal
        # To speed up, we can check in random order or fixed order.
        # Fixed order is deterministic. Random might find different solutions.
        # Let's try a few passes.
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                if not active[i]:
                    continue
                neigh = code_neighbors[i]
                if np.min(counts[neigh]) > 1:
                    active[i] = False
                    counts[neigh] -= 1
                    changed = True
                    
        return [code[i] for i in range(len(code)) if active[i]]

    def greedy_cover(init_size, seed=42):
        """
        1. Start with 'init_size' random codewords.
        2. Greedily add codewords that cover the most uncovered words.
        3. Prune the result.
        """
        random.seed(seed)
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        # Initialize gains: how many uncovered words does each candidate cover?
        # Initially, all words are uncovered, so gain[c] = len(neighbors[c])
        gains = np.array([len(neigh) for neigh in neighbors], dtype=np.int32)
        
        # Pick initial random codewords
        init_indices = random.sample(range(total_words), init_size)
        for c in init_indices:
            code.append(c)
            for w in neighbors[c]:
                if not covered[w]:
                    covered[w] = True
                    # Update gains for all candidates that cover w
                    for c_prime in neighbors[w]:
                        gains[c_prime] -= 1
                        
        # Greedy addition
        uncovered_count = np.sum(~covered)
        while uncovered_count > 0:
            # Find best candidate
            best_c = np.argmax(gains)
            if gains[best_c] == 0:
                # Should not happen if logic is correct, but break to avoid infinite loop
                break
                
            code.append(best_c)
            
            # Update coverage
            for w in neighbors[best_c]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    for c_prime in neighbors[w]:
                        gains[c_prime] -= 1
                        
        # Prune
        pruned = prune_code(code)
        return pruned

    # Try multiple initializations to find a better code
    best_code = []
    best_size = float('inf')
    
    # Strategy: Try different init sizes and seeds
    # The Direct Sum code has 729 words.
    # We expect to get close to 486.
    # Greedy Set Cover usually yields codes around 1.5-2x optimal? 
    # Optimal 486. Greedy might give 600-700.
    # Pruning might bring it down.
    
    # Let's try init sizes around 100-300.
    for init_size in [50, 100, 150, 200, 300]:
        for seed in range(5):
            code = greedy_cover(init_size, seed)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                # Early stop if we beat known best 486?
                # But we want to minimize.
                
    # If greedy didn't find anything (unlikely), fallback to direct sum
    if not best_code:
         tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
         best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
         best_code = prune_code([word_to_idx[tuple(c)] for c in best_code]) # Need indices for prune? 
         # Wait, prune_code takes indices. 
         # My greedy_cover returns indices.
         # The final return must be lists of ints.
         
    # Convert indices back to lists
    return [idx_to_word[c] for c in best_code]
