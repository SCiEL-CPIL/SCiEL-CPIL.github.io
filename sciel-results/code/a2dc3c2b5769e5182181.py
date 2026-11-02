import numpy as np
import itertools
import random

def construct(q, n, radius):
    # This problem is specifically for q=3, n=8, radius=1.
    # But the interface is general. We handle the specific case efficiently
    # and fall back to a general greedy if parameters differ.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: Direct sum for general case if n >= 4
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Brute force or simple construction for small n
            # For very small n, just return all words or a simple code
            return [list(x) for x in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, radius=1
    # 1. Generate initial code (Direct Sum)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to numpy arrays for faster manipulation if needed, 
    # but lists of ints are fine for the logic.
    
    # Total words in space
    total_words = 3**n
    word_to_idx = {}
    for i in range(total_words):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % 3)
            temp //= 3
        word_to_idx[tuple(w)] = i
        
    # Function to get neighbors of a word (including the word itself)
    def get_neighbors(word):
        idx = word_to_idx[tuple(word)]
        neighbors = [idx]
        for i in range(n):
            for delta in [-1, 1]:
                new_val = word[i] + delta
                if 0 <= new_val < 3:
                    new_word = word.copy()
                    new_word[i] = new_val
                    neighbors.append(word_to_idx[tuple(new_word)])
        return neighbors

    # Precompute neighbor lists for initial codewords
    # Note: We need to check if removing a codeword leaves any word uncovered.
    # We maintain a count array of size total_words.
    
    counts = np.zeros(total_words, dtype=np.int32)
    # Map codeword index to its neighbor list
    cw_neighbors = []
    for cw in initial_codewords:
        cw_neighbors.append(get_neighbors(cw))
        
    # Initialize counts
    for neigh_list in cw_neighbors:
        counts[neigh_list] += 1
        
    # Check if initial code covers everything (it should)
    if np.any(counts == 0):
        # If not, we have a bug in the direct sum logic or neighbor generation
        # But direct sum of perfect code covers everything.
        pass

    active = [True] * len(initial_codewords)
    
    # Iterative pruning
    # To make it faster, we can try removing codewords that have high overlap.
    # A codeword can be removed if all its neighbors have count > 1.
    
    # We'll loop until no more can be removed.
    # To avoid O(N^2) full scans every time, we can just scan.
    # N=729, N^2=530k. Each check is 17 ops. 530k * 17 is ~9M ops. 
    # In Python, this might take a few seconds.
    
    changed = True
    while changed:
        changed = False
        for i in range(len(initial_codewords)):
            if not active[i]:
                continue
            neigh = cw_neighbors[i]
            # Check if any neighbor has count 1
            # If min count in neighbors is > 1, we can remove
            if np.min(counts[neigh]) > 1:
                # Remove it
                active[i] = False
                counts[neigh] -= 1
                changed = True
                # We can break and restart, or continue. 
                # Continuing is fine, but might remove "bad" ones first?
                # Actually, removing any valid one is fine for greedy.
                
    # Extract final codewords
    final_codewords = [initial_codewords[i] for i in range(len(initial_codewords)) if active[i]]
    
    # Optional: Try a few random restarts or different orders?
    # The order of removal can affect the final size.
    # Let's try a few different removal orders (random shuffles of indices)
    # and keep the best result.
    
    best_size = len(final_codewords)
    best_code = final_codewords
    
    # Try 5 random orders to see if we can get smaller
    for _ in range(5):
        active = [True] * len(initial_codewords)
        counts = np.zeros(total_words, dtype=np.int32)
        for neigh_list in cw_neighbors:
            counts[neigh_list] += 1
            
        indices = list(range(len(initial_codewords)))
        random.shuffle(indices)
        
        changed = True
        while changed:
            changed = False
            for i in indices:
                if not active[i]:
                    continue
                neigh = cw_neighbors[i]
                if np.min(counts[neigh]) > 1:
                    active[i] = False
                    counts[neigh] -= 1
                    changed = True
                    
        current_code = [initial_codewords[i] for i in range(len(initial_codewords)) if active[i]]
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code
            
    return best_code
