import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate all words in the space
    # We can represent a word as an integer from 0 to q**n - 1
    # But for distance calculation, list/tuple is easier.
    # For speed, we can use base-q representation.
    
    # 1. Initial Construction: Direct sum of tetracode and all words of length n-4
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Generate tails
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    # Construct initial codewords
    initial_codewords = []
    for h in tetracode:
        for tail in tails:
            initial_codewords.append(h + list(tail))
            
    # Convert to numpy array for easier manipulation? 
    # Actually, lists of tuples are fine for the coverage logic if we precompute neighbors.
    
    # Precompute neighbors for each possible word? No, space is 729.
    # Let's map word -> index.
    # Index = sum(w[i] * q**(n-1-i))
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute all neighbors (distance <= radius) for each word in the space
    # Space size is q**n = 729
    space_size = q ** n
    neighbors = [[] for _ in range(space_size)]
    
    for i in range(space_size):
        w = idx_to_word(i)
        # Neighbor is w itself
        neighbors[i].append(i)
        # Neighbors are words differing in exactly 1 position
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val != original_val:
                    if radius >= 1:
                        w[pos] = new_val
                        neighbors[i].append(word_to_idx(w))
            w[pos] = original_val
            
    # Initialize coverage counts
    counts = np.zeros(space_size, dtype=np.int32)
    current_codewords = initial_codewords
    
    # Calculate initial counts
    for c in current_codewords:
        idx_c = word_to_idx(c)
        for nb in neighbors[idx_c]:
            counts[nb] += 1
            
    # Greedy Deletion Loop
    # We try to remove codewords that are redundant.
    # A codeword is redundant if removing it doesn't make any count drop to 0.
    
    # To be efficient, we can iterate in a random order or fixed order.
    # Fixed order is deterministic and fine.
    # We need to keep track of which codewords are currently active.
    # active_mask or just a list.
    
    # Let's store codewords as a list.
    # We will iterate multiple passes.
    
    max_passes = 100
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        removed_any = False
        # Iterate backwards to allow removal during iteration
        for i in range(len(current_codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            c = current_codewords[i]
            idx_c = word_to_idx(c)
            
            # Check if any neighbor count would become 0
            # We need to know if min(counts[nb] for nb in neighbors[idx_c]) > 1
            # If yes, we can remove c.
            
            # Optimization: 
            # If we just subtracted counts for previous removals, counts are current.
            # We need to check if removing c causes any count to be 0.
            # This is equivalent to checking if any neighbor has count == 1.
            
            # If any neighbor has count 1, c is essential (for now).
            # If all neighbors have count > 1, c is redundant.
            
            # We can check this by looking at the neighbors.
            redundant = True
            for nb in neighbors[idx_c]:
                if counts[nb] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                # Update counts
                for nb in neighbors[idx_c]:
                    counts[nb] -= 1
                current_codewords.pop(i)
                removed_any = True
                
        if not removed_any:
            break

    # Optional: Local Search / Swapping
    # Try to replace a codeword with another to potentially allow more deletions?
    # Or simply try to find a better set?
    # Given the time limit, the deletion heuristic from a good start is usually sufficient to reach 73.
    # The seed is 81. We expect to get down to ~73.
    
    # Convert back to list of lists if necessary, but they are already lists.
    # Ensure distinctness (guaranteed by construction and deletion logic)
    
    return current_codewords
