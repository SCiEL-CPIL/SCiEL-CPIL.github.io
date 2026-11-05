import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Construct a covering code for q^n with given radius.
    Optimized for q=3, n=7, radius=1.
    """
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task implies 3,7,1
        # Simple greedy for small sizes
        total = q ** n
        neighbors_list = []
        for w in range(total):
            # Convert to base q
            coords = []
            temp = w
            for _ in range(n):
                coords.append(temp % q)
                temp //= q
            neighbors = [w]
            for i in range(n):
                for d in range(-1, 2):
                    if d == 0:
                        continue
                    new_coord = (coords[i] + d) % q
                    # Reconstruct word
                    val = w
                    # Subtract old contribution
                    val -= coords[i] * (q ** i)
                    # Add new contribution
                    val += new_coord * (q ** i)
                    neighbors.append(val)
            neighbors_list.append(list(set(neighbors)))
        
        # Greedy
        cover_count = np.zeros(total, dtype=np.int32)
        chosen = []
        all_covered = False
        while not all_covered:
            uncovered = (cover_count == 0)
            # Calculate gain for each candidate
            gains = np.array([np.sum(uncovered[neighbors_list[c]]) for c in range(total)])
            best_c = np.argmax(gains)
            if gains[best_c] == 0:
                break
            chosen.append(best_c)
            cover_count[neighbors_list[best_c]] += 1
            if np.all(cover_count > 0):
                all_covered = True
            if len(chosen) > total: # Safety
                break
            
        # Local Search Removal
        # ... (Similar logic to below but generic is complex, stick to specific 3,7,1 optimization as per task hint)
        return chosen

    # Specific optimization for q=3, n=7, radius=1
    deadline = time.time() + 110.0 # 10s buffer for 120s limit
    
    # Precompute neighbors for all 2187 words
    # Each word has 1 + 7*2 = 15 neighbors
    total_words = 2187
    neighbors = np.zeros((total_words, 15), dtype=np.int32)
    
    # Helper to convert index to coords and back
    # Base 3 representation
    powers = [3**i for i in range(7)]
    
    for w in range(total_words):
        neighbors[w, 0] = w
        # Extract coords
        temp = w
        coords = [0]*7
        for i in range(7):
            coords[i] = temp % 3
            temp //= 3
        
        idx = 1
        for i in range(7):
            # Change coordinate i to +1
            new_coord = (coords[i] + 1) % 3
            # Delta = (new_coord - coords[i]) * 3^i
            delta = (new_coord - coords[i]) * powers[i]
            neighbors[w, idx] = w + delta
            idx += 1
            # Change coordinate i to -1
            new_coord = (coords[i] - 1) % 3
            delta = (new_coord - coords[i]) * powers[i]
            neighbors[w, idx] = w + delta
            idx += 1

    # Greedy Construction
    cover_count = np.zeros(total_words, dtype=np.int32)
    chosen = []
    
    while time.time() < deadline:
        uncovered = (cover_count == 0)
        if not np.any(uncovered):
            break
            
        # Vectorized gain calculation
        # uncovered[neighbors] creates a (2187, 15) boolean array
        # Sum along axis 1 gives count of uncovered neighbors for each candidate
        gains = uncovered[neighbors].sum(axis=1)
        
        best_c = np.argmax(gains)
        if gains[best_c] == 0:
            # Should not happen if not all covered, but safety
            break
            
        chosen.append(int(best_c))
        cover_count[neighbors[best_c]] += 1
        
    # Local Search: Try to remove codewords
    # A codeword c is removable if min(cover_count[neighbors[c]]) > 1
    # We iterate through chosen codewords in reverse order (most recently added are often less critical or we just try all)
    # Actually, iterating in random order or reverse is good.
    
    # To allow more search, we can do multiple passes of removal
    removal_pass = 0
    while time.time() < deadline:
        removed_any = False
        # Shuffle to avoid bias towards first elements
        # But shuffling every pass is expensive. Let's just iterate in reverse order first.
        # If we want to be thorough, we can randomly sample which ones to try.
        
        # Iterate in reverse order of addition
        for i in range(len(chosen) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            c = chosen[i]
            # Check if c is redundant
            # If all neighbors are covered by at least 2 codewords, we can remove c
            if np.min(cover_count[neighbors[c]]) > 1:
                # Remove c
                cover_count[neighbors[c]] -= 1
                chosen.pop(i)
                removed_any = True
                # Restart scan from end to catch new redundancies immediately
                break
                
        if not removed_any:
            # No more removals in this pass
            # Try a different order? Or stop?
            # For now, stop if no removals found in a full reverse pass.
            # A more advanced version might shuffle and try again.
            break

    # Convert indices back to lists of integers
    result = []
    for w in chosen:
        coords = []
        temp = w
        for _ in range(7):
            coords.append(temp % 3)
            temp //= 3
        result.append(coords)
        
    return result
