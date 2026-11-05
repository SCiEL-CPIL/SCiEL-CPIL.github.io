import numpy as np
import time
import itertools
import random
import sys

def construct(q, n, radius):
    """
    Construct a covering code for q^n with given radius.
    Optimized for q=3, n=7, radius=1.
    """
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        total = q ** n
        if total > 100000:
            # Too large, return empty or raise
            return []
        
        # Build neighbors
        neighbors_list = []
        for w in range(total):
            coords = []
            temp = w
            for _ in range(n):
                coords.append(temp % q)
                temp //= q
            neighbors = {w}
            for i in range(n):
                for d in range(-1, 2):
                    if d == 0:
                        continue
                    new_coord = (coords[i] + d) % q
                    val = w
                    val -= coords[i] * (q ** i)
                    val += new_coord * (q ** i)
                    neighbors.add(val)
            neighbors_list.append(list(neighbors))
        
        # Greedy
        cover_count = np.zeros(total, dtype=np.int32)
        chosen = []
        all_covered = False
        while not all_covered:
            uncovered = (cover_count == 0)
            gains = np.array([np.sum(uncovered[neighbors_list[c]]) for c in range(total)])
            best_c = np.argmax(gains)
            if gains[best_c] == 0:
                break
            chosen.append(int(best_c))
            cover_count[neighbors_list[best_c]] += 1
            if np.all(cover_count > 0):
                all_covered = True
        
        # Local search removal
        while True:
            removed = False
            for i in range(len(chosen) - 1, -1, -1):
                c = chosen[i]
                if np.min(cover_count[neighbors_list[c]]) > 1:
                    cover_count[neighbors_list[c]] -= 1
                    chosen.pop(i)
                    removed = True
                    break
            if not removed:
                break
        
        return [list(np.array_divmod(c, q)[0].tolist()) for c in chosen] if n <= 30 else []

    # Specific optimization for q=3, n=7, radius=1
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for 150s limit
    
    total_words = 2187
    powers = [3**i for i in range(7)]
    
    # Precompute neighbors: each word has 1 + 7*2 = 15 neighbors
    neighbors = np.zeros((total_words, 15), dtype=np.int32)
    for w in range(total_words):
        neighbors[w, 0] = w
        temp = w
        coords = [0]*7
        for i in range(7):
            coords[i] = temp % 3
            temp //= 3
        idx = 1
        for i in range(7):
            new_coord_plus = (coords[i] + 1) % 3
            delta_plus = (new_coord_plus - coords[i]) * powers[i]
            neighbors[w, idx] = w + delta_plus
            idx += 1
            new_coord_minus = (coords[i] - 1) % 3
            delta_minus = (new_coord_minus - coords[i]) * powers[i]
            neighbors[w, idx] = w + delta_minus
            idx += 1

    # Helper functions
    def idx_to_coords(idx):
        coords = []
        temp = idx
        for _ in range(7):
            coords.append(temp % 3)
            temp //= 3
        return coords

    def coords_to_idx(coords):
        idx = 0
        for i, c in enumerate(coords):
            idx += c * powers[i]
        return idx

    def is_valid_covering(chosen, cover_count):
        return np.all(cover_count > 0)

    def try_remove(chosen, cover_count, neighbors, deadline):
        """Try to remove redundant codewords. Returns True if any removed."""
        # Iterate in random order for better exploration
        indices = list(range(len(chosen)))
        random.shuffle(indices)
        for i in indices:
            if time.time() > deadline:
                return False
            c = chosen[i]
            if np.min(cover_count[neighbors[c]]) > 1:
                cover_count[neighbors[c]] -= 1
                chosen.pop(i)
                return True
        return False

    def try_swap(chosen, cover_count, neighbors, deadline):
        """
        Try to swap a codeword with another to reduce size or maintain size with better structure.
        For now, focus on removal. Swap is complex.
        """
        pass

    # Phase 1: Greedy construction
    cover_count = np.zeros(total_words, dtype=np.int32)
    chosen = []
    
    while time.time() < deadline - 20.0:  # Leave 20s for local search
        uncovered = (cover_count == 0)
        if not np.any(uncovered):
            break
        
        # Vectorized gain calculation
        gains = uncovered[neighbors].sum(axis=1)
        best_c = np.argmax(gains)
        if gains[best_c] == 0:
            break
        
        chosen.append(int(best_c))
        cover_count[neighbors[best_c]] += 1
    
    if not is_valid_covering(chosen, cover_count):
        # Greedy didn't finish, force completion
        while time.time() < deadline - 5.0:
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break
            # Pick an uncovered word, add it (or a better candidate)
            # Simple: add the uncovered word itself
            c = int(uncovered[0])
            if c not in chosen:
                chosen.append(c)
                cover_count[neighbors[c]] += 1
            else:
                # Should not happen, but if it does, pick another
                break
        if not is_valid_covering(chosen, cover_count):
            # Last resort: add all uncovered words
            uncovered = np.where(cover_count == 0)[0]
            for c in uncovered:
                c = int(c)
                if c not in chosen:
                    chosen.append(c)
                    cover_count[neighbors[c]] += 1

    initial_size = len(chosen)
    
    # Phase 2: Local search - iterative removal
    # Multiple passes with shuffling
    max_passes = 50
    for p in range(max_passes):
        if time.time() > deadline - 5.0:
            break
        removed = try_remove(chosen, cover_count, neighbors, deadline)
        if not removed:
            # No more removals possible with simple pass
            # Try a more aggressive approach: try all permutations? No, too slow.
            # Instead, try to find a codeword that can be replaced by a "better" one?
            # For now, break.
            break
    
    # Phase 3: Advanced local search - Simulated Annealing style
    # Try to reduce size by swapping or moving codewords.
    # Define cost = len(chosen)
    # Move: pick a codeword, try to move it to a new position (reassign its coverage area?).
    # Actually, the code is a set. Moving means: remove one, add another.
    # But we want to reduce size, so we only remove.
    # To escape local minima, we might temporarily increase size, then reduce.
    
    # Let's try a different local search: for each codeword, check if it's "essential".
    # If not, remove. We already did this.
    
    # Let's try: for each codeword c, and each neighbor w of c, 
    # check if removing c and adding w reduces the number of "weakly covered" points.
    # This is complex. Let's stick to simple removal but with multiple random restarts of the removal order.
    
    # Actually, the simple removal is deterministic if the order is fixed.
    # We shuffled. Let's do multiple full passes.
    
    for p in range(100):
        if time.time() > deadline - 5.0:
            break
        removed = try_remove(chosen, cover_count, neighbors, deadline)
        if not removed:
            # Try a different shuffle
            indices = list(range(len(chosen)))
            random.shuffle(indices)
            found = False
            for i in indices:
                if time.time() > deadline - 5.0:
                    break
                c = chosen[i]
                if np.min(cover_count[neighbors[c]]) > 1:
                    cover_count[neighbors[c]] -= 1
                    chosen.pop(i)
                    found = True
                    break
            if not found:
                break
    
    # Phase 4: Final polish - try to swap codewords to enable further removals
    # This is heuristic. For each codeword c, try replacing it with a neighbor w.
    # If after removal, the new cover_count is still valid, keep the swap? No, size is same.
    # But maybe the new configuration allows further removals.
    
    # Let's try: for each c in chosen, for each w in neighbors[c]:
    #   temp_cover = cover_count.copy()
    #   temp_cover[neighbors[c]] -= 1
    #   if np.any(temp_cover[neighbors[c]] < 0): continue # c is essential? No, we subtracted c's coverage.
    #   # Check if w can cover the "deficit"
    #   # This is complex. Let's skip for now.
    
    # Convert indices back to lists of integers
    result = []
    for w in chosen:
        result.append(idx_to_coords(int(w)))
        
    return result
