import time
import random
import itertools

def construct(q, n, radius):
    # The task specifies q=3, n=8, radius=1.
    # We assume these values are passed, but the logic is specific to this instance.
    # If q, n, radius differ, we fall back to a simple greedy or the seed logic.
    if (q, n, radius) != (3, 8, 1):
        # Fallback to seed logic for other parameters
        # Direct sum of tetracode (only valid for n>=4, q=3)
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Simple greedy for other cases (not optimized)
            return [] # Or some basic construction

    total_points = 3 ** 8
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    # 1. Precompute masks for all words
    # word_index: 0 to 6560
    # Convert index to base-3 digits
    def index_to_digits(idx, n=8, base=3):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % base
            idx //= base
        return digits

    def digits_to_index(digits, n=8, base=3):
        idx = 0
        for i in range(n):
            idx = idx * base + digits[i]
        return idx

    masks = []
    for w_idx in range(total_points):
        digits = index_to_digits(w_idx)
        neighbors = set()
        neighbors.add(w_idx)
        for pos in range(8):
            orig_val = digits[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_digits = digits.copy()
                    new_digits[pos] = new_val
                    neighbors.add(digits_to_index(new_digits))
        masks.append(list(neighbors))

    # 2. Precompute neighbors_of_point (inverse)
    neighbors_of_point = [[] for _ in range(total_points)]
    for w_idx in range(total_points):
        for p in masks[w_idx]:
            neighbors_of_point[p].append(w_idx)

    # 3. Initialize with 729-word code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            word = h + list(tail)
            initial_words.append(digits_to_index(word))
    
    active = set(initial_words)
    counts = [0] * total_points
    for w in active:
        for p in masks[w]:
            counts[p] += 1

    # 4. Local Search
    def removal_pass():
        changed = False
        # Iterate over a copy of active list to allow removal
        for c in list(active):
            # Check if c is redundant
            # c is redundant if all p in masks[c] have counts[p] > 1
            # i.e., no p has counts[p] == 1
            is_redundant = True
            for p in masks[c]:
                if counts[p] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove c
                active.remove(c)
                for p in masks[c]:
                    counts[p] -= 1
                changed = True
        return changed

    # Run initial removal
    removal_pass()
    removal_pass() # Second pass to catch chains

    while time.time() - start_time < time_limit:
        # Try to remove more
        if not removal_pass():
            # No more removals possible. Try a swap.
            if not active:
                break
                
            # Pick a random active codeword to swap
            c_old = random.choice(list(active))
            
            # Identify critical points
            crits = [p for p in masks[c_old] if counts[p] == 1]
            
            # Find candidates that cover all crits
            # Start with neighbors of first crit
            if crits:
                candidates = set(neighbors_of_point[crits[0]])
                for p in crits[1:]:
                    candidates.intersection_update(neighbors_of_point[p])
            else:
                # Should not happen if removal_pass is correct, but just in case
                candidates = set(range(total_points))
            
            # Remove c_old itself (we are replacing it)
            candidates.discard(c_old)
            
            # Remove currently active words (we can't duplicate)
            candidates.difference_update(active)
            
            if candidates:
                c_new = random.choice(list(candidates))
                
                # Perform swap
                # Remove c_old
                for p in masks[c_old]:
                    counts[p] -= 1
                active.remove(c_old)
                
                # Add c_new
                for p in masks[c_new]:
                    counts[p] += 1
                active.add(c_new)
                
                # Check validity (should be valid by construction of candidates, 
                # but let's be safe: ensure no count is 0)
                # Actually, candidates were chosen to cover crits.
                # Other points in masks[c_old] had count > 1, so after decrement, count >= 1.
                # So validity is guaranteed.
                
                # Try to remove again
                removal_pass()

    # Convert active indices back to list of lists of integers
    result = []
    for w in active:
        result.append(index_to_digits(w))
    
    return result
