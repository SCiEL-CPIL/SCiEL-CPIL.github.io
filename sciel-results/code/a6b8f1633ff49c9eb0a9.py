import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # The task specifies q=3, n=8, radius=1.
    N = q ** n
    
    # Helper to decode an integer to a list of digits
    def decode(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    # Precompute coverage bitsets
    # coverage[c] is an integer where bit i is 1 if point i is covered by codeword c
    coverage = [0] * N
    
    # Precompute neighbors for each point to speed up coverage calculation
    for c in range(N):
        bits = 0
        # The point itself
        bits |= (1 << c)
        
        # Neighbors at distance 1
        # We can compute neighbors more efficiently
        # For each position i, change digit
        x = c
        for i in range(n):
            # Extract digit at position i
            # Our encoding is: digit[0] is least significant?
            # decode: digits[i] = x % q; x //= q.
            # So digit 0 is LSB.
            # To change digit i:
            # base = q ** i
            # val = (c // base) % q
            # new_val in 0..q-1, new_val != val
            base = q ** i
            val = (c // base) % q
            for new_val in range(q):
                if new_val == val:
                    continue
                neighbor = c + (new_val - val) * base
                bits |= (1 << neighbor)
        
        coverage[c] = bits

    full_mask = (1 << N) - 1
    
    best_code = None
    best_size = N + 1
    
    # Run multiple iterations of Greedy Set Cover
    # We use random tie-breaking to explore different solutions
    num_iterations = 50
    
    for _ in range(num_iterations):
        if time.time() > deadline - 5:  # Leave time for final check/return
            break
            
        uncovered = full_mask
        selected = []
        
        while uncovered:
            best_c = -1
            best_gain = 0
            # To speed up, we don't shuffle the whole list.
            # We iterate all and use random tie-breaking.
            for c in range(N):
                gain = (uncovered & coverage[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                elif gain == best_gain and best_gain > 0:
                    # Random tie break: with probability 0.5, switch
                    if random.random() < 0.5:
                        best_c = c
            
            if best_gain == 0:
                # Should not happen if coverage is valid
                break
                
            selected.append(best_c)
            uncovered &= ~coverage[best_c]
            
        # Check if we have a valid code
        if uncovered != 0:
            continue
            
        # Post-processing: Try to remove redundant codewords
        # This is a local search step
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Iterate through selected codewords in random order to avoid bias
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(selected):
                    continue
                c = selected[i]
                # Check if removing c leaves all points covered
                # We need to check if `full_mask` is covered by `selected \ {c}`
                # Efficient check:
                # Let `union_others` be union of coverage of all other selected.
                # We can compute this, but it's O(K^2) per check if done naively.
                # Better: Maintain a count of how many times each point is covered?
                # Or just compute union of others. K is ~500. 500 * 500 = 250k bit ops. Fast enough.
                
                union_others = 0
                for j, other_c in enumerate(selected):
                    if j != i:
                        union_others |= coverage[other_c]
                
                if (union_others & full_mask) == full_mask:
                    selected.pop(i)
                    changed = True
                    break # Restart the loop as indices changed
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
            
    # Fallback if no valid code found (should not happen)
    if best_code is None:
        # Return a trivial valid code (all words) - this is huge but valid
        # Or use the first greedy run result if it was valid but not best?
        # Actually, the first greedy run should be valid.
        # Let's just return the first valid one we found in the loop.
        # If best_code is still None, it means we never finished a greedy run successfully?
        # That's unlikely.
        pass

    result = []
    for c in best_code:
        result.append(decode(c))
        
    return result
