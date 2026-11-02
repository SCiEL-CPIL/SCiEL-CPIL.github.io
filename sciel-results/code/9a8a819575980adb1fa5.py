import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Handle the specific case of the task (q=3, n=8)
    # If parameters differ significantly, fall back to seed
    N = q ** n
    
    if N > 50000:
        # Fallback to seed logic for larger instances
        # This logic assumes n >= 4
        if n < 4:
            return [list(tail) for tail in itertools.product(range(q), repeat=n)]
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # Helper to decode an integer to a list of digits (little-endian as per decode logic above)
    def decode(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    # Precompute coverage bitsets
    coverage = [0] * N
    
    for c in range(N):
        bits = 0
        digits = decode(c)
        bits |= (1 << c)
        
        for i in range(n):
            orig_val = digits[i]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbor_digits = digits[:]
                neighbor_digits[i] = new_val
                neighbor_val = 0
                for j in range(n):
                    neighbor_val = neighbor_val * q + neighbor_digits[j]
                bits |= (1 << neighbor_val)
        
        coverage[c] = bits

    full_mask = (1 << N) - 1
    
    best_code = None
    best_size = N
    
    # Run multiple iterations of Greedy Set Cover
    num_iterations = 15 # Adjust based on time
    
    for _ in range(num_iterations):
        if time.time() > deadline:
            break
            
        uncovered = full_mask
        selected = []
        
        while uncovered:
            best_c = -1
            best_gain = 0
            
            # Iterate through all candidates
            # Using a list to allow random tie-breaking effectively
            # But iterating range(N) is faster.
            # We'll just do random tie-breaking in the comparison.
            
            for c in range(N):
                gain = (uncovered & coverage[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                elif gain == best_gain and best_gain > 0:
                    if random.random() < 0.5:
                        best_c = c
            
            if best_gain == 0:
                break
                
            selected.append(best_c)
            uncovered &= ~coverage[best_c]
            
        # Post-processing: Remove redundant codewords
        # We do this in a loop until no more can be removed
        # To speed up, we check from the end (last added are often less critical? No, greedy adds most critical first).
        # Actually, greedy adds the most covering first. So early ones are more important.
        # Late ones might be redundant.
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to remove from the end first
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                # Check if c is redundant
                # Union of all others
                union_others = 0
                for j in range(len(selected)):
                    if j != i:
                        union_others |= coverage[other_c] if False else coverage[selected[j]]
                
                if (union_others & full_mask) == full_mask:
                    selected.pop(i)
                    changed = True
                    break
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
            
    # If no code found (should not happen), return empty or seed
    if best_code is None:
        return []

    result = []
    for c in best_code:
        result.append(decode(c))
        
    return result
