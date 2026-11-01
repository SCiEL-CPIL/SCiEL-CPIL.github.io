import numpy as np
import itertools
import time

def construct(q, n, radius):
    # For this specific task: q=3, n=7, radius=1
    # We assume the evaluator passes these, but we can hardcode logic for efficiency 
    # or generalize if needed. The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # However, the problem is specifically K_3(7,1). 
    # We will implement a general solver for q=3, n=7, r=1 using the heuristic described.
    
    start_time = time.time()
    deadline = 110.0 # 10s margin from 120s limit
    
    # Parameters
    N = q ** n
    # Map index to word
    # We need to generate neighbors for each word.
    # A word is an integer 0..N-1.
    # Base q representation.
    
    # Precompute neighbors
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    # For radius 1, neighbors are i itself and words differing in 1 position.
    
    # Efficiently generating neighbors:
    # Convert index to digits, change one digit, convert back.
    # Since N=2187, we can do this once.
    
    indices = np.arange(N)
    # Create a 2D array of digits for all indices
    # digits[i, k] is the k-th digit of index i in base q
    # We can use base conversion or just iterate.
    # For speed, let's generate the neighbor list directly.
    
    neighbors_list = []
    for i in range(N):
        # Get digits of i
        x = i
        digits = [0] * n
        for k in range(n):
            digits[k] = x % q
            x //= q
        
        neigh = [i]
        # Change each position
        for pos in range(n):
            orig_digit = digits[pos]
            for new_digit in range(q):
                if new_digit == orig_digit:
                    continue
                # Construct new index
                # This is slow in pure python. 
                # Let's use a precomputed table or vectorized approach if possible.
                # Given N is small, python loop is fine for setup.
                new_x = i
                # Remove old digit contribution and add new
                # i = d0 + d1*q + ...
                # Contribution of pos is digits[pos] * (q**pos)
                power = q ** pos
                new_x = i - orig_digit * power + new_digit * power
                neigh.append(new_x)
        neighbors_list.append(neigh)
        
    # Convert to numpy array for fast slicing
    # All neighbors lists have length 1 + n*(q-1)
    neigh_len = 1 + n * (q - 1)
    neighbors = np.array(neighbors_list, dtype=np.int32)
    
    # Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    # Optimization: 
    # gains = uncovered[neighbors].sum(axis=1)
    # This creates an intermediate array of size N x neigh_len.
    # 2187 x 15 is small.
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Calculate gain for each candidate
        # uncovered[neighbors] -> shape (N, neigh_len)
        # .sum(axis=1) -> shape (N,)
        gains = uncovered[neighbors].sum(axis=1)
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # Should not happen if uncovered.any() is true, unless logic error
            # But if we are stuck, break
            break
            
        code.append(int(best_idx))
        uncovered[neighbors[best_idx]] = False
        
    # Pruning Phase
    # Try to remove redundant words
    # Maintain coverage counts
    # counts[x] = number of codewords covering x
    
    # Function to check if a set of codewords is a valid cover
    def is_valid_cover(codewords):
        if not codewords:
            return N == 0
        counts = np.zeros(N, dtype=np.int32)
        for w in codewords:
            counts[neighbors[w]] += 1
        return np.all(counts > 0)
    
    # We can optimize pruning by maintaining the counts array.
    # Initialize counts
    counts = np.zeros(N, dtype=np.int32)
    for w in code:
        counts[neighbors[w]] += 1
        
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate in reverse to allow safe removal without index shifting issues
        # Or just iterate and mark for removal
        to_remove = []
        for i in range(len(code) - 1, -1, -1):
            w = code[i]
            # Tentatively remove w
            # Check if any neighbor of w has count == 1
            # If yes, removing w makes it 0 -> invalid
            # If no, all neighbors have count >= 2 -> valid
            if np.any(counts[neighbors[w]] == 1):
                continue
            # Safe to remove
            counts[neighbors[w]] -= 1
            to_remove.append(i)
            
        if to_remove:
            # Remove them
            # Sort indices to remove to avoid index confusion if we did it in place, 
            # but we are just building a new list or filtering.
            remove_set = set(to_remove)
            code = [w for i, w in enumerate(code) if i not in remove_set]
            changed = True
            
    # Final validity check (just in case)
    # The pruning logic maintains validity, but let's be safe.
    if not is_valid_cover(code):
        # Fallback to seed if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to lists of digits
    result = []
    for w in code:
        x = w
        digits = [0] * n
        for k in range(n):
            digits[k] = x % q
            x //= q
        result.append(digits)
        
    return result

# EVOLVE-BLOCK-START
# The code above is the complete program. 
# I need to wrap it in the EVOLVE-BLOCK format as requested? 
# "Output the COMPLETE program in a single python code block."
# The prompt shows the current program with EVOLVE-BLOCK markers. 
# I should provide the full code.
# EVOLVE-BLOCK-END
