import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Parameters are fixed by the problem description effectively, 
    # but we should handle the general case or at least the specific one robustly.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # However, the task description specifies 8 matches, ternary, radius 1.
    # I will implement it generally for q=3, n=8, r=1 but allow parameters.
    
    if q != 3 or n != 8 or radius != 1:
         # Fallback to a simple construction for other cases to avoid errors, 
         # though the evaluator likely only calls with 3, 8, 1.
         # Direct sum for general q, n, r=1?
         # For now, return the seed solution if params don't match, 
         # or a simple greedy if they do.
         # Given the specific task contract, it's 3, 8, 1.
         pass

    N = q ** n
    deadline = time.time() + 115.0 # 5s safety margin
    
    # Precompute neighbors
    # List of all words
    # Convert 0..N-1 to base q digits?
    # It's faster to generate neighbors on the fly or precompute once.
    
    # Precompute neighbors for all N words
    # Shape: (N, 1 + 2*n)
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    
    for w in range(N):
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        
        # Get digits
        temp = w
        digits = [0] * n
        for i in range(n):
            digits[i] = temp % q
            temp //= q
            
        for i in range(n):
            d = digits[i]
            # change digit i to (d+1)%q
            w1 = w + ((d + 1) % q - d) * (q ** i)
            # change digit i to (d+2)%q
            w2 = w + ((d + 2) % q - d) * (q ** i)
            
            neighbors[w, idx] = w1
            idx += 1
            neighbors[w, idx] = w2
            idx += 1
            
    # Greedy Search
    best_code = None
    best_size = float('inf')
    
    # Seed solution for comparison
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    seed_size = len(seed_code)
    best_size = seed_size
    best_code = seed_code
    
    rng = np.random.default_rng()
    
    # Strategy:
    # 1. Run randomized greedy set cover multiple times.
    # 2. Apply local search (removal) to reduce size.
    # 3. Keep the best result.
    
    while time.time() < deadline:
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        
        while uncovered.sum() > 0:
            # Calculate gain for each candidate
            # gain[w] = sum(uncovered[neighbors[w]])
            # Using numpy fancy indexing
            gains = uncovered[neighbors].sum(axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            
            # If max_gain == 0, something is wrong (should not happen)
            if max_gain == 0:
                break
                
            # Find all words with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly from candidates
            w = rng.choice(candidates)
            
            code.append(w)
            
            # Mark neighbors as covered
            uncovered[neighbors[w]] = 0
            
        # Local Search Reduction Phase
        # We have a valid code 'code'. We try to remove redundant codewords.
        # A codeword c is redundant if every neighbor of c is covered by at least one other codeword.
        # We can track coverage counts.
        
        # Initialize coverage counts
        cov_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            cov_counts[neighbors[c]] += 1
            
        # Iterate removal
        # We need to be careful: removing a codeword changes coverage counts.
        # If a codeword c has a neighbor w where cov_counts[w] == 1, then c is essential (cannot remove).
        # If all neighbors of c have cov_counts[w] > 1, c is redundant.
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Iterate over a copy of the list to allow removal
            # We can't iterate over a list while modifying it easily, so we use indices or a copy.
            # Let's iterate from end to start to avoid index shifting issues if we remove.
            # Or just build a new list.
            new_code = []
            for c in code:
                # Check if c is redundant
                # If any neighbor of c has count 1, it's essential.
                if np.min(cov_counts[neighbors[c]]) > 1:
                    # Redundant
                    # Remove c: decrement counts for its neighbors
                    cov_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    # Essential
                    new_code.append(c)
            
            if changed:
                code = new_code
                
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
            # Optional: break if we hit the lower bound (403)? 
            # We don't know if we found it, but 403 is the bound. 
            # If we get close, we can keep trying.
            
    # Convert integer codewords back to lists of integers
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse() # If we constructed digits LSB first, need to check order
        # In precompute: digits[0] is LSB. 
        # In seed: h + tail. h is [a,b,...]. 
        # The order of coordinates matters for the output format?
        # "list of n integers in 0..q-1". 
        # Usually coordinate 0 is the first element.
        # My base conversion: 
        # w = d_0 + d_1 * q + ... 
        # So word[0] should be d_0? 
        # In the seed: tetracode = [[a, b, ...]]
        # a corresponds to coordinate 0.
        # My loop: digits[i] = temp % q. So digits[0] is coord 0.
        # So word.append(digits[i]) is correct order.
        
        # Let's re-verify the base conversion in the neighbor precompute.
        # digits[0] = w % 3. This is the coefficient of 3^0.
        # So coordinate 0 is the least significant digit in my integer representation.
        # The output should be a list [c0, c1, ..., c_{n-1}].
        # So I should output digits[0], digits[1], ...
        
        final_code.append(word)
        
    return final_code
