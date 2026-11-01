import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    # Parameters specific to the task instance are passed, 
    # but we assume q=3, n=6, radius=1 based on the problem description.
    # The function signature is construct(q, n, radius).
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though the task is specific.
        # For now, return the seed construction generalized if possible, 
        # or just a simple greedy.
        # But the prompt implies this is for the specific 6-match football pool.
        pass

    deadline = time.time() + 115  # Safety margin
    
    # Precompute neighbors for all 729 words
    # Word is 0..728. Convert to base-3 digits.
    # Neighbors: change one digit to one of the other 2 values.
    
    def get_neighbors(word_int):
        # Convert int to tuple of 3 digits
        # word_int = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4 + 243*d5
        neighbors = []
        digits = [0]*n
        temp = word_int
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        
        for i in range(n):
            original = digits[i]
            for val in range(q):
                if val == original:
                    continue
                new_digits = digits[:]
                new_digits[i] = val
                # Convert back to int
                new_int = 0
                mult = 1
                for j in range(n):
                    new_int += new_digits[j] * mult
                    mult *= q
                neighbors.append(new_int)
        return neighbors

    # Precompute all neighbor lists
    all_neighbors = [get_neighbors(i) for i in range(q**n)]
    
    # Initialize with the 81-word code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            # Convert to int
            w_int = 0
            mult = 1
            for d in word:
                w_int += d * mult
                mult *= q
            initial_code.append(w_int)
            
    current_code = set(initial_code)
    
    # Calculate coverage counts
    # coverage[w] = number of codewords c such that dist(c, w) <= 1
    coverage = np.zeros(q**n, dtype=np.int8)
    
    def update_coverage(codeword, sign):
        # sign = 1 for add, -1 for remove
        neighbors = all_neighbors[codeword]
        # The ball includes the codeword itself
        coverage[codeword] += sign
        for nb in neighbors:
            coverage[nb] += sign

    for c in current_code:
        update_coverage(c, 1)
        
    best_code = set(current_code)
    best_size = len(best_code)
    
    # Helper to check validity
    def is_valid():
        return np.min(coverage) >= 1

    # Local Search
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove a codeword
        # Shuffle to avoid bias
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c in code_list:
            # Temporarily remove
            update_coverage(c, -1)
            if is_valid():
                current_code.remove(c)
                improved = True
                break
            else:
                # Revert
                update_coverage(c, 1)
                
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = set(current_code)
            continue # Restart removal phase
        
        # 2. If no removal, try swaps
        # Pick a random codeword to swap
        swaps_made = 0
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c_old in code_list:
            # Try a few random candidates
            found_swap = False
            # Candidate 1: Random word from space
            # Candidate 2: Neighbor of c_old
            # Candidate 3: Word that covers many points currently covered only by c_old?
            
            # Simple strategy: Try 10 random candidates
            for _ in range(10):
                if random.random() < 0.5:
                    c_new = random.choice(all_neighbors[c_old])
                else:
                    c_new = random.randint(0, q**n - 1)
                    
                if c_new in current_code:
                    continue
                    
                # Remove c_old, Add c_new
                update_coverage(c_old, -1)
                update_coverage(c_new, 1)
                
                if is_valid():
                    current_code.remove(c_old)
                    current_code.add(c_new)
                    swaps_made += 1
                    found_swap = True
                    break
                else:
                    # Revert
                    update_coverage(c_new, -1)
                    update_coverage(c_old, 1)
                    
            if found_swap:
                break # Restart removal phase after a successful swap?
                # Or continue swapping?
                # Usually, after a swap, the structure changes, so restart removal.
                break
        
        if swaps_made > 0:
            # Check if this swap led to a better state (unlikely immediately, but maybe enables removal)
            # We restart the loop to try removals again.
            pass
        else:
            # Stuck. Perturb.
            # Change 2 random codewords to random words, then repair?
            # Simpler: Just accept that we might be in a local minimum of size k.
            # Try a "double swap": remove 2, add 2.
            # This is complex. 
            # Alternative: Reset to best_code and try again with different random seeds?
            # Or just keep trying random swaps with a higher number of candidates.
            
            # Let's try a more aggressive perturbation:
            # Pick 2 codewords, replace them with 2 random words.
            # Check validity. If valid, keep.
            c_list = list(current_code)
            if len(c_list) >= 2:
                c1, c2 = random.sample(c_list, 2)
                for _ in range(20):
                    n1 = random.randint(0, q**n - 1)
                    n2 = random.randint(0, q**n - 1)
                    if n1 in current_code or n2 in current_code or n1 == n2:
                        continue
                        
                    update_coverage(c1, -1)
                    update_coverage(c2, -1)
                    update_coverage(n1, 1)
                    update_coverage(n2, 1)
                    
                    if is_valid():
                        current_code.remove(c1)
                        current_code.remove(c2)
                        current_code.add(n1)
                        current_code.add(n2)
                        break
                    else:
                        update_coverage(n1, -1)
                        update_coverage(n2, -1)
                        update_coverage(c1, 1)
                        update_coverage(c2, 1)

    # Convert best_code back to list of lists
    result = []
    for w_int in best_code:
        digits = [0]*n
        temp = w_int
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        result.append(digits)
        
    return result
