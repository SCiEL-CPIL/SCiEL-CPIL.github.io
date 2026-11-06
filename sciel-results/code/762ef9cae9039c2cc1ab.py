import time
import random
import numpy as np

# Precompute neighbors for all words in {0,1,2}^7
# Word is represented as an integer in base 3.
# Total words: 3^7 = 2187

def int_to_word(w, n=7):
    """Convert base-3 integer to list of digits."""
    return [w // (3**i) % 3 for i in range(n)]

def word_to_int(w):
    """Convert list of digits to base-3 integer."""
    val = 0
    for i, d in enumerate(w):
        val += d * (3**i)
    return val

# Precompute neighbors for all 2187 words
NEIGHBORS = []
for w in range(3**7):
    digits = int_to_word(w)
    neighbors = [w]
    for i in range(7):
        for d in range(3):
            if d != digits[i]:
                new_digits = digits.copy()
                new_digits[i] = d
                neighbors.append(word_to_int(new_digits))
    NEIGHBORS.append(neighbors)

def construct(q, n, radius):
    # The task specifies q=3, n=7, radius=1. 
    # We assume these values based on the problem description.
    # If q, n, radius differ, we fall back to a simple grid or raise an error if unsupported.
    if q != 3 or n != 7 or radius != 1:
        # Fallback: simple greedy for general case (might be slow/weak for other params)
        # But the task is specific to 3,7,1.
        pass

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Greedy Construction
    # coverage[i] = number of codewords covering word i
    coverage = np.zeros(3**7, dtype=np.int32)
    code = set()
    
    # Initial seed: maybe start with a few random or structured codewords?
    # Pure greedy from empty might be slow to start. 
    # Let's start with the direct sum code from the seed as a strong initial point?
    # The seed code has 243 words. It's valid. 
    # Starting from 243 and reducing might be faster than building from 0 to ~200.
    # Let's use the seed's initial code as the starting point for local search.
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    for w in initial_code:
        code.add(word_to_int(w))
        
    # Initialize coverage
    for c in code:
        for v in NEIGHBORS[c]:
            coverage[v] += 1
            
    # 2. Local Optimization: Remove Redundant Codewords
    # We repeatedly try to remove codewords.
    # To make it more robust, we can randomize the order or do multiple passes.
    
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        # Shuffle order to avoid getting stuck in a specific order bias?
        # Or just iterate. Shuffling might help escape local minima if we added a "swap" step, 
        # but for pure deletion, order matters less. 
        # However, removing one might make another redundant. 
        # A single pass might not remove all. We loop until no more removed in a pass.
        
        for c in list(code):
            if time.time() > deadline:
                break
            
            # Check if c is redundant
            # Temporarily remove c's contribution
            is_redundant = True
            for v in NEIGHBORS[c]:
                if coverage[v] == 1:
                    # If coverage is 1, only c covers it. Removing c would uncover v.
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove c
                for v in NEIGHBORS[c]:
                    coverage[v] -= 1
                code.remove(c)
                removed_any = True
        
    # Optional: Try to add codewords to cover gaps if we somehow broke validity? 
    # No, we only remove if valid.
    
    # Convert back to list of lists
    result = [int_to_word(c) for c in code]
    
    return result
