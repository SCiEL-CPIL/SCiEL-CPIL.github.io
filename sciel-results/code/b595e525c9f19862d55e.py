import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 17
    MIN_D = 6
    SIZE = 1 << N
    
    # Precompute weights
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)
    
    # 1. Construct Linear Code [17, 8, 6]
    basis = []
    code = np.array([0], dtype=np.int32)
    
    for w in range(1, SIZE):
        if len(basis) == 8:
            break
        if time.time() > deadline - 30:
            break
            
        # Check distance to all current codewords
        dists = weights[w ^ code]
        if dists.min() >= MIN_D:
            basis.append(w)
            new_words = code ^ w
            code = np.concatenate([code, new_words])
            
    lin_code = code.tolist()
    
    if len(lin_code) < 256:
        return lin_code
        
    lin_code_np = np.array(lin_code, dtype=np.int32)
    
    # 2. Find Candidates
    candidates = []
    # Vectorized check for all words?
    # We can do it in batches to save memory/time
    # But 131k * 256 is small enough for a loop with vectorized inner step
    
    for x in range(SIZE):
        if x in lin_code: # Optimization: skip existing
             continue
        dists = weights[x ^ lin_code_np]
        if dists.min() >= MIN_D:
            candidates.append(x)
            
    if not candidates:
        return lin_code
        
    # 3. Greedy Packing with Random Restarts
    best_size = len(lin_code)
    best_code = lin_code[:]
    
    # Estimate number of restarts based on time
    # Assume each restart takes T_restart
    # We have ~50s left.
    
    num_restarts = 1000 # Adjust dynamically
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        cands = candidates[:]
        random.shuffle(cands)
        
        current_code = lin_code[:]
        current_code_np = lin_code_np.copy()
        
        for x in cands:
            # Check distance to current_code
            dists = weights[x ^ current_code_np]
            if dists.min() >= MIN_D:
                current_code.append(x)
                # Update np array efficiently? 
                # Appending to list and converting every time is slow.
                # Better: maintain a list, and only convert to np for distance check?
                # Or just use a set/list and loop in Python for small sizes?
                # No, NumPy is better.
                # We can just rebuild the np array? No, too slow.
                # Let's just use the list for the check if it's small?
                # Actually, 256+ words.
                # Let's just use the list and a helper function.
                
        # Optimization for Step 3:
        # The above loop `for x in cands` with `weights[x ^ current_code_np]` 
        # requires `current_code_np` to be updated.
        # Updating a NumPy array by appending is O(N).
        # Doing this K times is O(K*N).
        # K is small.
        
        # Let's refine the inner loop for speed:
        # Instead of updating np array, we can just check against the list using a loop?
        # No, NumPy is faster for distance calc.
        
        # Let's try a different approach for the greedy step:
        # Keep `current_code` as a list.
        # For each x, check distance to all in `current_code`.
        # If `len(current_code)` is ~256, and we check ~100 candidates, 
        # that's 25600 distance checks.
        # Each check is `weights[x ^ y]`.
        # This is fast in Python if we use the precomputed weights array.
        
        pass # Replace with efficient implementation below

    return best_code
