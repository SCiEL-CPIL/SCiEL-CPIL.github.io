"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # 1. Pre-compute the incidence matrix M
    # M[k, i] = 1 if index i is in arithmetic progression k
    # We need to enumerate all APs in [0, n-1]
    
    # List of progressions as lists of indices
    progressions = []
    for step in range(1, n + 1):
        for r in range(step):
            if r >= n:
                continue
            # Indices: r, r+step, r+2*step, ... < n
            indices = list(range(r, n, step))
            if indices:
                progressions.append(indices)
    
    num_progs = len(progressions)
    
    # Create sparse-like structure or dense matrix?
    # Dense matrix: num_progs x n. 33k x 256 = 8.5M elements.
    # 8.5M * 8 bytes = 68 MB. This is acceptable.
    # However, creating it in a Python loop might be slow.
    # Let's try to build it efficiently.
    
    # Alternative: Store progressions as arrays and use advanced indexing for updates?
    # If we flip sign at index i, we need to add delta to sums[k] for all k containing i.
    # If we have M, sums += delta * M[:, i].
    # Building M:
    M = np.zeros((num_progs, n), dtype=np.int8) # int8 is enough for 0/1
    
    for k, indices in enumerate(progressions):
        M[k, indices] = 1
        
    # 2. Initial State
    rng = random.Random(42)
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int64)
    
    # Current sums for each progression
    # sums = M @ signs
    sums = M @ signs
    
    # Objective: Minimize max(|sums|)
    # We will use a hybrid objective or just max.
    # Let's stick to minimizing the max discrepancy, but use the fast evaluation to do SA.
    
    current_max = np.max(np.abs(sums))
    best_max = current_max
    best_signs = signs.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    start_time = time.time()
    iter_count = 0
    
    # Pre-compute columns of M for fast update?
    # M[:, i] is a view/slice. Accessing it is fast.
    
    while time.time() < deadline:
        # Pick random index to flip
        i = rng.randint(0, n - 1)
        
        # Flip sign
        old_val = signs[i]
        new_val = -old_val
        delta = new_val - old_val # +/- 2
        
        # Update sums
        # sums += delta * M[:, i]
        # M[:, i] is a column vector.
        # This is a vector operation of size num_progs (~33k).
        
        # Optimization: Check if this flip can possibly improve the max?
        # Not easily without computing.
        
        # Perform update
        M_col_i = M[:, i]
        sums += delta * M_col_i
        
        new_max = np.max(np.abs(sums))
        
        # Acceptance criterion
        if new_max <= current_max:
            # Always accept improvement or equal
            current_max = new_max
            signs[i] = new_val
            
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            # Worse move
            delta_obj = new_max - current_max
            # Accept with probability exp(-delta_obj / T)
            # Since delta_obj is integer >= 1, and T starts at 1.0
            if rng.random() < np.exp(-delta_obj / T):
                current_max = new_max
                signs[i] = new_val
            else:
                # Revert
                sums -= delta * M_col_i
                # signs[i] is already old_val, so no need to revert signs if we didn't update it?
                # Wait, I updated signs[i] = new_val only in the accept block?
                # No, I didn't update signs[i] in the code above before checking.
                # Let's fix the logic.
                
                # Corrected Logic:
                # 1. Calculate new sums
                # 2. Decide accept/reject
                # 3. If accept, update signs. If reject, revert sums.
                
                # So in the reject case, signs[i] is still old_val.
                # We just need to revert sums.
                pass
        
        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T
            # Optional: Restart or continue? Let's continue.
            
        iter_count += 1
        
        # Periodic restart to escape local minima?
        # If we are stuck (no improvement for a while), maybe perturb?
        # For now, simple SA.
        
        # Check time more frequently to avoid overshoot
        if iter_count % 1000 == 0:
             if time.time() > deadline - 2:
                 break

    return best_signs.tolist()
# EVOLVE-BLOCK-END
