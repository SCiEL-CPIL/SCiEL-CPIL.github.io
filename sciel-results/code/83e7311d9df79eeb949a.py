import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit

    # 1. Generate all arithmetic progressions
    # An AP is defined by (start, step). It includes indices start, start+step, start+2*step, ... < n
    # Note: The problem says "contained in [1, 256]". Indices 0..255 correspond to 1..256.
    # Any start in 0..255 and step in 1..255.
    progs = []
    for step in range(1, n + 1):
        for start in range(n):
            # If start + step >= n, the progression has length 1 (just start) if start < n.
            # Actually, any single element is an AP of length 1.
            # The problem says "length including one".
            # So for any i in 0..255, {i} is an AP.
            # Also {i, i+step, ...} for any step.
            # Let's just generate all unique sets of indices.
            # However, many duplicates might exist? No, (start, step) uniquely defines the set.
            # Wait, {0, 2} is step 2, start 0. {2, 0} is not an AP with positive step.
            # So (start, step) is unique.
            # Number of APs: sum_{step=1}^{255} 256 + 256 (for step=256? No, step=256 means length 1 only if start < 256.
            # Actually if step >= n, the AP is just {start}.
            # So for step >= n, there are n APs, all singletons.
            # But singletons are already covered by step=1? No.
            # Step 1: {0}, {1}, ..., {255} (length 1) and {0,1}, {1,2}...
            # Wait, if I iterate start 0..255 and step 1..255:
            # For step=1: {0}, {1}...{255} are included as length 1?
            # range(start, n, step).
            # If start=0, step=1: 0, 1, ..., 255.
            # The "progression" is the set of indices.
            # The sum is over the elements in the progression.
            # So for a given (start, step), the AP is {start, start+step, ...}.
            # This set is unique for each (start, step).
            # Total number of (start, step) pairs: 256 * 256 = 65536.
            # This is manageable.
            
            idx = range(start, n, step)
            # We need to store these for quick access.
            # Storing lists of indices for 65k APs is fine.
            progs.append(list(idx))

    N_p = len(progs)
    
    # 2. Precompute P_containing: for each index i, which APs contain it?
    # This is crucial for fast updates.
    P_containing = [[] for _ in range(n)]
    for pid, idx in enumerate(progs):
        for i in idx:
            P_containing[i].append(pid)
            
    # Convert to numpy arrays for faster indexing
    for i in range(n):
        P_containing[i] = np.array(P_containing[i], dtype=np.int32)
        
    # 3. Initial Solution: Thue-Morse or Random
    # Thue-Morse is a good deterministic start.
    v = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    
    # 4. Initial Sums
    sums = np.zeros(N_p, dtype=np.int32)
    for pid, idx in enumerate(progs):
        if idx:
            sums[pid] = v[idx].sum()
            
    current_max = np.max(np.abs(sums))
    best_v = v.copy()
    best_max = current_max
    
    # 5. Local Search with Simulated Annealing
    # We want to minimize the max absolute sum.
    # Flipping a sign at index i changes the sum of all APs containing i by +/- 2.
    
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    # Precompute the max absolute sum efficiently?
    # np.max(np.abs(sums)) is O(N_p). N_p ~ 65k. This is fast.
    
    while time.time() < deadline:
        i = np.random.randint(0, n)
        delta = -2 * v[i]
        pids = P_containing[i]
        
        if len(pids) == 0:
            # Should not happen, but safe
            continue
            
        old_vals = sums[pids].copy()
        
        # Apply flip
        sums[pids] += delta
        
        new_max = np.max(np.abs(sums))
        
        if new_max < current_max:
            # Accept
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_v = v.copy()
        else:
            # Accept with prob exp(-delta_E / T)
            dE = new_max - current_max
            if dE > 0 and np.random.rand() < np.exp(-dE / T):
                current_max = new_max
            else:
                # Revert
                sums[pids] = old_vals
                # Don't flip v
                v[i] = -v[i] # Wait, I need to be careful with the flip logic.
                # If I revert sums, I must also revert v.
                # But I haven't flipped v yet in the code above?
                # Let's restructure.
                
        # Correct logic:
        # 1. Save old sums
        # 2. Update sums
        # 3. Check acceptance
        # 4. If accepted, flip v. If rejected, revert sums and don't flip v.
        
        # My previous code was slightly tangled. Let's rewrite the loop body cleanly.
        
    # The loop above was messy. Let's do it properly in the final code.
    
    return best_v.tolist()

# Let's rewrite the whole thing to be clean and correct.

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit

    # 1. Generate all arithmetic progressions
    progs = []
    for step in range(1, n + 1):
        for start in range(n):
            idx = list(range(start, n, step))
            progs.append(idx)

    N_p = len(progs)
    
    # 2. Precompute P_containing
    P_containing = [[] for _ in range(n)]
    for pid, idx in enumerate(progs):
        for i in idx:
            P_containing[i].append(pid)
            
    for i in range(n):
        P_containing[i] = np.array(P_containing[i], dtype=np.int32)
        
    # 3. Initial Solution
    v = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    
    # 4. Initial Sums
    sums = np.zeros(N_p, dtype=np.int32)
    for pid, idx in enumerate(progs):
        if idx:
            sums[pid] = v[idx].sum()
            
    current_max = np.max(np.abs(sums))
    best_v = v.copy()
    best_max = current_max
    
    # 5. Local Search
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    while time.time() < deadline:
        i = np.random.randint(0, n)
        pids = P_containing[i]
        
        if len(pids) == 0:
            continue
            
        old_vals = sums[pids].copy()
        
        # Tentative flip
        delta = -2 * v[i]
        sums[pids] += delta
        
        new_max = np.max(np.abs(sums))
        
        accept = False
        if new_max < current_max:
            accept = True
        else:
            dE = new_max - current_max
            if dE > 0 and np.random.rand() < np.exp(-dE / T):
                accept = True
            elif dE == 0:
                accept = True # Neutral move, accept to explore
                
        if accept:
            v[i] = -v[i]
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_v = v.copy()
        else:
            sums[pids] = old_vals
            # v[i] remains unchanged
            
        T *= alpha
        if T < T_min:
            T = T_min

    return best_v.tolist()
