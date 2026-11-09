import time
import random
import numpy as np

# Global cache for precomputed APs to avoid recomputation if construct is called multiple times
# (Though typically it's called once per evaluation, this is good practice)
_AP_CACHE = None

def _precompute_aps(n):
    global _AP_CACHE
    if _AP_CACHE is not None and _AP_CACHE[0] == n:
        return _AP_CACHE[1], _AP_CACHE[2]
        
    aps = []
    aps_containing = [[] for _ in range(n)]
    
    for d in range(1, n):
        for s in range(n):
            # Generate the maximal AP starting at s with step d
            # Indices: s, s+d, s+2d, ... < n
            # We need to add all prefixes of this sequence as separate APs
            # Because any prefix is also a valid AP contained in [1, n]
            
            # Optimization: 
            # For a fixed d and s, the APs are:
            # [s]
            # [s, s+d]
            # [s, s+d, s+2d]
            # ...
            
            curr = s
            prefix = []
            while curr < n:
                prefix.append(curr)
                ap_id = len(aps)
                aps.append(prefix.copy()) # Store the indices
                for idx in prefix:
                    aps_containing[idx].append(ap_id)
                curr += d
                
    _AP_CACHE = (n, aps, aps_containing)
    return aps, aps_containing

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin from 60s
    
    aps, aps_containing = _precompute_aps(n)
    num_aps = len(aps)
    
    # Initial random signs
    rng = random.Random(42)
    x = [rng.choice([-1, 1]) for _ in range(n)]
    
    # Initial sums
    sums = np.zeros(num_aps, dtype=np.int32)
    for k, ap in enumerate(aps):
        s = 0
        for idx in ap:
            s += x[idx]
        sums[k] = s
        
    current_max = np.max(np.abs(sums))
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    # Pre-convert aps_containing to numpy arrays for faster indexing if needed?
    # List indexing is fine for the update step if we use numpy slicing on sums
    # sums[list_of_indices] += delta works well
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        i = rng.randint(0, n - 1)
        old_val = x[i]
        new_val = -old_val
        delta = 2 * new_val
        
        # Update sums
        indices = aps_containing[i]
        if indices:
            sums[indices] += delta
            new_max = np.max(np.abs(sums))
            delta_max = new_max - current_max
            
            # Acceptance probability
            if delta_max < 0 or (T > 0 and rng.random() < np.exp(-delta_max / T)):
                x[i] = new_val
                current_max = new_max
            else:
                # Reject
                sums[indices] -= delta
                # x[i] remains old_val
        
        T *= cooling
        
    return x
