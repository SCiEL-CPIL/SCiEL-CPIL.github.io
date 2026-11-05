import numpy as np
import time

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate divisors
    divisors = []
    for i in range(3, base + 1, 2):
        if base % i == 0:
            divisors.append(i)
    # Sort ascending for greedy (most constrained first)
    divisors.sort()
    
    # 2. Greedy Initialization
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    for m in divisors:
        if time.time() > deadline:
            break
        best_r = 0
        best_gain = -1
        # For each residue r, count how many new elements it covers
        # covered[r::m] gives the slice
        # We want count of 0s in that slice
        # Vectorizing this loop is tricky due to variable slice length, but m is small for small divisors
        # For large m, slice is short.
        # Python loop over r is acceptable.
        for r in range(m):
            # Count zeros in the slice
            # Using np.count_nonzero on a boolean array
            # slice_view = covered[r::m]
            # gain = np.count_nonzero(slice_view == 0)
            # To avoid creating boolean array, we can use sum(covered[r::m] == 0)
            # Or better: base_count = len(range(0, base, m))
            # covered_count = np.count_nonzero(covered[r::m])
            # gain = base_count - covered_count
            
            # Let's use the subtraction method, it's slightly faster
            # Length of slice r::m is either base//m or (base-1)//m + 1
            # Since m divides base, length is exactly base // m
            total_len = base // m
            cov_count = np.count_nonzero(covered[r::m])
            gain = total_len - cov_count
            
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        classes.append([m, best_r])
        covered[best_r::m] = 1

    # 3. Local Search Improvement
    # We maintain a 'counts' array to track how many classes cover each residue
    # This allows O(1) update of loss/gain for swaps
    
    # Initial counts
    counts = np.zeros(base, dtype=np.int16)
    for m, r in classes:
        counts[r::m] += 1
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through all classes
        for i in range(len(classes)):
            if time.time() > deadline:
                return classes
            m, r = classes[i]
            # Calculate loss: residues in S_r that are ONLY covered by this class
            # i.e., counts[x] == 1 for x in r::m
            loss = np.count_nonzero(counts[r::m] == 1)
            
            best_r_new = r
            best_delta = 0 # We only swap if delta > 0
            
            # Check all possible new residues r'
            # Gain for r' is count of x in r'::m where counts[x] == 0
            for r_new in range(m):
                gain = np.count_nonzero(counts[r_new::m] == 0)
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_r_new = r_new
            
            if best_delta > 0:
                # Perform swap
                # Update counts
                counts[r::m] -= 1
                counts[best_r_new::m] += 1
                classes[i] = [m, best_r_new]
                improved = True
                # Restart loop? Or continue?
                # Since counts changed, previous calculations for other i are stale.
                # However, the change is local.
                # For safety and simplicity, we can just let the while loop restart.
                # But we are in a for loop.
                # Let's break out of for loop to restart the while loop.
                break
                
    return classes
