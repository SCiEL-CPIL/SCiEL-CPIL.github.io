import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    num_aps = len(aps)
    # Convert to numpy array for vectorized ops
    A = np.array(aps, dtype=np.int32)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    # other_indices[ap_idx][pos_idx] = tuple of other 3 indices
    other_indices = []
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            idx_to_aps[idx].append(i)
        
        # Precompute others for this AP
        ap_others = []
        for pos_idx in range(4):
            others = tuple(ap[j] for j in range(4) if j != pos_idx)
            ap_others.append(others)
        other_indices.append(ap_others)
    
    # Precompute pos_in_ap mapping: pos_in_ap[ap_idx][idx] = position of idx in ap
    # This is a 2D list or array. Since num_aps is ~1000, and n=128, 
    # a list of dicts or a 2D array works. 
    # Let's use a 2D numpy array for speed in lookup? 
    # pos_in_ap[ap_idx][idx] is accessed in the inner loop.
    # Using a list of lists is fine.
    pos_in_ap = [[-1]*n for _ in range(num_aps)]
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            pos_in_ap[i][idx] = pos_idx

    def count_bad_colors_vectorized(colors):
        """Count APs with <= 2 distinct colors using numpy."""
        c = colors[A]
        # Sort each row
        sorted_c = np.sort(c, axis=1)
        diffs = np.diff(sorted_c, axis=1)
        # Count zeros in diffs
        # If there are >= 2 zeros, it means at least 3 elements are equal? 
        # Wait. 
        # [a,a,a,b] -> sorted [a,a,a,b] -> diffs [0,0,1] -> 2 zeros. Distinct=2. Bad.
        # [a,a,b,b] -> sorted [a,a,b,b] -> diffs [0,1,0] -> 2 zeros. Distinct=2. Bad.
        # [a,a,a,a] -> sorted [a,a,a,a] -> diffs [0,0,0] -> 3 zeros. Distinct=1. Bad.
        # [a,a,b,c] -> sorted [a,a,b,c] -> diffs [0,1,1] -> 1 zero. Distinct=3. Good.
        # [a,b,c,d] -> sorted [a,b,c,d] -> diffs [1,1,1] -> 0 zeros. Distinct=4. Good.
        # So condition for Bad is (diffs == 0).sum(axis=1) >= 2.
        zero_counts = (diffs == 0).sum(axis=1)
        return np.sum(zero_counts >= 2)

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5...
    for k in range(3, 15):
        if time.time() > deadline - 15:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts
        # Each restart should run for a fixed time or until convergence
        restart_time_limit = 0.5  # Reduce restart time to allow more restarts
        
        while time.time() < deadline - 10:
            restart_start = time.time()
            
            # Random initialization
            colors = np.random.randint(0, k, n, dtype=np.int32)
            current_cost = count_bad_colors_vectorized(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            # Simulated Annealing
            T = 1.0
            # Run until time limit or convergence
            while time.time() - restart_start < restart_time_limit:
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                # Pick random index
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost
                delta = 0
                for ap_idx in idx_to_aps[i]:
                    pos = pos_in_ap[ap_idx][i]
                    others = other_indices[ap_idx][pos]
                    
                    # Get colors of others
                    c0 = colors[others[0]]
                    c1 = colors[others[1]]
                    c2 = colors[others[2]]
                    
                    # Check distinct count of {old_color, c0, c1, c2}
                    # and {new_color, c0, c1, c2}
                    
                    # Helper to check if set size <= 2
                    # Size <= 2 if:
                    # 1. All 4 same
                    # 2. 3 same, 1 diff
                    # 3. 2 same, 2 same (different pair)
                    
                    # Faster check without set creation:
                    # Let's define a small inline check
                    def is_bad_4(a, b, c, d):
                        # a, b, c, d are ints
                        # Count distinct
                        # If a==b and a==c and a==d: True
                        # If a==b and a==c and d!=a: True (3 a's, 1 d)
                        # If a==b and c==d and a!=c: True (2 a's, 2 c's)
                        # If a==b and c!=a and d!=a and c!=d: False (a,a,c,d -> 3 distinct)
                        # ... this is getting complex. 
                        # Using set is actually quite fast for 4 elements in CPython.
                        # But let's try to avoid it if possible.
                        # Alternative:
                        # distinct_count = 1
                        # if b != a: distinct_count += 1
                        # if c != a and c != b: distinct_count += 1
                        # if d != a and d != b and d != c: distinct_count += 1
                        # return distinct_count <= 2
                        
                        if a == b and a == c and a == d: return True
                        if a == b:
                            if a == c: return True # a,a,a,d
                            if a == d: return True # a,a,d,a
                            if c == d: return True # a,a,c,c
                            return False # a,a,c,d
                        if a == c:
                            if a == d: return True # a,d,a,a
                            if b == d: return True # a,b,a,a -> wait, a==c. If b==d, then a,b,a,b. Distinct 2. True.
                            # If a==c, b!=a. 
                            # If d==a: a,b,a,a (True)
                            # If d==b: a,b,a,b (True)
                            # If d!=a and d!=b: a,b,a,d (False)
                            return b == d
                        if a == d:
                            # a,d,a? No a==d. 
                            # b!=a, c!=a.
                            # If b==c: a,b,b,a (True)
                            # If b!=c: a,b,c,a (False)
                            return b == c
                        # a is unique so far.
                        # b, c, d must be checked.
                        if b == c:
                            # a,b,b,d. If d==b: a,b,b,b (True). If d!=b: a,b,b,d (False)
                            return d == b
                        if b == d:
                            # a,b,c,b. If c==b: a,b,b,b (True). If c!=b: a,b,c,b (False)
                            return c == b
                        if c == d:
                            # a,b,c,c. If b==c: handled. If b!=c: a,b,c,c (False)
                            return False
                        return False

                    # This inline function is too heavy. 
                    # Let's stick to set but optimize the loop.
                    # Actually, pre-computing the "others" colors into a tuple and using set is fine.
                    
                    set_others = {c0, c1, c2}
                    
                    old_bad = 1 if (old_color in set_others or len(set_others) < 2) else 0
                    # Wait, if old_color is in set_others, size is len(set_others).
                    # If old_color not in set_others, size is len(set_others) + 1.
                    # Bad if size <= 2.
                    
                    size_others = len(set_others)
                    old_size = size_others + (0 if old_color in set_others else 1)
                    new_size = size_others + (0 if new_color in set_others else 1)
                    
                    old_bad = 1 if old_size <= 2 else 0
                    new_bad = 1 if new_size <= 2 else 0
                    delta += new_bad - old_bad
                    
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                        
                # Faster cool down
                T *= 0.99
                if T < 0.005:
                    T = 0.005

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            # Found a valid solution with k colors
            # Verify with vectorized check just in case
            if count_bad_colors_vectorized(best_k_solution) == 0:
                return list(best_k_solution)
            
    # Fallback: Greedy construction similar to seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
