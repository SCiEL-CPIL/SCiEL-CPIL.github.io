import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1
    
    best_A = None
    best_max = float('inf')
    
    def compute_counts(ind):
        """Compute unordered pair-sum representation counts for [T_min, T_max]."""
        conv = np.convolve(ind, ind, mode='full')
        counts = np.empty(R, dtype=np.int32)
        for i, s in enumerate(range(T_min, T_max + 1)):
            c_ord = conv[s]
            if s % 2 == 0:
                half = s // 2
                n_eq = 1 if (half < N_max and ind[half] == 1) else 0
            else:
                n_eq = 0
            counts[i] = (c_ord + n_eq) // 2
        return counts

    def max_count(counts):
        return int(np.max(counts))

    def is_valid(counts):
        return bool(np.all(counts > 0))

    def repair(A_set, ind, counts):
        """Repair gaps by adding elements with best gain."""
        for _ in range(200):
            if is_valid(counts):
                break
            gaps = np.where(counts == 0)[0]
            if len(gaps) == 0:
                break
            best_y = None
            best_gain = -1
            for y in range(N_max):
                if ind[y] == 1:
                    continue
                gain = 0
                for gi in gaps:
                    s = T_min + gi
                    a = s - y
                    if 0 <= a < N_max and (ind[a] == 1 or a == y):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_y = y
            if best_y is not None:
                A_set.add(best_y)
                ind[best_y] = 1
                counts = compute_counts(ind)
            else:
                break
        return is_valid(counts)

    def local_search(A_initial, time_budget):
        nonlocal best_A, best_max
        
        A_set = set(A_initial)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
        counts = compute_counts(ind)
        
        if not is_valid(counts):
            if not repair(A_set, ind, counts):
                return
        
        current_max = max_count(counts)
        if current_max < best_max:
            best_max = current_max
            best_A = set(A_set)
        
        A_cur = set(A_set)
        ind_cur = ind.copy()
        counts_cur = counts.copy()
        max_cur = current_max
        
        # Precompute list of elements in A for fast iteration
        A_list = list(A_cur)
        
        while time.time() < time_budget:
            if len(A_cur) == 0:
                break
            
            # Pick random x to remove and y to add
            x = random.choice(A_list)
            y = random.randint(0, N_max - 1)
            
            if y == x or ind_cur[y] == 1:
                continue
            
            # Perform swap x -> y
            # 1. Remove x
            # Update counts
            # Pairs involving x: {x, a} for a in A_cur, a != x, and {x, x}
            # Note: A_cur still contains x at this point in logic, but we are simulating removal.
            # Let's update counts_cur in place, then revert if rejected.
            
            # Store old values to revert if needed? 
            # Better: copy counts_cur? 
            # Copying an array of size 129 is cheap.
            counts_new = counts_cur.copy()
            
            # Remove x
            # {x, x}
            s_xx = 2 * x
            if T_min <= s_xx <= T_max:
                idx = s_xx - T_min
                counts_new[idx] -= 1
            
            # {x, a} for a in A_cur, a != x
            for a in A_list:
                if a == x:
                    continue
                s_xa = x + a
                if T_min <= s_xa <= T_max:
                    idx = s_xa - T_min
                    counts_new[idx] -= 1
            
            # Add y
            # {y, y}
            s_yy = 2 * y
            if T_min <= s_yy <= T_max:
                idx = s_yy - T_min
                counts_new[idx] += 1
            
            # {y, a} for a in A_cur, a != x
            for a in A_list:
                if a == x:
                    continue
                s_ya = y + a
                if T_min <= s_ya <= T_max:
                    idx = s_ya - T_min
                    counts_new[idx] += 1
            
            # Check validity
            if not is_valid(counts_new):
                continue
            
            m_new = max_count(counts_new)
            
            if m_new < max_cur:
                # Accept
                A_cur.remove(x)
                A_cur.add(y)
                A_list.remove(x)
                A_list.append(y)
                ind_cur[x] = 0
                ind_cur[y] = 1
                counts_cur = counts_new
                max_cur = m_new
                
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
                    
            elif m_new == max_cur:
                # Accept with some probability to explore plateaus
                # Or accept if it reduces the number of peaks
                old_peaks = np.sum(counts_cur == max_cur)
                new_peaks = np.sum(counts_new == max_cur)
                
                if new_peaks < old_peaks or random.random() < 0.1:
                    A_cur.remove(x)
                    A_cur.add(y)
                    A_list.remove(x)
                    A_list.append(y)
                    ind_cur[x] = 0
                    ind_cur[y] = 1
                    counts_cur = counts_new
                    # max_cur unchanged
            else:
                # Reject
                pass

    # Generate diverse initial sets
    starts = []
    
    # 1. Random sets of various sizes
    for k in range(15, 40):
        for _ in range(2):
            random.seed(k * 100)
            starts.append(random.sample(range(N_max), k))
            
    # 2. Structured sets
    starts.append(list(range(64, 129))) # Upper half
    starts.append(list(range(0, 129, 2))) # Evens
    starts.append(list(range(1, 129, 2))) # Odds
    starts.append(list(range(32, 129, 2))) # Upper evens
    
    # Shuffle starts
    random.shuffle(starts)
    
    for i, s in enumerate(starts):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate time based on index
        # Give more time to early, diverse starts
        time_budget = min(remaining * 0.8, 10.0)
        
        local_search(s, time_budget)
        
        # If we found a very good solution (e.g., max <= 2 or 3), we might stop early?
        # The theoretical minimum is likely around 2 or 3.
        if best_max <= 2:
            break

    if best_A is None:
        # Fallback
        return list(range(64, 129))
        
    return list(best_A)
