def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute K4s and which edges they contain
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    e1 = edge_index[(a, b)]
                    e2 = edge_index[(a, c)]
                    e3 = edge_index[(a, d)]
                    e4 = edge_index[(b, c)]
                    e5 = edge_index[(b, d)]
                    e6 = edge_index[(c, d)]
                    k4s.append((e1, e2, e3, e4, e5, e6))
                    
    num_k4s = len(k4s)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k_idx, k4 in enumerate(k4s):
        for e in k4:
            edge_to_k4s[e].append(k_idx)
            
    # State: colours array, distinct_count_per_k4 array, violation_count
    # distinct_count_per_k4[k] = number of distinct colours in K4 k
    
    # Initialize with random 14 colours
    colours = np.array([random.randint(0, 13) for _ in range(num_edges)], dtype=np.int32)
    
    # Calculate initial state
    distinct_counts = np.zeros(num_k4s, dtype=np.int32)
    for k_idx, k4 in enumerate(k4s):
        c_set = set(colours[k4])
        distinct_counts[k_idx] = len(c_set)
        
    violations = np.sum(distinct_counts < 5)
    num_distinct = len(np.unique(colours))
    
    best_colours = colours.copy()
    best_violations = violations
    best_distinct = num_distinct
    best_score = violations * 10000 + num_distinct
    
    if best_violations == 0 and best_distinct <= 14:
        return best_colours.tolist()
        
    T = 50.0
    cooling = 0.9995
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        idx = random.randint(0, num_edges - 1)
        old_c = colours[idx]
        new_c = random.randint(0, 13)
        
        if old_c == new_c:
            continue
            
        # Calculate change in violations
        # Only K4s containing idx are affected
        affected_k4s = edge_to_k4s[idx]
        
        old_viol_delta = 0
        new_viol_delta = 0
        
        # We need to update distinct_counts for these K4s
        # To do this efficiently, we need to know the count of each colour in the K4.
        # This is expensive if done from scratch every time.
        # However, 91 K4s * 6 edges is small.
        
        for k_idx in affected_k4s:
            k4 = k4s[k_idx]
            
            # Count colours in K4 before change
            # We can maintain a counter for each K4? 
            # Memory: 1820 K4s * 14 colours = 25k ints. Feasible.
            
            # Let's just recalculate for the affected K4s.
            # Get current colours of the 6 edges
            c_vals = colours[k4]
            
            # Remove old_c from consideration for idx, add new_c
            # Actually, just recalculate distinct count for this K4
            # But we need to update the global `distinct_counts` array
            
            # Count distinct in old state
            old_distinct = distinct_counts[k_idx]
            
            # Update colours array temporarily? No, just calculate new distinct.
            # c_vals has old_c at position of idx.
            # Replace old_c with new_c in a copy?
            # Or just count frequencies.
            
            # Frequency of colours in this K4
            # Since K4 size is 6, we can just use a small dict or array.
            freq = [0]*14
            for e in k4:
                freq[colours[e]] += 1
            
            # Apply change
            freq[old_c] -= 1
            freq[new_c] += 1
            
            new_distinct = sum(1 for f in freq if f > 0)
            
            # Update violation counts
            if old_distinct < 5 and new_distinct >= 5:
                old_viol_delta -= 1 # Violation removed
            elif old_distinct >= 5 and new_distinct < 5:
                old_viol_delta += 1 # Violation added
            # If both <5 or both >=5, no change in violation count for this K4
            
            # Update distinct_counts
            distinct_counts[k_idx] = new_distinct
            
        new_violations = violations + old_viol_delta
        new_num_distinct = len(np.unique(colours)) # This is expensive, O(E log E)
        # We can track num_distinct incrementally too.
        # But np.unique is fast for 120 elements.
        
        new_score = new_violations * 10000 + new_num_distinct
        current_score = violations * 10000 + num_distinct
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            colours[idx] = new_c
            violations = new_violations
            num_distinct = new_num_distinct
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_violations = new_violations
                best_distinct = new_num_distinct
                best_colours = colours.copy()
                
                if best_violations == 0 and best_distinct <= 14:
                    # We found a valid 14-colouring.
                    # We can break early or continue to find 13.
                    # Given bounds, 13 is unlikely but possible.
                    # Let's run for a bit more to see if we can improve.
                    pass
        else:
            # Reject
            # Revert distinct_counts for affected K4s
            for k_idx in affected_k4s:
                k4 = k4s[k_idx]
                freq = [0]*14
                for e in k4:
                    freq[colours[e]] += 1
                distinct_counts[k_idx] = sum(1 for f in freq if f > 0)
            
        T *= cooling
        
    if best_violations == 0:
        return best_colours.tolist()
        
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
