def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Scale factor for integer approximation
    # 10^7 is 10 million. 1/28 ~ 0.035. 1/48 ~ 0.02.
    # Deficit resolution ~ 10^-7.
    # If the true minimum deficit is smaller, we might miss it.
    # But likely the optimal deficit is not that small.
    # Let's try K = 10**7 first. If time allows, we can increase.
    K = 10**7
    
    # Pre-compute scaled values
    vals = [round(1.0 / d * K) for d in denominators]
    
    # Beam search
    # beam: list of (sum_int, bitmask)
    # We want to maximize sum_int < K
    beam = [(0, 0)]
    beam_width = 10000  # Start with 10k
    
    for i in range(n):
        val = vals[i]
        new_beam = []
        # Option: don't take i
        # We can just add all current states, and then the new ones
        # To avoid duplicates and keep it efficient, we can use a dict
        # But dict overhead is high. Let's use a list and sort/prune.
        
        # Actually, we can just extend the list with new states and prune.
        # The size of beam is W. New states are W. Total 2W.
        # Sorting 2W elements is O(W log W).
        
        # Add old states (not taking i)
        # Note: We must ensure we don't use the updated beam for the "take" part.
        # So we iterate over the current beam.
        
        candidates = list(beam) # Not taking i
        
        for s, m in beam:
            new_s = s + val
            if new_s < K:
                new_m = m | (1 << i)
                candidates.append((new_s, new_m))
        
        # Prune: keep top W sums
        # Sort by sum descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top W unique sums?
        # If multiple masks give the same sum, we only need one.
        # Let's keep a set of seen sums to deduplicate.
        seen = set()
        new_beam = []
        for s, m in candidates:
            if s in seen:
                continue
            seen.add(s)
            new_beam.append((s, m))
            if len(new_beam) >= beam_width:
                break
        
        beam = new_beam
        
        # Check time
        if time.time() > deadline:
            break
            
    # Find best
    if not beam:
        return []
        
    best_s, best_m = max(beam, key=lambda x: x[0])
    
    # Reconstruct indices
    indices = []
    for i in range(n):
        if best_m & (1 << i):
            indices.append(i)
            
    # Verify with Fraction
    from fractions import Fraction
    total = Fraction(0)
    for i in indices:
        total += Fraction(1, denominators[i])
        
    if total >= 1:
        # This shouldn't happen if K is large enough and we enforced < K
        # But rounding might have pushed it over.
        # Try to fix by removing the smallest reciprocal (largest denominator)
        # Or just return a safe fallback?
        # Let's try to remove one element to make it < 1.
        # Find the index with the largest denominator (smallest fraction)
        # Actually, removing any element reduces the sum.
        # We want to keep the sum as close to 1 as possible.
        # So remove the element with the smallest value (largest d).
        if indices:
            # Find index with max denominator
            max_d_idx = max(indices, key=lambda i: denominators[i])
            indices.remove(max_d_idx)
            # Recalculate
            total = Fraction(0)
            for i in indices:
                total += Fraction(1, denominators[i])
            # If still >= 1, remove more?
            # Unlikely.
            
    return indices
