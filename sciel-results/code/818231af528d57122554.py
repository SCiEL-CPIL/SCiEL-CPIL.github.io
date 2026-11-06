import bisect
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for the 90s hard limit

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves
    n1 = n // 2
    n2 = n - n1
    
    indices1 = list(range(n1))
    indices2 = list(range(n1, n))
    
    # Precompute float and Fraction values
    vals1 = [1.0 / denominators[i] for i in indices1]
    vals2 = [1.0 / denominators[i] for i in indices2]
    
    frac1 = [Fraction(1, denominators[i]) for i in indices1]
    frac2 = [Fraction(1, denominators[i]) for i in indices2]
    
    # Generate subset sums for half 1: (float_sum, mask)
    sums1 = [(0.0, 0)]
    for i in range(n1):
        v = vals1[i]
        # Use list comprehension or loop. Loop is clearer for time checks.
        new_sums = []
        # Pre-allocate if possible? No, size doubles.
        for s, m in sums1:
            new_sums.append((s, m))
            new_sums.append((s + v, m | (1 << i)))
        sums1 = new_sums
        
    if time.time() > deadline - 20:
        # Fallback if generation is too slow (unlikely for N=23)
        return _greedy_fallback()

    # Generate subset sums for half 2: (float_sum, mask)
    sums2 = [(0.0, 0)]
    for i in range(n2):
        v = vals2[i]
        new_sums = []
        for s, m in sums2:
            new_sums.append((s, m))
            new_sums.append((s + v, m | (1 << i)))
        sums2 = new_sums
        
    # Sort half 2 by float sum
    sums2.sort(key=lambda x: x[0])
    
    # Extract sorted floats and masks for binary search
    s2_floats = [s for s, m in sums2]
    s2_masks = [m for s, m in sums2]
    
    best_deficit_float = float('inf')
    best_exact_deficit = None
    best_indices = None
    
    # Iterate through half 1 sums
    for s1, m1 in sums1:
        if time.time() > deadline - 5:
            break
            
        target = 1.0 - s1
        if target <= 0:
            continue
            
        # Find position in s2_floats
        pos = bisect.bisect_left(s2_floats, target)
        
        # Check neighbors around pos
        candidates = set()
        if pos > 0:
            candidates.add(pos - 1)
        if pos < len(s2_floats):
            candidates.add(pos)
        if pos + 1 < len(s2_floats):
            candidates.add(pos + 1)
            
        for j in candidates:
            s2 = s2_floats[j]
            m2 = s2_masks[j]
            
            s_float = s1 + s2
            if s_float >= 1.0:
                continue
                
            def_float = 1.0 - s_float
            if def_float >= best_deficit_float:
                continue
                
            # Compute exact sum
            sum_frac = Fraction(0)
            
            # Sum bits in m1
            mm = m1
            i = 0
            while mm:
                if mm & 1:
                    sum_frac += frac1[i]
                mm >>= 1
                i += 1
                
            # Sum bits in m2
            mm = m2
            i = 0
            while mm:
                if mm & 1:
                    sum_frac += frac2[i]
                mm >>= 1
                i += 1
                
            if sum_frac < 1:
                def_exact = 1 - sum_frac
                if best_exact_deficit is None or def_exact < best_exact_deficit:
                    best_exact_deficit = def_exact
                    best_deficit_float = float(def_exact)
                    
                    # Construct indices
                    inds = []
                    mm = m1
                    i = 0
                    while mm:
                        if mm & 1:
                            inds.append(indices1[i])
                        mm >>= 1
                        i += 1
                    mm = m2
                    i = 0
                    while mm:
                        if mm & 1:
                            inds.append(indices2[i])
                        mm >>= 1
                        i += 1
                    best_indices = inds

    if best_indices is None:
        return _greedy_fallback()
        
    return best_indices

def _greedy_fallback():
    selected, total = [], Fraction()
    for i, d in enumerate(range(2, 49)):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    if not selected:
        return [0]
    return selected
