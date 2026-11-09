import numpy as np

def _can_add_fast(x, selected):
    m = len(selected)
    if m < 3:
        return True
    
    sel = np.array(selected, dtype=np.int64)
    
    # 1. Check if x divides sum of any 3 in selected
    if m >= 3:
        # Generate all pair sums
        # Pairs (i, j) with i < j
        # We need sums of 3, so we can just iterate triples?
        # Or: sum3 = u + v + w.
        # Is there a pair sum s = u+v such that (s+w) % x == 0?
        # For each w in selected, check if there is a pair sum s from selected\{w} such that (s+w) % x == 0.
        # This is still complex.
        
        # Simple vectorized triple check:
        # Get all combinations of 3 indices
        # This is hard to vectorize easily without itertools.
        # Given m <= 48, itertools.combinations is fine.
        # The modulo check is the slow part.
        
        sums = []
        for c in combinations(range(m), 3):
            sums.append(sel[c[0]] + sel[c[1]] + sel[c[2]])
        sums_arr = np.array(sums)
        if np.any(sums_arr % x == 0):
            return False
            
    # 2. Check if any a in selected divides sum of 3 from (selected \ {a} + {x})
    # Triples involving x: x + u + v, where u, v in selected \ {a}
    # Triples not involving x: already checked in step 1 (for the original set)
    # Wait, step 1 checked x dividing sums of 3 in selected.
    # Now we check a dividing sums.
    # The sums of 3 in (selected \ {a} + {x}) are:
    # 1. Sums of 3 from selected \ {a} (Subset of sums of 3 from selected)
    #    If a divided any of these, the set was already invalid?
    #    No, the set `selected` is valid. So a does NOT divide any sum of 3 from selected.
    #    Since selected \ {a} is a subset, a does not divide any sum of 3 from selected \ {a}.
    #    So we only need to check triples that include x.
    
    # Triples with x: {x, u, v} where u, v in selected, u != v, u != a, v != a.
    # Sum = x + u + v.
    # We need to check if (x + u + v) % a == 0 for any u, v in selected \ {a}.
    
    for i, a in enumerate(selected):
        # Get indices of selected excluding i
        mask = np.ones(m, dtype=bool)
        mask[i] = False
        others = sel[mask]
        
        if len(others) < 2:
            continue
            
        # Generate all pair sums of others
        # We need to check if (x + pair_sum) % a == 0
        # Pair sums of others:
        # Can vectorize?
        # others is array of size m-1.
        # We need all u+v for u<v in others.
        
        # Efficient way:
        # For each u in others, check if there is v in others (v!=u) such that (x+u+v)%a==0.
        # This is O(m^2).
        
        # Let's just use itertools for pairs, but check modulo with numpy?
        # Or just python loop. m is small.
        
        # Optimization:
        # We can precompute pair sums for the whole selected set?
        # But we need to exclude a.
        
        # Let's try a simple loop first, it might be fast enough.
        found = False
        o_len = len(others)
        for j in range(o_len):
            u = others[j]
            for k in range(j+1, o_len):
                v = others[k]
                if (x + u + v) % a == 0:
                    found = True
                    break
            if found:
                break
        if found:
            return False
            
    return True
