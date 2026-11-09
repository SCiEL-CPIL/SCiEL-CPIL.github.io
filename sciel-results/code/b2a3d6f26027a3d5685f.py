def construct():
    p = 1009
    
    # 1. Generate pairs (a, b) with a < b and a*b == 1 mod p
    # Exclude self-inverses 1 and p-1
    pairs = []
    for x in range(2, p - 1):
        y = pow(x, p - 2, p)
        if x < y:
            pairs.append((x, y))
        # If x > y, the pair (y, x) will be added when x=y's inverse is processed?
        # No, we iterate x from 2 to p-2.
        # If x > y, then y < x.
        # When we iterate to y, we will check y < x (which is y < y's inverse? No).
        # Let's just add (min(x,y), max(x,y)) and ensure no duplicates.
        # Since x != y for x in 2..p-2, we can just add (min, max).
        # But we might add duplicates if we are not careful?
        # No, each x has a unique inverse.
        # If we add (min(x,y), max(x,y)) for all x, we will add each pair twice?
        # No. If x=2, y=505. Add (2, 505).
        # If x=505, y=2. Add (2, 505).
        # So we will have duplicates.
        # We can use a set or just filter x < y.
        
    # Let's restart pair generation to be clean
    pairs = []
    for x in range(2, p - 1):
        y = pow(x, p - 2, p)
        if x < y:
            pairs.append((x, y))
            
    # Sort by a descending
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Extract a's and b's
    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    
    # Compute prefix minimums of b_vals
    # PMin[i] = min(b_vals[0..i])
    if not pairs:
        # This shouldn't happen for p>3
        return [2, 1]
        
    pmin = [0] * len(pairs)
    pmin[0] = b_vals[0]
    for i in range(1, len(pairs)):
        pmin[i] = min(pmin[i-1], b_vals[i])
        
    best_len = 0
    best_start = 2
    
    # Iterate s from 2 to p-2
    # We need to find for each s, the index k such that a_vals[k] >= s
    # Since a_vals is sorted descending, we can binary search.
    # We want the LAST index where a_vals[i] >= s.
    # If no such index, k = -1.
    
    import bisect
    
    # bisect works on ascending.
    # Let's create a list of negative a_vals or just use custom binary search.
    # Or sort ascending and adjust.
    # Let's stick to descending and write a small binary search.
    
    def find_last_ge(val, arr):
        # arr is descending
        # Find last index i such that arr[i] >= val
        lo = 0
        hi = len(arr) - 1
        ans = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] >= val:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans
        
    for s in range(2, p - 1):
        k = find_last_ge(s, a_vals)
        
        if k == -1:
            # No pairs with a >= s
            # Constraint from pairs is empty
            m_s = p # effectively infinity, limited by p-2 later
        else:
            m_s = pmin[k]
            
        # Max e is min(m_s - 1, p - 2)
        e_max = min(m_s - 1, p - 2)
        
        if e_max >= s:
            current_len = e_max - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
                
    return [best_start, best_len]
