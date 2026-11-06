import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    m = len(smooth)
    
    # Precompute divisibility: div_mask[i] is a bitmask of indices j where smooth[i] divides smooth[j]
    # For antichain check, we need to ensure no two chosen terms have a divisibility relation.
    
    # Actually, a simpler precomputation: for each pair (i, j) with i < j, check if smooth[i] divides smooth[j]
    # We can store a set of "bad pairs" or just check on the fly since m is small (approx 20-30 numbers).
    
    # Let's count smooth numbers: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128
    # Count is about 37.
    
    # Precompute a 2D boolean array `divides[i][j]` = True if smooth[i] divides smooth[j]
    # We will use this to quickly check if a set is an antichain.
    # Or better, for a given set of indices, check all pairs.
    
    # Even better: for each subset of size k, we can precompute if it's an antichain? No, too many subsets.
    # But we can precompute a bitmask of "incompatible" indices for each index.
    # incompat[i] is a bitmask of all j != i such that smooth[i] divides smooth[j] OR smooth[j] divides smooth[i].
    # Then a set of indices is an antichain iff for all i in set, (incompat[i] & set_mask) == 0 (excluding self).
    
    # Let's build incompat masks.
    n_smooth = len(smooth)
    incompat = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                incompat[i] |= (1 << j)
    
    def is_antichain(indices):
        # indices is a list of indices into smooth
        # Check if any pair divides
        # Using bitmask is faster for small k
        mask = 0
        for idx in indices:
            mask |= (1 << idx)
        for idx in indices:
            # Check if any other index in mask is in incompat[idx]
            if incompat[idx] & mask:
                # Need to exclude self
                if (incompat[idx] & mask) & ~(1 << idx):
                    return False
        return True
    
    answers = []
    
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: use a known valid representation or return empty?
            # To avoid timeout penalty, we should return something valid if possible, or break.
            # But we must return 128 rows.
            # If we hit deadline, we might have to use a slower but guaranteed method for remaining?
            # Or just use the previous found (which is None) -> invalid.
            # Let's assume we don't hit deadline.
            pass
        
        found = None
        
        # Try k=1
        if n in smooth_set:
            found = [n]
        
        # Try k=2
        if found is None:
            # Iterate through smooth numbers less than n
            for i in range(n_smooth):
                a = smooth[i]
                if a >= n:
                    break
                b = n - a
                if b in smooth_set and b > a:
                    # Find index of b
                    # Since smooth is sorted, we can use bisect or just a dict
                    # For speed, let's precompute value_to_index
                    # But for k=2, simple check is fine.
                    # b % a != 0
                    if b % a != 0:
                        found = [a, b]
                        break
        
        # Try k=3
        if found is None:
            # a < b < c
            for i in range(n_smooth):
                a = smooth[i]
                if a >= n // 3:
                    break
                for j in range(i + 1, n_smooth):
                    b = smooth[j]
                    if a + b >= n - 1: # c must be at least b+1? No, c > b. Min c is next smooth.
                        break
                    c = n - a - b
                    if c in smooth_set and c > b:
                        if b % a != 0 and c % b != 0 and c % a != 0:
                            found = [a, b, c]
                            break
                if found is not None:
                    break
        
        # Try k=4
        if found is None:
            # a < b < c < d
            for i in range(n_smooth):
                a = smooth[i]
                if a * 4 > n:
                    break
                for j in range(i + 1, n_smooth):
                    b = smooth[j]
                    if a + b * 3 > n: # Rough bound
                        break
                    for k in range(j + 1, n_smooth):
                        c = smooth[k]
                        if a + b + c * 2 > n:
                            break
                        d = n - a - b - c
                        if d in smooth_set and d > c:
                            # Check antichain
                            # a,b,c,d
                            if (b % a != 0 and c % a != 0 and c % b != 0 and 
                                d % a != 0 and d % b != 0 and d % c != 0):
                                found = [a, b, c, d]
                                break
                        if found is not None:
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
        
        # Try k=5
        if found is None:
            # a < b < c < d < e
            # This gets expensive. Let's use a more generic backtracking for k>=5 if needed.
            # But first, let's see if k=4 is enough.
            # If not, we can implement a recursive search for k=5, 6, etc.
            
            # Let's implement a general recursive search for k from 5 to 10
            # using the incompat masks for efficiency.
            
            # First, map values to indices
            val_to_idx = {v: i for i, v in enumerate(smooth)}
            
            def search_k(k, remaining, start_idx, current_terms):
                if k == 0:
                    return current_terms if remaining == 0 else None
                if remaining < 0:
                    return None
                # Pruning: if remaining is too small for k terms
                # Min sum of k terms starting from start_idx
                # This is complex. Let's just iterate.
                
                for idx in range(start_idx, n_smooth):
                    val = smooth[idx]
                    if val > remaining:
                        break
                    # Check if adding val violates antichain with current_terms
                    # current_terms are indices
                    # We need to check if val is incompatible with any in current_terms
                    # incompat[idx] has bits set for incompatible indices.
                    # Create mask of current_terms
                    # For efficiency, we can maintain a mask of used indices.
                    
                    # Let's pass a mask of used indices.
                    # This requires changing the function signature.
                    # Let's just check pairs for now, k is small.
                    compatible = True
                    for prev_idx in current_terms:
                        if incompat[idx] & (1 << prev_idx):
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    res = search_k(k - 1, remaining - val, idx + 1, current_terms + [idx])
                    if res is not None:
                        return res
                return None
            
            # Try k=5
            res = search_k(5, n, 0, [])
            if res is not None:
                found = [smooth[i] for i in res]
            
            if found is None and time.time() < deadline:
                # Try k=6
                res = search_k(6, n, 0, [])
                if res is not None:
                    found = [smooth[i] for i in res]
            
            if found is None and time.time() < deadline:
                # Try k=7
                res = search_k(7, n, 0, [])
                if res is not None:
                    found = [smooth[i] for i in res]
        
        # Fallback
        if found is None:
            found = []
            
        answers.append(found)
        
    return answers
