import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 85.0

    # 1. Generate 5-smooth numbers up to 128
    smooth_nums = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_nums.append(x)
    
    is_smooth = set(smooth_nums)
    m = len(smooth_nums)
    
    # 2. Precompute antichain conflict masks for each smooth number
    # conflict_mask[i] is a bitmask where bit j is set if smooth_nums[i] divides smooth_nums[j] or vice versa
    # (i.e., they cannot both be in an antichain)
    conflict_mask = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            a, b = smooth_nums[i], smooth_nums[j]
            if b % a == 0 or a % b == 0:
                conflict_mask[i] |= (1 << j)
    
    # 3. Precompute valid k-term antichain sums for k=1..5
    # For each k and target n, store the minimum-term antichain found
    # We'll do this lazily per target to save time
    
    answers = []
    
    def find_antichain_sum(n, max_k=5):
        """Find a valid antichain sum of up to max_k smooth numbers for target n."""
        # Try 1 term
        if n in is_smooth:
            return [n]
        
        # Try 2 terms
        for i in range(m):
            a = smooth_nums[i]
            if a >= n:
                break
            b = n - a
            if b <= a:
                continue
            if b in is_smooth:
                j = smooth_nums.index(b) if b in smooth_nums else -1
                # Find index of b
                j = -1
                for jj in range(m):
                    if smooth_nums[jj] == b:
                        j = jj
                        break
                if j >= 0 and i != j:
                    # Check antichain: no conflict between i and j
                    if not (conflict_mask[i] & (1 << j)):
                        return [a, b]
        
        # Try 3 terms
        for i in range(m):
            a = smooth_nums[i]
            if 3 * a >= n:
                break
            for j in range(i + 1, m):
                b = smooth_nums[j]
                if b <= a:
                    continue
                if a + 2 * b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c not in is_smooth:
                    continue
                k = -1
                for kk in range(m):
                    if smooth_nums[kk] == c:
                        k = kk
                        break
                if k < 0 or k <= j:
                    continue
                # Check all pairs are antichain
                if (conflict_mask[i] & (1 << j)) or (conflict_mask[i] & (1 << k)) or (conflict_mask[j] & (1 << k)):
                    continue
                return [a, b, c]
        
        # Try 4 terms
        for i in range(m):
            a = smooth_nums[i]
            if 4 * a >= n:
                break
            for j in range(i + 1, m):
                b = smooth_nums[j]
                if b <= a:
                    continue
                if a + 3 * b >= n:
                    break
                for k in range(j + 1, m):
                    c = smooth_nums[k]
                    if c <= b:
                        continue
                    if a + b + 2 * c >= n:
                        break
                    d = n - a - b - c
                    if d <= c:
                        continue
                    if d not in is_smooth:
                        continue
                    l = -1
                    for ll in range(m):
                        if smooth_nums[ll] == d:
                            l = ll
                            break
                    if l < 0 or l <= k:
                        continue
                    # Check all pairs
                    if ((conflict_mask[i] & (1 << j)) or (conflict_mask[i] & (1 << k)) or 
                        (conflict_mask[i] & (1 << l)) or (conflict_mask[j] & (1 << k)) or 
                        (conflict_mask[j] & (1 << l)) or (conflict_mask[k] & (1 << l))):
                        continue
                    return [a, b, c, d]
        
        # Try 5 terms
        for i in range(m):
            a = smooth_nums[i]
            if 5 * a >= n:
                break
            for j in range(i + 1, m):
                b = smooth_nums[j]
                if b <= a:
                    continue
                if a + 4 * b >= n:
                    break
                for k in range(j + 1, m):
                    c = smooth_nums[k]
                    if c <= b:
                        continue
                    if a + b + 3 * c >= n:
                        break
                    for l in range(k + 1, m):
                        d = smooth_nums[l]
                        if d <= c:
                            continue
                        if a + b + c + 2 * d >= n:
                            break
                        e = n - a - b - c - d
                        if e <= d:
                            continue
                        if e not in is_smooth:
                            continue
                        p = -1
                        for pp in range(m):
                            if smooth_nums[pp] == e:
                                p = pp
                                break
                        if p < 0 or p <= l:
                            continue
                        # Check all 10 pairs
                        pairs = [(i,j), (i,k), (i,l), (i,p), (j,k), (j,l), (j,p), (k,l), (k,p), (l,p)]
                        valid = True
                        for u, v in pairs:
                            if conflict_mask[u] & (1 << v):
                                valid = False
                                break
                        if valid:
                            return [a, b, c, d, e]
        
        # Fallback: backtracking search for any k
        return _find_any_antichain_sum(n, smooth_nums, conflict_mask)
    
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            # Fallback: append a known valid solution or empty if stuck
            if n in is_smooth:
                answers.append([n])
            else:
                break
            continue

        found = find_antichain_sum(n)
        
        if found is None:
            # Absolute fallback
            found = [n]
        
        answers.append(found)

    return answers

def _find_any_antichain_sum(n, smooth_nums, conflict_mask):
    # Backtracking search for any valid antichain sum
    m = len(smooth_nums)
    
    def backtrack(rem, start_idx, current_terms, current_mask):
        if rem == 0:
            if current_terms:
                return list(current_terms)
            return None
        if rem < 0:
            return None
        
        for i in range(start_idx, m):
            val = smooth_nums[i]
            if val > rem:
                break
            
            # Check if adding val keeps antichain property using conflict mask
            if current_mask & (1 << i):
                continue
            
            new_mask = current_mask | conflict_mask[i] | (1 << i)
            
            current_terms.append(val)
            res = backtrack(rem - val, i + 1, current_terms, new_mask)
            if res:
                return res
            current_terms.pop()
        
        return None

    return backtrack(n, 0, [], 0)

# EVOLVE-BLOCK-END
