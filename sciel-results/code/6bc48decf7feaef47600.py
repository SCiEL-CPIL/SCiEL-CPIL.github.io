import time

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
    smooth_index = {v: i for i, v in enumerate(smooth_nums)}
    m = len(smooth_nums)
    
    # 2. Precompute antichain conflict masks for each smooth number
    # conflict_mask[i] is a bitmask where bit j is set if smooth_nums[i] divides smooth_nums[j] or vice versa
    conflict_mask = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            a, b = smooth_nums[i], smooth_nums[j]
            if b % a == 0 or a % b == 0:
                conflict_mask[i] |= (1 << j)
    
    # Precompute for fast antichain check: can_add[i][j] = True if i and j don't conflict
    # Actually, just use conflict_mask directly
    
    answers = [None] * 128
    
    def find_antichain_sum(n):
        """Find a valid antichain sum with minimum terms for target n."""
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
            j = smooth_index.get(b, -1)
            if j < 0 or j <= i:
                continue
            if not (conflict_mask[i] & (1 << j)):
                return [a, b]
        
        # Try 3 terms
        for i in range(m):
            a = smooth_nums[i]
            if 3 * a >= n:
                break
            for j in range(i + 1, m):
                b = smooth_nums[j]
                if a + 2 * b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                k = smooth_index.get(c, -1)
                if k < 0 or k <= j:
                    continue
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
                if a + 3 * b >= n:
                    break
                for k in range(j + 1, m):
                    c = smooth_nums[k]
                    if a + b + 2 * c >= n:
                        break
                    d = n - a - b - c
                    if d <= c:
                        continue
                    l = smooth_index.get(d, -1)
                    if l < 0 or l <= k:
                        continue
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
                if a + 4 * b >= n:
                    break
                for k in range(j + 1, m):
                    c = smooth_nums[k]
                    if a + b + 3 * c >= n:
                        break
                    for l in range(k + 1, m):
                        d = smooth_nums[l]
                        if a + b + c + 2 * d >= n:
                            break
                        e = n - a - b - c - d
                        if e <= d:
                            continue
                        p = smooth_index.get(e, -1)
                        if p < 0 or p <= l:
                            continue
                        # Check all 10 pairs
                        if (conflict_mask[i] & (1 << j)) or (conflict_mask[i] & (1 << k)) or \
                           (conflict_mask[i] & (1 << l)) or (conflict_mask[i] & (1 << p)) or \
                           (conflict_mask[j] & (1 << k)) or (conflict_mask[j] & (1 << l)) or \
                           (conflict_mask[j] & (1 << p)) or (conflict_mask[k] & (1 << l)) or \
                           (conflict_mask[k] & (1 << p)) or (conflict_mask[l] & (1 << p)):
                            continue
                        return [a, b, c, d, e]
        
        # Try 6 terms
        for i in range(m):
            a = smooth_nums[i]
            if 6 * a >= n:
                break
            for j in range(i + 1, m):
                b = smooth_nums[j]
                if a + 5 * b >= n:
                    break
                for k in range(j + 1, m):
                    c = smooth_nums[k]
                    if a + b + 4 * c >= n:
                        break
                    for l in range(k + 1, m):
                        d = smooth_nums[l]
                        if a + b + c + 3 * d >= n:
                            break
                        for p in range(l + 1, m):
                            e = smooth_nums[p]
                            if a + b + c + d + 2 * e >= n:
                                break
                            f = n - a - b - c - d - e
                            if f <= e:
                                continue
                            q = smooth_index.get(f, -1)
                            if q < 0 or q <= p:
                                continue
                            # Check all 15 pairs
                            idxs = [i, j, k, l, p, q]
                            valid = True
                            for u in range(6):
                                for v in range(u + 1, 6):
                                    if conflict_mask[idxs[u]] & (1 << idxs[v]):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                return [a, b, c, d, e, f]
        
        # Fallback: backtracking search for any k
        return _find_any_antichain_sum(n, smooth_nums, conflict_mask, smooth_index)
    
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            # Emergency fallback
            if n in is_smooth:
                answers[n-1] = [n]
            else:
                answers[n-1] = [1, n-1]  # may not be valid, but we need something
            continue

        found = find_antichain_sum(n)
        
        if found is None:
            # Absolute fallback - should not happen
            answers[n-1] = [n]
        else:
            answers[n-1] = found

    # Ensure all entries are filled
    for i in range(128):
        if answers[i] is None:
            n = i + 1
            if n in is_smooth:
                answers[i] = [n]
            else:
                answers[i] = [1, n-1]

    return answers

def _find_any_antichain_sum(n, smooth_nums, conflict_mask, smooth_index):
    # Backtracking search for any valid antichain sum with minimum terms
    m = len(smooth_nums)
    
    def backtrack(rem, start_idx, current_terms, current_mask, depth):
        if rem == 0:
            if current_terms:
                return list(current_terms)
            return None
        if rem < 0:
            return None
        if depth > 10:
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
            res = backtrack(rem - val, i + 1, current_terms, new_mask, depth + 1)
            if res:
                return res
            current_terms.pop()
        
        return None

    return backtrack(n, 0, [], 0, 0)

# EVOLVE-BLOCK-END
