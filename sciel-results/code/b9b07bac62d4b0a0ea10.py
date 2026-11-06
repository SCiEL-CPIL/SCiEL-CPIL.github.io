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
    smooth_index = {v: i for i, v in enumerate(smooth_nums)}
    
    # 2. Precompute antichain conflict masks
    conflict_mask = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            a, b = smooth_nums[i], smooth_nums[j]
            if b % a == 0 or a % b == 0:
                conflict_mask[i] |= (1 << j)
    
    # 3. Precompute valid antichains of size 1, 2, 3 grouped by sum
    antichains_by_sum = {}
    
    for i in range(m):
        s = smooth_nums[i]
        if s <= 128:
            if s not in antichains_by_sum:
                antichains_by_sum[s] = []
            antichains_by_sum[s].append((i,))
    
    for i in range(m):
        for j in range(i+1, m):
            if conflict_mask[i] & (1 << j):
                continue
            s = smooth_nums[i] + smooth_nums[j]
            if s <= 128:
                if s not in antichains_by_sum:
                    antichains_by_sum[s] = []
                antichains_by_sum[s].append((i, j))
    
    for i in range(m):
        for j in range(i+1, m):
            if conflict_mask[i] & (1 << j):
                continue
            for k in range(j+1, m):
                if (conflict_mask[i] & (1 << k)) or (conflict_mask[j] & (1 << k)):
                    continue
                s = smooth_nums[i] + smooth_nums[j] + smooth_nums[k]
                if s <= 128:
                    if s not in antichains_by_sum:
                        antichains_by_sum[s] = []
                    antichains_by_sum[s].append((i, j, k))
    
    # 4. For each target, find minimum-term antichain
    answers = []
    
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            if n in is_smooth:
                answers.append([n])
            else:
                break
            continue
        
        best = None
        if n in antichains_by_sum:
            for entry in antichains_by_sum[n]:
                if best is None or len(entry) < len(best):
                    best = entry
                if len(best) == 1:
                    break
        
        if best is not None:
            answers.append([smooth_nums[idx] for idx in best])
        else:
            found = _find_antichain_size_4_5(n, smooth_nums, conflict_mask, smooth_index, is_smooth)
            if found is None:
                found = _find_any_antichain_sum(n, smooth_nums, conflict_mask)
            if found is None:
                found = [n]
            answers.append(found)
    
    return answers

def _find_antichain_size_4_5(n, smooth_nums, conflict_mask, smooth_index, is_smooth):
    """Try to find a size-4 or size-5 antichain summing to n."""
    m = len(smooth_nums)
    
    # Try size 4 first
    for i in range(m):
        a = smooth_nums[i]
        if 4 * a >= n:
            break
        for j in range(i + 1, m):
            b = smooth_nums[j]
            if conflict_mask[i] & (1 << j):
                continue
            if a + 3 * b >= n:
                break
            for k in range(j + 1, m):
                c = smooth_nums[k]
                if (conflict_mask[i] & (1 << k)) or (conflict_mask[j] & (1 << k)):
                    continue
                if a + b + 2 * c >= n:
                    break
                d = n - a - b - c
                if d <= c:
                    continue
                if d not in is_smooth:
                    continue
                l = smooth_index.get(d, -1)
                if l <= k:
                    continue
                if (conflict_mask[i] & (1 << l)) or (conflict_mask[j] & (1 << l)) or (conflict_mask[k] & (1 << l)):
                    continue
                return [a, b, c, d]
    
    # Try size 5
    for i in range(m):
        a = smooth_nums[i]
        if 5 * a >= n:
            break
        for j in range(i + 1, m):
            b = smooth_nums[j]
            if conflict_mask[i] & (1 << j):
                continue
            if a + 4 * b >= n:
                break
            for k in range(j + 1, m):
                c = smooth_nums[k]
                if (conflict_mask[i] & (1 << k)) or (conflict_mask[j] & (1 << k)):
                    continue
                if a + b + 3 * c >= n:
                    break
                for l in range(k + 1, m):
                    d = smooth_nums[l]
                    if ((conflict_mask[i] & (1 << l)) or (conflict_mask[j] & (1 << l)) or 
                        (conflict_mask[k] & (1 << l))):
                        continue
                    if a + b + c + 2 * d >= n:
                        break
                    e = n - a - b - c - d
                    if e <= d:
                        continue
                    if e not in is_smooth:
                        continue
                    p = smooth_index.get(e, -1)
                    if p <= l:
                        continue
                    if ((conflict_mask[i] & (1 << p)) or (conflict_mask[j] & (1 << p)) or 
                        (conflict_mask[k] & (1 << p)) or (conflict_mask[l] & (1 << p))):
                        continue
                    return [a, b, c, d, e]
    
    return None

def _find_any_antichain_sum(n, smooth_nums, conflict_mask):
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
