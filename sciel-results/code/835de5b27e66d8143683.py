# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n_val = 5040
    target = 4979
    
    # Hardcoded divisors of 5040 (60 divisors)
    divisors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 48, 56, 60, 63, 70, 72, 80, 84, 90, 105, 112, 120, 126, 140, 144, 168, 180, 189, 210, 224, 240, 252, 280, 315, 336, 360, 378, 420, 504, 560, 630, 720, 840, 1008, 1260, 1512, 2016, 2520, 5040]
    n_divs = len(divisors)
    divisor_set = set(divisors)
    
    # Precompute prefix sums
    prefix = [0] * (n_divs + 1)
    for i in range(n_divs):
        prefix[i + 1] = prefix[i] + divisors[i]
    total_sum = prefix[n_divs]
    
    # k=1
    if target in divisor_set:
        return [target]
        
    # k=2: O(N) using set
    for d in divisors:
        complement = target - d
        if complement in divisor_set and complement > d:
            return [d, complement]
            
    # k=3: O(N^2) with early termination
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            s = a + b
            if s >= target:
                break
            c = target - s
            if c in divisor_set and c > b:
                return [a, b, c]
                
    # k=4: Meet-in-the-middle with early termination
    # Build pair sums dict, but stop early if we find a match
    pair_sums = {}
    for i in range(n_divs):
        a = divisors[i]
        if a >= target:
            break
        for j in range(i + 1, n_divs):
            b = divisors[j]
            s = a + b
            if s >= target:
                break
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((i, j))
            
    # Look for two disjoint pairs that sum to target
    for s1 in pair_sums:
        s2 = target - s1
        if s2 not in pair_sums:
            continue
        if s2 < s1:
            continue
        pairs1 = pair_sums[s1]
        pairs2 = pair_sums[s2]
        
        if s1 == s2:
            for p1_idx in range(len(pairs1)):
                i1, j1 = pairs1[p1_idx]
                for p2_idx in range(p1_idx + 1, len(pairs1)):
                    i2, j2 = pairs1[p2_idx]
                    if i2 != i1 and i2 != j1 and j2 != i1 and j2 != j1:
                        return [divisors[i1], divisors[j1], divisors[i2], divisors[j2]]
        else:
            for i1, j1 in pairs1:
                for i2, j2 in pairs2:
                    if i2 != i1 and i2 != j1 and j2 != i1 and j2 != j1:
                        return [divisors[i1], divisors[j1], divisors[i2], divisors[j2]]
    
    # k=5: Meet-in-the-middle (2 + 3)
    if time.time() < deadline:
        triple_sums = {}
        for i in range(n_divs):
            a = divisors[i]
            if a >= target:
                break
            for j in range(i + 1, n_divs):
                b = divisors[j]
                s2 = a + b
                if s2 >= target:
                    break
                for k in range(j + 1, n_divs):
                    c = divisors[k]
                    s = s2 + c
                    if s >= target:
                        break
                    if s not in triple_sums:
                        triple_sums[s] = []
                    triple_sums[s].append((i, j, k))
        
        for s_t in triple_sums:
            s_p = target - s_t
            if s_p not in pair_sums:
                continue
            triples = triple_sums[s_t]
            pairs = pair_sums[s_p]
            for i1, j1, k1 in triples:
                used = {i1, j1, k1}
                for i2, j2 in pairs:
                    if i2 not in used and j2 not in used:
                        return [divisors[i1], divisors[j1], divisors[k1], divisors[i2], divisors[j2]]
    
    # k=6: Meet-in-the-middle (3 + 3)
    if time.time() < deadline:
        for s1 in triple_sums:
            s2 = target - s1
            if s2 not in triple_sums:
                continue
            if s2 < s1:
                continue
            triples1 = triple_sums[s1]
            triples2 = triple_sums[s2]
            
            if s1 == s2:
                for t1_idx in range(len(triples1)):
                    i1, j1, k1 = triples1[t1_idx]
                    used1 = {i1, j1, k1}
                    for t2_idx in range(t1_idx + 1, len(triples1)):
                        i2, j2, k2 = triples1[t2_idx]
                        if i2 not in used1 and j2 not in used1 and k2 not in used1:
                            return [divisors[i1], divisors[j1], divisors[k1], divisors[i2], divisors[j2], divisors[k2]]
            else:
                for i1, j1, k1 in triples1:
                    used1 = {i1, j1, k1}
                    for i2, j2, k2 in triples2:
                        if i2 not in used1 and j2 not in used1 and k2 not in used1:
                            return [divisors[i1], divisors[j1], divisors[k1], divisors[i2], divisors[j2], divisors[k2]]
    
    # k >= 7: Backtracking with strong pruning
    def find_k_divisors(k, start_idx, remaining_sum):
        if k == 0:
            return [] if remaining_sum == 0 else None
        
        if start_idx >= n_divs:
            return None
            
        if n_divs - start_idx < k:
            return None
            
        min_possible = prefix[start_idx + k] - prefix[start_idx]
        if remaining_sum < min_possible:
            return None
            
        max_possible = total_sum - prefix[n_divs - k]
        if remaining_sum > max_possible:
            return None
            
        for i in range(start_idx, n_divs):
            d = divisors[i]
            new_remaining = remaining_sum - d
            if new_remaining < 0:
                break
            
            if i + 1 + (k - 1) <= n_divs:
                min_next = prefix[i + 1 + (k - 1)] - prefix[i + 1]
                if new_remaining < min_next:
                    break
            else:
                continue

            res = find_k_divisors(k - 1, i + 1, new_remaining)
            if res is not None:
                return [d] + res
                
        return None
    
    for k in range(7, 257):
        if time.time() > deadline:
            break
            
        min_sum_k = prefix[k]
        max_sum_k = total_sum - prefix[n_divs - k]
        if target < min_sum_k or target > max_sum_k:
            continue
            
        res = find_k_divisors(k, 0, target)
        if res is not None:
            return res
            
    # Fallback: Greedy
    remainder = target
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder and d not in chosen:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
                
    return chosen
# EVOLVE-BLOCK-END
