import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin

    n = 5040
    target = 4979
    
    # Precompute divisors
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    # Filter divisors <= target
    divisors = [d for d in divisors if d <= target]
    # Sort descending to help with pruning in DFS if used, but for combinations order doesn't matter for sum
    # However, for MITM or DFS, sorted helps.
    divs = sorted(divisors, reverse=True)
    
    # We want to minimize k.
    # Max sum of 1: max(divs)
    # Max sum of 2: sum of top 2
    # etc.
    
    # Check if target is possible with small k using specific strategies
    
    # Strategy:
    # 1. Try k=1, 2, 3, 4 with itertools.combinations (fast enough)
    # 2. Try k=5 with Meet-in-the-Middle (2+3)
    # 3. If not found, use a generic DFS with pruning for k=6, 7...
    
    def find_combo(k):
        if time.time() > deadline:
            return None
        
        if k == 1:
            if target in divisors:
                return [target]
            return None
        
        if k == 2:
            # Use set for O(1) lookup
            seen = set()
            for d in divs:
                if target - d in seen:
                    return [d, target - d]
                seen.add(d)
            return None

        if k <= 4:
            # Direct enumeration is fast for k<=4 (max ~500k combos)
            for combo in itertools.combinations(divs, k):
                if sum(combo) == target:
                    return list(combo)
            return None
        
        if k == 5:
            # Meet in the middle: 2 + 3
            # Generate all pairs (sum, tuple)
            pair_sums = {}
            for i in range(len(divs)):
                for j in range(i+1, len(divs)):
                    s = divs[i] + divs[j]
                    # We only need one pair per sum? No, we need to ensure disjointness.
                    # But if we store the pair, we can check disjointness later.
                    # To keep it fast, we can just store the sum and one pair. 
                    # If there are multiple pairs for the same sum, we might miss a solution 
                    # if the stored pair overlaps with the triple. 
                    # Given the sparsity, storing a list or just checking existence is risky.
                    # Better: Store all pairs in a dict: sum -> list of pairs.
                    # Or, since k is small, we can just iterate triples and check if (target - s3) is in pair_sums.
                    # Let's store sum -> (d1, d2). If collision, we might need more.
                    # For safety, let's use a set of pairs? No, too many.
                    # Let's just use a set of sums for pairs first to see if it's possible.
                    pass
            
            # Refined MITM for k=5:
            # 1. Create set of all pair sums.
            pair_sums_set = set()
            pair_sums_dict = {} # sum -> (d1, d2) for one example
            for i in range(len(divs)):
                for j in range(i+1, len(divs)):
                    s = divs[i] + divs[j]
                    pair_sums_set.add(s)
                    if s not in pair_sums_dict:
                        pair_sums_dict[s] = (divs[i], divs[j])
            
            # 2. Iterate over all triples
            for i in range(len(divs)):
                for j in range(i+1, len(divs)):
                    for m in range(j+1, len(divs)):
                        s3 = divs[i] + divs[j] + divs[m]
                        rem = target - s3
                        if rem in pair_sums_set:
                            # Found a sum match. Need to check disjointness.
                            # Get one pair from dict. It might overlap.
                            # If it overlaps, we need to find another pair for 'rem'.
                            # This can get complicated.
                            # Alternative: Just use itertools.combinations for k=5 if we optimize the sum check?
                            # 5.5M combos. In Python, a loop of 5.5M taking 1 microsecond each is 5.5s.
                            # The previous code was slow because of time.time(). 
                            # Without time.time(), itertools.combinations is C-speed for generation.
                            # sum(combo) is the cost.
                            # Let's try to make k=5 fast with a different approach or just accept the time if we remove the time check.
                            pass
            
            # Let's stick to a robust DFS for k>=5 if k<=4 fails, but optimized.
            # Or, simply, since k=3,4 are very likely, we might not even reach k=5.
            # Let's just return None for now and let the caller handle higher k with a better method.
            return None
        
        return None

    # Try k=1, 2, 3, 4
    for k in range(1, 5):
        res = find_combo(k)
        if res:
            return res
        
        if time.time() > deadline:
            break

    # If not found in 1-4, try k=5 with a dedicated MITM or optimized search
    # Given the constraints, a simple optimized DFS with pruning is safer for arbitrary k.
    # But let's try to improve the DFS from the original.
    
    # Optimized DFS for k >= 5
    # Pruning:
    # 1. If remaining elements < needed, fail.
    # 2. If sum of largest 'needed' elements < remaining, fail.
    # 3. If sum of smallest 'needed' elements > remaining, fail.
    
    divs = sorted(divisors, reverse=True) # Largest first for pruning max sum
    
    def dfs(idx, remaining, needed, chosen):
        if time.time() > deadline:
            return None
        if needed == 0:
            if remaining == 0:
                return list(chosen)
            return None
        if idx >= len(divs):
            return None
        
        # Pruning 1: Not enough items left
        if len(divs) - idx < needed:
            return None
            
        # Pruning 2: Max possible sum is too small
        # Sum of divs[idx : idx+needed]
        max_sum = sum(divs[idx:idx+needed])
        if max_sum < remaining:
            return None
            
        # Pruning 3: Min possible sum is too large
        # Sum of the smallest 'needed' items available? 
        # Since divs is sorted descending, the smallest 'needed' items from idx onwards are divs[len(divs)-needed : len(divs)]
        # But we must pick from idx onwards. The absolute smallest 'needed' divisors in the whole list might not be available if idx is high?
        # No, we can pick any from idx onwards. The smallest available are at the end of the list.
        min_sum = sum(divs[len(divs)-needed:len(divs)])
        if min_sum > remaining:
            return None
            
        # Option 1: Take divs[idx]
        if divs[idx] <= remaining:
            chosen.append(divs[idx])
            res = dfs(idx + 1, remaining - divs[idx], needed - 1, chosen)
            if res:
                return res
            chosen.pop()
            
        # Option 2: Skip divs[idx]
        res = dfs(idx + 1, remaining, needed, chosen)
        return res

    # Try k=5, 6, ...
    for k in range(5, len(divs) + 1):
        res = dfs(0, target, k, [])
        if res:
            return res
        if time.time() > deadline:
            break
            
    # Fallback: should not happen if solution exists
    # Return a valid set if we find one later? 
    # The problem guarantees a solution exists.
    # If we run out of time, we might return an invalid set or crash.
    # We must ensure we return *something* valid if possible, or let it timeout?
    # "An invalid result... scores 0".
    # So we must return a valid result.
    # If we haven't found one, we are in trouble.
    # However, k=3 or 4 is extremely likely.
    
    return [] # This would be invalid.
