"""Program for cardinality-separated subset sums using reverse greedy construction."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    
    Uses a multi-strategy search: for each candidate size n from 60 down,
    try several construction methods to find a valid set.
    """
    N = 256
    
    def check_set(s):
        """Check if sorted set s satisfies all cardinality separation constraints."""
        n = len(s)
        if n <= 1:
            return True
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s[i]
        for k in range(2, n + 1):
            min_k_sum = prefix[k]
            max_km1_sum = prefix[n] - prefix[n - k + 1]
            if min_k_sum <= max_km1_sum:
                return False
        return True
    
    def try_consecutive(n):
        """Try consecutive integers starting at various positions near the top."""
        for start in range(257 - n, max(1, 257 - n - 100), -1):
            s = list(range(start, start + n))
            if s[-1] > N:
                continue
            if check_set(s):
                return s
        return None
    
    def try_spread(n):
        """Try a spread construction using linear interpolation."""
        for base in range(1, 50):
            for span in range(n, N + 1 - base + 1):
                if span < n:
                    continue
                s = []
                for i in range(n):
                    val = base + int(round(i * (span - 1) / max(1, n - 1))) if n > 1 else base
                    if val < 1 or val > N:
                        break
                    s.append(val)
                if len(s) != n:
                    continue
                s = sorted(set(s))
                if len(s) != n:
                    continue
                if check_set(s):
                    return s
        return None
    
    def try_boosted(n):
        """Try boosting the smallest elements of the n-largest set."""
        base = list(range(257 - n, 257))
        for boost in range(0, 50):
            s = base.copy()
            # Boost the smallest elements
            for i in range(min(boost, n)):
                new_val = s[i] + (boost - i)
                if new_val > N:
                    break
                # Ensure it's still less than the next element
                if i + 1 < n and new_val >= s[i + 1]:
                    break
                s[i] = new_val
            else:
                s.sort()
                if len(set(s)) == n and s[-1] <= N and s[0] >= 1:
                    if check_set(s):
                        return s
        return None
    
    def try_reduced_top(n):
        """Try reducing the largest elements of the n-largest set."""
        base = list(range(257 - n, 257))
        for reduce in range(0, 50):
            s = base.copy()
            for i in range(min(reduce, n)):
                idx = n - 1 - i
                new_val = s[idx] - (reduce - i)
                if new_val < 1:
                    break
                if idx - 1 >= 0 and new_val <= s[idx - 1]:
                    break
                s[idx] = new_val
            else:
                s.sort()
                if len(set(s)) == n and s[-1] <= N and s[0] >= 1:
                    if check_set(s):
                        return s
        return None
    
    # Try n from 60 down to 1
    for n in range(60, 0, -1):
        result = try_consecutive(n)
        if result is not None:
            return result
        
        result = try_boosted(n)
        if result is not None:
            return result
        
        result = try_reduced_top(n)
        if result is not None:
            return result
        
        result = try_spread(n)
        if result is not None:
            return result
    
    # Fallback: return the seed
    return list(range(128, 136))

# EVOLVE-BLOCK-END
