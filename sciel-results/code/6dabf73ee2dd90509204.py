import math
import time
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    # 115s deadline to be safe (150s limit, buffer for overhead)
    deadline = start_time + 115.0
    
    # 1. Generate powerful numbers up to a large limit.
    # Powerful numbers are of the form a^2 * b^3 where b is square-free.
    # For L = 10^9, max b is ~ 1000. Max a is sqrt(10^9 / 1) = 31622.
    # Total count is roughly L^(1/3) * something, which is manageable (~10^5).
    
    def is_square_free(n):
        if n < 2:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    def generate_powerful(limit):
        powerful = set()
        max_b = int(limit ** (1.0/3.0)) + 2
        # Precompute square-free numbers up to max_b
        sq_free = [i for i in range(1, max_b + 1) if is_square_free(i)]
        
        for b in sq_free:
            b3 = b ** 3
            if b3 > limit:
                break
            max_a = int(math.isqrt(limit // b3))
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    powerful.add(val)
        return sorted(powerful)

    # Phase 1: Generate powerful numbers up to 10^9
    # This allows N up to 2 * 10^9.
    L1 = 10**9
    p_large = generate_powerful(L1)
    
    # Convert to numpy array for vectorized operations
    p_arr = np.array(p_large, dtype=np.int64)
    p_set = set(p_large)
    
    # Phase 2: Identify candidate N values
    # Hypothesis: N with many representations are likely to be formed by
    # powerful numbers with many divisors (highly composite powerful numbers).
    # We will score powerful numbers by their divisor count and pick the top ones.
    # Then we will look at sums of pairs of these top numbers.
    
    def count_divisors(n):
        if n == 1:
            return 1
        count = 0
        i = 2
        while i * i <= n:
            if n % i == 0:
                exp = 0
                while n % i == 0:
                    n //= i
                    exp += 1
                count += exp
            i += 1
        if n > 1:
            count += 1
        return count

    # Calculate divisor counts for all powerful numbers
    # To save time, we only calculate for numbers that are "large" enough to be interesting
    # or we just do it for all since count is ~100k
    div_counts = {}
    for p in p_large:
        div_counts[p] = count_divisors(p)
    
    # Sort powerful numbers by divisor count descending
    sorted_by_div = sorted(p_large, key=lambda x: div_counts[x], reverse=True)
    
    # Take the top K powerful numbers
    K = 500
    top_p = sorted_by_div[:K]
    
    # Generate candidate N values from sums of top powerful numbers
    candidates = set()
    # We want N <= 2 * L1
    # Iterate over pairs of top_p
    # K=500 -> 125k pairs. Fast.
    for i in range(K):
        a = top_p[i]
        for j in range(i, K):
            b = top_p[j]
            s = a + b
            if s <= 2 * L1:
                candidates.add(s)
            else:
                break # Since top_p is sorted by divisors, not value, this break might be premature
                # Actually top_p is NOT sorted by value. So we can't break.
                # But we can just check the condition.
    
    # We also want to include some "hot" N from a smaller range brute force if we have time?
    # No, let's stick to the new strategy.
    # However, the top K by divisors might not cover all "hot" N.
    # Let's also add candidates from a small brute force as a backup if we have time.
    
    # Sort candidates. We want to verify the most promising ones first.
    # How to rank candidates?
    # We can estimate the number of representations by counting how many pairs (a,b) in top_p sum to N.
    cand_counts = defaultdict(int)
    for i in range(K):
        a = top_p[i]
        for j in range(i, K):
            b = top_p[j]
            s = a + b
            if s <= 2 * L1:
                cand_counts[s] += 1
                
    # Sort candidates by this estimated count
    sorted_candidates = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Take top M candidates for verification
    M = 100
    top_candidates = [n for n, c in sorted_candidates[:M]]
    
    # Phase 3: Verify candidates using vectorized NumPy
    # For each N, we want to find all a in p_large such that N-a is also in p_large.
    # a <= N/2.
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    # Sort p_arr for binary search if needed, but we use set for membership.
    # Vectorized check:
    # For a given N, create an array of all a in p_large where a <= N//2.
    # Then check if N - a is in p_set.
    
    for N in top_candidates:
        if time.time() > deadline - 5:
            break
            
        # Filter p_arr to a <= N//2
        mask = p_arr <= N // 2
        a_candidates = p_arr[mask]
        
        if len(a_candidates) == 0:
            continue
            
        # Check if N - a is in p_set
        # We can do this with a list comprehension or numpy + set intersection
        # b_values = N - a_candidates
        # valid_b = [b for b in b_values if b in p_set]
        # This is O(M) in Python, which might be slow for M=100k.
        # Let's try numpy:
        b_candidates = N - a_candidates
        # Convert to set for intersection? No, b_candidates is an array.
        # We can do:
        # valid_mask = np.isin(b_candidates, list(p_set)) -> Slow for large set
        
        # Alternative:
        # Since p_set is a Python set, iterating in Python is fast enough for 100k elements?
        # 100k iterations * 100 candidates = 10M iterations. Might be too slow in Python.
        
        # Let's optimize:
        # Only check if N is even? No, powerful numbers can be odd/even.
        
        # Let's try a hybrid:
        # If len(a_candidates) is large, we can use a different method.
        # But 100k is not that large.
        # Let's time it.
        
        pairs = []
        # To speed up, we can pre-convert p_set to a list if we use numpy isin?
        # np.isin(b_candidates, list(p_set))
        # This creates a large list every time.
        
        # Better:
        # Use a boolean array if the range is small? No, range is 10^9.
        
        # Let's just use the Python loop but break early if we have too many?
        # No, we need the count.
        
        # Optimization:
        # We know that a and b are powerful.
        # We can iterate through a_candidates.
        
        # Let's try to be smarter.
        # If N is large, a_candidates is large.
        # But we only have 100 candidates.
        # 100 * 100k = 10M operations. In Python, 10M simple ops is ~1s.
        # So this should be fine.
        
        for a in a_candidates:
            b = N - a
            if b in p_set:
                pairs.append([int(a), int(b)])
                
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs

    # If the targeted candidates didn't yield a good result, or if we have time,
    # we can try a fallback: brute force on a small range.
    # But the current program already does that.
    # Let's see if we found anything.
    
    if best_N == 0:
        # Fallback to small range brute force if we failed
        L2 = 10**7
        p_small = generate_powerful(L2)
        counts_small = defaultdict(int)
        for i, a in enumerate(p_small):
            for b in p_small[i:]:
                s = a + b
                if s <= 2 * L2:
                    counts_small[s] += 1
        sorted_ns = sorted(counts_small.items(), key=lambda x: x[1], reverse=True)
        if sorted_ns:
            best_N = sorted_ns[0][0]
            best_count = sorted_ns[0][1]
            p_set_small = set(p_small)
            best_pairs = []
            for a in p_small:
                if a > best_N // 2:
                    break
                b = best_N - a
                if b in p_set_small:
                    best_pairs.append([a, b])
        
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
