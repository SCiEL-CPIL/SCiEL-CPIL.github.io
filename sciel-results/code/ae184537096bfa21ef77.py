"""Stronger approach: focus on N=2*10^18 which has many representations."""

def construct():
    # The key insight: N = 2*10^18 = 2 * (10^9)^2
    # If a = k^2 * 10^18 (for k=1,2,...,1000000), then a is powerful (since 10^18 is powerful)
    # and b = N - a = (2 - k^2) * 10^18
    # For b to be powerful and non-negative, we need k^2 < 2, so only k=1 gives b=10^18 (powerful)
    # That gives only 1 pair. Not enough.
    
    # Better approach: N = 2 * M where M is a powerful number.
    # If a = M and b = M, that's one pair.
    # More generally, for N = 2*M, any powerful a <= M with N-a powerful gives a pair.
    
    # Even better: use N that is a multiple of a highly composite powerful structure.
    
    # Strategy: N = 2 * P where P is powerful.
    # Pairs: (a, b) where a+b = 2P, both powerful, a <= b.
    # a = P - d, b = P + d for various d.
    # We need P-d and P+d both powerful.
    
    # Alternative: Use N = 2 * (2^30) = 2^31. 
    # a = 2^k * (something), b = ...
    
    # Let's think differently. The number of powerful numbers up to X is about C*X^{1/2}.
    # For X = 10^9, that's about 31628 powerful numbers.
    # O(n^2) with n=31628 is about 10^9 operations - too slow.
    
    # For X = 10^6, about 1000 powerful numbers. O(n^2) = 10^6 - fast.
    # But N is limited to ~2*10^6, small range.
    
    # Let me use a smarter approach:
    # Generate powerful numbers up to some limit L.
    # For each powerful number a, compute b = N - a for various candidate N values.
    # But we don't know N in advance.
    
    # Better: Use the structure of powerful numbers.
    # Powerful numbers have the form prod(p_i^{e_i}) with e_i >= 2.
    # Small powerful numbers: 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, ...
    
    # Key insight: If N is even and large, and N/2 is powerful, then (N/2, N/2) is one pair.
    # Also, if N = a + b where a and b are both perfect squares (hence powerful), we get pairs.
    # But not all sums of squares give powerful numbers.
    
    # Let's use a computational approach with a manageable set of powerful numbers.
    # Generate powerful numbers up to 10^8 (about 10,000 of them).
    # Then use O(n^2) to count all sums and find the N with max representations.
    
    import time
    from collections import defaultdict
    
    start = time.time()
    DEADLINE = start + 110  # 10 second safety margin
    
    def generate_powerful(limit):
        """Generate all powerful numbers up to limit."""
        ps = set()
        # n = a^2 * b^3 where b is squarefree
        max_b = int(limit ** (1/3)) + 2
        # Sieve for squarefree
        is_sqfree = [True] * (max_b + 1)
        is_sqfree[0] = False
        if max_b >= 1:
            is_sqfree[1] = True
        for p in range(2, int(max_b**0.5) + 1):
            p2 = p * p
            for j in range(p2, max_b + 1, p2):
                is_sqfree[j] = False
        
        sqfree = [b for b in range(1, max_b + 1) if is_sqfree[b]]
        
        for b in sqfree:
            b3 = b ** 3
            if b3 > limit:
                continue
            max_a = int((limit // b3) ** 0.5)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n > limit:
                    break
                ps.add(n)
        
        return sorted(ps)
    
    # Start with a moderate limit
    limit = 10**8
    pow_list = generate_powerful(limit)
    pow_set = set(pow_list)
    n_pow = len(pow_list)
    
    # Count sums - use O(n^2) but with early termination
    sum_counts = defaultdict(int)
    
    # To speed up, only consider pairs where a <= b and a+b <= 2*limit
    # Use the sorted list for bounds
    for i in range(n_pow):
        a = pow_list[i]
        max_b_val = 2 * limit - a  # We want N <= 2*limit
        # Find upper bound in pow_list
        j = i
        while j < n_pow and pow_list[j] <= max_b_val:
            b = pow_list[j]
            s = a + b
            sum_counts[s] += 1
            j += 1
    
    # Find best N
    best_n = 2
    best_count = 0
    for n, count in sum_counts.items():
        if count > best_count:
            best_count = count
            best_n = n
    
    # Collect pairs for best_n
    pairs = []
    for i in range(n_pow):
        a = pow_list[i]
        b = best_n - a
        if b >= a and b in pow_set:
            pairs.append([a, b])
    
    # Check time and try to improve with larger limit if time permits
    elapsed = time.time() - start
    if elapsed < 80 and best_count < 20:
        limit2 = min(10**9, limit * 10)
        pow_list2 = generate_powerful(limit2)
        pow_set2 = set(pow_list2)
        n_pow2 = len(pow_list2)
        
        sum_counts2 = defaultdict(int)
        for i in range(n_pow2):
            a = pow_list2[i]
            max_b_val = 2 * limit2 - a
            j = i
            while j < n_pow2 and pow_list2[j] <= max_b_val:
                b = pow_list2[j]
                s = a + b
                sum_counts2[s] += 1
                j += 1
        
        best_n2 = best_n
        best_count2 = best_count
        for n, count in sum_counts2.items():
            if count > best_count2:
                best_count2 = count
                best_n2 = n
        
        if best_count2 > best_count:
            best_n = best_n2
            best_count = best_count2
            pow_list = pow_list2
            pow_set = pow_set2
            n_pow = n_pow2
            pairs = []
            for i in range(n_pow):
                a = pow_list[i]
                b = best_n - a
                if b >= a and b in pow_set:
                    pairs.append([a, b])
    
    # Ensure we have at least one valid pair
    if not pairs:
        if best_n == 2:
            pairs = [[1, 1]]
        else:
            # Find any valid pair
            for a in pow_list:
                b = best_n - a
                if b >= a and b in pow_set:
                    pairs.append([a, b])
                    break
            if not pairs:
                return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_n, 'pairs': pairs}
