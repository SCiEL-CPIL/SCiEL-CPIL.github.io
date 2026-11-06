def construct():
    import math
    from collections import defaultdict
    
    LIMIT = 10**15
    # Generate powerful numbers up to LIMIT
    # x = i^2 * j^3
    # j^3 <= LIMIT => j <= LIMIT^(1/3)
    max_j = int(round(LIMIT ** (1/3)))
    powerful_set = set()
    
    for j in range(1, max_j + 1):
        j_cubed = j ** 3
        # i^2 <= LIMIT / j^3
        max_i = int(math.isqrt(LIMIT // j_cubed))
        for i in range(1, max_i + 1):
            val = i * i * j_cubed
            if val <= LIMIT:
                powerful_set.add(val)
    
    powerful = sorted(powerful_set)
    n_pow = len(powerful)
    
    # Count sums
    counts = defaultdict(int)
    best_N = 0
    max_count = 0
    
    # Optimization: Use list for faster access
    # Nested loop
    for i in range(n_pow):
        a = powerful[i]
        for j in range(i, n_pow):
            b = powerful[j]
            s = a + b
            if s > 10**18:
                break
            counts[s] += 1
            if counts[s] > max_count:
                max_count = counts[s]
                best_N = s
                
    # Retrieve pairs for best_N
    pairs = []
    for a in powerful:
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
