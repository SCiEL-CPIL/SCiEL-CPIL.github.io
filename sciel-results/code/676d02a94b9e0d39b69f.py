def construct():
    max_n = 1000000
    window_size = 24
    limit = max_n + window_size + 10
    
    # 1. Compute Smallest Prime Factor (SPF)
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # 2. Compute g[m] = product of prime factors < 24
    g = [1] * (limit + 1)
    for m in range(2, limit + 1):
        p = spf[m]
        if p < 24:
            g[m] = g[m // p] * p
        else:
            g[m] = g[m // p]
            
    # 3. Sliding Window with List-based Counting
    # Max value of g[m] is m, so limit is safe for count array size
    counts = [0] * (limit + 1)
    
    start_idx = 1001
    end_idx = start_idx + window_size
    
    current_distinct = 0
    for i in range(start_idx, end_idx):
        val = g[i]
        if counts[val] == 0:
            current_distinct += 1
        counts[val] += 1
        
    min_distinct = current_distinct
    best_n = 1000
    
    for n in range(1000, max_n):
        out_val = g[n+1]
        counts[out_val] -= 1
        if counts[out_val] == 0:
            current_distinct -= 1
            
        in_val = g[n+25]
        if counts[in_val] == 0:
            current_distinct += 1
        counts[in_val] += 1
        
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = n + 1
            
    return best_n
