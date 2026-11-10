def construct():
    import time
    from bisect import bisect_right
    
    n_max = 10000
    sieve = [True] * (n_max + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    primes = [i for i in range(2, n_max + 1) if sieve[i]]
    n = len(primes)
    
    if n <= 2:
        return primes[:]
    
    # L[i][j] = length of longest valid nondecreasing-gap subsequence ending with primes[i], primes[j]
    # prev[i][j] = k such that subsequence is ...primes[k], primes[i], primes[j]
    
    L = [[2] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]
    
    # For each idx, sorted_entries[idx] = [(gap, k), ...] sorted by gap, for k < idx
    # prefix_max[idx] = prefix max of L[k][idx]
    sorted_entries = [[] for _ in range(n)]
    prefix_max = [[] for _ in range(n)]
    
    best_len = 2
    best_i, best_j = 0, 1
    
    for j in range(1, n):
        for i in range(j):
            gap_ij = primes[j] - primes[i]
            L[i][j] = 2
            prev[i][j] = -1
            
            if sorted_entries[i]:
                # Extract gaps for binary search
                gaps_only = [e[0] for e in sorted_entries[i]]
                pos = bisect_right(gaps_only, gap_ij)
                if pos > 0:
                    best_L = prefix_max[i][pos - 1]
                    if best_L + 1 > L[i][j]:
                        L[i][j] = best_L + 1
                        # Find the k that achieves best_L
                        for p in range(pos - 1, -1, -1):
                            k = sorted_entries[i][p][1]
                            if L[k][i] == best_L:
                                prev[i][j] = k
                                break
            
            if L[i][j] > best_len:
                best_len = L[i][j]
                best_i, best_j = i, j
        
        # Build data structure for index j
        entries = []
        for i in range(j):
            gap_ij = primes[j] - primes[i]
            entries.append((gap_ij, i))
        entries.sort(key=lambda x: x[0])
        sorted_entries[j] = entries
        pm = []
        cur = 0
        for gap, k in entries:
            cur = max(cur, L[k][j])
            pm.append(cur)
        prefix_max[j] = pm
    
    # Reconstruct
    seq = []
    i, j = best_i, best_j
    while True:
        seq.append(primes[j])
        if prev[i][j] == -1:
            seq.append(primes[i])
            break
        i, j = prev[i][j], i
    seq.reverse()
    
    return seq
