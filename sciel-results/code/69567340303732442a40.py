import time

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)
    
    dp = [None] * n
    
    for i in range(n):
        if time.time() - t0 >= DEADLINE:
            break
        pi_val = primes[i]
        cands = []
        
        for j in range(i):
            gap = pi_val - primes[j]
            cands.append((gap, 2, j, -1))
            
            dj = dp[j]
            if dj is not None:
                ln_dj = len(dj)
                if ln_dj > 0:
                    lo, hi = 0, ln_dj - 1
                    best_k = -1
                    while lo <= hi:
                        mid = (lo + hi) >> 1
                        if dj[mid][0] <= gap:
                            best_k = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    
                    if best_k != -1:
                        cands.append((gap, dj[best_k][1] + 1, j, best_k))
        
        if not cands:
            dp[i] = []
        elif len(cands) == 1:
            dp[i] = cands
        else:
            cands.sort(key=lambda x: (x[0], -x[1]))
            frontier = []
            max_len = 0
            last_gap = -1
            for item in cands:
                g, l, pi_idx, pei = item
                if g != last_gap:
                    if l > max_len:
                        frontier.append(item)
                        max_len = l
                    last_gap = g
            dp[i] = frontier

    best_len = 1
    best_i = 0
    best_k = -1
    for i in range(n):
        di = dp[i]
        if di is None:
            continue
        for k in range(len(di)):
            l = di[k][1]
            if l > best_len:
                best_len = l
                best_i = i
                best_k = k

    if best_len <= 1:
        return [primes[0]]

    result = []
    cur_i = best_i
    cur_k = best_k
    while True:
        result.append(primes[cur_i])
        if cur_k == -1:
            break
        entry = dp[cur_i][cur_k]
        cur_k = entry[3]
        cur_i = entry[2]
    
    result.reverse()
    return result
