import time
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [p for p in range(n+1) if is_prime[p]]
    
    primes = sieve(10000)
    n = len(primes)
    
    if n <= 1:
        return primes
    
    # states[i] is a Pareto list of (gap, length, prev_idx, prev_gap)
    # sorted by gap ascending, length strictly increasing
    states = [[] for _ in range(n)]
    states[0].append((0, 1, -1, 0))
    
    best_length = 1
    best_end = 0
    
    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break
        
        candidates = []
        # State: just primes[i]
        candidates.append((0, 1, -1, 0))
        
        for j in range(i):
            gap = primes[i] - primes[j]
            sj = states[j]
            if not sj:
                continue
            
            # Find best state in sj with last_gap <= gap
            # sj is sorted by gap, lengths increasing
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            
            if lo > 0:
                prev_gap, prev_len, prev_idx, prev_prev_gap = sj[lo-1]
                candidates.append((gap, prev_len + 1, j, prev_gap))
        
        if candidates:
            # Pareto filter: sort by gap asc, length desc; keep increasing length
            candidates.sort(key=lambda x: (x[0], -x[1]))
            filtered = []
            max_len_so_far = 0
            for gap, length, prev_idx, prev_gap in candidates:
                if length > max_len_so_far:
                    filtered.append((gap, length, prev_idx, prev_gap))
                    max_len_so_far = length
            states[i] = filtered
            
            if filtered:
                last_len = filtered[-1][1]
                if last_len > best_length:
                    best_length = last_len
                    best_end = i
    
    # Reconstruct
    if best_length <= 1:
        return primes[:best_length]
    
    seq = []
    idx = best_end
    while idx != -1:
        seq.append(primes[idx])
        sj = states[idx]
        # Find the state with length == best_length (or the max length at this idx)
        # Since lengths are strictly increasing in the Pareto front, there's exactly one
        # state with the maximum length.
        target_len = best_length
        # Walk forward from idx: at each step, the state's length should decrease by 1
        # Actually, we need to track which state we're on. Let me redo this.
        pass
    
    # Better reconstruction: track the path during DP
    # Let me redo with explicit path tracking
    
    # Reset
    states2 = [[] for _ in range(n)]
    states2[0].append((0, 1, -1, 0))
    
    for i in range(1, n):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        candidates = []
        candidates.append((0, 1, -1, 0))
        
        for j in range(i):
            gap = primes[i] - primes[j]
            sj = states2[j]
            if not sj:
                continue
            
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            
            if lo > 0:
                prev_gap, prev_len, prev_idx, prev_prev_gap = sj[lo-1]
                candidates.append((gap, prev_len + 1, j, prev_gap))
        
        if candidates:
            candidates.sort(key=lambda x: (x[0], -x[1]))
            filtered = []
            max_len_so_far = 0
            for gap, length, prev_idx, prev_gap in candidates:
                if length > max_len_so_far:
                    filtered.append((gap, length, prev_idx, prev_gap))
                    max_len_so_far = length
            states2[i] = filtered
    
    # Now reconstruct using states2
    best_length2 = 1
    best_end2 = 0
    for i in range(n):
        if states2[i]:
            last_len = states2[i][-1][1]
            if last_len > best_length2:
                best_length2 = last_len
                best_end2 = i
    
    if best_length2 <= 1:
        return [primes[0]]
    
    seq = []
    idx = best_end2
    target_len = best_length2
    while idx != -1 and target_len >= 1:
        seq.append(primes[idx])
        sj = states2[idx]
        # Find the state with length == target_len
        target_state = None
        for s in sj:
            if s[1] == target_len:
                target_state = s
                break
        if target_state is None:
            # Shouldn't happen, but fallback
            target_state = sj[-1]
            target_len = target_state[1]
        prev_idx = target_state[2]
        target_len -= 1
        idx = prev_idx
    
    seq.reverse()
    
    # Verify
    if len(seq) < 2:
        return [primes[0]]
    
    valid = True
    for k in range(len(seq) - 1):
        if seq[k] >= seq[k+1] or seq[k] > 10000:
            valid = False
            break
    if valid and len(seq) >= 3:
        for k in range(2, len(seq)):
            gap_curr = seq[k] - seq[k-1]
            gap_prev = seq[k-1] - seq[k-2]
            if gap_curr < gap_prev:
                valid = False
                break
    
    if not valid:
        return [primes[0]]
    
    return seq
