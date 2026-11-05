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
    
    # states[i] is a list of (gap, length, prev_gap)
    # prev_gap is the gap of the state in states[prev_idx] that was used.
    # If prev_idx is -1, prev_gap is 0 (or unused).
    # We need prev_idx to find the previous prime.
    # So state is (gap, length, prev_idx, prev_gap)
    
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
                # sj[lo-1] is the best state
                prev_gap, prev_len, prev_idx, prev_prev_gap = sj[lo-1]
                # New state at i
                # gap is current gap
                # length is prev_len + 1
                # prev_idx is j
                # prev_gap is the gap of the state used at j, which is prev_gap
                candidates.append((gap, prev_len + 1, j, prev_gap))
        
        if candidates:
            # Filter Pareto
            # Sort by gap, then by length descending (to handle duplicates)
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
        if not sj:
            break
        # Find the state in sj that has the max length (last in Pareto front)
        # But we need to track which specific state was used.
        # The state stored has (gap, length, prev_idx, prev_gap)
        # We need to find the state at idx that has length = best_length for the first step,
        # then for subsequent steps, we need the state with the specific gap.
        # Actually, let's just trace back using the stored prev_idx and prev_gap.
        # At each step, we need to find the state in states[idx] that matches.
        # For the first step, it's the max length state (sj[-1]).
        # For subsequent steps, we need to find the state with gap = prev_gap.
        if len(seq) == 1:
            # First step: take the max length state
            _, _, prev_idx, prev_gap = sj[-1]
        else:
            # Find state with gap = prev_gap (stored from previous step)
            # Binary search for prev_gap in sj
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] < prev_gap:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(sj) and sj[lo][0] == prev_gap:
                _, _, prev_idx, prev_gap = sj[lo]
            else:
                # Fallback: take last state
                _, _, prev_idx, prev_gap = sj[-1]
        idx = prev_idx
    
    seq.reverse()
    
    # Verification
    is_valid = True
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        for k in range(len(gaps) - 1):
            if gaps[k+1] < gaps[k]:
                is_valid = False
                break
    for p in seq:
        if p < 2 or p > 10000:
            is_valid = False
            break
    
    if not is_valid:
        return [primes[0]]
        
    return seq
