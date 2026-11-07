import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N = 256
    
    # 1. Sieve primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Calculate M_p (largest power of p <= N)
    M = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        M[p] = power
        
    # 3. Identify candidates
    # A candidate x > 1 is a number whose prime factors P_x allow a "gain"
    # Gain = x - sum(M_p for p in P_x)
    # We only care about x where Gain > 0.
    # Note: If x is a prime power p^k, P_x={p}. Gain = p^k - M_p = 0 (since M_p is max power).
    # So prime powers yield 0 gain relative to baseline. We can ignore them in the "improvement" search,
    # as they are already in the baseline.
    # We only need to search for composites with >= 2 distinct prime factors.
    
    candidates = [] # List of (gain, x, frozenset_of_primes)
    
    # Precompute prime factors for all numbers 2..256 for speed
    # Actually, just factor on the fly for 2..256 is fast enough.
    
    for x in range(2, N + 1):
        # Find distinct prime factors
        pf = set()
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                pf.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            pf.add(temp)
            
        # If x is a prime power, pf has size 1.
        # Gain is 0. We skip these as they don't improve over baseline M_p.
        if len(pf) == 1:
            continue
            
        # Calculate gain
        baseline_cost = sum(M[p] for p in pf)
        gain = x - baseline_cost
        
        if gain > 0:
            candidates.append((gain, x, frozenset(pf)))
            
    # Sort by gain descending to help B&B pruning
    candidates.sort(key=lambda item: item[0], reverse=True)
    num_candidates = len(candidates)
    
    # 4. Branch and Bound
    best_gain = 0
    best_selected = [] # List of indices into candidates
    
    # Suffix sums of gains for upper bound pruning
    suffix_gains = [0] * (num_candidates + 1)
    for i in range(num_candidates - 1, -1, -1):
        suffix_gains[i] = suffix_gains[i + 1] + candidates[i][0]
        
    def backtrack(idx, used_primes, current_gain, selected):
        nonlocal best_gain, best_selected
        
        # Check time limit
        if time.time() > deadline:
            return
            
        # Pruning: Even if we take all remaining, can we beat best?
        if current_gain + suffix_gains[idx] <= best_gain:
            return
            
        if idx == num_candidates:
            if current_gain > best_gain:
                best_gain = current_gain
                best_selected = list(selected)
            return

        # Option 1: Skip candidate idx
        backtrack(idx + 1, used_primes, current_gain, selected)

        # Option 2: Take candidate idx
        gain, x, pf = candidates[idx]
        # Check if primes in x are already used
        if used_primes.isdisjoint(pf):
            new_used = used_primes | pf
            selected.append(idx)
            backtrack(idx + 1, new_used, current_gain + gain, selected)
            selected.pop()

    backtrack(0, frozenset(), 0, [])
    
    # 5. Construct Result
    selected_primes = set()
    result = [1]
    
    # Add selected composites
    for idx in best_selected:
        gain, x, pf = candidates[idx]
        result.append(x)
        selected_primes.update(pf)
        
    # Add M_p for all primes NOT used in selected composites
    for p in primes:
        if p not in selected_primes:
            result.append(M[p])
            
    # 6. Verification and Fallback
    # The logic guarantees distinctness and coprimality if the sets are disjoint.
    # But let's be safe.
    
    # Check distinct
    if len(set(result)) != len(result):
        result = [1] + [M[p] for p in primes]
        
    # Check range
    valid = True
    for x in result:
        if x < 1 or x > N:
            valid = False
            break
    if not valid:
        result = [1] + [M[p] for p in primes]
        
    # Check coprime
    if valid:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if gcd(result[i], result[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        result = [1] + [M[p] for p in primes]

    return [int(x) for x in result]
