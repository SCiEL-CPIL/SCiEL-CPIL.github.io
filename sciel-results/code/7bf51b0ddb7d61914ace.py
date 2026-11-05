import time
import random
import math

MAX_VAL = 20000
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(get_prime_factors(s))
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin within 60s limit
    
    # Precompute: for a given allowed prime set, check if n is P-smooth
    def make_smooth_checker(P):
        primes = sorted(P)
        def is_smooth(n):
            temp = n
            for p in primes:
                while temp % p == 0:
                    temp //= p
            return temp == 1
        return is_smooth
    
    # Try different prime sets, starting with smallest
    prime_sets = [
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
    ]
    
    best_arr = None
    best_cost = float('inf')
    
    # For each prime set, try to find 12 numbers with all pairwise sums P-smooth
    for P in prime_sets:
        if time.time() > deadline - 15:
            break
        
        is_smooth = make_smooth_checker(P)
        P_set = set(P)
        
        # Precompute for each x in [1,10000], the set of y values such that x+y is P-smooth
        # This is expensive, so we'll use a different approach:
        # For greedy construction, maintain a list of valid candidates incrementally
        
        best_for_P = None
        best_cost_P = float('inf')
        
        # Multiple restarts with greedy + local repair
        for restart in range(300):
            if time.time() > deadline - 10:
                break
            
            # Greedy construction
            arr = []
            used = set()
            
            # Pick first element randomly
            arr.append(random.randint(1, 10000))
            used.add(arr[0])
            
            success = True
            for step in range(11):
                # Find valid candidates: x not in used, x+arr[i] is P-smooth for all i in arr
                # Sample candidates more intelligently
                valid = []
                
                # Try to find candidates by checking smooth numbers near existing arr values
                # For each smooth number s, and each a in arr, x = s - a must be positive and <= 10000
                # Collect all such x values
                candidate_set = set()
                for a in arr:
                    # x = s - a where s is P-smooth, 1 <= x <= 10000
                    # s ranges from a+1 to a+10000
                    s_min = a + 1
                    s_max = a + 10000
                    # Generate P-smooth numbers in [s_min, s_max]
                    # Use recursive generation
                    def gen_smooth(lo, hi, idx=0, val=1):
                        if idx == len(primes):
                            if lo <= val <= hi:
                                yield val
                            return
                        p = primes[idx]
                        v = val
                        while v <= hi:
                            yield from gen_smooth(lo, hi, idx + 1, v)
                            if v > hi // p:
                                break
                            v *= p
                    
                    for s in gen_smooth(s_min, s_max):
                        x = s - a
                        if 1 <= x <= 10000 and x not in used:
                            # Check all other arr elements
                            ok = True
                            for b in arr:
                                if b == a:
                                    continue
                                if not is_smooth(b + x):
                                    ok = False
                                    break
                            if ok:
                                candidate_set.add(x)
                
                valid = list(candidate_set)
                
                if not valid:
                    success = False
                    break
                
                # Choose randomly from valid candidates
                x = random.choice(valid)
                arr.append(x)
                used.add(x)
            
            if success:
                cost = count_distinct_primes(arr)
                if cost < best_cost_P:
                    best_cost_P = cost
                    best_for_P = arr[:]
                
                # Local search to try to reduce cost further
                if cost <= len(P):
                    # Already using only primes from P, can't do better than len(P)
                    # But maybe we can find a set with fewer primes
                    pass
            
            # Also try local search from a random starting point
            if time.time() > deadline - 10:
                break
            
            # Random starting point + local search
            arr = [random.randint(1, 10000) for _ in range(12)]
            # Ensure distinct
            for i in range(12):
                for j in range(i + 1, 12):
                    if arr[i] == arr[j]:
                        arr[j] = arr[i] + 1
                        if arr[j] > 10000:
                            arr[j] = 1
        
        if best_for_P is not None:
            if best_cost_P < best_cost:
                best_cost = best_cost_P
                best_arr = best_for_P[:]
        
        # If we found a solution with cost <= len(P), we can stop trying larger P sets
        # (since we're going in order of increasing |P|)
        if best_cost <= len(P):
            break
    
    # Final local search polish on best_arr
    if best_arr is None:
        best_arr = [3 + 6 * k for k in range(12)]
        best_cost = count_distinct_primes(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    # Simulated annealing with incremental cost update
    T = 5.0
    cooling_rate = 0.9995
    
    while time.time() < deadline - 2:
        if T < 0.01:
            break
        
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        new_val = random.randint(1, 10000)
        
        if new_val in current_arr and new_val != old_val:
            continue
        
        # Compute delta cost: only pairs involving idx change
        old_primes = set()
        new_primes = set()
        for j in range(12):
            if j == idx:
                continue
            old_primes.update(get_prime_factors(old_val + current_arr[j]))
            new_primes.update(get_prime_factors(new_val + current_arr[j]))
        
        # Get all primes for full set (need to account for pairs not involving idx)
        all_primes_old = set()
        all_primes_new = set()
        for i in range(12):
            for j in range(i + 1, 12):
                if i == idx or j == idx:
                    continue
                all_primes_old.update(get_prime_factors(current_arr[i] + current_arr[j]))
                all_primes_new.update(get_prime_factors(current_arr[i] + current_arr[j]))
        
        old_total = all_primes_old | old_primes
        new_total = all_primes_new | new_primes
        
        old_cost = len(old_total)
        new_cost = len(new_total)
        
        current_arr[idx] = new_val
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_arr = current_arr[:]
        else:
            delta = new_cost - current_cost
            if delta > 0 and random.random() < math.exp(-delta / T):
                current_cost = new_cost
            else:
                current_arr[idx] = old_val
        
        T *= cooling_rate
    
    return best_arr
