"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import isqrt

# EVOLVE-BLOCK-START

def construct():
    """Return 12 distinct positive integers <= 10000 minimizing distinct prime
    factors of pairwise sums."""
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute primes up to 20000 (max sum is 20000)
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(limit) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    primes = sieve(20000)
    # Map from prime to index for fast lookup
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute smallest prime factor for numbers up to 20000
    spf = list(range(20001))
    for i in range(2, isqrt(20000) + 1):
        if spf[i] == i:
            for j in range(i*i, 20001, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
        """Return set of prime factors of n."""
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def count_distinct_primes(arr):
        """Count distinct prime factors across all pairwise sums."""
        all_primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                all_primes.update(factorize(s))
        return len(all_primes)
    
    # Strategy: use multiples of a small base to guarantee common prime factors
    # If all numbers are multiples of 6, then all sums are multiples of 6,
    # so primes 2 and 3 are always present. We then minimize additional primes.
    
    # Base set: multiples of 6 up to 10000
    # We pick 12 values from {6, 12, 18, ..., 9996}
    # That's 1666 possible values
    
    # Initial solution: evenly spaced multiples of 6
    def initial_solution():
        # Pick 12 well-spread multiples of 6
        base = 6
        max_val = 10000
        # Evenly spaced
        step = (max_val - base) // 11
        result = []
        for i in range(12):
            val = base + i * step
            if val > max_val:
                val = max_val
            if val % 6 != 0:
                val = (val // 6) * 6
            if val < 6:
                val = 6
            result.append(val)
        # Ensure distinct
        result = sorted(set(result))
        while len(result) < 12:
            result.append(result[-1] + 6)
        result = result[:12]
        return result
    
    # Better initial: try to minimize prime factors
    # Use local search with simulated annealing
    
    def local_search(arr, max_time=5.0):
        """Local search to minimize distinct prime factors."""
        current = arr[:]
        current_cost = count_distinct_primes(current)
        best = current[:]
        best_cost = current_cost
        start_ls = time.time()
        
        random.seed(42)
        
        while time.time() - start_ls < max_time and time.time() < deadline - 5:
            # Pick a random element and try to change it
            idx = random.randint(0, 11)
            old_val = current[idx]
            
            # Try random new values that are multiples of 6
            new_val = random.randint(1, 1666) * 6
            if new_val > 10000:
                new_val = 9996
            
            if new_val == old_val:
                continue
            
            # Check distinctness
            if new_val in current and new_val != old_val:
                continue
            
            current[idx] = new_val
            new_cost = count_distinct_primes(current)
            
            if new_cost <= current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best = current[:]
            else:
                # Revert
                current[idx] = old_val
        
        return best, best_cost
    
    def local_search_diverse(arr, max_time=5.0):
        """Local search allowing any value, not just multiples of 6."""
        current = arr[:]
        current_cost = count_distinct_primes(current)
        best = current[:]
        best_cost = current_cost
        start_ls = time.time()
        
        random.seed(123)
        
        while time.time() - start_ls < max_time and time.time() < deadline - 5:
            idx = random.randint(0, 11)
            old_val = current[idx]
            
            # Try random new values
            new_val = random.randint(1, 10000)
            
            if new_val == old_val:
                continue
            
            if new_val in current and new_val != old_val:
                continue
            
            current[idx] = new_val
            new_cost = count_distinct_primes(current)
            
            if new_cost <= current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best = current[:]
            else:
                current[idx] = old_val
        
        return best, best_cost
    
    def local_search_swap(arr, max_time=5.0):
        """Local search with swap moves."""
        current = arr[:]
        current_cost = count_distinct_primes(current)
        best = current[:]
        best_cost = current_cost
        start_ls = time.time()
        
        random.seed(777)
        
        while time.time() - start_ls < max_time and time.time() < deadline - 5:
            i = random.randint(0, 11)
            j = random.randint(0, 11)
            if i == j:
                continue
            
            old_i, old_j = current[i], current[j]
            new_i = random.randint(1, 10000)
            new_j = random.randint(1, 10000)
            
            # Check distinctness
            new_arr = current[:]
            new_arr[i] = new_i
            new_arr[j] = new_j
            if len(set(new_arr)) != 12:
                continue
            
            current = new_arr
            new_cost = count_distinct_primes(current)
            
            if new_cost <= current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best = current[:]
            else:
                current[i] = old_i
                current[j] = old_j
        
        return best, best_cost
    
    # Multiple restart strategies
    best_overall = None
    best_cost_overall = float('inf')
    
    # Strategy 1: Multiples of 6
    arr1 = initial_solution()
    best1, cost1 = local_search(arr1, max_time=15.0)
    if cost1 < best_cost_overall:
        best_cost_overall = cost1
        best_overall = best1
    
    if time.time() > deadline - 10:
        return best_overall
    
    # Strategy 2: Multiples of 30 (guarantees primes 2, 3, 5)
    def initial_solution_30():
        base = 30
        max_val = 10000
        step = (max_val - base) // 11
        result = []
        for i in range(12):
            val = base + i * step
            if val > max_val:
                val = max_val
            if val % 30 != 0:
                val = (val // 30) * 30
            if val < 30:
                val = 30
            result.append(val)
        result = sorted(set(result))
        while len(result) < 12:
            result.append(result[-1] + 30)
        result = result[:12]
        return result
    
    arr2 = initial_solution_30()
    best2, cost2 = local_search_diverse(arr2, max_time=10.0)
    if cost2 < best_cost_overall:
        best_cost_overall = cost2
        best_overall = best2
    
    if time.time() > deadline - 10:
        return best_overall
    
    # Strategy 3: Random restarts with multiples of 6
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        random.seed(1000 + restart)
        arr_r = []
        while len(arr_r) < 12:
            v = random.randint(1, 1666) * 6
            if v <= 10000 and v not in arr_r:
                arr_r.append(v)
        arr_r.sort()
        best_r, cost_r = local_search(arr_r, max_time=8.0)
        if cost_r < best_cost_overall:
            best_cost_overall = cost_r
            best_overall = best_r
    
    if time.time() > deadline - 10:
        return best_overall
    
    # Strategy 4: Try to get very low prime count with structured approach
    # If we can make all sums have only primes from a small set, that's ideal
    # Try multiples of 210 (2*3*5*7) - guarantees 4 primes minimum
    def initial_solution_210():
        base = 210
        max_val = 10000
        step = (max_val - base) // 11
        result = []
        for i in range(12):
            val = base + i * step
            if val > max_val:
                val = max_val
            if val % 210 != 0:
                val = (val // 210) * 210
            if val < 210:
                val = 210
            result.append(val)
        result = sorted(set(result))
        while len(result) < 12:
            result.append(result[-1] + 210)
        result = result[:12]
        return result
    
    arr4 = initial_solution_210()
    best4, cost4 = local_search_diverse(arr4, max_time=8.0)
    if cost4 < best_cost_overall:
        best_cost_overall = cost4
        best_overall = best4
    
    if time.time() > deadline - 10:
        return best_overall
    
    # Final polish: intensive local search on best solution
    if best_overall is not None:
        best_final, cost_final = local_search_swap(best_overall, max_time=min(10.0, deadline - time.time() - 2))
        if cost_final < best_cost_overall:
            best_cost_overall = cost_final
            best_overall = best_final
    
    if best_overall is None:
        best_overall = initial_solution()
    
    # Verify validity
    assert len(best_overall) == 12
    assert len(set(best_overall)) == 12
    assert all(1 <= x <= 10000 for x in best_overall)
    
    return best_overall

# EVOLVE-BLOCK-END
