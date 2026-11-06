"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

def _prime_factors(n):
    """Return set of prime factors of n using trial division."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _distinct_primes_in_pair_sums(arr):
    """Count distinct primes in all pair sums (distinct indices)."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            primes |= _prime_factors(arr[i] + arr[j])
    return len(primes)

def _evaluate(arr):
    """Evaluate the number of distinct prime factors in pair sums."""
    return _distinct_primes_in_pair_sums(arr)

def _generate_candidate_from_primes(target_primes, max_val=10000):
    """
    Try to construct 12 distinct positive integers such that all pair sums
    are divisible only by primes in target_primes.
    
    Strategy: pick numbers from a structured set that ensures pair sums
    have limited prime factors.
    """
    if not target_primes:
        return None
    
    # Sort primes for determinism
    primes = sorted(target_primes)
    
    # Try to find 12 numbers where all pair sums are composed of these primes
    # Start with small multiples and check
    
    candidates = []
    used = set()
    
    # Try numbers that are multiples of small products of the target primes
    # For example, if target_primes = {2, 3}, try multiples of 6
    
    # Generate candidate values
    base_values = []
    for p in primes:
        for k in range(1, 50):
            base_values.append(p * k)
    
    # Also try combinations
    if len(primes) >= 2:
        p1, p2 = primes[0], primes[1]
        for k in range(1, 30):
            base_values.append(p1 * p2 * k)
    
    # Filter to valid range
    base_values = [v for v in base_values if 1 <= v <= max_val]
    base_values = list(set(base_values))
    random.shuffle(base_values)
    
    # Greedily select 12 numbers
    for val in base_values:
        if len(candidates) >= 12:
            break
        if val in used:
            continue
        # Check if adding this value keeps all pair sums within target primes
        valid = True
        for existing in candidates:
            s = existing + val
            pf = _prime_factors(s)
            if not pf.issubset(set(target_primes)):
                valid = False
                break
        if valid:
            candidates.append(val)
            used.add(val)
    
    if len(candidates) == 12:
        return sorted(candidates)
    return None

def _simulated_annealing(arr, start_time, deadline, target_primes_hint=None):
    """
    Simulated annealing to minimize distinct prime factors in pair sums.
    """
    current = arr[:]
    current_cost = _evaluate(current)
    best = current[:]
    best_cost = current_cost
    
    if target_primes_hint is None:
        # Derive hint from current solution
        target_primes_hint = set()
        n = len(current)
        for i in range(n):
            for j in range(i + 1, n):
                target_primes_hint |= _prime_factors(current[i] + current[j])
    
    temperature = 5.0
    cooling_rate = 0.995
    min_temperature = 0.01
    
    while time.time() < deadline - 2 and temperature > min_temperature:
        # Generate a neighbor
        new_arr = current[:]
        idx = random.randint(0, 11)
        
        # Choose a mutation strategy
        strategy = random.random()
        
        if strategy < 0.3:
            # Small perturbation: add/subtract small amount
            delta = random.choice([-2, -1, 1, 2, -4, -3, 3, 4, -6, -5, 5, 6])
            new_val = current[idx] + delta
        elif strategy < 0.5:
            # Replace with a value that shares factors with existing sums
            # Try values near current
            new_val = current[idx] + random.randint(-20, 20)
        elif strategy < 0.7:
            # Try a value from the target primes hint structure
            if target_primes_hint:
                p = random.choice(list(target_primes_hint))
                new_val = p * random.randint(1, max(1, 10000 // p))
            else:
                new_val = random.randint(1, 10000)
        else:
            # Larger jump
            new_val = random.randint(1, 10000)
        
        # Validate
        if new_val < 1 or new_val > 10000:
            continue
        if new_val in current and current.count(new_val) > 1:
            continue
        if new_val in [v for v in current if v != current[idx]]:
            continue
            
        new_arr[idx] = new_val
        
        # Check distinctness
        if len(set(new_arr)) != 12:
            continue
        
        new_cost = _evaluate(new_arr)
        
        # Accept if better or with probability based on temperature
        if new_cost < current_cost:
            current = new_arr
            current_cost = new_cost
            if current_cost < best_cost:
                best = new_arr[:]
                best_cost = current_cost
        else:
            delta = new_cost - current_cost
            if delta > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current = new_arr
                    current_cost = new_cost
        
        temperature *= cooling_rate
    
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    random.seed(42)  # For reproducibility
    
    # Initial best
    best_arr = list(range(2, 25, 2))  # 2,4,6,...,24
    best_count = _evaluate(best_arr)
    
    # Strategy 1: Try to construct from small prime sets
    # The key insight: if all numbers are even, all pair sums are even,
    # so 2 is always a factor. If we can make all other factors shared,
    # we minimize the count.
    
    # Try different prime target sets
    small_prime_sets = [
        {2},
        {2, 3},
        {2, 5},
        {2, 7},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {2, 3, 5, 7},
        {2, 3, 5, 11},
        {3, 5},
        {3, 7},
        {5, 7},
    ]
    
    for pset in small_prime_sets:
        if time.time() > deadline - 5:
            break
        
        # Try to construct a solution with only these primes
        candidate = _generate_candidate_from_primes(pset)
        if candidate is not None:
            cost = _evaluate(candidate)
            if cost < best_count:
                best_count = cost
                best_arr = candidate
    
    # Strategy 2: Simulated annealing from multiple starting points
    if time.time() < deadline - 10:
        # Start from current best
        sa_best, sa_cost = _simulated_annealing(
            best_arr, start_time, deadline - 5
        )
        if sa_cost < best_count:
            best_count = sa_cost
            best_arr = sa_best
    
    # Strategy 3: More targeted search
    # Try arithmetic progressions with different steps
    if time.time() < deadline - 5:
        for step in range(2, 50):
            if time.time() > deadline - 3:
                break
            for start in range(1, min(10000 - 11*step, 100), step):
                candidate = [start + i * step for i in range(12)]
                if max(candidate) > 10000:
                    break
                if min(candidate) < 1:
                    continue
                if len(set(candidate)) != 12:
                    continue
                cost = _evaluate(candidate)
                if cost < best_count:
                    best_count = cost
                    best_arr = candidate
    
    # Final refinement with simulated annealing
    if time.time() < deadline - 2:
        sa_best, sa_cost = _simulated_annealing(
            best_arr, start_time, deadline - 1
        )
        if sa_cost < best_count:
            best_count = sa_cost
            best_arr = sa_best
    
    return best_arr

# EVOLVE-BLOCK-END
