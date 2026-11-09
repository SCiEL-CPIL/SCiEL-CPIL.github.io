import time
import random
import math
from collections import Counter

# Precompute small primes for faster factorization
_SMALL_PRIMES = []
def _init_primes():
    is_prime = [True] * 200
    is_prime[0] = is_prime[1] = False
    for i in range(2, 200):
        if is_prime[i]:
            _SMALL_PRIMES.append(i)
            for j in range(i*i, 200, i):
                is_prime[j] = False
_init_primes()

# Precompute prime factorization for all possible sums (2 to 20000)
_prime_factors_cache = [None] * 20001
def _precompute_prime_factors():
    for n in range(2, 20001):
        primes = set()
        temp = n
        for p in _SMALL_PRIMES:
            if p * p > temp:
                break
            while temp % p == 0:
                primes.add(p)
                temp //= p
        if temp > 1:
            primes.add(temp)
        _prime_factors_cache[n] = primes
    # Handle n=1
    _prime_factors_cache[1] = set()
_precompute_prime_factors()

def get_primes(n):
    """Return the set of prime factors of n."""
    if n == 1:
        return set()
    primes = set()
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            primes.add(p)
            n //= p
    if n > 1:
        primes.add(n)
    return primes

def get_primes_cached(n):
    """Return the set of prime factors of n with precomputed cache."""
    if n < 1 or n >= len(_prime_factors_cache):
        return get_primes(n)
    return _prime_factors_cache[n]

def evaluate(set_a):
    """Calculate the union of prime factors of all pairwise sums."""
    all_primes = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            all_primes.update(get_primes_cached(s))
    return all_primes

def evaluate_with_prime_counts(set_a):
    """Calculate the union of prime factors and count occurrences."""
    prime_counts = Counter()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            for p in get_primes_cached(s):
                prime_counts[p] += 1
    return prime_counts

# Precompute 7-smooth check: a number is 7-smooth if all its prime factors are in {2,3,5,7}
def _is_7_smooth(n):
    """Check if n is 7-smooth (all prime factors in {2,3,5,7})."""
    if n <= 1:
        return True
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1

# Precompute which sums (2..20000) are 7-smooth
_7smooth_sums = [False] * 20001
def _precompute_7smooth():
    for s in range(2, 20001):
        _7smooth_sums[s] = _is_7_smooth(s)
_precompute_7smooth()

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    def time_left():
        return deadline - time.time()

    # Phase 0: Try to find a 12-clique in the 7-smooth sum compatibility graph
    # If found, we get at most 4 distinct primes {2,3,5,7}
    
    # Build compatibility: for each number 1..10000, which other numbers are compatible
    # (i.e., their sum is 7-smooth)
    # We'll use a greedy backtracking approach
    
    # First, let's try a direct construction approach:
    # Numbers that are multiples of a highly composite number tend to have smooth sums
    # Let's try: all numbers of the form k * L where L is lcm of small primes
    
    # Strategy: Try to find 12 numbers such that all pairwise sums are 7-smooth
    # This means for any pair (a,b): a+b must only have prime factors 2,3,5,7
    
    # Key observation: if a and b are both multiples of M, then a+b is a multiple of M
    # If M is 7-smooth, then a+b = M*(a/M + b/M), and we need a/M + b/M to be 7-smooth too
    
    # Let's try a backtracking search with the 7-smooth constraint
    
    # Precompute: for each number n in 1..10000, the set of numbers m in 1..10000
    # such that n+m is 7-smooth. But this is too large (10000*10000).
    # Instead, use forward checking in backtracking.
    
    # Better approach: generate candidate numbers that are "likely" to work
    # Numbers that are multiples of 2, 3, 5, 7 (i.e., 7-smooth themselves) are good candidates
    # because if a and b are 7-smooth, a+b might also be 7-smooth
    
    # Actually, let's just do a smart backtracking:
    # 1. Start with a pool of candidate numbers
    # 2. Greedily add numbers that are compatible with all previously chosen ones
    
    # Generate a pool of candidate numbers (7-smooth numbers up to 10000)
    smooth_candidates = []
    for n in range(1, 10001):
        if _is_7_smooth(n):
            smooth_candidates.append(n)
    
    # Also include some non-7-smooth numbers that might work
    # But start with 7-smooth pool
    
    # Try greedy: pick numbers one by one, ensuring compatibility with all previous
    def try_greedy_7smooth(pool, target_size=12):
        """Greedily build a set where all pairwise sums are 7-smooth."""
        if len(pool) < target_size:
            return None
        result = []
        remaining = list(pool)
        while len(result) < target_size and remaining:
            # Pick the number with maximum compatibility with remaining
            best_idx = -1
            best_compat = -1
            for i, candidate in enumerate(remaining):
                compat = 0
                for j in range(len(remaining)):
                    if i == j:
                        continue
                    if _7smooth_sums[candidate + remaining[j]]:
                        compat += 1
                # Also check compatibility with already chosen
                ok = True
                for chosen in result:
                    if not _7smooth_sums[candidate + chosen]:
                        ok = False
                        break
                if not ok:
                    continue
                if compat > best_compat:
                    best_compat = compat
                    best_idx = i
            if best_idx == -1:
                break
            result.append(remaining[best_idx])
            remaining.pop(best_idx)
        if len(result) >= target_size:
            return result[:target_size]
        return None
    
    # Try with 7-smooth candidates
    best_set = None
    best_count = float('inf')
    best_primes = None
    
    greedy_result = try_greedy_7smooth(smooth_candidates)
    if greedy_result:
        primes = evaluate(greedy_result)
        best_count = len(primes)
        best_primes = primes
        best_set = greedy_result[:]
    
    # If greedy didn't work well, try backtracking
    if best_set is None or best_count > 4:
        # Backtracking with forward checking
        def backtrack_7smooth(selected, available, target_size=12):
            if len(selected) == target_size:
                return selected[:]
            if not available:
                return None
            if time_left() < 5:
                return None
            
            # Sort available by degree (number of compatible neighbors in available)
            # for better pruning
            for candidate in available:
                # Check compatibility with all selected
                ok = True
                for s in selected:
                    if not _7smooth_sums[candidate + s]:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Forward check: can we still complete?
                # Count how many remaining candidates are compatible with candidate
                # and with all selected
                new_selected = selected + [candidate]
                new_available = [x for x in available if x != candidate and all(_7smooth_sums[x + s] for s in new_selected)]
                
                if len(new_available) >= target_size - len(new_selected) - 1:
                    result = backtrack_7smooth(new_selected, new_available, target_size)
                    if result is not None:
                        return result
            return None
        
        # Use a smaller pool for backtracking to keep it tractable
        # Focus on smaller 7-smooth numbers
        small_smooth = [x for x in smooth_candidates if x <= 5000]
        bt_result = backtrack_7smooth([], small_smooth)
        if bt_result:
            primes = evaluate(bt_result)
            if len(primes) < best_count:
                best_count = len(primes)
                best_primes = primes
                best_set = bt_result[:]
    
    # If we found a 4-prime solution, we're done
    if best_set is not None and best_count <= 4:
        return best_set
    
    # Fallback: use the original heuristic approach
    if best_set is None:
        best_set = [2**i for i in range(12)]
        best_primes = evaluate(best_set)
        best_count = len(best_primes)
    
    # Phase 1: Local search to improve
    current_set = best_set[:]
    current_primes = best_primes
    current_count = best_count
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time_left() > 10:
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        r = random.random()
        if r < 0.4:
            new_val = old_val + random.randint(-100, 100)
        elif r < 0.7:
            new_val = random.randint(1, 10000)
        else:
            other_idx = random.randint(0, 11)
            if other_idx == idx:
                other_idx = (idx + 1) % 12
            base = current_set[other_idx]
            step = random.choice([6, 12, 30, 60])
            new_val = base + random.randint(-5, 5) * step
        
        if new_val < 1 or new_val > 10000 or new_val == old_val:
            continue
        if new_val in current_set and current_set.index(new_val) != idx:
            continue
        
        new_set = current_set[:]
        new_set[idx] = new_val
        new_primes = evaluate(new_set)
        new_count = len(new_primes)
        
        if new_count < current_count:
            current_set = new_set
            current_primes = new_primes
            current_count = new_count
            no_improve_count = 0
            if current_count < best_count:
                best_count = current_count
                best_primes = current_primes
                best_set = current_set[:]
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve and time_left() > 20:
            no_improve_count = 0
    
    # Phase 2: Targeted prime elimination
    current_set = best_set[:]
    current_primes = set(best_primes)
    current_count = best_count
    
    while time_left() > 5:
        prime_counts = evaluate_with_prime_counts(current_set)
        sorted_primes = sorted(prime_counts.items(), key=lambda x: x[1])
        
        improved = False
        for prime_to_elim, count in sorted_primes:
            if time_left() < 5:
                break
            if prime_to_elim not in current_primes:
                continue
            
            for idx in range(12):
                if time_left() < 5:
                    break
                old_val = current_set[idx]
                for d in range(-50, 51):
                    new_val = old_val + d
                    if new_val < 1 or new_val > 10000 or new_val == old_val:
                        continue
                    if new_val in set(current_set) - {old_val}:
                        continue
                    
                    new_set = current_set[:]
                    new_set[idx] = new_val
                    
                    # Quick check
                    has_target = False
                    for j in range(12):
                        if j == idx:
                            continue
                        s = new_val + current_set[j]
                        if prime_to_elim in get_primes_cached(s):
                            has_target = True
                            break
                    if has_target:
                        continue
                    
                    new_primes = evaluate(new_set)
                    new_count = len(new_primes)
                    
                    if new_count < current_count and prime_to_elim not in new_primes:
                        current_set = new_set
                        current_primes = new_primes
                        current_count = new_count
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_primes = current_primes
                            best_set = current_set[:]
                        break
                if improved:
                    break
            if improved:
                continue
            else:
                break
    
    return best_set
