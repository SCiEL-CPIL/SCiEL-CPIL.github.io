import time
import math
import itertools

DEADLINE = 140.0
MAX_N = 10**18

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def _divisors_of(n):
    """Generate all divisors of n in sorted order."""
    # Factorize n
    factors = []
    temp = n
    for p in _SMALL_PRIMES:
        if p * p > temp:
            break
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors.append((p, count))
    if temp > 1:
        factors.append((temp, 1))
        
    divs = [1]
    for p, e in factors:
        new_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(divs):
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def _score_n(n):
    if n > MAX_N:
        return -1
    divs = _divisors_of(n)
    return _tau_perp(divs)

def _divisors_of_squarefree(primes):
    """Generate all divisors of a squarefree number given its prime list."""
    divs = [1]
    for p in primes:
        divs = divs + [d * p for d in divs]
    divs.sort()
    return divs

def _score_primes(primes):
    """Score for squarefree number with given primes."""
    n = 1
    for p in primes:
        n *= p
        if n > MAX_N:
            return -1, n
    divs = _divisors_of_squarefree(primes)
    return _tau_perp(divs), n

def _local_search_exponents(n, primes):
    """Try to improve n by increasing exponents of primes in 'primes'."""
    # n is assumed to be squarefree with respect to primes in 'primes'
    # We will try to increase exponents of small primes
    # and remove large primes if needed
    
    best_n = n
    best_score = _score_n(n)
    
    # Factorize n to get current exponents
    # We know the primes. Let's just try multiplying by p
    # and dividing by q
    
    # Strategy:
    # 1. Try increasing exponent of smallest primes
    # 2. Try swapping a large prime for a square of a small prime
    
    changed = True
    while changed:
        changed = False
        # Try increasing exponent of each prime
        for p in primes:
            if n % p == 0:
                # p is in n
                if n * p <= MAX_N:
                    new_n = n * p
                    new_score = _score_n(new_n)
                    if new_score > best_score:
                        best_score = new_score
                        best_n = new_n
                        n = new_n
                        changed = True
            else:
                # p is not in n, maybe we can add it?
                # But we are focusing on improving existing structure
                pass
        
        # Try swapping: remove a prime q, add p^2?
        # This is more complex. Let's stick to simple exponent increases first.
        
    return best_n, best_score

def construct():
    start = time.time()
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # seed value
    
    # First, evaluate the seed
    seed_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    s, n = _score_primes(seed_primes)
    best_score = s
    best_n = n
    
    # Search for best squarefree base
    # We will use a hill-climbing approach on the set of prime indices.
    # State: a set of indices into _SMALL_PRIMES.
    # Objective: Maximize score, subject to product <= MAX_N.
    
    # Initial candidates: Greedy smallest primes for various k
    candidates = []
    for k in range(2, 16):
        if time.time() - start > DEADLINE - 10:
            break
        primes = _SMALL_PRIMES[:k]
        n = 1
        valid = True
        for p in primes:
            n *= p
            if n > MAX_N:
                valid = False
                break
        if valid:
            candidates.append(set(range(k)))
    
    # Also add some random or swapped starts?
    # Let's just run hill climbing on the greedy starts.
    
    for start_set in candidates:
        if time.time() - start > DEADLINE - 5:
            break
            
        current_set = set(start_set)
        current_primes = [_SMALL_PRIMES[i] for i in current_set]
        current_score, current_n = _score_primes(current_primes)
        
        if current_score > best_score:
            best_score = current_score
            best_n = current_n
            
        improved = True
        while improved:
            improved = False
            # Try single swaps: remove i, add j
            # j must not be in current_set
            # i must be in current_set
            
            # Pre-calculate product of current_set to check constraint quickly?
            # Product can be large, but we only have ~15 elements.
            
            # To speed up, we can just try all pairs (i, j) where i in set, j not in set.
            # Number of swaps is |S| * (Total - |S|).
            # |S| <= 15, Total ~ 25. 15 * 10 = 150 swaps.
            
            for i in list(current_set):
                if time.time() - start > DEADLINE - 5:
                    break
                p_out = _SMALL_PRIMES[i]
                for j in range(len(_SMALL_PRIMES)):
                    if j in current_set:
                        continue
                    p_in = _SMALL_PRIMES[j]
                    
                    # Check constraint: new_n = current_n / p_out * p_in
                    # We need current_n. We can maintain it.
                    # But we don't have it in the loop scope easily without recalculating or passing it.
                    # Let's just calculate new_n.
                    
                    # Optimization: If p_in > p_out, n increases. If p_in < p_out, n decreases.
                    # We know current_n <= MAX_N.
                    # If p_in > p_out, we must check if current_n // p_out * p_in <= MAX_N.
                    # Integer division is safe if p_out divides current_n (which it does).
                    
                    new_n = (current_n // p_out) * p_in
                    if new_n > MAX_N:
                        continue
                    
                    new_set = current_set - {i} | {j}
                    new_primes = [_SMALL_PRIMES[idx] for idx in new_set]
                    new_score, _ = _score_primes(new_primes)
                    
                    if new_score > current_score:
                        current_set = new_set
                        current_primes = new_primes
                        current_score = new_score
                        current_n = new_n
                        improved = True
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                        break # Restart loop with new state to be thorough? Or just continue?
                        # Restarting is safer for hill climbing.
            
            # Try double swaps?
            # Remove i, k and add j, l.
            # This is O(|S|^2 * (Total-|S|)^2) which is ~ 15^2 * 10^2 = 22500 evaluations per step.
            # Too slow for Python if we do many steps.
            # Maybe only try double swaps if single swaps fail?
            
            if not improved:
                # Try a few random double swaps to escape local minima
                import random
                random.seed(42 + len(current_set)) # Deterministic-ish but varied
                for _ in range(50):
                    if time.time() - start > DEADLINE - 5:
                        break
                    if len(current_set) < 2:
                        break
                    # Pick 2 to remove
                    to_remove = random.sample(list(current_set), 2)
                    # Pick 2 to add
                    available = [j for j in range(len(_SMALL_PRIMES)) if j not in current_set]
                    if len(available) < 2:
                        continue
                    to_add = random.sample(available, 2)
                    
                    # Check constraint
                    # new_n = current_n / (p_i * p_k) * (p_j * p_l)
                    num = current_n
                    for idx in to_remove:
                        num //= _SMALL_PRIMES[idx]
                    for idx in to_add:
                        num *= _SMALL_PRIMES[idx]
                        
                    if num > MAX_N:
                        continue
                        
                    new_set = (current_set - set(to_remove)) | set(to_add)
                    new_primes = [_SMALL_PRIMES[idx] for idx in new_set]
                    new_score, _ = _score_primes(new_primes)
                    
                    if new_score > current_score:
                        current_set = new_set
                        current_primes = new_primes
                        current_score = new_score
                        current_n = num
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                        break

    return best_n
