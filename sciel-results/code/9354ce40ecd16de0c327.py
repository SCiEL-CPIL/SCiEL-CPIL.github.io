"""Rewritten program for maximizing tau_perp(n)."""
import math
import time
import random
from functools import lru_cache
from itertools import product

# EVOLVE-BLOCK-START

def _factorize(n):
    """Factorize n into {prime: exponent}."""
    if n < 2:
        return {}
    factors = {}
    d = 2
    x = n
    while d * d <= x:
        while x % d == 0:
            factors[d] = factors.get(d, 0) + 1
            x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        factors[x] = factors.get(x, 0) + 1
    return factors


def _score_fast(primes, exps):
    """
    Compute tau_perp directly from prime factorization.
    Optimization: Sort divisors by their logarithm to avoid large integer multiplications.
    """
    k = len(primes)
    if k == 0:
        return 0
    
    # Precompute log primes
    log_primes = [math.log(p) for p in primes]
    
    # Generate all divisor exponent tuples
    ranges = [range(e + 1) for e in exps]
    div_tuples = list(product(*ranges))
    
    if not div_tuples:
        return 0

    # Calculate log values for sorting
    # We store (log_val, tuple) to sort by value
    # Using log avoids overflow and is faster than big int multiplication for sorting
    # Note: Distinct divisors have distinct log values (since primes are distinct)
    # Floating point precision: sum of ~20 logs of small primes.
    # Max exponent is ~60 for 2. 60 * log(2) ~ 41. 
    # Sum for 15 primes with max exps is manageable.
    # To be safe against floating point collisions (unlikely for distinct integers),
    # we could use tuple of (int_val) if we were worried, but log is standard for this size.
    # Actually, let's stick to integer values if the number of divisors is small, 
    # but for large counts (10^5+), log is better.
    # Let's use a mixed approach or just log. Log is sufficient for ordering distinct integers < 10^18.
    
    def get_log_val(t):
        v = 0.0
        for i in range(k):
            if t[i]:
                v += t[i] * log_primes[i]
        return v

    # Sort by log value
    # We need to recover the tuple for the coprime check
    # We can zip them
    indexed = [(get_log_val(t), t) for t in div_tuples]
    indexed.sort(key=lambda x: x[0])
    
    # Extract sorted tuples
    sorted_tuples = [item[1] for item in indexed]
    
    # Count adjacent coprime pairs
    score = 0
    for i in range(len(sorted_tuples) - 1):
        a = sorted_tuples[i]
        b = sorted_tuples[i + 1]
        coprime = True
        for j in range(k):
            if a[j] > 0 and b[j] > 0:
                coprime = False
                break
        if coprime:
            score += 1
    return score


def _score(n):
    """Compute tau_perp(n) from the integer n."""
    if n < 2:
        return 0
    factors = _factorize(n)
    primes = sorted(factors.keys())
    exps = [factors[p] for p in primes]
    return _score_fast(primes, exps)


def _generate_initial_candidates():
    """Generate a diverse set of starting points for optimization."""
    LIMIT = 10**18
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    candidates = []
    
    # 1. Squarefree products
    n = 1
    for k in range(2, len(primes_list) + 1):
        p = primes_list[k-1]
        if n * p > LIMIT:
            break
        n *= p
        candidates.append((primes_list[:k], [1]*k, n))
        
    # 2. Some structured non-squarefree
    # Try increasing exponents of small primes
    for k in range(3, 10):
        base_primes = primes_list[:k]
        # Start with [1]*k and try boosting first few
        for boost in range(2, 6):
            exps = [1]*k
            exps[0] = boost
            n_val = 1
            valid = True
            for i, p in enumerate(base_primes):
                n_val *= p ** exps[i]
                if n_val > LIMIT:
                    valid = False
                    break
            if valid:
                candidates.append((base_primes, exps, n_val))
                
            if k > 2:
                exps2 = [1]*k
                exps2[0] = 2
                exps2[1] = 2
                n_val = 1
                valid = True
                for i, p in enumerate(base_primes):
                    n_val *= p ** exps2[i]
                    if n_val > LIMIT:
                        valid = False
                        break
                if valid:
                    candidates.append((base_primes, exps2, n_val))
                    
    # 3. Random valid configurations
    random.seed(42)
    for _ in range(50):
        k = random.randint(3, 12)
        primes_k = primes_list[:k]
        exps = [1] * k
        n_val = 1
        # Randomly increase some exponents
        for i in range(k):
            if random.random() < 0.5:
                max_e = int(math.log(LIMIT / n_val) / math.log(primes_k[i]))
                if max_e > 1:
                    exps[i] = random.randint(1, max_e)
                    n_val *= primes_k[i] ** (exps[i] - 1)
        if n_val <= LIMIT:
             candidates.append((primes_k, exps, n_val))

    return candidates

def _simulated_annealing(primes, exps, n, time_limit=20.0):
    """
    Perform Simulated Annealing on the exponent vector.
    State: exps list.
    Move: Change one exponent by +/- 1.
    Constraint: n <= 10^18.
    """
    LIMIT = 10**18
    start_time = time.time()
    
    # Calculate initial log_n to track constraint efficiently
    log_limit = math.log(LIMIT)
    
    def calc_log_n(exps):
        l = 0.0
        for i, e in enumerate(exps):
            if e:
                l += e * math.log(primes[i])
        return l

    current_exps = list(exps)
    current_log_n = calc_log_n(current_exps)
    current_score = _score_fast(primes, current_exps)
    
    best_exps = list(current_exps)
    best_score = current_score
    best_n = n
    
    # Cooling schedule
    T_start = 1.0
    T_end = 0.01
    steps = 5000 # Approx steps, we will loop until time limit
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        # Calculate current temperature
        # Linear cooling
        T = T_start * (T_end / T_start) ** (step / steps)
        
        # Choose a random prime index to modify
        i = random.randint(0, len(primes) - 1)
        
        # Choose delta: +1 or -1
        delta = random.choice([1, -1])
        
        new_exp = current_exps[i] + delta
        
        # Check bounds
        if new_exp < 0:
            continue
            
        # Check size constraint using logs (fast)
        # new_log_n = current_log_n + delta * log(primes[i])
        new_log_n = current_log_n + delta * math.log(primes[i])
        
        if new_log_n > log_limit:
            # If we try to go up and exceed limit, reject
            if delta > 0:
                continue
            # If we try to go down, it's always safe from limit perspective
            # But we must ensure new_exp >= 0 (checked above)
            
        # Apply move
        current_exps[i] = new_exp
        current_log_n = new_log_n
        
        # Calculate new score
        # Optimization: If the move is small, we might be able to update score incrementally?
        # No, the sorting order changes globally. We must recompute.
        new_score = _score_fast(primes, current_exps)
        
        # Metropolis criterion
        if new_score > current_score:
            # Accept
            pass
        else:
            # Calculate delta
            delta_score = new_score - current_score
            if delta_score < 0:
                # Accept with probability exp(delta_score / T)
                prob = math.exp(delta_score / T)
                if random.random() > prob:
                    # Reject
                    current_exps[i] -= delta
                    current_log_n -= delta * math.log(primes[i])
                    continue
            # Accept
            
        # Update best
        if new_score > best_score:
            best_score = new_score
            best_exps = list(current_exps)
            # Calculate best_n
            best_n = 1
            for j, p in enumerate(primes):
                if best_exps[j] > 0:
                    best_n *= p ** best_exps[j]
                    
        current_score = new_score
        
    return best_n, best_score, best_exps

def construct():
    deadline = time.time() + 145  # Safety margin
    
    # Phase 1: Generate initial candidates
    candidates = _generate_initial_candidates()
    
    best_n = 2
    best_score = 0
    best_primes = []
    best_exps = []
    
    # Evaluate candidates to find a good starting point
    for primes, exps, n in candidates:
        if time.time() > deadline - 10:
            break
        s = _score_fast(primes, exps)
        if s > best_score:
            best_score = s
            best_n = n
            best_primes = primes
            best_exps = exps
            
    # Phase 2: Simulated Annealing from the best candidate
    if time.time() < deadline - 20 and best_primes:
        sa_n, sa_score, sa_exps = _simulated_annealing(best_primes, best_exps, best_n, time_limit=120)
        if sa_score > best_score:
            best_score = sa_score
            best_n = sa_n
            best_primes = best_primes # Primes don't change in this SA variant
            best_exps = sa_exps
            
    # Phase 3: Try SA from a few other diverse candidates if time permits
    if time.time() < deadline - 30:
        # Pick 2-3 other top candidates
        scored_candidates = []
        for primes, exps, n in candidates:
            if (primes, tuple(exps)) == (best_primes, tuple(best_exps)):
                continue
            s = _score_fast(primes, exps)
            scored_candidates.append((s, primes, exps, n))
            
        scored_candidates.sort(reverse=True)
        
        for s, primes, exps, n in scored_candidates[:3]:
            if time.time() > deadline - 15:
                break
            # Run shorter SA
            sa_n, sa_score, sa_exps = _simulated_annealing(primes, exps, n, time_limit=10)
            if sa_score > best_score:
                best_score = sa_score
                best_n = sa_n
                best_primes = primes
                best_exps = sa_exps
                
    # Final safety check
    if best_n < 2 or best_n > 10**18:
        return 2
        
    return best_n

# EVOLVE-BLOCK-END
