import math
import time
import random
from math import gcd

def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_up_to(60)

def _compute_score(primes, exponents):
    """Compute tau_perp score efficiently from prime factorization."""
    # Generate all divisors
    divisors = [1]
    for idx, (p, e) in enumerate(zip(primes, exponents)):
        if e == 0:
            continue
        current = list(divisors)
        pk = 1
        for k in range(1, e + 1):
            pk *= p
            for d in current:
                divisors.append(d * pk)
    divisors.sort()
    if len(divisors) < 2:
        return 0
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _compute_n(primes, exponents):
    n = 1
    for p, e in zip(primes, exponents):
        if e > 0:
            n *= p ** e
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    max_n = 10**18
    primes = _PRIMES
    
    # Determine how many primes we can use with all exponent 1
    prod = 1
    max_k = 0
    for i, p in enumerate(primes):
        if prod * p <= max_n:
            prod *= p
            max_k = i + 1
        else:
            break
    
    # Initial: primorial of first max_k primes
    current_exps = [1] * max_k
    current_n = _compute_n(primes, current_exps)
    current_score = _compute_score(primes, current_exps)
    
    best_score = current_score
    best_n = current_n
    best_exps = current_exps[:]
    
    # --- Phase 1: Greedy local search (fast, converges quickly) ---
    improved = True
    while improved and time.time() < deadline - 60:
        improved = False
        for i in range(len(current_exps)):
            if time.time() > deadline - 60:
                break
            p = primes[i]
            
            # Try increasing exponent
            if current_exps[i] < 8:
                new_exps = current_exps[:]
                new_exps[i] += 1
                new_n = _compute_n(primes, new_exps)
                if new_n <= max_n:
                    new_score = _compute_score(primes, new_exps)
                    if new_score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_exps = current_exps[:]
            
            # Try decreasing exponent (if > 1)
            if current_exps[i] > 1:
                new_exps = current_exps[:]
                new_exps[i] -= 1
                new_n = _compute_n(primes, new_exps)
                if new_n <= max_n:
                    new_score = _compute_score(primes, new_exps)
                    if new_score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_exps = current_exps[:]
    
    # --- Phase 2: Simulated Annealing (structural change - can escape local optima) ---
    if time.time() < deadline - 10:
        sa_start = time.time()
        sa_deadline = deadline - 5.0
        
        # Start from best found so far
        sa_exps = best_exps[:]
        sa_n = _compute_n(primes, sa_exps)
        sa_score = best_score
        sa_best_score = best_score
        sa_best_n = best_n
        sa_best_exps = best_exps[:]
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        n_steps = 20000
        
        step = 0
        while step < n_steps and time.time() < sa_deadline:
            step += 1
            # Linear cooling
            T = T_start * (T_end / T_start) ** (step / n_steps)
            
            # Random move: change one exponent by ±1, or add/remove a prime
            move_type = random.random()
            if move_type < 0.7:
                # Change exponent of existing prime
                i = random.randint(0, len(sa_exps) - 1)
                if sa_exps[i] == 0:
                    delta_e = 1
                elif sa_exps[i] >= 6:
                    delta_e = -1
                else:
                    delta_e = random.choice([-1, 1, 1])  # bias toward increasing
                new_exps = sa_exps[:]
                new_exps[i] = max(0, sa_exps[i] + delta_e)
                new_n = _compute_n(primes, new_exps)
            elif move_type < 0.85 and len(sa_exps) < max_k:
                # Add a new prime
                new_exps = sa_exps + [1]
                new_n = _compute_n(primes, new_exps)
            elif move_type < 0.95 and len(sa_exps) > 3:
                # Remove a prime
                i = random.randint(0, len(sa_exps) - 1)
                if sa_exps[i] == 0:
                    new_exps = sa_exps[:]
                    new_exps[i] = 0  # no-op
                    new_n = sa_n
                else:
                    new_exps = sa_exps[:]
                    new_exps[i] = 0
                    new_n = _compute_n(primes, new_exps)
            else:
                # Swap: remove one prime, add another
                if len(sa_exps) < 4:
                    continue
                i = random.randint(0, len(sa_exps) - 1)
                if sa_exps[i] == 0:
                    continue
                new_exps = sa_exps[:]
                new_exps[i] = 0
                # Find a prime to add
                used = set()
                for idx, e in enumerate(new_exps):
                    if e > 0:
                        used.add(idx)
                candidates = [j for j in range(max_k) if j not in used]
                if not candidates:
                    continue
                j = random.choice(candidates)
                # Insert at position j (maintain ordering)
                if j < len(new_exps):
                    new_exps.insert(j, 1)
                else:
                    new_exps.append(1)
                new_n = _compute_n(primes, new_exps)
            
            if new_n > max_n or new_n < 2:
                continue
            
            new_score = _compute_score(primes, new_exps)
            delta = new_score - sa_score
            
            if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                sa_exps = new_exps
                sa_n = new_n
                sa_score = new_score
                
                if sa_score > sa_best_score:
                    sa_best_score = sa_score
                    sa_best_n = sa_n
                    sa_best_exps = new_exps[:]
                    if sa_best_score > best_score:
                        best_score = sa_best_score
                        best_n = sa_best_n
                        best_exps = sa_best_exps[:]
    
    # --- Phase 3: Final greedy polish from SA result ---
    if time.time() < deadline - 5:
        polish_exps = best_exps[:]
        polish_n = _compute_n(primes, polish_exps)
        polish_score = best_score
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(polish_exps)):
                if time.time() > deadline - 2:
                    break
                # Try +1
                if polish_exps[i] < 10:
                    new_exps = polish_exps[:]
                    new_exps[i] += 1
                    new_n = _compute_n(primes, new_exps)
                    if new_n <= max_n:
                        new_score = _compute_score(primes, new_exps)
                        if new_score > polish_score:
                            polish_exps = new_exps
                            polish_n = new_n
                            polish_score = new_score
                            improved = True
                            if polish_score > best_score:
                                best_score = polish_score
                                best_n = polish_n
                                best_exps = polish_exps[:]
                # Try -1
                if polish_exps[i] > 0:
                    new_exps = polish_exps[:]
                    new_exps[i] -= 1
                    new_n = _compute_n(primes, new_exps)
                    if new_n <= max_n and new_n >= 2:
                        new_score = _compute_score(primes, new_exps)
                        if new_score > polish_score:
                            polish_exps = new_exps
                            polish_n = new_n
                            polish_score = new_score
                            improved = True
                            if polish_score > best_score:
                                best_score = polish_score
                                best_n = polish_n
                                best_exps = polish_exps[:]
    
    # Ensure validity
    if best_n < 2 or best_n > max_n:
        # Fallback to primorial
        n = 1
        for p in _PRIMES:
            if n * p <= max_n:
                n *= p
            else:
                break
        return n
    
    return best_n
