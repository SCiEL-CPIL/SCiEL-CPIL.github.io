"""Constructs an integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Strategy: Use the product of the first 15 primes (primorial P_15) as a base,
then apply a greedy interval-density heuristic to select prime exponents that
maximize the density of support-disjoint boundaries within the 10^18 budget,
followed by a local hill-climbing polish pass.

For square-free numbers, the divisor lattice is a hypercube. Adjacent
divisors in sorted order often differ by a single prime factor, and many
adjacent pairs have disjoint prime supports (gcd 1). Maximizing k maximizes
the number of potential disjoint support transitions. The greedy heuristic
models the sorted divisor sequence as intervals where coprime transitions
occur, and selects exponents that maximize the density of such transitions
per unit of log-budget consumed.
"""

# EVOLVE-BLOCK-START
import time
from math import gcd, log
from collections import defaultdict

def _factorize(n):
    """Factorize n, returning dict of prime -> exponent."""
    factors = {}
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    return factors

def _generate_divisors(primes, exps):
    """Generate all divisors from prime list and exponents."""
    divs = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        pw = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pw)
            pw *= p
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp_from_factors(factors):
    """Compute tau_perp from factorization dict."""
    if not factors:
        return 0
    primes = sorted(factors.keys())
    exps = [factors[p] for p in primes]
    k = len(primes)
    
    divs = _generate_divisors(primes, exps)
    
    if all(e == 1 for e in exps):
        # Square-free: use bitmask
        divisor_mask = {}
        for d in divs:
            mask = 0
            for j, p in enumerate(primes):
                if d % p == 0:
                    mask |= (1 << j)
            divisor_mask[d] = mask
        count = 0
        for i in range(len(divs) - 1):
            if divisor_mask[divs[i]] & divisor_mask[divs[i+1]] == 0:
                count += 1
        return count
    else:
        count = 0
        for i in range(len(divs) - 1):
            if gcd(divs[i], divs[i + 1]) == 1:
                count += 1
        return count

def _tau_perp_fast(n):
    """Compute tau_perp for integer n."""
    if n < 2:
        return 0
    factors = _factorize(n)
    return _tau_perp_from_factors(factors)

def _greedy_exponent_selection(primes, budget_log, deadline):
    """
    Greedy interval-density heuristic: select exponents that maximize
    the density of coprime-support transitions per unit log-budget.
    
    For each prime, increasing its exponent adds (e+1) divisors per existing
    divisor, but consumes log(p) per exponent level. The marginal benefit
    of adding one more exponent to prime p is approximately the number of
    new adjacent pairs that will be coprime. We approximate this by the
    fraction of divisors that don't share p with their neighbors.
    
    Returns a list of exponents for the given primes.
    """
    k = len(primes)
    exps = [1] * k  # Start with square-free
    current_log = sum(log(p) for p in primes)
    
    # Precompute: for square-free, estimate coprime density
    # A new divisor d*p is coprime with its neighbor if the neighbor
    # doesn't contain p. Roughly, half the neighbors don't contain p
    # for large k.
    
    # Greedy: repeatedly add the exponent that gives best score improvement
    # per unit log cost. We use a surrogate: marginal tau_perp gain / log(p).
    # Since exact computation is expensive, we use a heuristic:
    # - Smaller primes give more divisors per log unit, so they're better
    #   for increasing divisor count (and thus potential coprime pairs)
    # - But we need to balance: too many exponents on small primes reduces
    #   the number of distinct primes we can include
    
    # Strategy: try adding exponents to primes in order of smallest log(p)
    # (i.e., smallest primes first), but stop when budget is exhausted.
    # This is a greedy approximation to the interval-density objective.
    
    # First, try to maximize number of distinct primes (k)
    # Then use remaining budget for exponents on smallest primes.
    
    # Sort primes by log value (ascending)
    sorted_indices = sorted(range(k), key=lambda i: log(primes[i]))
    
    # Phase 1: ensure all primes have exponent >= 1 (already done)
    # Phase 2: greedily add exponents to smallest primes
    for idx in sorted_indices:
        if time.time() > deadline:
            break
        p = primes[idx]
        log_p = log(p)
        while current_log + log_p <= budget_log:
            # Check if adding this exponent is beneficial
            # Heuristic: always beneficial for small primes since they
            # create many new divisors that can be coprime with neighbors
            exps[idx] += 1
            current_log += log_p
            if exps[idx] >= 8:  # Cap to avoid extreme imbalances
                break
    
    return exps

def _local_hill_climb(n, primes, exps, deadline, max_iter=50):
    """
    Local hill-climbing polish: try incrementing/decrementing exponents
    to find a locally optimal configuration.
    """
    current_score = _tau_perp_from_factors({p: e for p, e in zip(primes, exps) if e > 0})
    current_n = n
    current_exps = exps[:]
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(primes)):
            if time.time() > deadline:
                break
            p = primes[i]
            # Try increment
            if current_exps[i] < 10:
                new_exps = current_exps[:]
                new_exps[i] += 1
                new_n = current_n * p
                if new_n <= 10**18:
                    new_score = _tau_perp_from_factors({pr: ne for pr, ne in zip(primes, new_exps) if ne > 0})
                    if new_score > current_score:
                        current_score = new_score
                        current_n = new_n
                        current_exps = new_exps
                        improved = True
            # Try decrement (if exponent > 1)
            if current_exps[i] > 1:
                new_exps = current_exps[:]
                new_exps[i] -= 1
                new_n = current_n // p
                new_score = _tau_perp_from_factors({pr: ne for pr, ne in zip(primes, new_exps) if ne > 0})
                if new_score > current_score:
                    current_score = new_score
                    current_n = new_n
                    current_exps = new_exps
                    improved = True
        if not improved:
            break
    
    return current_n, current_score, current_exps

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Base: first 15 primes
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    P15 = 1
    for p in base_primes:
        P15 *= p
    
    best_n = P15
    best_score = _tau_perp_fast(P15)
    
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    # Helper to compute n from prime list and exponents
    def make_n(pk, exps):
        n = 1
        for p, e in zip(pk, exps):
            n *= (p ** e)
        return n
    
    # Greedy interval-density heuristic for exponent selection
    budget_log = log(10**18)
    
    # Try different k values with greedy exponent selection
    for k in [15, 14, 13, 12]:
        if time.time() > deadline - 30:
            break
        pk = primes_all[:k]
        base_val = 1
        for p in pk:
            base_val *= p
        
        if base_val > 10**18:
            continue
        
        # Greedy exponent selection
        greedy_exps = _greedy_exponent_selection(pk, budget_log, deadline)
        greedy_n = make_n(pk, greedy_exps)
        if greedy_n <= 10**18:
            greedy_score = _tau_perp_from_factors({p: e for p, e in zip(pk, greedy_exps) if e > 0})
            if greedy_score > best_score:
                best_score = greedy_score
                best_n = greedy_n
        
        # Local hill-climb polish on the greedy solution
        if time.time() < deadline - 20:
            polished_n, polished_score, polished_exps = _local_hill_climb(
                greedy_n, pk, greedy_exps, deadline - 10, max_iter=30
            )
            if polished_score > best_score:
                best_score = polished_score
                best_n = polished_n
    
    # Beam search for k distinct primes (kept from original, with improved limits)
    for k in range(12, 16):
        if time.time() > deadline - 15:
            break
        pk = primes_all[:k]
        base_val = 1
        for p in pk:
            base_val *= p
        
        if base_val > 10**18:
            continue
        
        # Beam search: start with all 1s, explore exponent increments
        candidates = [([1] * k, base_val)]
        seen_states = {tuple([1] * k)}
        
        max_steps = 200
        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            
            new_candidates = []
            for exps_c, val_c in candidates:
                for i in range(k):
                    if exps_c[i] >= 5:  # Limit exponent to keep search manageable
                        continue
                    new_exps = exps_c[:]
                    new_exps[i] += 1
                    new_val = val_c * pk[i]
                    if new_val > 10**18:
                        continue
                    key = tuple(new_exps)
                    if key not in seen_states:
                        seen_states.add(key)
                        new_candidates.append((new_exps, new_val))
            
            if not new_candidates:
                break
            
            # Score a subset of candidates
            scored = []
            limit = 20
            for exps_c, val_c in new_candidates[:limit]:
                if time.time() > deadline - 8:
                    break
                n_c = make_n(pk, exps_c)
                s = _tau_perp_from_factors({p: e for p, e in zip(pk, exps_c) if e > 0})
                scored.append((s, n_c, exps_c, val_c))
            
            scored.sort(key=lambda x: -x[0])
            
            for s, n_c, exps_c, val_c in scored:
                if s > best_score:
                    best_score = s
                    best_n = n_c
            
            # Keep top candidates for next round
            if scored:
                candidates = [(x[2], x[3]) for x in scored[:6]]
            else:
                break
    
    # Targeted searches with specific prime sets and exponent patterns
    if time.time() < deadline - 10:
        # 15 primes with extra 2s
        rest_15 = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        rest_prod_15 = 1
        for p in rest_15:
            rest_prod_15 *= p
        for a in range(1, 7):
            n_try = (2 ** a) * rest_prod_15
            if n_try <= 10**18:
                s = _tau_perp_fast(n_try)
                if s > best_score:
                    best_score = s
                    best_n = n_try
    
    if time.time() < deadline - 10:
        # 14 primes (drop 47)
        primes_14 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        prod_14 = 1
        for p in primes_14:
            prod_14 *= p
        budget = 10**18 // prod_14
        for a in range(1, min(8, budget // 2 + 1)):
            for b in range(1, min(5, budget // (2**a) // 3 + 1)):
                n_try = prod_14 * (2 ** a) * (3 ** b)
                if n_try <= 10**18:
                    s = _tau_perp_fast(n_try)
                    if s > best_score:
                        best_score = s
                        best_n = n_try
    
    if time.time() < deadline - 10:
        # 13 primes (drop 43, 47)
        primes_13 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        prod_13 = 1
        for p in primes_13:
            prod_13 *= p
        budget = 10**18 // prod_13
        for a in range(1, min(10, budget // 2 + 1)):
            for b in range(1, min(6, budget // (2**a) // 3 + 1)):
                for c in range(1, min(4, budget // (2**a) // (3**b) // 5 + 1)):
                    n_try = prod_13 * (2 ** a) * (3 ** b) * (5 ** c)
                    if n_try <= 10**18:
                        s = _tau_perp_fast(n_try)
                        if s > best_score:
                            best_score = s
                            best_n = n_try
    
    if time.time() < deadline - 5:
        # 12 primes (drop 41, 43, 47)
        primes_12 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        prod_12 = 1
        for p in primes_12:
            prod_12 *= p
        budget = 10**18 // prod_12
        for a in range(1, min(12, budget // 2 + 1)):
            for b in range(1, min(7, budget // (2**a) // 3 + 1)):
                for c in range(1, min(5, budget // (2**a) // (3**b) // 5 + 1)):
                    for d in range(1, min(3, budget // (2**a) // (3**b) // (5**c) // 7 + 1)):
                        n_try = prod_12 * (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                        if n_try <= 10**18:
                            s = _tau_perp_fast(n_try)
                            if s > best_score:
                                best_score = s
                                best_n = n_try
    
    return best_n
# EVOLVE-BLOCK-END
