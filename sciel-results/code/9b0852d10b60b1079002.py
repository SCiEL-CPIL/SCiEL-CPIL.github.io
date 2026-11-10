"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

# Precompute primes up to 40
def _get_primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

N = P['n']
MAX_SIZE = P['maximum_size']
PRIMES = _get_primes(N)
N_PRIMES = len(PRIMES)
CANDIDATES = list(range(2, N + 1))

# Precompute prime-exponent vectors for all candidates
def _factorize(n):
    fac = {}
    for p in PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            fac[p] = fac.get(p, 0) + 1
            n //= p
    if n > 1:
        fac[n] = fac.get(n, 0) + 1
    return fac

ALL_VECS = {}
for num in CANDIDATES:
    fac = _factorize(num)
    vec = [0] * N_PRIMES
    for i, p in enumerate(PRIMES):
        if p in fac:
            vec[i] = fac[p]
    ALL_VECS[num] = vec

# Complexity metrics for ordering heuristics
def _total_prime_factors(num):
    n = num
    count = 0
    for p in PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            count += 1
            n //= p
    if n > 1:
        count += 1
    return count

def _distinct_prime_factors(num):
    n = num
    count = 0
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            count += 1
        while n % p == 0:
            n //= p
    if n > 1:
        count += 1
    return count

TOTAL_PF = {x: _total_prime_factors(x) for x in CANDIDATES}
DISTINCT_PF = {x: _distinct_prime_factors(x) for x in CANDIDATES}

# Precompute a few deterministic orderings
ORDER_TOTAL_PF = sorted(CANDIDATES, key=lambda x: (TOTAL_PF[x], x))
ORDER_ASC = sorted(CANDIDATES)
ORDER_DESC = sorted(CANDIDATES, reverse=True)
ORDER_PRIMES_FIRST = sorted(CANDIDATES, key=lambda x: (DISTINCT_PF[x], TOTAL_PF[x], x))

ZERO_VEC = (0,) * N_PRIMES

def _add_element_check(existing_products, new_vec):
    """Return (True, new_prods) if adding new_vec causes no collision, else (False, None)."""
    new_products = set()
    for p in existing_products:
        np = tuple(p[i] + new_vec[i] for i in range(N_PRIMES))
        if np in existing_products:
            return (False, None)
        new_products.add(np)
    return (True, new_products)

def _greedy_fill(ordered, max_size, deadline):
    """Greedy fill following ordered candidates, using vector-based collision check."""
    selected = []
    products = {ZERO_VEC}
    
    for cand in ordered:
        if time.time() > deadline:
            break
        if len(selected) >= max_size:
            break
        
        vec = ALL_VECS[cand]
        valid, new_prods = _add_element_check(products, vec)
        if valid:
            selected.append(cand)
            products.update(new_prods)
    
    return selected, products

def _rebuild_products(selected):
    """Rebuild product set from a selected list."""
    products = {ZERO_VEC}
    for x in selected:
        vec = ALL_VECS[x]
        new_prods = set()
        for p in products:
            np = tuple(p[i] + vec[i] for i in range(N_PRIMES))
            new_prods.add(np)
        products.update(new_prods)
    return products

def _local_search_single_swap(selected, candidates, max_size, deadline):
    """Try removing one element and greedily re-adding to improve."""
    best = list(selected)
    best_len = len(best)
    
    for i in range(len(best)):
        if time.time() > deadline:
            break
        if best_len >= max_size:
            break
        
        removed = best[i]
        trial = best[:i] + best[i+1:]
        trial_set = set(trial)
        
        products = _rebuild_products(trial)
        
        # Greedily add from candidates not in trial
        for cand in candidates:
            if time.time() > deadline:
                break
            if len(trial) >= max_size:
                break
            if cand in trial_set:
                continue
            vec = ALL_VECS[cand]
            valid, new_prods = _add_element_check(products, vec)
            if valid:
                trial.append(cand)
                trial_set.add(cand)
                products.update(new_prods)
        
        if len(trial) > best_len:
            best = trial
            best_len = len(trial)
    
    return best, best_len

def _local_search_double_swap(selected, candidates, max_size, deadline):
    """Try removing two elements and greedily re-adding to improve."""
    best = list(selected)
    best_len = len(best)
    
    for i in range(len(best)):
        if time.time() > deadline:
            break
        if best_len >= max_size:
            break
        for j in range(i + 1, len(best)):
            if time.time() > deadline:
                break
            if best_len >= max_size:
                break
            
            trial = best[:i] + best[i+1:j] + best[j+1:]
            trial_set = set(trial)
            
            products = _rebuild_products(trial)
            
            for cand in candidates:
                if time.time() > deadline:
                    break
                if len(trial) >= max_size:
                    break
                if cand in trial_set:
                    continue
                vec = ALL_VECS[cand]
                valid, new_prods = _add_element_check(products, vec)
                if valid:
                    trial.append(cand)
                    trial_set.add(cand)
                    products.update(new_prods)
            
            if len(trial) > best_len:
                best = trial
                best_len = len(trial)
    
    return best, best_len

def _validate(selected):
    """Final validation: ensure all 2^k subset products are truly distinct."""
    k = len(selected)
    if k == 0:
        return True
    if k > 20:
        return False
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= selected[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin within 60s limit
    
    candidates = CANDIDATES
    max_size = MAX_SIZE
    
    best_set = []
    best_len = 0
    
    # Phase 1: Deterministic greedy with different orderings
    orderings = [ORDER_TOTAL_PF, ORDER_ASC, ORDER_DESC, ORDER_PRIMES_FIRST]
    
    for order in orderings:
        if time.time() > deadline:
            break
        sel, _ = _greedy_fill(order, max_size, deadline)
        if len(sel) > best_len:
            best_len = len(sel)
            best_set = sel[:]
        if best_len >= max_size:
            break
    
    # Phase 2: Random restarts with mixed strategies
    num_restarts = 50000
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        if best_len >= max_size:
            break
        
        r = random.random()
        if r < 0.2:
            ordered = ORDER_TOTAL_PF[:]
        elif r < 0.4:
            ordered = ORDER_ASC[:]
        elif r < 0.6:
            ordered = ORDER_DESC[:]
        else:
            ordered = candidates[:]
            random.shuffle(ordered)
        
        sel, _ = _greedy_fill(ordered, max_size, deadline)
        if len(sel) > best_len:
            best_len = len(sel)
            best_set = sel[:]
    
    # Phase 3: Local search on best solution
    if best_len < max_size and time.time() < deadline:
        best_set, best_len = _local_search_single_swap(best_set, candidates, max_size, deadline)
    
    # Phase 4: More random restarts if still not at max
    if best_len < max_size and time.time() < deadline:
        for _ in range(20000):
            if time.time() > deadline:
                break
            if best_len >= max_size:
                break
            
            r = random.random()
            if r < 0.3:
                ordered = ORDER_TOTAL_PF[:]
            else:
                ordered = candidates[:]
                random.shuffle(ordered)
            
            sel, _ = _greedy_fill(ordered, max_size, deadline)
            if len(sel) > best_len:
                best_len = len(sel)
                best_set = sel[:]
        
        # Another round of local search
        if best_len < max_size and time.time() < deadline:
            best_set, best_len = _local_search_single_swap(best_set, candidates, max_size, deadline)
            if best_len < max_size and time.time() < deadline:
                best_set, best_len = _local_search_double_swap(best_set, candidates, max_size, deadline)
    
    # Phase 5: Final validation
    if best_set and not _validate(best_set):
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
