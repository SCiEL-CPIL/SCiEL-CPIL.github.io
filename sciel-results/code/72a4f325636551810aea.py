# EVOLVE-BLOCK-START
import time
import math
import random
import heapq

DEADLINE = 145.0
MAX_N = 10 ** 18


def _sieve(limit):
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            for j in range(i * i, limit + 1, i):
                s[j] = False
    return [i for i in range(limit + 1) if s[i]]


PRIMES = _sieve(2000)
# Use a larger pool for search
POOL = [p for p in PRIMES if p <= 2000]
POOLSET = set(POOL)
POOL_INDEX = {p: i for i, p in enumerate(POOL)}
LARGE_POOL = POOL # Same pool for now


def _product(primes):
    n = 1
    for p in primes:
        n *= p
        if n > MAX_N:
            return n
    return n


_div_cache = {}

def _tau_perp(primes):
    """Count adjacent coprime pairs in the sorted divisor list of a squarefree n.
    Uses heap-merge of scaled divisor lists."""
    key = primes
    if key in _div_cache:
        return _div_cache[key]
    
    # Heap elements: (value, last_prime_index)
    h = [(1, -1)]
    g = math.gcd
    c = 0
    prev_div = 1
    
    while h:
        val, idx = heapq.heappop(h)
        
        if g(prev_div, val) == 1:
            c += 1
        prev_div = val
        
        for j in range(idx + 1, len(primes)):
            heapq.heappush(h, (val * primes[j], j))
            
    _div_cache[key] = c
    return c


def construct():
    start = time.time()

    def rem():
        return DEADLINE - time.time()

    # Start with primorial of first 15 primes as initial solution
    init_primes = tuple(p for p in PRIMES if p <= 47)  # first 15 primes
    init_n = _product(init_primes)
    if init_n > MAX_N:
        init_primes = list(init_primes)
        while init_n > MAX_N and init_primes:
            init_primes.pop()
            init_n = _product(tuple(init_primes))
        init_primes = tuple(init_primes)
    
    best_p = init_primes
    best_n = _product(best_p)
    best_s = _tau_perp(best_p)

    rng = random.Random(42)

    # Simulated Annealing
    T = 8.0
    T_min = 0.005
    cooling = 0.9997

    cur_p = best_p
    cur_s = best_s
    cur_n = best_n

    while rem() > 30 and T > T_min:
        move_type = rng.random()
        cur_set = set(cur_p)
        cur_list = list(cur_p)
        
        if move_type < 0.65 and cur_list:
            # Swap
            i = rng.randrange(len(cur_list))
            old_p = cur_list[i]
            max_q = MAX_N // (cur_n // old_p)
            candidates = [q for q in POOL if q not in cur_set and q <= max_q]
            if not candidates:
                T *= cooling
                continue
            new_p = rng.choice(candidates)
            new_list = cur_list[:]
            new_list[i] = new_p
            new_list.sort()
            new_tuple = tuple(new_list)
        elif move_type < 0.85:
            # Add
            max_q = MAX_N // cur_n
            candidates = [q for q in POOL if q not in cur_set and q <= max_q]
            if not candidates:
                T *= cooling
                continue
            new_p = rng.choice(candidates)
            new_list = cur_list + [new_p]
            new_list.sort()
            new_tuple = tuple(new_list)
        else:
            # Remove
            if len(cur_list) <= 2:
                T *= cooling
                continue
            i = rng.randrange(len(cur_list))
            new_list = cur_list[:]
            new_list.pop(i)
            new_tuple = tuple(new_list)

        new_s = _tau_perp(new_tuple)
        new_n = _product(new_tuple)
        
        delta = new_s - cur_s
        if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
            cur_p = new_tuple
            cur_s = new_s
            cur_n = new_n
            if cur_s > best_s:
                best_s = cur_s
                best_p = cur_p
                best_n = cur_n
        
        T *= cooling

    # Local search refinement - stochastic
    if rem() > 15:
        cur = best_p
        improved = True
        while improved and rem() > 10:
            improved = False
            cur_set = set(cur)
            cur_n = _product(cur)
            cur_s = _tau_perp(cur)
            
            # Try random moves
            best_move = None
            best_move_s = cur_s
            
            # Number of random tries
            tries = 200
            for _ in range(tries):
                move_type = rng.random()
                cur_list = list(cur)
                
                if move_type < 0.6 and cur_list:
                    # Swap
                    i = rng.randrange(len(cur_list))
                    old_p = cur_list[i]
                    max_q = MAX_N // (cur_n // old_p)
                    candidates = [q for q in POOL if q not in cur_set and q <= max_q]
                    if not candidates:
                        continue
                    new_p = rng.choice(candidates)
                    new_list = cur_list[:]
                    new_list[i] = new_p
                    new_list.sort()
                    new_tuple = tuple(new_list)
                elif move_type < 0.8:
                    # Add
                    max_q = MAX_N // cur_n
                    candidates = [q for q in POOL if q not in cur_set and q <= max_q]
                    if not candidates:
                        continue
                    new_p = rng.choice(candidates)
                    new_list = cur_list + [new_p]
                    new_list.sort()
                    new_tuple = tuple(new_list)
                else:
                    # Remove
                    if len(cur_list) <= 2:
                        continue
                    i = rng.randrange(len(cur_list))
                    new_list = cur_list[:]
                    new_list.pop(i)
                    new_tuple = tuple(new_list)
                
                s = _tau_perp(new_tuple)
                if s > best_move_s:
                    best_move_s = s
                    best_move = new_tuple
            
            if best_move is not None:
                cur = best_move
                improved = True
        
        if _tau_perp(cur) > best_s:
            best_s = _tau_perp(cur)
            best_p = cur
            best_n = _product(best_p)

    return best_n
# EVOLVE-BLOCK-END
