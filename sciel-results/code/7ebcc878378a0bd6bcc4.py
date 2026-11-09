"""Maximize coprime adjacent divisor pairs for n <= 1e18 (multi-start SA + local search)."""
# EVOLVE-BLOCK-START
import time
import math
import random

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


PRIMES = _sieve(500)
# Pool of primes to consider
POOL = [p for p in PRIMES if p <= 200]
POOLSET = set(POOL)
POOL_INDEX = {p: i for i, p in enumerate(POOL)}


def _product(primes):
    n = 1
    for p in primes:
        n *= p
        if n > MAX_N:
            return n
    return n


def _tau_perp_fast(primes):
    """Count adjacent coprime pairs in the sorted divisor list of a squarefree n."""
    divs = [1]
    for p in primes:
        divs += [d * p for d in divs]
    divs.sort()
    g = math.gcd
    c = 0
    for i in range(1, len(divs)):
        if g(divs[i - 1], divs[i]) == 1:
            c += 1
    return c


def _initial_solution(k, seed=42):
    """Generate an initial solution with k primes using greedy heuristic."""
    rng = random.Random(seed)
    # Start with smallest k primes
    primes = PRIMES[:k]
    n = _product(primes)
    if n > MAX_N:
        # Remove largest until fits
        while n > MAX_N and len(primes) > 2:
            primes.pop()
            n = _product(primes)
        return tuple(sorted(primes))
    
    # Try to improve by swapping larger primes with unused ones
    used = set(primes)
    improved = True
    while improved:
        improved = False
        for i in range(len(primes)):
            for q in POOL:
                if q in used or q <= primes[i]:
                    continue
                new_n = n // primes[i] * q
                if new_n > MAX_N:
                    continue
                new_primes = list(primes)
                new_primes[i] = q
                new_primes.sort()
                new_primes = tuple(new_primes)
                if _tau_perp_fast(new_primes) > _tau_perp_fast(primes):
                    primes = new_primes
                    n = new_n
                    used = set(primes)
                    improved = True
                    break
            if improved:
                break
    return tuple(sorted(primes))


def construct():
    start = time.time()

    def rem():
        return DEADLINE - time.time()

    # Generate multiple initial solutions with different k values
    best_p = None
    best_n = 0
    best_s = 0
    
    for k in range(10, 16):
        if rem() < 20:
            break
        init = _initial_solution(k, seed=42 + k)
        s = _tau_perp_fast(init)
        n = _product(init)
        if s > best_s:
            best_s = s
            best_p = init
            best_n = n
    
    # Also try the primorial of first 15 primes as a baseline
    primorial = tuple(PRIMES[:15])
    s = _tau_perp_fast(primorial)
    n = _product(primorial)
    if s > best_s:
        best_s = s
        best_p = primorial
        best_n = n

    rng = random.Random(12345)

    # Multi-start simulated annealing
    T = 3.0
    T_min = 0.01
    cooling = 0.999

    cur_p = best_p
    cur_s = best_s
    cur_n = best_n

    while rem() > 15 and T > T_min:
        # Generate a move: swap one prime, add one, or remove one
        move_type = rng.random()
        cur_set = set(cur_p)
        cur_list = list(cur_p)
        
        if move_type < 0.7 and cur_list:
            # Swap: replace one prime with another
            i = rng.randrange(len(cur_list))
            old_p = cur_list[i]
            candidates = [q for q in POOL if q not in cur_set]
            if not candidates:
                T *= cooling
                continue
            new_candidates = []
            for q in candidates:
                new_prod = cur_n // old_p * q
                if new_prod <= MAX_N:
                    new_candidates.append(q)
            if not new_candidates:
                T *= cooling
                continue
            new_p = rng.choice(new_candidates)
            new_list = cur_list[:]
            new_list[i] = new_p
            new_list.sort()
            new_tuple = tuple(new_list)
        elif move_type < 0.85:
            # Add a prime
            candidates = [q for q in POOL if q not in cur_set and cur_n * q <= MAX_N]
            if not candidates:
                T *= cooling
                continue
            new_p = rng.choice(candidates)
            new_list = cur_list + [new_p]
            new_list.sort()
            new_tuple = tuple(new_list)
        else:
            # Remove a prime
            if len(cur_list) <= 2:
                T *= cooling
                continue
            i = rng.randrange(len(cur_list))
            new_list = cur_list[:]
            new_list.pop(i)
            new_tuple = tuple(new_list)

        new_s = _tau_perp_fast(new_tuple)
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

    # Final local search refinement
    if rem() > 10:
        cur = best_p
        improved = True
        while improved and rem() > 5:
            improved = False
            cur_set = set(cur)
            cur_n = _product(cur)
            best_move = None
            best_move_s = _tau_perp_fast(cur)
            
            # Try all single swaps
            for i, p in enumerate(cur):
                for q in POOL:
                    if q in cur_set:
                        continue
                    new_n = cur_n // p * q
                    if new_n > MAX_N:
                        continue
                    cand = list(cur)
                    cand[i] = q
                    cand.sort()
                    cand = tuple(cand)
                    s = _tau_perp_fast(cand)
                    if s > best_move_s:
                        best_move_s = s
                        best_move = cand
            # Try adds
            for q in POOL:
                if q in cur_set:
                    continue
                if cur_n * q > MAX_N:
                    continue
                cand = tuple(sorted(cur + (q,)))
                s = _tau_perp_fast(cand)
                if s > best_move_s:
                    best_move_s = s
                    best_move = cand
            # Try removes
            for i in range(len(cur)):
                cand = tuple(cur[:i] + cur[i+1:])
                s = _tau_perp_fast(cand)
                if s > best_move_s:
                    best_move_s = s
                    best_move = cand
            
            if best_move is not None:
                cur = best_move
                improved = True
        
        if _tau_perp_fast(cur) > best_s:
            best_s = _tau_perp_fast(cur)
            best_p = cur
            best_n = _product(best_p)

    return best_n
# EVOLVE-BLOCK-END
