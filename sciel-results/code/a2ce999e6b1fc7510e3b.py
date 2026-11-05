import time
import random
import sys

N = 5000
DEADLINE = 310.0

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for multiple in range(p * p, limit + 1, p):
                sieve[multiple] = False
    return [i for i, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)

def _construct_once(order, primes):
    S = []
    S_set = set()
    residue_sets = {p: set() for p in primes}
    
    # Greedy phase
    for x in order:
        new_size = len(S) + 1
        valid = True
        for p in primes:
            if p > new_size:
                break
            r = x % p
            if r not in residue_sets[p]:
                if len(residue_sets[p]) == p - 1:
                    valid = False
                    break
        
        if valid:
            S.append(x)
            S_set.add(x)
            for p in primes:
                r = x % p
                if r not in residue_sets[p]:
                    residue_sets[p].add(r)
            
    # Polish phase
    # Single pass is sufficient as per logic
    for x in order:
        if x in S_set:
            continue
        new_size = len(S) + 1
        valid = True
        for p in primes:
            if p > new_size:
                break
            r = x % p
            if r not in residue_sets[p]:
                if len(residue_sets[p]) == p - 1:
                    valid = False
                    break
        
        if valid:
            S.append(x)
            S_set.add(x)
            for p in primes:
                r = x % p
                if r not in residue_sets[p]:
                    residue_sets[p].add(r)
            
    return S

def construct():
    start_time = time.time()
    best_S = []
    best_len = 0
    rng = random.Random(42)
    
    # Initial run with natural order
    order = list(range(N + 1))
    S = _construct_once(order, PRIMES)
    best_S = S
    best_len = len(S)
    
    # Multiple restarts
    while time.time() < DEADLINE - 5:
        order = list(range(N + 1))
        rng.shuffle(order)
        S = _construct_once(order, PRIMES)
        if len(S) > best_len:
            best_len = len(S)
            best_S = S
    
    return best_S
