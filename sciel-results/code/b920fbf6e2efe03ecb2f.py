import time
import random
from fractions import Fraction

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def _compute_L_fast(covered, cap=100000):
    L = 0
    for n in range(1, cap + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _build_covered(classes, cap=100000):
    covered = bytearray(cap + 1)
    for p, r in classes:
        start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        if start < 1:
            start = 1
        for n in range(start, cap + 1, p):
            covered[n] = 1
    return covered

def _sum_recip(classes):
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    all_primes = _primes_upto(1000)
    
    # Initial solution
    primes_seed = [p for p in all_primes if p >= 13]
    best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
    
    cap = 100000
    best_covered = _build_covered(best_classes, cap)
    best_L = _compute_L_fast(best_covered, cap)
    
    random.seed(42)
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        candidate = [list(c) for c in best_classes]
        mutation_type = random.random()
        
        if mutation_type < 0.5:
            idx = random.randint(0, len(candidate) - 1)
            p = candidate[idx][0]
            candidate[idx][1] = random.randint(0, p - 1)
        elif mutation_type < 0.7:
            if len(candidate) >= 2:
                i, j = random.sample(range(len(candidate)), 2)
                candidate[i][1], candidate[j][1] = candidate[j][1], candidate[i][1]
        elif mutation_type < 0.85:
            used_primes = set(c[0] for c in candidate)
            available = [p for p in all_primes if p not in used_primes]
            if available:
                idx = random.randint(0, len(candidate) - 1)
                old_p = candidate[idx][0]
                new_p = random.choice(available)
                new_sum = _sum_recip([c for c in candidate if c[0] != old_p] + [[new_p, 0]])
                if new_sum <= 1:
                    candidate[idx] = [new_p, random.randint(0, new_p - 1)]
        else:
            used_primes = set(c[0] for c in candidate)
            available = [p for p in all_primes if p not in used_primes]
            if available:
                new_p = random.choice(available)
                new_sum = _sum_recip(candidate) + Fraction(1, new_p)
                if new_sum <= 1:
                    candidate.append([new_p, random.randint(0, new_p - 1)])
        
        if _sum_recip(candidate) > 1:
            continue
        
        covered = _build_covered(candidate, cap)
        L = _compute_L_fast(covered, cap)
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in candidate]
            best_covered = covered
    
    return best_classes
