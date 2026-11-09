import time
from math import gcd
from fractions import Fraction
import random

# EVOLVE-BLOCK-START

def _is_disjoint(classes, m_new, r_new):
    for m, r in classes:
        g = gcd(m_new, m)
        if r_new % g == r % g:
            return False
    return True

def _score(classes):
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return total

def _greedy(candidates):
    classes = []
    for m, r in candidates:
        if _is_disjoint(classes, m, r):
            classes.append((m, r))
    return classes

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # All candidates
    all_candidates = [(m, r) for m in range(20, 201) for r in range(m)]
    
    # Strategy 1: Sort by m ascending
    c1 = sorted(all_candidates, key=lambda x: (x[0], x[1]))
    best_classes = _greedy(c1)
    best_score = _score(best_classes)
    
    # Strategy 2: Sort by m ascending, r descending
    c2 = sorted(all_candidates, key=lambda x: (x[0], -x[1]))
    trial = _greedy(c2)
    s = _score(trial)
    if s > best_score:
        best_score = s
        best_classes = trial
    
    # Strategy 3: Randomized restarts
    rng = random.Random(12345)
    for i in range(200):
        if time.time() > deadline - 5:
            break
        shuffled = all_candidates[:]
        rng.shuffle(shuffled)
        # Bias toward small m
        shuffled.sort(key=lambda x: x[0] + rng.random() * 10)
        trial = _greedy(shuffled)
        s = _score(trial)
        if s > best_score:
            best_score = s
            best_classes = trial[:]
    
    return [[m, r] for m, r in best_classes]

# EVOLVE-BLOCK-END
