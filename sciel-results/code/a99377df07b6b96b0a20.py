"""Direct constructive approach for minimum maximal no-isosceles set."""
import itertools
import time
import numpy as np
from itertools import combinations

N = 16
DEADLINE = 325.0  # safety margin before 330s kill


def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def _isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def construct():
    start_time = time.time()
    
    # Greedy maximal construction in row-major order.
    chosen = []
    for p in itertools.product(range(N), repeat=2):
        if time.time() - start_time > DEADLINE:
            break
        if all(not _isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    
    # Try to improve by removing points that don't break maximality
    # and then re-adding them in different orders
    best_set = chosen[:]
    
    # Try multiple random restarts with different orderings
    import random
    rng = random.Random(42)
    
    for attempt in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Shuffle the order of points
        points = list(itertools.product(range(N), repeat=2))
        rng.shuffle(points)
        
        chosen = []
        for p in points:
            if time.time() - start_time > DEADLINE:
                break
            if all(not _isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
                chosen.append(p)
        
        if len(chosen) < len(best_set):
            best_set = chosen[:]
    
    return best_set
# EVOLVE-BLOCK-END
