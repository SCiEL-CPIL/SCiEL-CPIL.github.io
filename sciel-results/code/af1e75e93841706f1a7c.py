import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50
    MAX_MOD = 128
    
    def compatible(r1, m1, r2, m2):
        g = gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def greedy(moduli_order):
        selected = []
        for m in moduli_order:
            for r in range(m):
                ok = True
                for (r_sel, m_sel) in selected:
                    if not compatible(r, m, r_sel, m_sel):
                        ok = False
                        break
                if ok:
                    selected.append((r, m))
                    break
        return selected
    
    best = []
    
    # Deterministic orderings
    for order in [list(range(2, MAX_MOD+1)), list(range(MAX_MOD, 1, -1))]:
        result = greedy(order)
        if len(result) > len(best):
            best = result
    
    # Random restarts
    random.seed(42)
    for _ in range(5000):
        if time.time() > deadline:
            break
        order = list(range(2, MAX_MOD+1))
        random.shuffle(order)
        result = greedy(order)
        if len(result) > len(best):
            best = result
    
    return [[r, m] for (r, m) in best]
