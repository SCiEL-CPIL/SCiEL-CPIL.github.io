import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def calc_score(pairs):
        return sum(1.0 / m for m, r in pairs)

    def greedy_random(seed=None):
        if seed is not None:
            random.seed(seed)
        selected = []
        for m in range(20, 201):
            valid_residues = []
            for r in range(m):
                ok = True
                for m_j, r_j in selected:
                    g = gcd(m, m_j)
                    if r % g == r_j % g:
                        ok = False
                        break
                if ok:
                    valid_residues.append(r)
            
            if valid_residues:
                if seed is None:
                    r = valid_residues[0]
                else:
                    r = random.choice(valid_residues)
                selected.append([m, r])
        return selected

    best_pairs = None
    best_score = 0.0

    # Deterministic first pass
    p = greedy_random(None)
    s = calc_score(p)
    if s > best_score:
        best_score = s
        best_pairs = p

    # Random restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        p = greedy_random(restarts)
        s = calc_score(p)
        if s > best_score:
            best_score = s
            best_pairs = p

    return best_pairs
