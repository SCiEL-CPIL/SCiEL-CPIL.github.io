"""Improved covering-system search for moduli dividing 51975."""
import time
import random
import numpy as np

BASE = 51975  # 3^3 * 5^2 * 7 * 11

def _divisors(n):
    """Return all divisors of n > 1, sorted."""
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(d for d in divs if d > 1)

def _score(classes):
    """Return number of uncovered residues modulo BASE."""
    covered = np.zeros(BASE, dtype=np.int8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.sum(1 - covered))

def _greedy_build(moduli_order, covered, classes):
    """Greedy: for each modulus in order, pick residue covering most uncovered."""
    for m in moduli_order:
        best_r = 0
        best_count = -1
        for r in range(m):
            cnt = int(np.sum(1 - covered[r::m]))
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes

def _local_search(classes, time_limit, rng):
    """Local search: try swapping residues or removing redundant classes."""
    current_score = _score(classes)
    covered = np.zeros(BASE, dtype=np.int8)
    for m, r in classes:
        covered[r::m] = 1
    
    moduli_used = [m for m, r in classes]
    residues = [r for m, r in classes]
    
    deadline = time.time() + time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try changing residue for each class
        for i, m in enumerate(moduli_used):
            old_r = residues[i]
            covered[old_r::m] = 0
            best_r = old_r
            best_local_score = _score_from_covered(covered)
            # Sample some residues if m is large
            if m > 50:
                candidates = [old_r] + rng.sample(range(m), min(m-1, 30))
            else:
                candidates = list(range(m))
            for r in candidates:
                if r == old_r:
                    continue
                covered[r::m] = 1
                s = _score_from_covered(covered)
                covered[r::m] = 0
                if s < best_local_score:
                    best_local_score = s
                    best_r = r
            covered[best_r::m] = 1
            if best_r != old_r:
                residues[i] = best_r
                current_score = best_local_score
                improved = True
        
        # Try removing a class that doesn't reduce coverage
        for i in range(len(moduli_used)):
            m = moduli_used[i]
            r = residues[i]
            covered[r::m] = 0
            s = _score_from_covered(covered)
            covered[r::m] = 1
            if s <= current_score:
                # Removing this class doesn't hurt
                del moduli_used[i]
                del residues[i]
                current_score = s
                improved = True
                break
    
    return [[m, r] for m, r in zip(moduli_used, residues)]

def _score_from_covered(covered):
    """Count uncovered residues from a covered array."""
    return int(np.sum(1 - covered))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    all_moduli = _divisors(BASE)
    rng = random.Random(42)
    
    best_classes = None
    best_score = BASE + 1
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        all_moduli,                    # increasing
        list(reversed(all_moduli)),    # decreasing
        sorted(all_moduli, key=lambda m: BASE // m),  # by density
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        covered = np.zeros(BASE, dtype=np.int8)
        classes = []
        _greedy_build(order, covered, classes)
        s = _score(classes)
        if s < best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Strategy 2: Local search from best greedy result
    if best_classes is not None and time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 5:
            best_classes = _local_search([list(c) for c in best_classes], min(remaining - 5, 60), rng)
            s = _score(best_classes)
            if s < best_score:
                best_score = s
    
    # Strategy 3: Random restarts with greedy + local
    while time.time() < deadline - 10:
        # Random subset of moduli
        k = rng.randint(5, len(all_moduli))
        subset = rng.sample(all_moduli, k)
        rng.shuffle(subset)
        
        covered = np.zeros(BASE, dtype=np.int8)
        classes = []
        _greedy_build(subset, covered, classes)
        s = _score(classes)
        
        if s < best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
        
        # Quick local search if promising
        if s <= best_score + 50 and time.time() < deadline - 15:
            classes = _local_search([list(c) for c in classes], min(5, deadline - time.time() - 15), rng)
            s = _score(classes)
            if s < best_score:
                best_score = s
                best_classes = [list(c) for c in classes]
    
    # Final polish: try all residues for small moduli in best solution
    if best_classes is not None and time.time() < deadline - 5:
        covered = np.zeros(BASE, dtype=np.int8)
        for m, r in best_classes:
            covered[r::m] = 1
        for i, (m, r) in enumerate(best_classes):
            if m > 20:
                continue
            covered[r::m] = 0
            best_r = r
            best_s = _score_from_covered(covered)
            for tr in range(m):
                if tr == r:
                    continue
                covered[tr::m] = 1
                s = _score_from_covered(covered)
                covered[tr::m] = 0
                if s < best_s:
                    best_s = s
                    best_r = tr
            covered[best_r::m] = 1
            best_classes[i][1] = best_r
    
    return best_classes if best_classes is not None else [[3, 0]]
