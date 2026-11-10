"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _get_divisors(n):
    """Get all divisors of n greater than 1."""
    divisors = []
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.append(n)
    return sorted(divisors)

def _coverage_count(covered, m, r):
    """Count how many uncovered residues are covered by class (m, r)."""
    # Indices: r, r+m, r+2m, ... < base
    idx = np.arange(r, len(covered), m)
    return int(np.sum(covered[idx] == 0))

def _update_coverage(covered, m, r, add=True):
    """Add or remove coverage for class (m, r)."""
    idx = np.arange(r, len(covered), m)
    if add:
        covered[idx] = 1
    else:
        covered[idx] = 0

def _greedy_construct(base, divisors):
    """Greedy construction: process moduli in order, pick best residue."""
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    for m in divisors:
        # Find residue that covers most uncovered
        best_r = 0
        best_count = -1
        for r in range(m):
            cnt = _coverage_count(covered, m, r)
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append([m, best_r])
        _update_coverage(covered, m, best_r, add=True)
    return classes, covered

def _score(classes, base):
    """Count uncovered residues."""
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        _update_coverage(covered, m, r, add=True)
    return int(np.sum(covered == 0))

def _local_search(classes, base, divisors, max_time=10.0):
    """Local search: try changing residues and adding/removing classes."""
    start_time = time.time()
    covered = np.zeros(base, dtype=np.uint8)
    class_map = {}
    for m, r in classes:
        _update_coverage(covered, m, r, add=True)
        class_map[m] = r
    
    current_score = int(np.sum(covered == 0))
    best_score = current_score
    best_classes = [list(c) for c in classes]
    
    used_moduli = set(m for m, r in classes)
    available_moduli = [m for m in divisors if m not in used_moduli]
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        
        # Try changing residue for each existing class
        for m in list(class_map.keys()):
            old_r = class_map[m]
            best_r = old_r
            best_gain = 0
            
            for r in range(m):
                if r == old_r:
                    continue
                # Compute marginal gain: new coverage - old coverage
                # Remove old, add new, count change
                old_idx = np.arange(old_r, base, m)
                new_idx = np.arange(r, base, m)
                
                # Gain = (# newly covered by new) - (# uncovered that were covered by old)
                newly_covered = np.sum((covered[new_idx] == 0))
                lost = np.sum((covered[old_idx] == 1))
                gain = newly_covered - lost
                
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                # Apply change
                _update_coverage(covered, m, old_r, add=False)
                _update_coverage(covered, m, best_r, add=True)
                class_map[m] = best_r
                current_score -= best_gain
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [[m, class_map[m]] for m in class_map]
        
        # Try adding a new class
        if available_moduli:
            for m in available_moduli:
                best_r = 0
                best_gain = 0
                for r in range(m):
                    cnt = _coverage_count(covered, m, r)
                    if cnt > best_gain:
                        best_gain = cnt
                        best_r = r
                
                if best_gain > 0:
                    _update_coverage(covered, m, best_r, add=True)
                    class_map[m] = best_r
                    available_moduli.remove(m)
                    current_score -= best_gain
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [[m, class_map[m]] for m in class_map]
        
        # Try removing a class (if it doesn't help, skip for now)
        # Could add removal logic if beneficial
    
    return best_classes, best_score

def construct():
    base = 51975
    divisors = _get_divisors(base)
    
    deadline = time.time() + 110  # Leave 10s margin
    
    best_classes = None
    best_score = base  # worst case
    
    # Phase 1: Greedy construction
    classes, covered = _greedy_construct(base, divisors)
    score = _score(classes, base)
    best_classes = [list(c) for c in classes]
    best_score = score
    
    # Phase 2: Local search on greedy result
    if time.time() < deadline:
        ls_classes, ls_score = _local_search(classes, base, divisors, max_time=30.0)
        if ls_score < best_score:
            best_classes = ls_classes
            best_score = ls_score
    
    # Phase 3: Random restarts with local search
    random.seed(42)
    while time.time() < deadline:
        # Random initialization: random subset of divisors with random residues
        random_divs = random.sample(divisors, k=min(len(divisors), random.randint(5, len(divisors))))
        random_classes = [[m, random.randint(0, m - 1)] for m in random_divs]
        
        rs_classes, rs_score = _local_search(random_classes, base, divisors, max_time=15.0)
        if rs_score < best_score:
            best_classes = rs_classes
            best_score = rs_score
    
    return best_classes
# EVOLVE-BLOCK-END
