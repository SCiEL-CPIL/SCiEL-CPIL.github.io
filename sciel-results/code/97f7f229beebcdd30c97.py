import time
import numpy as np
import random

BASE = 51975

def get_divisors():
    divs = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            divs.append(d)
    return divs

def compute_covered(classes):
    """Compute coverage array from list of [m, r] classes."""
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return covered

def get_score(covered):
    """Score = number of uncovered residues."""
    return int(np.sum(covered == 0))

def greedy_construct(moduli_order, seed=0):
    """Greedy: for each modulus, pick residue that maximizes newly covered residues."""
    rng = random.Random(seed)
    covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in moduli_order:
        best_r = 0
        best_gain = -1
        for r in range(m):
            positions = np.arange(r, BASE, m)
            gain = int(np.sum(covered[positions] == 0))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes, covered

def local_search(classes, max_time=None):
    """Iterative improvement: try changing one residue at a time."""
    covered = compute_covered(classes)
    best_score = get_score(covered)
    classes_copy = [list(c) for c in classes]
    
    improved = True
    while improved and (max_time is None or time.time() < max_time):
        improved = False
        for i in range(len(classes_copy)):
            m, r = classes_copy[i]
            best_r = r
            best_new_score = best_score
            
            for r_try in range(m):
                if r_try == r:
                    continue
                # Temporarily change
                covered[r::m] = 0
                covered[r_try::m] = 1
                new_score = get_score(covered)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_r = r_try
                else:
                    # Revert
                    covered[r_try::m] = 0
                    covered[r::m] = 1
            
            if best_r != r:
                covered[r::m] = 0
                covered[best_r::m] = 1
                best_score = best_new_score
                classes_copy[i][1] = best_r
                improved = True
    
    return classes_copy, best_score

def simulated_annealing(classes, max_time=None, start_temp=10.0, end_temp=0.1):
    """SA to escape local optima."""
    covered = compute_covered(classes)
    current_score = get_score(covered)
    best_score = current_score
    best_classes = [list(c) for c in classes]
    
    classes_copy = [list(c) for c in classes]
    n = len(classes_copy)
    
    if max_time is None:
        max_time = time.time() + 10
    
    start_t = time.time()
    total_time = max(1, max_time - start_t)
    current_temp = start_temp
    
    while time.time() < max_time:
        # Pick random class to modify
        i = random.randint(0, n - 1)
        m, r = classes_copy[i]
        
        if m <= 1:
            continue
        
        # Pick random new residue
        r_new = random.randint(0, m - 1)
        if r_new == r:
            continue
        
        # Apply change
        covered[r::m] = 0
        covered[r_new::m] = 1
        new_score = get_score(covered)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / max(current_temp, 1e-9)):
            # Accept
            current_score = new_score
            classes_copy[i][1] = r_new
            
            if current_score < best_score:
                best_score = current_score
                best_classes = [list(c) for c in classes_copy]
        else:
            # Revert
            covered[r_new::m] = 0
            covered[r::m] = 1
        
        # Cool down
        elapsed = time.time() - start_t
        frac = min(1.0, elapsed / total_time)
        current_temp = start_temp * (end_temp / start_temp) ** frac
    
    return best_classes, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    divs = get_divisors()
    n_divs = len(divs)
    
    best_classes = None
    best_score = BASE + 1
    
    # Strategy 1: Greedy with different orderings + local search
    orderings = [
        sorted(divs),                          # increasing
        sorted(divs, reverse=True),            # decreasing
        sorted(divs, key=lambda x: -BASE//x),  # by coverage density
    ]
    
    # Add a few random orderings
    rng = random.Random(42)
    for _ in range(3):
        shuffled = divs[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        
        classes, cov = greedy_construct(order, seed=42)
        score = get_score(cov)
        
        if score < best_score:
            best_score = score
            best_classes = [list(c) for c in classes]
        
        # Local search
        if time.time() < deadline - 10:
            ls_classes, ls_score = local_search(best_classes, max_time=deadline - 10)
            if ls_score < best_score:
                best_score = ls_score
                best_classes = ls_classes
    
    # Strategy 2: Simulated annealing on best solution
    if time.time() < deadline - 5 and best_classes is not None:
        sa_classes, sa_score = simulated_annealing(
            best_classes, 
            max_time=deadline - 5,
            start_temp=20.0,
            end_temp=0.05
        )
        if sa_score < best_score:
            best_score = sa_score
            best_classes = sa_classes
    
    # Strategy 3: More local search passes
    if time.time() < deadline - 3 and best_classes is not None:
        ls_classes, ls_score = local_search(best_classes, max_time=deadline - 3)
        if ls_score < best_score:
            best_score = ls_score
            best_classes = ls_classes
    
    # Strategy 4: Another SA round if time permits
    if time.time() < deadline - 2 and best_classes is not None:
        sa_classes, sa_score = simulated_annealing(
            best_classes,
            max_time=deadline - 2,
            start_temp=10.0,
            end_temp=0.02
        )
        if sa_score < best_score:
            best_score = sa_score
            best_classes = sa_classes
    
    return best_classes
