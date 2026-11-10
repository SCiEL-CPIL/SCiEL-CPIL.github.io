"""Combined program: fast numpy-based greedy construction with local search refinement."""
import time
import numpy as np

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# All divisors > 1 (all are odd since base is odd)

def _get_divisors(n):
    divs = []
    for i in range(2, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs

DIVISORS = _get_divisors(BASE)
DIVISORS.sort()  # ascending order


def _evaluate_score(classes):
    """Return the number of uncovered residues modulo BASE for a given list of classes."""
    if not classes:
        return BASE
    cov = np.zeros(BASE, dtype=bool)
    for m, r in classes:
        cov[r::m] = True
    return int(np.sum(~cov))


def _greedy_construct():
    """Greedy construction: iterate moduli in ascending order, pick best residue each time."""
    covered = np.zeros(BASE, dtype=bool)
    classes = []
    
    for m in DIVISORS:
        # Reshape to compute uncovered count per residue class efficiently
        reshaped = covered.reshape(-1, m)
        covered_counts = reshaped.sum(axis=0)
        total_in_class = BASE // m
        uncovered_counts = total_in_class - covered_counts
        
        best_r = int(np.argmax(uncovered_counts))
        best_count = uncovered_counts[best_r]
        
        if best_count > 0:
            covered[best_r::m] = True
            classes.append([m, best_r])
        else:
            # No new coverage from this modulus; still add it? 
            # Actually if best_count == 0, adding doesn't help. Skip.
            pass
    
    return classes, covered


def _local_search(classes, deadline):
    """Local search to improve the score: try swapping residues, removing redundant classes."""
    covered = np.zeros(BASE, dtype=bool)
    for m, r in classes:
        covered[r::m] = True
    current_score = int(np.sum(~covered))
    
    if current_score == 0:
        return classes, covered
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Try removing redundant classes
        new_classes = list(classes)
        i = 0
        while i < len(new_classes):
            if time.time() >= deadline:
                break
            m, r = new_classes[i]
            # Check if this class is redundant: all residues it covers are covered by others
            class_indices = np.arange(r, BASE, m)
            if len(class_indices) > 0 and np.all(covered[class_indices]):
                # Check if removing this class leaves them covered
                # Build coverage without this class
                temp_cov = np.zeros(BASE, dtype=bool)
                for j, (mm, rr) in enumerate(new_classes):
                    if j != i:
                        temp_cov[rr::mm] = True
                if np.all(temp_cov[class_indices]):
                    new_classes.pop(i)
                    improved = True
                    # Recompute covered
                    covered = np.zeros(BASE, dtype=bool)
                    for mm, rr in new_classes:
                        covered[rr::mm] = True
                    current_score = int(np.sum(~covered))
                    continue
            i += 1
        classes = new_classes
        
        # Pass 2: Try changing residue for each class
        for idx in range(len(classes)):
            if time.time() >= deadline:
                break
            m, r = classes[idx]
            # Try all residues, find one that maximizes new coverage
            reshaped = covered.reshape(-1, m)
            covered_counts = reshaped.sum(axis=0)
            total_in_class = BASE // m
            uncovered_counts = total_in_class - covered_counts
            
            # But we need to account for the fact that this class currently covers r::m
            # The "uncovered" count above already reflects current state
            # We want to pick r' that covers the most currently-uncovered residues
            best_r_new = int(np.argmax(uncovered_counts))
            best_gain = uncovered_counts[best_r_new]
            
            if best_r_new != r and best_gain > 0:
                # Check if this actually improves overall score
                # Temporarily swap
                old_r = classes[idx][1]
                classes[idx][1] = best_r_new
                new_score = _evaluate_score(classes)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Update covered
                    covered = np.zeros(BASE, dtype=bool)
                    for mm, rr in classes:
                        covered[rr::mm] = True
                else:
                    classes[idx][1] = old_r
        
        # Pass 3: Try adding a class from unused divisors
        used_moduli = set(m for m, r in classes)
        for m in DIVISORS:
            if time.time() >= deadline:
                break
            if m in used_moduli:
                continue
            reshaped = covered.reshape(-1, m)
            covered_counts = reshaped.sum(axis=0)
            total_in_class = BASE // m
            uncovered_counts = total_in_class - covered_counts
            best_r = int(np.argmax(uncovered_counts))
            best_gain = uncovered_counts[best_r]
            if best_gain > 0:
                # Test adding this class
                test_classes = classes + [[m, best_r]]
                new_score = _evaluate_score(test_classes)
                if new_score < current_score:
                    classes = test_classes
                    current_score = new_score
                    improved = True
                    used_moduli.add(m)
                    covered = np.zeros(BASE, dtype=bool)
                    for mm, rr in classes:
                        covered[rr::mm] = True
    
    return classes, covered


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Step 1: Greedy construction
    classes, covered = _greedy_construct()
    current_score = int(np.sum(~covered))
    
    if current_score == 0:
        return classes
    
    # Step 2: Local search refinement
    if time.time() < deadline:
        classes, covered = _local_search(classes, deadline)
        current_score = int(np.sum(~covered))
    
    return classes
