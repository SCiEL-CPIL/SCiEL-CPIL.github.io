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

def score_solution(classes, covered_arr):
    # covered_arr is np array of 0/1, score = number of 0s
    return int(np.sum(covered_arr == 0))

def apply_class(covered_arr, m, r):
    # Set covered_arr[r::m] = 1
    covered_arr[r::m] = 1
    return covered_arr

def remove_class(covered_arr, m, r, other_classes):
    # This is tricky - we need to recompute coverage without this class
    # Better to just recompute from scratch for local search
    pass

def greedy_construct(moduli_order, seed=0):
    rng = random.Random(seed)
    covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in moduli_order:
        best_r = 0
        best_gain = -1
        for r in range(m):
            # Count uncovered residues in r::m
            positions = np.arange(r, BASE, m)
            gain = int(np.sum(covered[positions] == 0))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes, covered

def local_search(classes, max_time=None):
    # Try to improve by changing residues
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    
    best_score = int(np.sum(covered == 0))
    classes_copy = [list(c) for c in classes]
    
    improved = True
    while improved and (max_time is None or time.time() < max_time):
        improved = False
        for i, (m, r) in enumerate(classes_copy):
            # Try all residues for this modulus
            best_r = r
            best_gain = 0
            # Current coverage contribution
            current_positions = np.arange(r, BASE, m)
            current_uncovered = int(np.sum(covered[current_positions] == 0))
            
            for r_try in range(m):
                if r_try == r:
                    continue
                new_positions = np.arange(r_try, BASE, m)
                # Gain: uncovered residues in new positions that are NOT covered by other classes
                # This is complex; simpler: recompute score
                # For efficiency, only try a subset or use approximation
                # Let's do a quick check: count uncovered in new positions
                gain = int(np.sum(covered[new_positions] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r_try
            
            if best_r != r:
                # Update covered
                covered[r::m] = 0
                covered[best_r::m] = 1
                new_score = int(np.sum(covered == 0))
                if new_score < best_score:
                    best_score = new_score
                    classes_copy[i][1] = best_r
                    improved = True
                else:
                    covered[best_r::m] = 0
                    covered[r::m] = 1
    
    return classes_copy, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    divs = get_divisors()
    
    # Try multiple orderings
    best_classes = None
    best_score = BASE + 1
    
    # Ordering 1: increasing moduli (original)
    order1 = sorted(divs)
    classes1, cov1 = greedy_construct(order1, seed=42)
    score1 = int(np.sum(cov1 == 0))
    if score1 < best_score:
        best_score = score1
        best_classes = [list(c) for c in classes1]
    
    # Local search on best
    if time.time() < deadline:
        ls_classes, ls_score = local_search(best_classes, max_time=deadline - 5)
        if ls_score < best_score:
            best_score = ls_score
            best_classes = ls_classes
    
    # Ordering 2: decreasing moduli
    if time.time() < deadline:
        order2 = sorted(divs, reverse=True)
        classes2, cov2 = greedy_construct(order2, seed=42)
        score2 = int(np.sum(cov2 == 0))
        if score2 < best_score:
            best_score = score2
            best_classes = [list(c) for c in classes2]
        
        if time.time() < deadline - 5:
            ls_classes, ls_score = local_search(best_classes, max_time=deadline - 5)
            if ls_score < best_score:
                best_score = ls_score
                best_classes = ls_classes
    
    # Ordering 3: random
    if time.time() < deadline:
        order3 = divs[:]
        random.shuffle(order3)
        classes3, cov3 = greedy_construct(order3, seed=123)
        score3 = int(np.sum(cov3 == 0))
        if score3 < best_score:
            best_score = score3
            best_classes = [list(c) for c in classes3]
        
        if time.time() < deadline - 5:
            ls_classes, ls_score = local_search(best_classes, max_time=deadline - 5)
            if ls_score < best_score:
                best_score = ls_score
                best_classes = ls_classes
    
    # More local search iterations
    if time.time() < deadline - 5:
        ls_classes, ls_score = local_search(best_classes, max_time=deadline - 5)
        if ls_score < best_score:
            best_score = ls_score
            best_classes = ls_classes
    
    return best_classes
