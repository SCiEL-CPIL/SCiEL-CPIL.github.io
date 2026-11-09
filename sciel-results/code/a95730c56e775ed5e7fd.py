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
    return int(np.sum(covered_arr == 0))

def greedy_construct(moduli_order, seed=0):
    rng = random.Random(seed)
    covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in moduli_order:
        # Find best residue for m
        # Count uncovered in each residue class
        zero_idx = np.where(covered == 0)[0]
        if len(zero_idx) == 0:
            best_r = 0
        else:
            counts = np.bincount(zero_idx % m, minlength=m)
            best_r = int(np.argmax(counts))
        
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes, covered

def fast_local_search(classes, deadline):
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    
    best_score = int(np.sum(covered == 0))
    classes_copy = [list(c) for c in classes]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order of moduli to escape symmetry?
        # Or just iterate sequentially
        for i, (m, r) in enumerate(classes_copy):
            zero_idx = np.where(covered == 0)[0]
            if len(zero_idx) == 0:
                break
            
            counts = np.bincount(zero_idx % m, minlength=m)
            best_r_new = int(np.argmax(counts))
            
            if best_r_new != r:
                # Check if this change actually improves score
                # It should, because best_r_new has max uncovered count.
                # But we need to be careful: 
                # Current uncovered in r is counts[r].
                # New uncovered in best_r_new is counts[best_r_new].
                # Since counts[best_r_new] >= counts[r], we cover at least as many.
                # So score should not increase.
                
                covered[r::m] = 0
                covered[best_r_new::m] = 1
                new_score = int(np.sum(covered == 0))
                
                if new_score < best_score:
                    best_score = new_score
                    classes_copy[i][1] = best_r_new
                    improved = True
                else:
                    # Revert (shouldn't happen if logic is correct, but safe)
                    covered[best_r_new::m] = 0
                    covered[r::m] = 1
    
    return classes_copy, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    divs = get_divisors()
    
    best_classes = None
    best_score = BASE + 1
    
    # Try multiple orderings
    orderings = [
        sorted(divs),
        sorted(divs, reverse=True),
        divs[:]
    ]
    random.shuffle(orderings[2])
    
    for i, order in enumerate(orderings):
        if time.time() > deadline - 10:
            break
            
        classes, cov = greedy_construct(order, seed=42 + i)
        score = int(np.sum(cov == 0))
        
        # Local search
        if time.time() < deadline - 5:
            ls_classes, ls_score = fast_local_search(classes, deadline - 5)
            if ls_score < score:
                classes = ls_classes
                score = ls_score
        
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # If time remains, do more random restarts
    while time.time() < deadline - 5:
        order = divs[:]
        random.shuffle(order)
        classes, cov = greedy_construct(order, seed=random.randint(0, 10000))
        score = int(np.sum(cov == 0))
        
        if time.time() < deadline - 2:
            ls_classes, ls_score = fast_local_search(classes, deadline - 2)
            if ls_score < score:
                classes = ls_classes
                score = ls_score
        
        if score < best_score:
            best_score = score
            best_classes = classes
    
    return best_classes
