import time
from math import log
import itertools
import random

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def generate_smart_candidates(U, max_val, n_candidates=20):
    """Generate candidates that are likely to improve the difference/sum ratio"""
    U_set = set(U)
    U_sorted = sorted(U)
    max_u = max(U_sorted)
    
    candidates = []
    
    # Elements that create new differences but few new sums
    # Focus on elements far from existing ones to maximize differences
    for x in range(1, max_val + 1):
        if x in U_set:
            continue
        
        # Score: prefer elements that are far from existing elements
        # and create many new differences
        min_dist = min(abs(x - u) for u in U_sorted)
        # Prefer elements that are roughly at 2*max_u or beyond current max
        # to create large differences
        candidates.append((min_dist, x))
    
    # Sort by minimum distance (prefer elements far from existing)
    candidates.sort(key=lambda t: -t[0])
    return [x for _, x in candidates[:n_candidates]]

def local_search(U, deadline, max_iter=500):
    """Hill-climbing local search with smart candidate generation"""
    U = sorted(set(U))
    if 0 not in U:
        return U
    best_score = score_set(U)
    best_set = list(U)
    max_val = max(U) + 30
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding a new element using smart candidates
        smart_cands = generate_smart_candidates(U, max_val, n_candidates=15)
        if smart_cands:
            for new_elem in smart_cands:
                if time.time() > deadline:
                    break
                new_U = sorted(set(U) | {new_elem})
                s = score_set(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(max_val, new_elem + 15)
                    improved = True
                    break
        
        # Also try random additions
        if not improved:
            candidates_to_add = [x for x in range(1, max_val + 1) if x not in U]
            if candidates_to_add:
                for _ in range(min(3, len(candidates_to_add))):
                    if time.time() > deadline:
                        break
                    new_elem = random.choice(candidates_to_add)
                    new_U = sorted(set(U) | {new_elem})
                    s = score_set(new_U)
                    if s > best_score:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        max_val = max(max_val, new_elem + 10)
                        improved = True
                        break
        
        # Try removing an element (not 0)
        if len(U) > 2:
            for _ in range(min(3, len(U) - 1)):
                if time.time() > deadline:
                    break
                idx = random.randint(1, len(U) - 1)
                new_U = sorted(set(U) - {U[idx]})
                s = score_set(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    improved = True
                    break
        
        # Try swapping an element
        if len(U) > 2:
            for _ in range(min(3, len(U) - 1)):
                if time.time() > deadline:
                    break
                idx = random.randint(1, len(U) - 1)
                old_elem = U[idx]
                # Try smart candidates for swap
                smart_cands = generate_smart_candidates(U, max_val, n_candidates=10)
                new_elem = random.choice(smart_cands) if smart_cands else random.randint(1, max_val)
                if new_elem == old_elem or new_elem in U:
                    continue
                new_U = sorted(set(U) - {old_elem} | {new_elem})
                s = score_set(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(max_val, new_elem + 10)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_set

def polish_set(U, deadline):
    """Final high-precision polish: try all single-element modifications"""
    U = sorted(set(U))
    if 0 not in U:
        return U
    best_score = score_set(U)
    best_set = list(U)
    max_val = max(U) + 50
    
    # Try all possible additions
    for x in range(1, max_val + 1):
        if time.time() > deadline:
            break
        if x in U:
            continue
        new_U = sorted(set(U) | {x})
        s = score_set(new_U)
        if s > best_score:
            best_score = s
            best_set = new_U
            U = new_U
            max_val = max(max_val, x + 20)
    
    # Try all possible removals
    if len(U) > 2:
        for idx in range(1, len(U)):
            if time.time() > deadline:
                break
            new_U = sorted(set(U) - {U[idx]})
            s = score_set(new_U)
            if s > best_score:
                best_score = s
                best_set = new_U
                U = new_U
    
    # Try all possible single swaps
    if len(U) > 2:
        for idx in range(1, len(U)):
            if time.time() > deadline:
                break
            old_elem = U[idx]
            for new_elem in range(1, max_val + 1):
                if time.time() > deadline:
                    break
                if new_elem == old_elem or new_elem in U:
                    continue
                new_U = sorted(set(U) - {old_elem} | {new_elem})
                s = score_set(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(max_val, new_elem + 20)
    
    return best_set

def construct():
    deadline = time.time() + 110  # Safety margin
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Try some known good constructions first
    initial_candidates = [
        [0, 1, 2, 4, 8],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 9, 16],
        [0, 1, 4, 16, 64],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 5, 10, 20],
        [0, 1, 4, 10, 28],
        [0, 1, 3, 10, 30],
        [0, 1, 5, 15, 45],
        [0, 1, 2, 6, 12, 24],
        [0, 1, 4, 9, 16, 25],
        [0, 1, 4, 10, 21, 43],
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 5, 11, 23, 47],
    ]
    
    for cand in initial_candidates:
        if time.time() > deadline:
            break
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Exhaustive search for small sets (sizes 3-5)
    for n in range(3, 6):
        if time.time() > deadline:
            break
        
        max_val = 40 if n <= 4 else 30
        
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                break
            
            U = (0,) + combo
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Local search to improve the best found set
    if time.time() < deadline:
        improved = local_search(best_set, deadline, max_iter=300)
        s = score_set(improved)
        if s > best_score:
            best_score = s
            best_set = improved
    
    # Try more random starting points with local search
    for _ in range(8):
        if time.time() > deadline:
            break
        n = random.randint(4, 9)
        max_val = random.randint(25, 80)
        rand_set = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
        improved = local_search(rand_set, deadline, max_iter=150)
        s = score_set(improved)
        if s > best_score:
            best_score = s
            best_set = improved
    
    # Final polish pass with exhaustive single-element modifications
    if time.time() < deadline - 5:
        polished = polish_set(best_set, deadline)
        s = score_set(polished)
        if s > best_score:
            best_score = s
            best_set = polished
    
    return best_set
