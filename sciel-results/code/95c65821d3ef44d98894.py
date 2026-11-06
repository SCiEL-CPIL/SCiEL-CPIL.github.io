"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
_BASE = 51975
_DEADLINE = 115.0
_START = time.time()

def _divisors(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _divisors(_BASE)
_DIVS_SORTED = sorted(_DIVS)

def _score(classes):
    """Exact uncovered count."""
    covered = bytearray(_BASE)
    for m, r in classes:
        for i in range(r, _BASE, m):
            covered[i] = 1
    return _BASE - sum(covered)

def _build_greedy(order=None):
    """Greedy construction: moduli in given order, each picks best residue."""
    if order is None:
        order = _DIVS_SORTED
    covered = bytearray(_BASE)
    classes = []
    for m in order:
        best_r = 0
        best_count = -1
        for r in range(m):
            count = 0
            for i in range(r, _BASE, m):
                if covered[i] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        for i in range(best_r, _BASE, m):
            covered[i] = 1
    return classes

def _local_search(classes, max_rounds=100):
    """Local search: for each modulus, find best residue given others fixed."""
    classes = [c[:] for c in classes]
    n = len(classes)
    
    for round_num in range(max_rounds):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for i in range(n):
            m = classes[i][0]
            r_old = classes[i][1]
            # Compute coverage from all other classes
            other_covered = bytearray(_BASE)
            for j in range(n):
                if j == i:
                    continue
                mj, rj = classes[j]
                for idx in range(rj, _BASE, mj):
                    other_covered[idx] = 1
            
            # For each residue r, compute how many newly covered
            best_r = r_old
            best_gain = -1
            for r in range(m):
                gain = 0
                for idx in range(r, _BASE, m):
                    if other_covered[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r != r_old:
                classes[i][1] = best_r
                improved = True
        
        if not improved:
            break
    
    return classes

def _pairwise_swap_search(classes, max_rounds=50):
    """Try swapping residues between pairs of moduli to improve score."""
    classes = [c[:] for c in classes]
    n = len(classes)
    current_score = _score(classes)
    
    for round_num in range(max_rounds):
        if time.time() > _DEADLINE - 10:
            break
        improved = False
        for i in range(n):
            if time.time() > _DEADLINE - 10:
                break
            for j in range(i+1, n):
                if time.time() > _DEADLINE - 10:
                    break
                mi, ri = classes[i]
                mj, rj = classes[j]
                
                # Try swapping residues
                classes[i][1] = rj % mi
                classes[j][1] = ri % mj
                new_score = _score(classes)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    classes[i][1] = ri
                    classes[j][1] = rj
        
        if not improved:
            break
    
    return classes

def construct():
    best_classes = None
    best_score = _BASE + 1
    
    # Start with greedy in sorted order
    classes = _build_greedy(_DIVS_SORTED)
    best_score = _score(classes)
    best_classes = [c[:] for c in classes]
    
    # Local search on greedy result
    classes_ls = _local_search(classes)
    score_ls = _score(classes_ls)
    if score_ls < best_score:
        best_score = score_ls
        best_classes = classes_ls
    
    # Pairwise swap search
    classes_ps = _pairwise_swap_search(best_classes)
    score_ps = _score(classes_ps)
    if score_ps < best_score:
        best_score = score_ps
        best_classes = classes_ps
    
    # Try greedy in reverse order
    if time.time() < _DEADLINE - 20:
        classes_rev = _build_greedy(sorted(_DIVS, reverse=True))
        score_rev = _score(classes_rev)
        if score_rev < best_score:
            best_score = score_rev
            best_classes = [c[:] for c in classes_rev]
        
        # Local search on reverse greedy
        classes_rev_ls = _local_search(classes_rev)
        score_rev_ls = _score(classes_rev_ls)
        if score_rev_ls < best_score:
            best_score = score_rev_ls
            best_classes = classes_rev_ls
    
    # Random restarts
    rng = random.Random(42)
    for _ in range(3):
        if time.time() > _DEADLINE - 15:
            break
        random_order = rng.sample(_DIVS, len(_DIVS))
        classes_rand = _build_greedy(random_order)
        score_rand = _score(classes_rand)
        if score_rand < best_score:
            best_score = score_rand
            best_classes = [c[:] for c in classes_rand]
        
        # Local search on random greedy
        classes_rand_ls = _local_search(classes_rand)
        score_rand_ls = _score(classes_rand_ls)
        if score_rand_ls < best_score:
            best_score = score_rand_ls
            best_classes = classes_rand_ls
    
    return best_classes
# EVOLVE-BLOCK-END
