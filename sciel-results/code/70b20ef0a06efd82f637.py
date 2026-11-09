"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import sys

BASE = 51975

def _get_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

def _compute_covered(classes):
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return covered

def _score_solution(classes):
    covered = _compute_covered(classes)
    return int(np.sum(1 - covered))

def _greedy_construct(divisors, order=None, seed=0):
    """Greedy: for each modulus in given order, pick residue maximizing newly covered residues."""
    rng = random.Random(seed)
    covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    
    if order is None:
        order = sorted(divisors)
    elif isinstance(order, str) and order == "random":
        order = divisors[:]
        rng.shuffle(order)
    elif isinstance(order, str) and order == "decreasing":
        order = sorted(divisors, reverse=True)
    elif isinstance(order, str) and order == "by_coverage":
        # Sort by number of positions covered (BASE/m), descending
        order = sorted(divisors, key=lambda m: -BASE // m)
    
    for m in order:
        best_r = 0
        best_gain = -1
        # For efficiency, if m is large, we can be smarter
        if m <= 100:
            for r in range(m):
                positions = np.arange(r, BASE, m)
                gain = int(np.sum(covered[positions] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
        else:
            # For large m, sample residues first, then refine
            candidates = list(range(0, m, max(1, m // 50)))
            if 0 not in candidates:
                candidates.append(0)
            for r in candidates:
                positions = np.arange(r, BASE, m)
                gain = int(np.sum(covered[positions] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            # Refine around best candidate
            for r in range(max(0, best_r - 5), min(m, best_r + 6)):
                if r == best_r:
                    continue
                positions = np.arange(r, BASE, m)
                gain = int(np.sum(covered[positions] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
        
        classes.append([m, best_r])
        covered[best_r::m] = 1
    
    return classes, covered

def _local_search(classes, max_time=None, deadline=None):
    """Iterative improvement: try changing one residue at a time, then pair swaps."""
    covered = _compute_covered(classes)
    best_score = int(np.sum(covered == 0))
    classes_copy = [list(c) for c in classes]
    n = len(classes_copy)
    
    if deadline is None:
        deadline = time.time() + (max_time or 10.0)
    
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        
        # Single-residue moves
        for i in range(n):
            m, r = classes_copy[i]
            
            # Remove this class temporarily
            covered[r::m] = 0
            base_uncovered = int(np.sum(covered == 0))
            
            best_new_r = r
            best_new_score = base_uncovered
            
            if m <= 200:
                for new_r in range(m):
                    newly_covered = int(np.sum(covered[new_r::m] == 0))
                    new_score = base_uncovered - newly_covered
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_r = new_r
            else:
                # Sample for large moduli
                candidates = list(range(0, m, max(1, m // 100)))
                if r not in candidates:
                    candidates.append(r)
                for new_r in candidates:
                    newly_covered = int(np.sum(covered[new_r::m] == 0))
                    new_score = base_uncovered - newly_covered
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_r = new_r
            
            # Re-add with best residue
            covered[best_new_r::m] = 1
            if best_new_r != r:
                classes_copy[i][1] = best_new_r
                if best_new_score < best_score:
                    best_score = best_new_score
                    improved = True
                break
        
        if time.time() >= deadline - 1.0:
            break
        
        # Pair-swap moves: try swapping residues between two moduli
        if time.time() < deadline - 2.0:
            for i in range(n):
                if time.time() >= deadline - 2.0:
                    break
                for j in range(i + 1, n):
                    mi, ri = classes_copy[i]
                    mj, rj = classes_copy[j]
                    
                    # Try a few candidate residues for each
                    candidates_i = [ri] + list(range(max(0, ri - 2), min(mi, ri + 3)))
                    candidates_j = [rj] + list(range(max(0, rj - 2), min(mj, rj + 3)))
                    
                    best_score_pair = best_score
                    best_ri = ri
                    best_rj = rj
                    
                    for new_ri in set(candidates_i):
                        for new_rj in set(candidates_j):
                            if new_ri == ri and new_rj == rj:
                                continue
                            
                            # Temporarily remove both
                            covered[ri::mi] = 0
                            covered[rj::mj] = 0
                            
                            # Add new residues
                            covered[new_ri::mi] = 1
                            covered[new_rj::mj] = 1
                            
                            new_score = int(np.sum(covered == 0))
                            
                            # Restore
                            covered[new_ri::mi] = 0
                            covered[new_rj::mj] = 0
                            covered[ri::mi] = 1
                            covered[rj::mj] = 1
                            
                            if new_score < best_score_pair:
                                best_score_pair = new_score
                                best_ri = new_ri
                                best_rj = new_rj
                    
                    if best_ri != ri or best_rj != rj:
                        classes_copy[i][1] = best_ri
                        classes_copy[j][1] = best_rj
                        if best_score_pair < best_score:
                            best_score = best_score_pair
                            improved = True
                        break
    
    return classes_copy, best_score

def _smart_local_search(classes, deadline):
    """More sophisticated local search with focus on uncovered residues."""
    covered = _compute_covered(classes)
    best_score = int(np.sum(covered == 0))
    classes_copy = [list(c) for c in classes]
    n = len(classes_copy)
    
    # Get list of uncovered residues
    uncovered = np.where(covered == 0)[0]
    
    # For each uncovered residue, find which moduli could cover it
    # and try to reassign those moduli to cover more uncovered residues
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        uncovered = np.where(covered == 0)[0]
        
        if len(uncovered) == 0:
            break
        
        # Try to cover one of the uncovered residues
        target = uncovered[0]
        
        # Find all moduli that could cover target
        for i in range(n):
            m, r = classes_copy[i]
            # Check if any residue for modulus m covers target
            # target % m is the residue that would cover target
            target_r = target % m
            
            if target_r == r:
                continue  # Already covers target
            
            # Try changing this modulus to cover target
            old_r = r
            covered[r::m] = 0
            covered[target_r::m] = 1
            
            new_score = int(np.sum(covered == 0))
            
            if new_score < best_score:
                best_score = new_score
                classes_copy[i][1] = target_r
                improved = True
            else:
                covered[target_r::m] = 0
                covered[old_r::m] = 1
            
            if improved:
                break
    
    return classes_copy, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    divisors = _get_divisors(BASE)
    
    best_classes = None
    best_score = BASE + 1
    
    # Try multiple greedy initializations with different orderings
    orderings = [
        ("increasing", None),
        ("decreasing", "decreasing"),
        ("by_coverage", "by_coverage"),
    ]
    
    for name, order in orderings:
        if time.time() > deadline - 20.0:
            break
        
        classes, covered = _greedy_construct(divisors, order=order, seed=42)
        score = int(np.sum(covered == 0))
        
        if score < best_score:
            best_score = score
            best_classes = classes[:]
    
    # Also try a few random orderings
    for seed in range(5):
        if time.time() > deadline - 20.0:
            break
        classes, covered = _greedy_construct(divisors, order="random", seed=seed)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes[:]
    
    # Local search on best solution
    if best_classes is not None and time.time() < deadline - 5.0:
        ls_classes, ls_score = _local_search(best_classes, deadline=deadline - 2.0)
        if ls_score < best_score:
            best_score = ls_score
            best_classes = ls_classes
    
    # Smart local search
    if best_classes is not None and time.time() < deadline - 5.0:
        sls_classes, sls_score = _smart_local_search(best_classes, deadline - 2.0)
        if sls_score < best_score:
            best_score = sls_score
            best_classes = sls_classes
    
    # Try more local search iterations
    if best_classes is not None and time.time() < deadline - 5.0:
        ls_classes, ls_score = _local_search(best_classes, deadline=deadline - 1.0)
        if ls_score < best_score:
            best_score = ls_score
            best_classes = ls_classes
    
    # Final verification
    if best_classes is None:
        # Fallback: simple greedy
        best_classes, _ = _greedy_construct(divisors, order=None, seed=0)
    
    return best_classes
# EVOLVE-BLOCK-END
