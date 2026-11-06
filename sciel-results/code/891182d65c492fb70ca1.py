"""Optimized program: local search over residue assignments for odd covering systems."""
import time
import random
import math
from collections import defaultdict

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# All odd divisors > 1
_DIVISORS = [d for d in range(3, BASE + 1, 2) if BASE % d == 0]
_DIVISORS_SORTED_DESC = sorted(_DIVISORS, reverse=True)  # larger moduli first
_DIVISORS_SORTED_ASC = sorted(_DIVISORS)  # smaller moduli first


def _uncovered_count(classes):
    """Count residues mod BASE covered by no class. classes = list of (m, r)."""
    covered = bytearray(BASE)
    for m, r in classes:
        for x in range(r, BASE, m):
            covered[x] = 1
    return BASE - sum(covered)


def _build_initial_greedy_desc():
    """Greedy construction: for each modulus (largest first), pick residue covering most uncovered."""
    covered = bytearray(BASE)
    classes = []
    for m in _DIVISORS_SORTED_DESC:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = 0
            for x in range(r, BASE, m):
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append((m, best_r))
        for x in range(best_r, BASE, m):
            covered[x] = 1
    return classes


def _build_initial_greedy_asc():
    """Greedy construction: for each modulus (smallest first), pick residue covering most uncovered."""
    covered = bytearray(BASE)
    classes = []
    for m in _DIVISORS_SORTED_ASC:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = 0
            for x in range(r, BASE, m):
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append((m, best_r))
        for x in range(best_r, BASE, m):
            covered[x] = 1
    return classes


def _fast_uncovered(classes):
    """Fast uncovered count using bytearray."""
    covered = bytearray(BASE)
    for m, r in classes:
        for x in range(r, BASE, m):
            covered[x] = 1
    return BASE - sum(covered)


def _local_search(classes, time_limit=8.0):
    """Local search: try changing residues, try adding/removing moduli."""
    start_time = time.time()
    best_classes = list(classes)
    best_score = _uncovered_count(best_classes)
    
    if best_score == 0:
        return best_classes, 0
    
    # Maintain a covered bitmask for quick evaluation
    covered = bytearray(BASE)
    for m, r in best_classes:
        for x in range(r, BASE, m):
            covered[x] = 1
    
    # Convert to dict for easy lookup
    mod_to_res = {}
    for m, r in best_classes:
        mod_to_res[m] = r
    
    used_mods = set(mod_to_res.keys())
    
    # Available moduli not yet used
    available_mods = [m for m in _DIVISORS if m not in used_mods]
    
    no_improve_streak = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try changing residue of an existing modulus
        for m in list(used_mods):
            old_r = mod_to_res[m]
            
            # Uncover old
            for x in range(old_r, BASE, m):
                covered[x] = 0
            
            best_new_r = old_r
            best_new_score = best_score
            
            # Try all residues for small moduli, sample for large
            if m <= 50:
                candidates = range(m)
            else:
                candidates = [random.randint(0, m - 1) for _ in range(50)]
            
            for r in candidates:
                if r == old_r:
                    continue
                # Apply
                for x in range(r, BASE, m):
                    covered[x] = 1
                score = BASE - sum(covered)
                if score < best_new_score:
                    best_new_score = score
                    best_new_r = r
                # Undo
                for x in range(r, BASE, m):
                    covered[x] = 0
            
            if best_new_r != old_r and best_new_score < best_score:
                # Commit
                for x in range(best_new_r, BASE, m):
                    covered[x] = 1
                mod_to_res[m] = best_new_r
                best_score = best_new_score
                improved = True
            else:
                # Re-apply old
                for x in range(old_r, BASE, m):
                    covered[x] = 1
        
        # Try adding a new modulus
        if available_mods:
            for m in random.sample(available_mods, min(5, len(available_mods))):
                best_r = 0
                best_gain = 0
                if m <= 50:
                    r_range = range(m)
                else:
                    r_range = [random.randint(0, m - 1) for _ in range(30)]
                for r in r_range:
                    gain = 0
                    for x in range(r, BASE, m):
                        if not covered[x]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                if best_gain > 0:
                    # Check if this actually helps (score decreases)
                    for x in range(best_r, BASE, m):
                        covered[x] = 1
                    new_score = BASE - sum(covered)
                    if new_score < best_score:
                        used_mods.add(m)
                        mod_to_res[m] = best_r
                        available_mods.remove(m)
                        best_score = new_score
                        improved = True
                    else:
                        for x in range(best_r, BASE, m):
                            covered[x] = 0
        
        # Try removing a modulus that doesn't help
        for m in list(used_mods):
            r = mod_to_res[m]
            for x in range(r, BASE, m):
                covered[x] = 0
            new_score = BASE - sum(covered)
            if new_score < best_score:
                # Keep it removed
                used_mods.discard(m)
                del mod_to_res[m]
                available_mods.append(m)
                best_score = new_score
                improved = True
            else:
                # Put it back
                for x in range(r, BASE, m):
                    covered[x] = 1
        
        if not improved:
            no_improve_streak += 1
            if no_improve_streak >= 15:
                break
        else:
            no_improve_streak = 0
        
        if best_score == 0:
            break
    
    best_classes = [[m, mod_to_res[m]] for m in sorted(used_mods)]
    return best_classes, best_score


def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    # Phase 1: Try both greedy orders, keep the better
    classes_desc = _build_initial_greedy_desc()
    score_desc = _fast_uncovered(classes_desc)
    
    if time.time() - start_time > DEADLINE:
        return [[m, r] for m, r in classes_desc]
    
    classes_asc = _build_initial_greedy_asc()
    score_asc = _fast_uncovered(classes_asc)
    
    if time.time() - start_time > DEADLINE:
        best_classes = classes_desc if score_desc <= score_asc else classes_asc
        return [[m, r] for m, r in best_classes]
    
    best_classes = classes_desc if score_desc <= score_asc else classes_asc
    best_score = min(score_desc, score_asc)
    
    # Phase 2: Local search on the best initial solution
    classes, best_score = _local_search(best_classes, time_limit=min(5.0, DEADLINE - (time.time() - start_time)))
    
    if time.time() - start_time > DEADLINE:
        return classes
    
    best_classes = classes
    best_score_final = best_score
    
    # Phase 3: Multiple restarts with random perturbations
    while time.time() - start_time < DEADLINE:
        # Perturb: randomly change some residues
        perturbed = list(best_classes)
        for i in range(len(perturbed)):
            if random.random() < 0.3:
                m = perturbed[i][0]
                perturbed[i] = [m, random.randint(0, m - 1)]
        
        # Local search on perturbed
        result_classes, result_score = _local_search(
            perturbed,
            time_limit=min(2.0, DEADLINE - (time.time() - start_time))
        )
        
        if result_score < best_score_final:
            best_score_final = result_score
            best_classes = result_classes
        
        if best_score_final == 0:
            break
    
    return best_classes
