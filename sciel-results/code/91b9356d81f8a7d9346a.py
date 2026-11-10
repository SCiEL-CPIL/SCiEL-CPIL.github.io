"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

DEADLINE = 140.0  # safety margin before 150s kill

def _score(classes):
    """Compute exact rational score, return float."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _compatible(m, r, selected):
    """Check if (m, r) is compatible with all in selected."""
    for m2, r2 in selected:
        g = gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True

def _greedy_construction():
    """Greedy: process moduli in increasing order, pick smallest compatible residue."""
    selected = []
    for m in range(20, 201):
        for r in range(m):
            if _compatible(m, r, selected):
                selected.append([m, r])
                break
    return selected

def _try_add_excluded(selected, excluded):
    """Try to add excluded moduli with any compatible residue."""
    improved = True
    while improved:
        improved = False
        for i, m in enumerate(excluded):
            for r in range(m):
                if _compatible(m, r, selected):
                    selected.append([m, r])
                    excluded.pop(i)
                    improved = True
                    break
    return selected, excluded

def _local_search(selected, excluded, deadline):
    """Local search: perturb residues, swap residues, try adding excluded."""
    best = [c[:] for c in selected]
    best_score = _score(best)
    
    # Track which moduli are selected
    selected_moduli = set(m for m, r in selected)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        made_progress = False
        
        # Try changing residue of each selected class
        for i in range(len(selected)):
            m, r = selected[i]
            for new_r in range(m):
                if new_r == r:
                    continue
                # Check compatibility with all other selected
                compatible = True
                for j in range(len(selected)):
                    if j == i:
                        continue
                    m2, r2 = selected[j]
                    g = gcd(m, m2)
                    if new_r % g == r2 % g:
                        compatible = False
                        break
                if compatible:
                    # Check if this allows adding an excluded modulus
                    old_r = selected[i][1]
                    selected[i][1] = new_r
                    # Try adding excluded
                    added = False
                    for k in range(len(excluded)):
                        me = excluded[k]
                        for re in range(me):
                            if _compatible(me, re, selected):
                                selected.append([me, re])
                                excluded.pop(k)
                                added = True
                                break
                        if added:
                            break
                    if added:
                        new_score = _score(selected)
                        if new_score > best_score + 1e-15:
                            best = [c[:] for c in selected]
                            best_score = new_score
                            made_progress = True
                        else:
                            # Revert
                            selected[i][1] = old_r
                            if added:
                                # Remove the added one
                                for j in range(len(selected)):
                                    if selected[j][0] == me and selected[j][1] == re:
                                        selected.pop(j)
                                        break
                                excluded.append(me)
                    else:
                        selected[i][1] = old_r
        
        # Try swapping residues between two selected classes
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                m1, r1 = selected[i]
                m2, r2 = selected[j]
                # Swap residues (only valid if r1 < m2 and r2 < m1)
                if r1 < m2 and r2 < m1:
                    selected[i][1] = r2
                    selected[j][1] = r1
                    compatible = True
                    for k in range(len(selected)):
                        mk, rk = selected[k]
                        for l in range(k + 1, len(selected)):
                            ml, rl = selected[l]
                            g = gcd(mk, ml)
                            if rk % g == rl % g:
                                compatible = False
                                break
                        if not compatible:
                            break
                    if compatible:
                        new_score = _score(selected)
                        if new_score > best_score + 1e-15:
                            best = [c[:] for c in selected]
                            best_score = new_score
                            made_progress = True
                        else:
                            # Revert
                            selected[i][1] = r1
                            selected[j][1] = r2
                    else:
                        selected[i][1] = r1
                        selected[j][1] = r2
        
        # Try adding excluded moduli
        for i in range(len(excluded)):
            m = excluded[i]
            for r in range(m):
                if _compatible(m, r, selected):
                    selected.append([m, r])
                    excluded.pop(i)
                    new_score = _score(selected)
                    if new_score > best_score + 1e-15:
                        best = [c[:] for c in selected]
                        best_score = new_score
                        made_progress = True
                    else:
                        selected.pop()
                        excluded.append(m)
                    break
        
        if not made_progress:
            # No progress, try random restarts
            if random.random() < 0.1:
                # Randomly perturb a few residues
                for _ in range(3):
                    if selected:
                        idx = random.randint(0, len(selected) - 1)
                        m, r = selected[idx]
                        new_r = random.randint(0, m - 1)
                        old_r = selected[idx][1]
                        selected[idx][1] = new_r
                        # Check if still valid
                        valid = True
                        for j in range(len(selected)):
                            if j == idx:
                                continue
                            m2, r2 = selected[j]
                            g = gcd(m, m2)
                            if new_r % g == r2 % g:
                                valid = False
                                break
                        if valid:
                            new_score = _score(selected)
                            if new_score > best_score + 1e-15:
                                best = [c[:] for c in selected]
                                best_score = new_score
                        else:
                            selected[idx][1] = old_r
    
    return best

def construct():
    start_time = time.time()
    
    # Start with greedy construction
    selected = _greedy_construction()
    selected_moduli = set(m for m, r in selected)
    excluded = [m for m in range(20, 201) if m not in selected_moduli]
    
    # Try to add excluded
    selected, excluded = _try_add_excluded(selected, excluded)
    
    # Local search with time budget
    best = _local_search(selected, excluded, DEADLINE)
    
    # Final check: ensure we have a valid result
    # Verify all constraints
    for i in range(len(best)):
        m1, r1 = best[i]
        if m1 < 20 or m1 > 200 or r1 < 0 or r1 >= m1:
            # Fallback to greedy
            return _greedy_construction()
        for j in range(i + 1, len(best)):
            m2, r2 = best[j]
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                return _greedy_construction()
    
    return best
# EVOLVE-BLOCK-END
