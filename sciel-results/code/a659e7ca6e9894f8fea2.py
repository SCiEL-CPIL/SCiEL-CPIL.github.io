import math
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _pair_ok(m1, r1, m2, r2):
    g = gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    for i in range(len(classes)):
        m1, r1 = classes[i]
        for j in range(i + 1, len(classes)):
            m2, r2 = classes[j]
            if not _pair_ok(m1, r1, m2, r2):
                return False
    return True

def _score(classes):
    return sum(1.0 / m for m, _ in classes)

def _greedy_construct(moduli, residue_order_shuffler):
    selected = []
    score = 0.0
    for m in moduli:
        # Get residues in shuffled order
        residues = list(range(m))
        residue_order_shuffler(residues)
        
        found = False
        for r in residues:
            ok = True
            for (m2, r2) in selected:
                if not _pair_ok(m, r, m2, r2):
                    ok = False
                    break
            if ok:
                selected.append((m, r))
                score += 1.0 / m
                found = True
                break
        if not found:
            continue
    return selected, score

def _local_search(selected, moduli_set, time_limit):
    # Convert to list of [m, r]
    current = [[m, r] for m, r in selected]
    current_score = _score(current)
    
    # Precompute moduli not in current
    selected_moduli = {m for m, _ in current}
    candidates = [m for m in range(20, 201) if m not in selected_moduli]
    # Sort candidates by weight (descending 1/m -> ascending m)
    candidates.sort()
    
    improved = True
    while improved and _time_left() > time_limit:
        improved = False
        # Try to add a candidate
        for m_cand in candidates:
            # Find a valid residue for m_cand
            valid_r = None
            for r in range(m_cand):
                ok = True
                for (m2, r2) in current:
                    if not _pair_ok(m_cand, r, m2, r2):
                        ok = False
                        break
                if ok:
                    valid_r = r
                    break
            
            if valid_r is not None:
                current.append([m_cand, valid_r])
                current_score += 1.0 / m_cand
                selected_moduli.add(m_cand)
                candidates.remove(m_cand)
                improved = True
                break # Restart candidate loop? Or continue?
        
        # Try to swap: Remove a selected m_swap, add m_cand if m_cand < m_swap
        # This is complex. Let's stick to "Add if possible" first.
        # The greedy already added what it could.
        # The "Add if possible" in local search might find things greedy missed due to residue choice.
        
        # Try to change residues to allow adding a high-value candidate
        # Pick the highest value candidate (smallest m) that is still not in current
        if candidates:
            m_target = candidates[0]
            # Try to modify one of the current residues to make m_target fit
            # We need to find a current pair (m_i, r_i) such that changing r_i
            # allows m_target to have a valid residue.
            
            # This is an NP-hard subproblem if done generally.
            # Heuristic: Try changing r_i for each current pair.
            # For each current pair, try all possible new residues r_i'
            # Check if (m_i, r_i') is valid with others.
            # Then check if m_target can be added.
            
            # This is $N_{selected} \times M_{avg} \times (N_{selected} + 1)$ checks.
            # ~ 30 * 100 * 31 ~ 100,000 ops. Fast enough.
            
            for i in range(len(current)):
                m_i, r_i = current[i]
                # Try all new residues for m_i
                for new_r_i in range(m_i):
                    if new_r_i == r_i: continue
                    
                    # Check if (m_i, new_r_i) is compatible with all OTHERS
                    ok_i = True
                    for j in range(len(current)):
                        if i == j: continue
                        m_j, r_j = current[j]
                        if not _pair_ok(m_i, new_r_i, m_j, r_j):
                            ok_i = False
                            break
                    
                    if not ok_i: continue
                    
                    # Check if m_target can be added with this new configuration
                    # Temporarily set current[i] to (m_i, new_r_i)
                    old_r = current[i][1]
                    current[i][1] = new_r_i
                    
                    valid_target_r = None
                    for r_t in range(m_target):
                        ok_t = True
                        for j in range(len(current)):
                            m_j, r_j = current[j]
                            if not _pair_ok(m_target, r_t, m_j, r_j):
                                ok_t = False
                                break
                        if ok_t:
                            valid_target_r = r_t
                            break
                    
                    if valid_target_r is not None:
                        # Accept the change and add target
                        current.append([m_target, valid_target_r])
                        current_score += 1.0 / m_target
                        selected_moduli.add(m_target)
                        candidates.remove(m_target)
                        improved = True
                        break
                    else:
                        # Revert
                        current[i][1] = old_r
                        break # No point trying other new_r_i for this m_i if target didn't fit?
                        # Actually, maybe another new_r_i would fit. But let's break to save time.
                
                if improved:
                    break
                    
    return current, current_score

def construct():
    moduli = list(range(20, 201))
    best_score = 0.0
    best_set = []
    
    # 1. Deterministic Greedy (First Fit)
    selected, score = _greedy_construct(moduli, lambda x: None)
    if score > best_score:
        best_score = score
        best_set = selected
    
    # 2. Randomized Greedy Restarts
    restarts = 0
    while _time_left() > 20.0: # Leave 20s for local search
        # Shuffle residues
        def shuffle_residues(res):
            random.shuffle(res)
        
        selected, score = _greedy_construct(moduli, shuffle_residues)
        if score > best_score:
            best_score = score
            best_set = selected
        restarts += 1
        
        # Occasionally run local search on the current best
        if restarts % 100 == 0 and _time_left() > 5.0:
            ls_set, ls_score = _local_search(best_set, set(moduli), 2.0)
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set
                
    # Final local search on the very best
    if _time_left() > 1.0:
        ls_set, ls_score = _local_search(best_set, set(moduli), _time_left() - 1.0)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
            
    return [[m, r] for m, r in best_set]
