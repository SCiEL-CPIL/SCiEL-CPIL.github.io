import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    moduli = list(range(20, 201))
    
    # Precompute gcd matrix for speed
    n_mod = len(moduli)
    m_to_idx = {m: i for i, m in enumerate(moduli)}
    gcd_mat = [[gcd(moduli[i], moduli[j]) for j in range(n_mod)] for i in range(n_mod)]

    def can_assign(moduli_list, residues_list, new_m):
        """Check if there exists a residue r for new_m such that it's disjoint from all existing."""
        for r_new in range(new_m):
            ok = True
            for j in range(len(moduli_list)):
                m_j = moduli_list[j]
                r_j = residues_list[j]
                g = gcd(new_m, m_j)
                if r_new % g == r_j % g:
                    ok = False
                    break
            if ok:
                return True, r_new
        return False, None

    def is_valid(classes):
        n = len(classes)
        for i in range(n):
            m1, r1 = classes[i]
            for j in range(i + 1, n):
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    def score(moduli_list):
        total = Fraction(0, 1)
        for m in moduli_list:
            total += Fraction(1, m)
        return total

    def local_search(moduli_list, residues_list):
        """Try to swap out a modulus for a better unselected one."""
        selected_set = set(moduli_list)
        unselected = [m for m in moduli if m not in selected_set]
        # Sort unselected by decreasing 1/m (increasing m) to prioritize small moduli
        unselected.sort()
        
        improved = True
        while improved:
            improved = False
            # Try swapping each selected modulus with each unselected modulus
            for i in range(len(moduli_list)):
                if time.time() > deadline:
                    return moduli_list, residues_list
                for new_m in unselected:
                    # If new_m is larger than moduli_list[i], it's not an improvement
                    if new_m >= moduli_list[i]:
                        break
                    # Try to add new_m to the list without moduli_list[i]
                    trial_mods = moduli_list[:i] + moduli_list[i + 1:]
                    trial_res = residues_list[:i] + residues_list[i + 1:]
                    can, r = can_assign(trial_mods, trial_res, new_m)
                    if can:
                        # Perform swap
                        moduli_list = trial_mods + [new_m]
                        residues_list = trial_res + [r]
                        selected_set.remove(moduli_list[i])  # The original m_i
                        selected_set.add(new_m)
                        unselected.remove(new_m)
                        unselected.append(moduli_list[i])  # The original m_i becomes unselected
                        unselected.sort()
                        improved = True
                        break
                if time.time() > deadline:
                    return moduli_list, residues_list
                if improved:
                    break
        return moduli_list, residues_list

    # Multi-start greedy
    best_moduli = []
    best_residues = []
    best_score = Fraction(0, 1)
    
    # Initial deterministic run (increasing m)
    mods = []
    res = []
    for m in moduli:
        if time.time() > deadline:
            break
        can, r = can_assign(mods, res, m)
        if can:
            mods.append(m)
            res.append(r)
    if mods:
        mods, res = local_search(mods, res)
        s = score(mods)
        if s > best_score:
            best_score = s
            best_moduli = mods[:]
            best_residues = res[:]

    # Randomized runs
    rng = random.Random(42)
    for _ in range(1000):  # Enough iterations, but time-bounded
        if time.time() > deadline:
            break
        
        order = moduli[:]
        rng.shuffle(order)
        
        mods = []
        res = []
        for m in order:
            can, r = can_assign(mods, res, m)
            if can:
                mods.append(m)
                res.append(r)
        
        if not mods:
            continue
        
        mods, res = local_search(mods, res)
        s = score(mods)
        if s > best_score:
            best_score = s
            best_moduli = mods[:]
            best_residues = res[:]
            if time.time() > deadline:
                break

    # Build the result
    result = [[best_moduli[i], best_residues[i]] for i in range(len(best_moduli))]
    
    # Verify validity
    if not is_valid(result):
        # Fallback to a simple valid set
        result = [[20 * j, j - 1] for j in range(1, 11)]
    
    return result

