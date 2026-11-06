"""Optimized program: constraint satisfaction with arc consistency + local search."""
import math
import time
import random

# EVOLVE-BLOCK-START

def _is_compatible(m1, r1, m2, r2):
    """Check if two residue classes are disjoint."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _score(chosen):
    """Compute the score (sum of 1/m) for a set of chosen classes."""
    return sum(1.0 / m for m, r in chosen)

def _greedy_construct(deadline):
    """Greedy construction: try moduli in ascending order, pick best residue."""
    moduli = list(range(20, 201))
    chosen = []
    for m in moduli:
        if time.time() > deadline:
            break
        best_r = None
        best_conf = m + 1
        for r in range(m):
            conf = 0
            for (m2, r2) in chosen:
                g = math.gcd(m, m2)
                if (r - r2) % g == 0:
                    conf += 1
                    if conf >= best_conf:
                        break
            if conf < best_conf:
                best_conf = conf
                best_r = r
                if best_conf == 0:
                    break
        if best_conf == 0:
            chosen.append([m, best_r])
    return chosen

def _arc_consistency_prune(moduli, chosen, deadline):
    """
    For unchosen moduli, prune residues that conflict with chosen classes.
    Returns dict: m -> set of valid residues.
    """
    valid = {}
    for m in moduli:
        if any(m == cm for cm, _ in chosen):
            continue
        valid_residues = set(range(m))
        for (m2, r2) in chosen:
            g = math.gcd(m, m2)
            # Remove residues r where (r - r2) % g == 0
            bad_residues = set()
            for r in range(m):
                if (r - r2) % g == 0:
                    bad_residues.add(r)
            valid_residues -= bad_residues
            if not valid_residues:
                break
        if valid_residues:
            valid[m] = valid_residues
    return valid

def _local_improve(chosen, deadline, max_iter=500):
    """
    Local search: try to add missing moduli or swap residues to improve score.
    """
    chosen_set = set((m, r) for m, r in chosen)
    chosen_mods = set(m for m, r in chosen)
    all_mods = set(range(20, 201))
    missing_mods = list(all_mods - chosen_mods)
    random.shuffle(missing_mods)
    
    current_score = _score(chosen)
    best_chosen = [list(x) for x in chosen]
    best_score = current_score
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try to add a missing modulus
        added = False
        for m in missing_mods:
            if m in chosen_mods:
                continue
            # Find best residue for m
            best_r = None
            best_conf = m + 1
            for r in range(m):
                conf = 0
                for (m2, r2) in chosen:
                    g = math.gcd(m, m2)
                    if (r - r2) % g == 0:
                        conf += 1
                        if conf >= best_conf:
                            break
                if conf < best_conf:
                    best_conf = conf
                    best_r = r
                    if best_conf == 0:
                        break
            if best_conf == 0:
                new_chosen = chosen + [[m, best_r]]
                new_score = _score(new_chosen)
                if new_score > best_score:
                    best_chosen = [list(x) for x in new_chosen]
                    best_score = new_score
                    chosen = new_chosen
                    chosen_mods.add(m)
                    missing_mods.remove(m) if m in missing_mods else None
                    added = True
                    break
        
        # Try to swap residues of existing classes
        if not added and len(chosen) >= 2:
            i, j = random.sample(range(len(chosen)), 2)
            m1, r1 = chosen[i]
            m2, r2 = chosen[j]
            # Try all pairs of new residues
            improved = False
            for new_r1 in range(m1):
                for new_r2 in range(m2):
                    if new_r1 == r1 and new_r2 == r2:
                        continue
                    # Check compatibility with each other
                    g = math.gcd(m1, m2)
                    if (new_r1 - new_r2) % g == 0:
                        continue
                    # Check compatibility with all others
                    valid = True
                    for k, (mk, rk) in enumerate(chosen):
                        if k == i or k == j:
                            continue
                        g1 = math.gcd(m1, mk)
                        if (new_r1 - rk) % g1 == 0:
                            valid = False
                            break
                        g2 = math.gcd(m2, mk)
                        if (new_r2 - rk) % g2 == 0:
                            valid = False
                            break
                    if valid:
                        new_chosen = [list(x) for x in chosen]
                        new_chosen[i] = [m1, new_r1]
                        new_chosen[j] = [m2, new_r2]
                        new_score = _score(new_chosen)
                        if new_score > best_score:
                            best_chosen = new_chosen
                            best_score = new_score
                            chosen = new_chosen
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
        
        # Try to replace a high-modulus class with a lower one
        if not added and not (len(chosen) >= 2):
            pass  # fallback
    
    return best_chosen

def _cs_construct(deadline):
    """
    Constraint satisfaction approach:
    1. Greedy initial solution
    2. Arc consistency to prune
    3. Local improvement
    """
    # Step 1: Greedy initial
    chosen = _greedy_construct(deadline - 30)
    
    if time.time() > deadline:
        return chosen
    
    # Step 2: Local improvement
    chosen = _local_improve(chosen, deadline, max_iter=1000)
    
    return chosen

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s
    
    # Run CS-based approach
    result = _cs_construct(deadline)
    
    # Ensure we have at least a valid solution
    if not result:
        result = [[20, 0]]
    
    return result

# EVOLVE-BLOCK-END
