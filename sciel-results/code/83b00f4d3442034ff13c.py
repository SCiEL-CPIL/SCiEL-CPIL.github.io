import time
from math import gcd

MAX_M = 128
DEADLINE = 85.0  # safety margin before 90s kill

def _all_classes():
    """Precompute every valid (residue, modulus) pair for m in 2..128."""
    classes_by_mod = {}
    for m in range(2, MAX_M + 1):
        classes_by_mod[m] = list(range(m))  # residues 0..m-1
    return classes_by_mod

def _conflict(r1, m1, r2, m2):
    """True if classes (r1,m1) and (r2,m2) are NOT disjoint."""
    g = gcd(m1, m2)
    return (r1 - r2) % g == 0

def _greedy_order():
    """
    Return a static ordering of all (m, r) candidates, sorted by a heuristic:
    prefer moduli with fewer total classes (harder to place), and among them
    prefer residues that conflict with fewer other candidates.
    """
    all_cands = []
    for m in range(2, MAX_M + 1):
        for r in range(m):
            all_cands.append((m, r))
    # Conflict count with all other candidates (approximate difficulty)
    # This is O(N^2) but N is at most sum(m for m in 2..128) = 8191
    # That's ~67M pairs... too slow. Use a cheaper heuristic.
    # Heuristic: sort by modulus ascending (smaller moduli have fewer residues,
    # and small moduli share more gcds with others, making them harder).
    all_cands.sort(key=lambda x: x[0])
    return all_cands

def _greedy_construct(order, deadline):
    """Greedy: iterate candidates in order, add if compatible with all chosen."""
    chosen = []  # list of (m, r)
    chosen_m = set()
    for m, r in order:
        if time.time() > deadline:
            break
        if m in chosen_m:
            continue
        # Check compatibility with all chosen
        ok = True
        for (cm, cr) in chosen:
            if _conflict(r, m, cr, cm):
                ok = False
                break
        if ok:
            chosen.append((m, r))
            chosen_m.add(m)
    return chosen

def _local_search(initial, deadline):
    """
    Try to improve by:
    1. For each modulus not in the set, try all residues to add.
    2. For each modulus in the set, try swapping to a different residue
       that allows adding a new modulus.
    """
    chosen = list(initial)
    chosen_m = set(m for m, r in chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding a missing modulus
        for m in range(2, MAX_M + 1):
            if m in chosen_m:
                continue
            for r in range(m):
                ok = True
                for (cm, cr) in chosen:
                    if _conflict(r, m, cr, cm):
                        ok = False
                        break
                if ok:
                    chosen.append((m, r))
                    chosen_m.add(m)
                    improved = True
                    break
        # Try swapping: for each chosen (m, r), try a different r' for same m
        # that still allows all other chosen, and then check if we can add a new one
        for i, (m, r) in enumerate(chosen):
            if time.time() > deadline:
                break
            for r_new in range(m):
                if r_new == r:
                    continue
                # Check r_new is compatible with all other chosen
                ok = True
                for j, (cm, cr) in enumerate(chosen):
                    if j == i:
                        continue
                    if _conflict(r_new, m, cr, cm):
                        ok = False
                        break
                if not ok:
                    continue
                # Try swap
                chosen[i] = (m, r_new)
                # Now try to add a missing modulus
                added = False
                for m2 in range(2, MAX_M + 1):
                    if m2 in chosen_m:
                        continue
                    for r2 in range(m2):
                        ok2 = True
                        for (cm, cr) in chosen:
                            if _conflict(r2, m2, cr, cm):
                                ok2 = False
                                break
                        if ok2:
                            chosen.append((m2, r2))
                            chosen_m.add(m2)
                            added = True
                            break
                    if added:
                        break
                if added:
                    improved = True
                else:
                    chosen[i] = (m, r)  # revert
    return chosen

def construct():
    t0 = time.time()
    # Phase 1: Greedy construction
    order = _greedy_order()
    result = _greedy_construct(order, DEADLINE)
    
    if time.time() > DEADLINE:
        return [[r, m] for m, r in result]
    
    # Phase 2: Local search to improve
    result = _local_search(result, DEADLINE)
    
    return [[r, m] for m, r in result]
