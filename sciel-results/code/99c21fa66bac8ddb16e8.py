"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _is_valid(classes):
    """Check pairwise disjointness: r_i != r_j mod gcd(m_i, m_j)."""
    n = len(classes)
    for i in range(n):
        for j in range(i + 1, n):
            mi, ri = classes[i]
            mj, rj = classes[j]
            g = gcd(mi, mj)
            if ri % g == rj % g:
                return False
    return True

def _score(classes):
    return sum(1.0 / m for m, r in classes)

def _try_add(classes, m):
    """Try to find a valid residue for modulus m given existing classes."""
    for r in range(m):
        ok = True
        for mi, ri in classes:
            g = gcd(m, mi)
            if r % g == ri % g:
                ok = False
                break
        if ok:
            return r
    return None

def _local_search(classes, deadline):
    """Try to add more moduli and improve residues via local search."""
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding missing moduli
        for m in range(20, 201):
            if any(c[0] == m for c in classes):
                continue
            r = _try_add(classes, m)
            if r is not None:
                classes.append([m, r])
                improved = True
        # Try swapping residues to free up space
        for i, (mi, ri) in enumerate(classes):
            for j, (mj, rj) in enumerate(classes):
                if i == j:
                    continue
                g = gcd(mi, mj)
                # Try all residues for class i
                for new_r in range(mi):
                    if new_r == ri:
                        continue
                    # Check against all other classes
                    ok = True
                    for k, (mk, rk) in enumerate(classes):
                        if k == i:
                            continue
                        gk = gcd(mi, mk)
                        if new_r % gk == rk % gk:
                            ok = False
                            break
                    if ok:
                        classes[i][1] = new_r
                        improved = True
                        break
    return classes

def _simulated_annealing(classes, deadline):
    """SA to escape local optima."""
    classes = [c[:] for c in classes]
    current_score = _score(classes)
    best_classes = [c[:] for c in classes]
    best_score = current_score
    T = 0.1
    T_min = 0.001
    cooling = 0.995
    
    while time.time() < deadline - 10 and T > T_min:
        # Pick a random class and change its residue
        i = random.randint(0, len(classes) - 1)
        mi = classes[i][0]
        old_r = classes[i][1]
        new_r = random.randint(0, mi - 1)
        classes[i][1] = new_r
        
        if _is_valid(classes):
            new_score = _score(classes)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in classes]
            else:
                classes[i][1] = old_r
        else:
            # Invalid: revert and try to fix
            classes[i][1] = old_r
        
        T *= cooling
    
    # After SA, try to add more moduli
    for m in range(20, 201):
        if any(c[0] == m for c in best_classes):
            continue
        r = _try_add(best_classes, m)
        if r is not None:
            best_classes.append([m, r])
    
    return best_classes

def _polish(classes, deadline):
    """Final polish: systematically try all residues for each class to maximize additions."""
    classes = [c[:] for c in classes]
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding missing moduli
        for m in range(20, 201):
            if any(c[0] == m for c in classes):
                continue
            r = _try_add(classes, m)
            if r is not None:
                classes.append([m, r])
                improved = True
        # Systematically try each residue for each class
        for i in range(len(classes)):
            mi = classes[i][0]
            for r in range(mi):
                if r == classes[i][1]:
                    continue
                old_r = classes[i][1]
                classes[i][1] = r
                if _is_valid(classes):
                    # Check if this allows adding more moduli
                    added = False
                    for m in range(20, 201):
                        if any(c[0] == m for c in classes):
                            continue
                        nr = _try_add(classes, m)
                        if nr is not None:
                            classes.append([m, nr])
                            added = True
                    if added:
                        improved = True
                    else:
                        classes[i][1] = old_r
                else:
                    classes[i][1] = old_r
    return classes

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    random.seed(42)
    
    # Phase 1: Greedy - add moduli in order of 1/m (descending)
    classes = []
    for m in range(20, 201):
        if time.time() > deadline - 30:
            break
        r = _try_add(classes, m)
        if r is not None:
            classes.append([m, r])
    
    # Phase 2: Local search
    classes = _local_search(classes, deadline)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 20:
        classes = _simulated_annealing(classes, deadline)
    
    # Phase 4: Final polish
    classes = _polish(classes, deadline)
    
    # Sort by modulus for consistency
    classes.sort(key=lambda x: x[0])
    
    return [[m, r] for m, r in classes]
# EVOLVE-BLOCK-END
