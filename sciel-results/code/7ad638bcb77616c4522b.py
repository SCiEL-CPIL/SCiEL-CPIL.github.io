"""Optimized disjoint residue construction: maximize sum 1/m with pairwise disjointness."""
import time
import math
import random
from fractions import Fraction

def _disjoint(m1, r1, m2, r2):
    """Check if residue classes are pairwise disjoint: r1 != r2 mod gcd(m1, m2)."""
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    """Check pairwise disjointness for a list of [m, r] pairs with distinct moduli."""
    if not classes:
        return True
    moduli = [c[0] for c in classes]
    if len(moduli) != len(set(moduli)):
        return False
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if not _disjoint(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                return False
    return True

def _score(classes):
    """Exact rational score, returned as float."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _find_valid_residues(m, selected):
    """Find all valid residues for modulus m given already selected classes."""
    valid = []
    for r in range(m):
        ok = True
        for (m2, r2) in selected:
            g = math.gcd(m, m2)
            if (r % g) == (r2 % g):
                ok = False
                break
        if ok:
            valid.append(r)
    return valid

def _local_search(classes, deadline):
    """Enhanced local search: try adding, removing, and swapping moduli."""
    best = [c[:] for c in classes]
    best_sc = _score(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_moduli = set(m for m, r in best)
        
        # Try to add a new modulus with various residue choices
        for m in range(20, 201):
            if m in current_moduli:
                continue
            valid_rs = _find_valid_residues(m, best)
            if valid_rs:
                for r in valid_rs[:3]:  # Try first few valid residues
                    new_classes = best + [[m, r]]
                    if _is_valid(new_classes):
                        sc = _score(new_classes)
                        if sc > best_sc:
                            best_sc = sc
                            best = new_classes
                            current_moduli = set(m2 for m2, r2 in best)
                            improved = True
                            break
                if improved:
                    break
        
        # Try to swap: remove one modulus, add a smaller one
        for i in range(len(best)):
            if time.time() > deadline:
                break
            m_remove = best[i][0]
            remaining = best[:i] + best[i+1:]
            remaining_moduli = set(m2 for m2, r2 in remaining)
            for m_add in range(20, m_remove):
                if m_add in remaining_moduli:
                    continue
                valid_rs = _find_valid_residues(m_add, remaining)
                if valid_rs:
                    for r in valid_rs[:2]:
                        new_classes = remaining + [[m_add, r]]
                        if _is_valid(new_classes):
                            sc = _score(new_classes)
                            if sc > best_sc:
                                best_sc = sc
                                best = new_classes
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        
        # Try to reassign residues of existing moduli
        for i in range(len(best)):
            if time.time() > deadline:
                break
            m, r_old = best[i]
            others = best[:i] + best[i+1:]
            valid_rs = _find_valid_residues(m, others)
            for r_new in valid_rs:
                if r_new == r_old:
                    continue
                new_classes = others + [[m, r_new]]
                if _is_valid(new_classes):
                    sc = _score(new_classes)
                    if sc > best_sc:
                        best_sc = sc
                        best = new_classes
                        improved = True
                        break
            if improved:
                break
    
    return best, best_sc

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    random.seed(42)
    
    best_classes = None
    best_score = 0.0
    
    # Strategy 1: Greedy in order of increasing modulus
    selected = []
    for m in range(20, 201):
        if time.time() > deadline - 20:
            break
        valid_rs = _find_valid_residues(m, selected)
        if valid_rs:
            selected.append((m, valid_rs[0]))
    
    if selected:
        classes = [[m, r] for m, r in selected]
        if _is_valid(classes):
            sc = _score(classes)
            if sc > best_score:
                best_score = sc
                best_classes = classes[:]
    
    # Strategy 2: Randomized multi-start greedy with local search
    # This is the key improvement: try many random orderings and residue choices
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Create a random ordering with bias toward smaller moduli
        moduli_list = list(range(20, 201))
        # Weight smaller moduli more heavily to be picked earlier
        weights = [1.0 / m for m in moduli_list]
        order = []
        remaining = moduli_list[:]
        remaining_weights = weights[:]
        while remaining:
            total_w = sum(remaining_weights)
            if total_w == 0:
                idx = 0
            else:
                r = random.random() * total_w
                cum = 0
                idx = 0
                for i, w in enumerate(remaining_weights):
                    cum += w
                    if r <= cum:
                        idx = i
                        break
            order.append(remaining[idx])
            remaining.pop(idx)
            remaining_weights.pop(idx)
        
        # Greedy build with random residue selection
        selected = []
        for m in order:
            if time.time() > deadline - 30:
                break
            valid_rs = _find_valid_residues(m, selected)
            if valid_rs:
                # Randomly pick a valid residue
                r = random.choice(valid_rs)
                selected.append((m, r))
        
        if selected:
            classes = [[m, r] for m, r in selected]
            if _is_valid(classes):
                sc = _score(classes)
                # Apply local search
                improved_classes, improved_sc = _local_search(classes, deadline - 30)
                if improved_sc > best_score:
                    best_score = improved_sc
                    best_classes = improved_classes
    
    # Strategy 3: Structured families with local search
    for g in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]:
        if time.time() > deadline - 10:
            break
        multiples = [m for m in range(20, 201) if m % g == 0]
        num_classes = min(g, len(multiples))
        selected_moduli = multiples[:num_classes]
        selected = [(m, i) for i, m in enumerate(selected_moduli)]
        classes = [[m, r] for m, r in selected]
        if _is_valid(classes):
            sc = _score(classes)
            # Apply local search
            improved_classes, improved_sc = _local_search(classes, deadline - 10)
            if improved_sc > best_score:
                best_score = improved_sc
                best_classes = improved_classes
    
    # Final local search on best solution
    if best_classes and time.time() < deadline - 5:
        improved_classes, improved_sc = _local_search(best_classes, deadline - 2)
        if improved_sc > best_score:
            best_score = improved_sc
            best_classes = improved_classes
    
    if best_classes is None:
        # Fallback to seed
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]
    
    return best_classes
