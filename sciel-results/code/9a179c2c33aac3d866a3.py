"""Optimized disjoint residue construction: maximize sum 1/m with pairwise disjointness."""
import time
import math
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

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_classes = None
    best_score = 0.0
    
    # Key insight: If gcd(m1,m2)=1, then r1 mod 1 = r2 mod 1 = 0, so they're NEVER disjoint.
    # Therefore, we CANNOT use coprime moduli! We need moduli with gcd > 1.
    # Strategy: Greedily add moduli in order of increasing m (decreasing 1/m),
    # assigning the smallest valid residue.
    
    # Strategy 1: Greedy in order of increasing modulus
    selected = []
    for m in range(20, 201):
        if time.time() > deadline - 10:
            break
        valid_rs = _find_valid_residues(m, selected)
        if valid_rs:
            # Pick the smallest valid residue
            selected.append((m, valid_rs[0]))
    
    if selected:
        classes = [[m, r] for m, r in selected]
        if _is_valid(classes):
            sc = _score(classes)
            if sc > best_score:
                best_score = sc
                best_classes = classes[:]
    
    # Strategy 2: Greedy but try different residue choices (smallest vs largest)
    for residue_strategy in range(3):
        if time.time() > deadline - 10:
            break
        selected = []
        for m in range(20, 201):
            if time.time() > deadline - 10:
                break
            valid_rs = _find_valid_residues(m, selected)
            if valid_rs:
                if residue_strategy == 0:
                    r = valid_rs[0]  # smallest
                elif residue_strategy == 1:
                    r = valid_rs[-1]  # largest
                else:
                    r = valid_rs[len(valid_rs) // 2]  # middle
                selected.append((m, r))
        
        if selected:
            classes = [[m, r] for m, r in selected]
            if _is_valid(classes):
                sc = _score(classes)
                if sc > best_score:
                    best_score = sc
                    best_classes = classes[:]
    
    # Strategy 3: Backtracking with most-constrained-first heuristic
    # For each modulus, compute how many valid residues it has given current selection.
    # Process moduli with fewest options first to avoid dead ends.
    
    def backtrack_greedy(order):
        """Try to build a valid set by processing moduli in given order."""
        selected = []
        for m in order:
            valid_rs = _find_valid_residues(m, selected)
            if valid_rs:
                selected.append((m, valid_rs[0]))
            # If no valid residue, skip this modulus
        return selected
    
    # Try processing in natural order (already done in Strategy 1)
    # Try processing larger moduli first (they have more residues to choose from)
    order_desc = list(range(200, 19, -1))
    selected = backtrack_greedy(order_desc)
    if selected:
        classes = [[m, r] for m, r in selected]
        if _is_valid(classes):
            sc = _score(classes)
            if sc > best_score:
                best_score = sc
                best_classes = classes[:]
    
    # Strategy 4: Local search - start from best, try to add more moduli or swap
    if best_classes:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_moduli = set(m for m, r in best_classes)
            # Try to add a new modulus
            for m in range(20, 201):
                if m in current_moduli:
                    continue
                valid_rs = _find_valid_residues(m, best_classes)
                if valid_rs:
                    new_classes = best_classes + [[m, valid_rs[0]]]
                    if _is_valid(new_classes):
                        sc = _score(new_classes)
                        if sc > best_score:
                            best_score = sc
                            best_classes = new_classes
                            improved = True
                            current_moduli = set(m2 for m2, r2 in best_classes)
                            break
            # Try to swap: remove one modulus, add a smaller one
            for i in range(len(best_classes)):
                if time.time() > deadline - 5:
                    break
                m_remove = best_classes[i][0]
                remaining = best_classes[:i] + best_classes[i+1:]
                for m_add in range(20, m_remove):
                    if m_add in set(m2 for m2, r2 in remaining):
                        continue
                    valid_rs = _find_valid_residues(m_add, remaining)
                    if valid_rs:
                        new_classes = remaining + [[m_add, valid_rs[0]]]
                        if _is_valid(new_classes):
                            sc = _score(new_classes)
                            if sc > best_score:
                                best_score = sc
                                best_classes = new_classes
                                improved = True
                                break
    
    # Strategy 5: Try structured families
    # All multiples of g with distinct residues mod g
    for g in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]:
        if time.time() > deadline - 5:
            break
        # Find multiples of g in [20, 200]
        multiples = [m for m in range(20, 201) if m % g == 0]
        # We can have at most g classes with distinct residues mod g
        # Pick the smallest min(g, len(multiples)) multiples
        num_classes = min(g, len(multiples))
        selected_moduli = multiples[:num_classes]
        # Assign residues 0, 1, ..., num_classes-1
        selected = [(m, i) for i, m in enumerate(selected_moduli)]
        classes = [[m, r] for m, r in selected]
        if _is_valid(classes):
            sc = _score(classes)
            if sc > best_score:
                best_score = sc
                best_classes = classes[:]
    
    # Strategy 6: Combined approach - use multiples of small g but also try to add
    # non-multiples that fit
    for g in [2, 3, 4, 5, 6, 10, 12, 20]:
        if time.time() > deadline - 5:
            break
        multiples = [m for m in range(20, 201) if m % g == 0]
        num_classes = min(g, len(multiples))
        selected_moduli = multiples[:num_classes]
        selected = [(m, i) for i, m in enumerate(selected_moduli)]
        # Now try to add non-multiples
        for m in range(20, 201):
            if m % g == 0:
                continue
            if m in set(m2 for m2, r2 in selected):
                continue
            valid_rs = _find_valid_residues(m, selected)
            if valid_rs:
                selected.append((m, valid_rs[0]))
        
        classes = [[m, r] for m, r in selected]
        if _is_valid(classes):
            sc = _score(classes)
            if sc > best_score:
                best_score = sc
                best_classes = classes[:]
    
    if best_classes is None:
        # Fallback to seed
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]
    
    return best_classes
