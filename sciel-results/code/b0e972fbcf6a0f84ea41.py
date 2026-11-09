"""Optimized disjoint residue class construction maximizing sum of 1/m."""
import time
import random
import math
from fractions import Fraction

def _disjoint(m1, r1, m2, r2):
    """Check if residue classes are pairwise disjoint: r1 != r2 mod gcd(m1, m2)."""
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _score(classes):
    """Exact rational score, returned as float."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _find_residue(m, selected):
    """Find a residue r in [0, m) that is disjoint from all selected classes.
    Returns None if no valid residue exists."""
    for r in range(m):
        ok = True
        for (m2, r2) in selected:
            if not _disjoint(m, r, m2, r2):
                ok = False
                break
        if ok:
            return r
    return None

def _try_add(m, selected):
    """Try to add modulus m with some residue to selected list. Returns True if added."""
    r = _find_residue(m, selected)
    if r is not None:
        selected.append((m, r))
        return True
    return False

def _local_search(selected, deadline, max_rounds=50):
    """Local search: try to replace larger moduli with smaller compatible ones."""
    selected = sorted(selected, key=lambda x: x[0])
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        for idx in range(len(selected) - 1, -1, -1):
            m_old, r_old = selected[idx]
            if m_old <= 20:
                break
            others = [s for i, s in enumerate(selected) if i != idx]
            # Try to find a smaller modulus that can be added
            for m_new in range(20, m_old):
                if any(m_new == s[0] for s in others):
                    continue
                r_new = _find_residue(m_new, others)
                if r_new is not None:
                    selected[idx] = (m_new, r_new)
                    improved = True
                    break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_classes = None
    best_score = 0.0

    # Strategy 1: Greedy ascending - add smallest moduli first
    # This naturally gives high score since small moduli have large reciprocals
    for attempt in range(5):
        if time.time() > deadline - 20:
            break

        selected = []
        moduli_to_try = list(range(20, 201))

        if attempt == 0:
            # Pure greedy ascending
            for m in moduli_to_try:
                if time.time() > deadline - 20:
                    break
                _try_add(m, selected)
        else:
            # Slightly randomized: shuffle moduli above 30
            random.seed(42 + attempt)
            moduli_shuffled = list(range(20, 31)) + list(range(31, 201))
            random.shuffle(moduli_shuffled[11:])
            for m in moduli_shuffled:
                if time.time() > deadline - 20:
                    break
                _try_add(m, selected)

        if selected:
            # Apply local search
            selected = _local_search(selected, deadline - 15)
            classes = [[m, r] for m, r in selected]
            sc = _score(classes)
            if sc > best_score:
                best_score = sc
                best_classes = classes[:]

    # Strategy 2: Start with pairwise coprime small moduli, then fill gaps
    if time.time() < deadline - 15:
        # Find a good set of pairwise coprime moduli
        coprime_set = []
        for m in range(20, 201):
            ok = True
            for p in coprime_set:
                if math.gcd(m, p) != 1:
                    ok = False
                    break
            if ok:
                coprime_set.append(m)

        # Start with these (residue 0 works for coprime set)
        selected = [(m, 0) for m in coprime_set]

        # Now try to add non-coprime moduli that still fit
        for m in range(20, 201):
            if time.time() > deadline - 15:
                break
            if m in [s[0] for s in selected]:
                continue
            if _try_add(m, selected):
                pass

        # Local search
        selected = _local_search(selected, deadline - 10)
        classes = [[m, r] for m, r in selected]
        sc = _score(classes)
        if sc > best_score:
            best_score = sc
            best_classes = classes[:]

    # Strategy 3: Aggressive local search from multiple random starts
    for attempt in range(30):
        if time.time() > deadline - 5:
            break

        random.seed(10000 + attempt)
        selected = []

        # Randomized greedy: pick moduli in random order but biased toward small
        available = list(range(20, 201))
        # Weight by 1/m so smaller moduli are more likely
        weights = [1.0 / m for m in available]

        for _ in range(80):
            if time.time() > deadline - 5:
                break
            if not available:
                break

            # Weighted random selection
            total_w = sum(weights)
            if total_w <= 0:
                break
            r = random.random() * total_w
            cumsum = 0
            chosen_idx = len(available) - 1
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    chosen_idx = i
                    break

            m = available[chosen_idx]
            if _try_add(m, selected):
                available.pop(chosen_idx)
                weights.pop(chosen_idx)

        if selected:
            # Local search
            selected = _local_search(selected, deadline - 3)
            classes = [[m, r] for m, r in selected]
            sc = _score(classes)
            if sc > best_score:
                best_score = sc
                best_classes = classes[:]

    # Strategy 4: Targeted improvement - try to add very small moduli (20-25)
    # that weren't included, by removing some larger ones if necessary
    if time.time() < deadline - 3 and best_classes:
        current_moduli = set(m for m, r in best_classes)
        for small_m in range(20, 26):
            if small_m in current_moduli:
                continue
            # Try adding small_m by removing the largest modulus if needed
            test_selected = [(m, r) for m, r in best_classes]
            if _try_add(small_m, test_selected):
                classes = [[m, r] for m, r in test_selected]
                sc = _score(classes)
                if sc > best_score:
                    best_score = sc
                    best_classes = classes[:]

    if best_classes is None:
        # Fallback: seed construction
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]

    return best_classes
