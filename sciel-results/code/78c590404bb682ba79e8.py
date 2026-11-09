"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# Divisors:
# 3^a * 5^b * 7^c * 11^d
# a in 0..3, b in 0..2, c in 0..1, d in 0..1

def _get_divisors():
    divs = []
    for a in range(4):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    val = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if val > 1:
                        divs.append(val)
    return sorted(divs)

DIVISORS = _get_divisors()

def _score(classes):
    """Count uncovered residues."""
    covered = bytearray(BASE)
    for m, r in classes:
        if r < m:
            covered[r::m] = b'\x01' * len(range(r, BASE, m))
    return covered.count(0)

def _greedy_order(moduli_order, covered=None):
    """Greedy: pick residue for each modulus in given order to maximize new coverage."""
    if covered is None:
        covered = bytearray(BASE)
    classes = []
    for m in moduli_order:
        best_r = 0
        best_gain = -1
        # For small m, try all residues
        for r in range(m):
            # Count how many uncovered residues we'd cover
            # covered[r::m].count(0)
            gain = covered[r::m].count(0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        if best_gain > 0:
            covered[best_r::m] = b'\x01' * len(range(best_r, BASE, m))
    return classes, _score(classes)

def _greedy_with_local_search(classes, max_iter=5000, deadline=None):
    """Local search: try changing residues of existing classes, or swapping which modulus is used."""
    # We'll keep a fixed set of moduli (all divisors) and just optimize residues
    # Actually, let's allow adding/removing classes
    # For simplicity, start with all divisors and optimize residues via local search
    
    score = _score(classes)
    best_score = score
    best_classes = [c[:] for c in classes]
    
    # Map modulus to index
    m_to_idx = {c[0]: i for i, c in enumerate(classes)}
    
    it = 0
    while it < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random class to modify
        idx = random.randint(0, len(classes) - 1)
        m, old_r = classes[idx]
        # Try a random new residue
        new_r = random.randint(0, m - 1)
        if new_r == old_r:
            continue
        # Compute score change
        # Remove old, add new
        covered = bytearray(BASE)
        for j, (mm, rr) in enumerate(classes):
            if j == idx:
                rr = new_r
            covered[rr::mm] = b'\x01' * len(range(rr, BASE, mm))
        new_score = covered.count(0)
        if new_score < best_score:
            best_score = new_score
            best_classes = [c[:] for c in classes]
            classes[idx][1] = new_r
        it += 1
    
    return best_classes, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Start with greedy in increasing modulus order
    classes, score = _greedy_order(DIVISORS)
    best_classes = [c[:] for c in classes]
    best_score = score
    
    # Try greedy in decreasing modulus order
    classes2, score2 = _greedy_order(list(reversed(DIVISORS)))
    if score2 < best_score:
        best_score = score2
        best_classes = [c[:] for c in classes2]
    
    # Local search on the best
    if time.time() < deadline - 5.0:
        best_classes, best_score = _greedy_with_local_search(
            best_classes, max_iter=20000, deadline=deadline - 5.0
        )
    
    # Additional local search passes if time remains
    while time.time() < deadline - 2.0:
        before = best_score
        best_classes, best_score = _greedy_with_local_search(
            best_classes, max_iter=5000, deadline=deadline - 2.0
        )
        if best_score == before:
            break
    
    return best_classes
# EVOLVE-BLOCK-END
