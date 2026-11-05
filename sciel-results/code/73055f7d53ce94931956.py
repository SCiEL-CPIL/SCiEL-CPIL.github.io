"""Improved program: greedy (large moduli first) + local search refinement."""
import time
import random
from itertools import combinations

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# All odd divisors > 1 of 51975
def _get_moduli():
    mods = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            mods.append(d)
    return mods

MODULI = _get_moduli()
# Sort in descending order for greedy
MODULI_DESC = sorted(MODULI, reverse=True)
MODULI_SET = set(MODULI)

def _compute_uncovered(classes):
    """Compute the number of uncovered residues given a list of [m, r] classes."""
    covered = bytearray(BASE)
    for m, r in classes:
        covered[r::m] = b'\x01' * len(range(r, BASE, m))
    return BASE - sum(covered)

def _greedy_init():
    """Greedy: process moduli in descending order, pick residue covering most uncovered."""
    covered = bytearray(BASE)
    classes = []
    for m in MODULI_DESC:
        best_r = 0
        best_count = -1
        for r in range(m):
            cnt = sum(1 for i in range(r, BASE, m) if not covered[i])
            if cnt > best_count:
                best_count = cnt
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = b'\x01' * len(range(best_r, BASE, m))
    return classes, _compute_uncovered(classes)

def _local_search(classes, uncovered_count, time_limit):
    """Hill-climbing: try changing residues of individual moduli to reduce uncovered."""
    best_classes = [c[:] for c in classes]
    best_uncovered = uncovered_count
    deadline = time.time() + time_limit
    
    while time.time() < deadline:
        improved = False
        # Try each modulus in a random order
        indices = list(range(len(best_classes)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() >= deadline:
                break
            m = best_classes[idx][0]
            old_r = best_classes[idx][1]
            
            # Compute current coverage
            covered = bytearray(BASE)
            for i, (mm, rr) in enumerate(best_classes):
                if i != idx:
                    covered[rr::mm] = b'\x01' * len(range(rr, BASE, mm))
            
            # Try all residues for this modulus
            best_r = old_r
            best_cnt = -1
            for r in range(m):
                cnt = sum(1 for i in range(r, BASE, m) if not covered[i])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_r = r
            
            if best_r != old_r and best_cnt > 0:
                # Check if this actually reduces uncovered
                new_classes = [c[:] for c in best_classes]
                new_classes[idx] = [m, best_r]
                new_uncovered = _compute_uncovered(new_classes)
                if new_uncovered < best_uncovered:
                    best_classes = new_classes
                    best_uncovered = new_uncovered
                    improved = True
        if not improved:
            break
    
    return best_classes, best_uncovered

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Greedy initialization
    classes, uncovered = _greedy_init()
    
    # Local search refinement
    remaining = deadline - time.time()
    if remaining > 5:
        classes, uncovered = _local_search(classes, uncovered, remaining - 5)
    
    return classes
