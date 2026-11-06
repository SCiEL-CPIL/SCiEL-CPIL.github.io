"""Search-based covering system optimizer with local search."""
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
_BASE = 51975
_DEADLINE = 115.0
_START = time.time()
_rng = random.Random(20240517)

def _divisors(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _divisors(_BASE)  # 41 distinct odd divisors > 1

def _score(classes):
    """Exact uncovered count."""
    covered = bytearray(_BASE)
    for m, r in classes:
        for i in range(r, _BASE, m):
            covered[i] = 1
    return _BASE - sum(covered)

def _build_greedy():
    """Greedy construction: moduli in increasing order, each picks best residue."""
    covered = bytearray(_BASE)
    classes = []
    for m in sorted(_DIVS):
        best_r = 0
        best_count = -1
        for r in range(m):
            count = sum(1 for i in range(r, _BASE, m) if covered[i] == 0)
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        for i in range(best_r, _BASE, m):
            covered[i] = 1
    return classes

def _local_search(classes, max_iters=5000):
    """Iteratively improve by changing residues of individual moduli."""
    best_score = _score(classes)
    best_classes = [c[:] for c in classes]
    covered = bytearray(_BASE)
    for m, r in best_classes:
        for i in range(r, _BASE, m):
            covered[i] = 1
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        improved = False
        for idx, (m, r) in enumerate(best_classes):
            # Try all possible residues for this modulus
            best_r = r
            best_gain = 0
            # Current coverage by this class
            cur_count = sum(1 for i in range(r, _BASE, m) if covered[i] == 0)
            for new_r in range(m):
                if new_r == r:
                    continue
                # Compute gain: how many newly covered - how many lost
                gain = 0
                for i in range(new_r, _BASE, m):
                    if covered[i] == 0:
                        gain += 1
                for i in range(r, _BASE, m):
                    if covered[i] == 0:
                        gain -= 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = new_r
            if best_gain > 0:
                # Update covered array
                for i in range(r, _BASE, m):
                    if covered[i] == 0:
                        covered[i] = 0  # will be reset
                # Actually need to recompute properly
                covered_new = bytearray(_BASE)
                for j, (mj, rj) in enumerate(best_classes):
                    if j == idx:
                        for i in range(best_r, _BASE, mj):
                            covered_new[i] = 1
                    else:
                        for i in range(rj, _BASE, mj):
                            covered_new[i] = 1
                covered = covered_new
                best_classes[idx] = [m, best_r]
                best_score = _BASE - sum(covered)
                improved = True
                break
        if not improved:
            break
    return best_classes, best_score

def construct():
    classes = _build_greedy()
    best_score = _score(classes)
    best_classes = [c[:] for c in classes]
    
    # Local search
    improved_classes, improved_score = _local_search(classes, max_iters=10000)
    if improved_score < best_score:
        best_classes = improved_classes
        best_score = improved_score
    
    return best_classes
# EVOLVE-BLOCK-END
