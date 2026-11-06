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
_DIVS_SORTED = sorted(_DIVS)

def _score(classes):
    """Exact uncovered count."""
    covered = bytearray(_BASE)
    for m, r in classes:
        for i in range(r, _BASE, m):
            covered[i] = 1
    return _BASE - sum(covered)

def _compute_covered(classes):
    """Compute coverage bytearray from a list of [m, r] classes."""
    covered = bytearray(_BASE)
    for m, r in classes:
        for i in range(r, _BASE, m):
            covered[i] = 1
    return covered

def _build_greedy():
    """Greedy construction: moduli in increasing order, each picks best residue."""
    covered = bytearray(_BASE)
    classes = []
    for m in _DIVS_SORTED:
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
    best_classes = [c[:] for c in classes]
    covered = _compute_covered(best_classes)
    best_score = _BASE - sum(covered)
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for idx, (m, r) in enumerate(best_classes):
            # Try all possible residues for this modulus
            best_r = r
            best_gain = 0
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
                covered_new = _compute_covered(best_classes)
                # Temporarily set idx to best_r
                best_classes[idx] = [m, best_r]
                covered_new = _compute_covered(best_classes)
                covered = covered_new
                best_score = _BASE - sum(covered)
                improved = True
                break
        if not improved:
            break
    return best_classes, best_score

def _simulated_annealing(classes, max_iters=20000):
    """Simulated annealing to escape local optima."""
    best_classes = [c[:] for c in classes]
    covered = _compute_covered(best_classes)
    best_score = _BASE - sum(covered)
    cur_classes = [c[:] for c in best_classes]
    cur_score = best_score
    
    T = 100.0
    T_min = 1.0
    cooling = 0.99
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        T = max(T * cooling, T_min)
        
        # Pick a random class and change its residue
        idx = _rng.randint(0, len(cur_classes) - 1)
        m, r = cur_classes[idx]
        if m <= 1:
            continue
        new_r = _rng.randint(0, m - 1)
        if new_r == r:
            continue
        
        # Compute new score
        old_classes = cur_classes[idx]
        cur_classes[idx] = [m, new_r]
        new_covered = _compute_covered(cur_classes)
        new_score = _BASE - sum(new_covered)
        delta = new_score - cur_score
        
        if delta < 0 or _rng.random() < pow(2.718281828, -delta / T):
            cur_classes[idx] = [m, new_r]
            cur_score = new_score
            covered = new_covered
            if new_score < best_score:
                best_score = new_score
                best_classes = [c[:] for c in cur_classes]
        else:
            cur_classes[idx] = old_classes
    
    return best_classes, best_score

def construct():
    # Build initial solution
    classes = _build_greedy()
    best_score = _score(classes)
    best_classes = [c[:] for c in classes]
    
    # Local search
    improved_classes, improved_score = _local_search(classes, max_iters=10000)
    if improved_score < best_score:
        best_classes = improved_classes
        best_score = improved_score
    
    # Simulated annealing to escape local optima
    if time.time() < _DEADLINE - 20:
        sa_classes, sa_score = _simulated_annealing(best_classes, max_iters=30000)
        if sa_score < best_score:
            best_classes = sa_classes
            best_score = sa_score
        
        # Another local search pass
        if time.time() < _DEADLINE - 5:
            final_classes, final_score = _local_search(best_classes, max_iters=10000)
            if final_score < best_score:
                best_classes = final_classes
                best_score = final_score
    
    return best_classes
# EVOLVE-BLOCK-END
