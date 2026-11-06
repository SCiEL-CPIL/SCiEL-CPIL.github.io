"""Search-based covering system optimizer with local search."""
import time
import random
import math
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
_N_DIVS = len(_DIVS)

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
            count = 0
            for i in range(r, _BASE, m):
                if covered[i] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        for i in range(best_r, _BASE, m):
            covered[i] = 1
    return classes

def _build_random(seed_offset=0):
    """Random initial solution with fixed seed offset."""
    rng = random.Random(20240517 + seed_offset)
    classes = []
    for m in sorted(_DIVS):
        r = rng.randint(0, m - 1)
        classes.append([m, r])
    return classes

def _local_search(classes, max_iters=5000, deadline=_DEADLINE - 10):
    """Iteratively improve by changing residues of individual moduli."""
    best_score = _score(classes)
    best_classes = [c[:] for c in classes]
    
    # Precompute coverage mask for current solution
    covered = bytearray(_BASE)
    for m, r in best_classes:
        for i in range(r, _BASE, m):
            covered[i] = 1
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(_N_DIVS):
            m, r = best_classes[idx]
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
                # Update covered array by recomputing
                covered = bytearray(_BASE)
                for j, (mj, rj) in enumerate(best_classes):
                    if j == idx:
                        for i in range(best_r, _BASE, mj):
                            covered[i] = 1
                    else:
                        for i in range(rj, _BASE, mj):
                            covered[i] = 1
                best_classes[idx] = [m, best_r]
                best_score = _BASE - sum(covered)
                improved = True
                break
        if not improved:
            break
    return best_classes, best_score

def _simulated_annealing(classes, max_iters=3000, deadline=_DEADLINE - 15):
    """Simulated annealing to escape local optima."""
    current_classes = [c[:] for c in classes]
    current_score = _score(current_classes)
    best_classes = [c[:] for c in current_classes]
    best_score = current_score
    
    # Coverage mask
    covered = bytearray(_BASE)
    for m, r in current_classes:
        for i in range(r, _BASE, m):
            covered[i] = 1
    
    T = 50.0  # Initial temperature
    T_min = 0.01
    cooling = 0.98
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if T <= T_min:
            break
            
        # Pick a random class and random new residue
        idx = _rng.randint(0, _N_DIVS - 1)
        m, r = current_classes[idx]
        new_r = _rng.randint(0, m - 1)
        if new_r == r:
            continue
            
        # Compute delta score
        delta = 0
        for i in range(new_r, _BASE, m):
            if covered[i] == 0:
                delta -= 1  # new coverage
        for i in range(r, _BASE, m):
            if covered[i] == 0:
                delta += 1  # lost coverage
        
        # Accept or reject
        if delta < 0 or _rng.random() < math.exp(-delta / T):
            # Update coverage
            for i in range(r, _BASE, m):
                if covered[i] == 0:
                    covered[i] = 0
            for i in range(new_r, _BASE, m):
                covered[i] = 1
            current_classes[idx] = [m, new_r]
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_classes = [c[:] for c in current_classes]
        else:
            # Revert: do nothing since we didn't update
            pass
        
        T *= cooling
    
    # Final local search polish
    best_classes, best_score = _local_search(best_classes, max_iters=500, deadline=deadline)
    return best_classes, best_score

def _polish(classes, deadline=_DEADLINE - 5):
    """Final high-precision polish: multiple passes of local search."""
    best_classes = [c[:] for c in classes]
    best_score = _score(best_classes)
    
    for _ in range(3):
        if time.time() > deadline:
            break
        new_classes, new_score = _local_search(best_classes, max_iters=2000, deadline=deadline)
        if new_score < best_score:
            best_classes = new_classes
            best_score = new_score
        else:
            break
    
    return best_classes, best_score

def construct():
    # Phase 1: Greedy initial solution
    classes = _build_greedy()
    best_score = _score(classes)
    best_classes = [c[:] for c in classes]
    
    # Phase 2: Local search on greedy
    ls_classes, ls_score = _local_search(classes, max_iters=5000, deadline=_DEADLINE - 30)
    if ls_score < best_score:
        best_classes = ls_classes
        best_score = ls_score
    
    # Phase 3: Simulated annealing from best solution
    if time.time() < _DEADLINE - 40:
        sa_classes, sa_score = _simulated_annealing(best_classes, max_iters=5000, deadline=_DEADLINE - 20)
        if sa_score < best_score:
            best_classes = sa_classes
            best_score = sa_score
    
    # Phase 4: Try random starts with local search
    if time.time() < _DEADLINE - 25:
        for offset in range(1, 6):
            if time.time() > _DEADLINE - 20:
                break
            rand_classes = _build_random(offset)
            rc_classes, rc_score = _local_search(rand_classes, max_iters=3000, deadline=_DEADLINE - 15)
            if rc_score < best_score:
                best_classes = rc_classes
                best_score = rc_score
    
    # Phase 5: Final polish
    best_classes, best_score = _polish(best_classes, deadline=_DEADLINE - 3)
    
    return best_classes
# EVOLVE-BLOCK-END
