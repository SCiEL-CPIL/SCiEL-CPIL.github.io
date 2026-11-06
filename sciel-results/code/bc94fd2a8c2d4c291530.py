import time
import random
from collections import defaultdict

_BASE = 51975
_DEADLINE = 115.0
_START = time.time()

# Precompute all divisors > 1 of 51975
def _get_divisors(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _get_divisors(_BASE)
_DIVS_SORTED = sorted(_DIVS)

# Precompute position lists for each (m, r) pair
# For each modulus m, positions r, r+m, r+2m, ... mod _BASE
_POS_CACHE = {}
def _get_positions(m, r):
    key = (m, r)
    if key not in _POS_CACHE:
        _POS_CACHE[key] = tuple(range(r, _BASE, m))
    return _POS_CACHE[key]

# Precompute for each modulus m, a list of all residue position tuples
_MOD_RESIDUE_POSITIONS = {}
for m in _DIVS_SORTED:
    _MOD_RESIDUE_POSITIONS[m] = [_get_positions(m, r) for r in range(m)]

def _count_uncovered(covered):
    return _BASE - sum(covered)

def _build_covered(classes):
    """Build a bytearray showing which positions are covered."""
    covered = bytearray(_BASE)
    for m, r in classes:
        for pos in _get_positions(m, r):
            covered[pos] = 1
    return covered

def _greedy_build(order=None):
    """Greedy: add moduli in given order (default ascending), each picks best residue."""
    if order is None:
        order = _DIVS_SORTED
    
    covered = bytearray(_BASE)
    classes = []
    
    for m in order:
        # Find residue r that covers the most currently uncovered positions
        best_r = 0
        best_count = -1
        for r in range(m):
            count = 0
            for pos in _get_positions(m, r):
                if covered[pos] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
            if best_count == 0:
                break
        
        classes.append([m, best_r])
        for pos in _get_positions(m, best_r):
            covered[pos] = 1
    
    return classes, _count_uncovered(covered)

def _local_refine(classes, max_time=None):
    """Fast local refinement using coverage count array."""
    classes = [c[:] for c in classes]
    n_classes = len(classes)
    
    # Build coverage count: how many classes cover each position
    cov_count = bytearray(_BASE)  # 0-255 should be enough
    for m, r in classes:
        for pos in _get_positions(m, r):
            cov_count[pos] += 1
    
    uncovered = _BASE - sum(1 for c in cov_count if c > 0)
    if max_time is None:
        max_time = time.time() - _START + 10  # default: use remaining time
    
    improved = True
    while improved:
        if time.time() > max_time:
            break
        improved = False
        
        for i in range(n_classes):
            m, r = classes[i]
            positions_old = _get_positions(m, r)
            
            best_r2 = r
            best_delta = 0  # delta = gain - loss (we want to maximize)
            
            for r2 in range(m):
                if r2 == r:
                    continue
                positions_new = _get_positions(m, r2)
                
                # gain: positions in new but not old that are currently uncovered
                gain = 0
                for pos in positions_new:
                    if pos not in set(positions_old) and cov_count[pos] == 0:
                        gain += 1
                # loss: positions in old but not new that are only covered by this class
                loss = 0
                for pos in positions_old:
                    if pos not in set(positions_new) and cov_count[pos] == 1:
                        loss += 1
                
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_r2 = r2
            
            if best_r2 != r and best_delta > 0:
                # Apply change
                positions_old = _get_positions(m, r)
                positions_new = _get_positions(m, best_r2)
                old_set = set(positions_old)
                new_set = set(positions_new)
                
                for pos in positions_old:
                    if pos not in new_set:
                        cov_count[pos] -= 1
                for pos in positions_new:
                    if pos not in old_set:
                        cov_count[pos] += 1
                
                classes[i] = [m, best_r2]
                uncovered -= best_delta
                improved = True
                if time.time() > max_time:
                    break
    
    return classes, uncovered

def _random_restart_build(rng):
    """Build with random ordering of moduli."""
    order = _DIVS_SORTED[:]
    rng.shuffle(order)
    return _greedy_build(order)

def construct():
    global _START
    
    # Start with greedy build
    classes, uncovered = _greedy_build()
    best_classes = [c[:] for c in classes]
    best_uncovered = uncovered
    
    # Time check
    if time.time() - _START > _DEADLINE - 10:
        return best_classes
    
    # Local refinement on the greedy result
    refined_classes, refined_uncovered = _local_refine(best_classes)
    if refined_uncovered < best_uncovered:
        best_classes = [c[:] for c in refined_classes]
        best_uncovered = refined_uncovered
    
    # Time check
    if time.time() - _START > _DEADLINE - 5:
        return best_classes
    
    # Try random restarts
    rng = random.Random(42)
    while time.time() - _START < _DEADLINE - 5:
        rc, ru = _random_restart_build(rng)
        if ru < best_uncovered:
            best_classes = [c[:] for c in rc]
            best_uncovered = ru
            # Local refine the new best
            refined_classes, refined_uncovered = _local_refine(best_classes)
            if refined_uncovered < best_uncovered:
                best_classes = [c[:] for c in refined_classes]
                best_uncovered = refined_uncovered
        
        if time.time() - _START > _DEADLINE - 5:
            break
    
    return best_classes
