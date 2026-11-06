import time
import random
import numpy as np

_BASE = 51975
_DEADLINE = 115.0
_START = time.time()

# Precompute all divisors > 1 of 51975
def _get_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _get_divisors(_BASE)
_DIVS_SORTED = sorted(_DIVS)

def _count_uncovered(covered):
    """Count how many positions are 0 (uncovered) in the bytearray."""
    return covered.count(0)

def _greedy_build(order):
    """Greedy: add moduli in given order, each picks residue covering most uncovered."""
    covered = bytearray(_BASE)  # 0 = uncovered
    classes = []
    
    for m in order:
        # Find residue r that covers the most currently uncovered positions
        best_r = 0
        best_count = -1
        for r in range(m):
            count = 0
            for pos in range(r, _BASE, m):
                if covered[pos] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
            if best_count == 0:
                break
        
        classes.append([m, best_r])
        # Mark as covered
        for pos in range(best_r, _BASE, m):
            covered[pos] = 1
    
    return classes, _count_uncovered(covered)

def _local_refine(classes, covered, uncovered_count):
    """Try to improve by changing residues of existing classes."""
    classes = [c[:] for c in classes]
    improved = True
    iterations = 0
    
    while improved and iterations < 20:
        improved = False
        iterations += 1
        
        for i in range(len(classes)):
            m, r = classes[i]
            # Try all other residues for this modulus
            for r2 in range(m):
                if r2 == r:
                    continue
                
                # Calculate improvement: how many new positions covered, minus how many lost
                gain = 0
                loss = 0
                
                # Positions that would be lost (covered by this class at r but not r2)
                for pos in range(r, _BASE, m):
                    if covered[pos] == 1:
                        # Check if this position is covered by any other class
                        only_here = True
                        for j, (m2, r2o) in enumerate(classes):
                            if j == i:
                                continue
                            if (pos - r2o) % m2 == 0:
                                only_here = False
                                break
                        if only_here:
                            loss += 1
                
                # Positions that would be gained (uncovered, but covered by r2)
                for pos in range(r2, _BASE, m):
                    if covered[pos] == 0:
                        gain += 1
                
                if gain > loss:
                    # Apply change
                    for pos in range(r, _BASE, m):
                        # Only unmark if not covered by other classes
                        only_here = True
                        for j, (m2, r2o) in enumerate(classes):
                            if j == i:
                                continue
                            if (pos - r2o) % m2 == 0:
                                only_here = False
                                break
                        if only_here:
                            covered[pos] = 0
                    for pos in range(r2, _BASE, m):
                        covered[pos] = 1
                    classes[i] = [m, r2]
                    uncovered_count += loss - gain
                    improved = True
                    break
    
    return classes, uncovered_count

def _random_restart_build(rng, order=None):
    """Build with random ordering of moduli."""
    if order is None:
        order = _DIVS_SORTED[:]
        rng.shuffle(order)
    
    covered = bytearray(_BASE)
    classes = []
    
    for m in order:
        best_r = 0
        best_count = -1
        for r in range(m):
            count = 0
            for pos in range(r, _BASE, m):
                if covered[pos] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
            if best_count == 0:
                break
        
        classes.append([m, best_r])
        for pos in range(best_r, _BASE, m):
            covered[pos] = 1
    
    return classes, _count_uncovered(covered)

def construct():
    global _START
    
    # Start with greedy build using sorted order
    classes, uncovered = _greedy_build(_DIVS_SORTED)
    best_classes = [c[:] for c in classes]
    best_uncovered = uncovered
    
    # Time check
    if time.time() - _START > _DEADLINE - 10:
        return best_classes
    
    # Local refinement on the greedy result
    covered = bytearray(_BASE)
    for m, r in best_classes:
        for pos in range(r, _BASE, m):
            covered[pos] = 1
    
    refined_classes, refined_uncovered = _local_refine(best_classes, covered, best_uncovered)
    if refined_uncovered < best_uncovered:
        best_classes = [c[:] for c in refined_classes]
        best_uncovered = refined_uncovered
    
    # Time check
    if time.time() - _START > _DEADLINE - 5:
        return best_classes
    
    # Try random restarts with different orderings
    rng = random.Random(42)
    restarts = 0
    while time.time() - _START < _DEADLINE - 5:
        restarts += 1
        order = _DIVS_SORTED[:]
        rng.shuffle(order)
        rc, ru = _random_restart_build(rng, order)
        if ru < best_uncovered:
            best_classes = [c[:] for c in rc]
            best_uncovered = ru
            # Local refine the new best
            covered = bytearray(_BASE)
            for m, r in best_classes:
                for pos in range(r, _BASE, m):
                    covered[pos] = 1
            refined_classes, refined_uncovered = _local_refine(best_classes, covered, best_uncovered)
            if refined_uncovered < best_uncovered:
                best_classes = [c[:] for c in refined_classes]
                best_uncovered = refined_uncovered
        
        if time.time() - _START > _DEADLINE - 5:
            break
    
    return best_classes
