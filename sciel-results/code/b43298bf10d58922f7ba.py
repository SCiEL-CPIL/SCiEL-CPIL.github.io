"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

_DEADLINE = 115.0  # seconds, safety margin before 120s/150s kill

def _is_compatible(m, r, placed):
    """Check if residue r for modulus m is compatible with all placed moduli."""
    for m2, r2 in placed:
        g = math.gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True

def _try_place(placed, m):
    """Try to place modulus m with some compatible residue. Returns (success, r)."""
    for r in range(m):
        if _is_compatible(m, r, placed):
            return True, r
    return False, None

def _score(placed):
    """Compute the sum of 1/m for placed moduli as a float."""
    s = 0.0
    for m, r in placed:
        s += 1.0 / m
    return s

def _greedy_build(order):
    """Build a solution by placing moduli in the given order."""
    placed = []
    for m in order:
        ok, r = _try_place(placed, m)
        if ok:
            placed.append((m, r))
    return placed

def _local_search(placed, all_moduli, time_limit=2.0):
    """Try to improve by swapping or replacing moduli. Returns improved placed list."""
    placed_set = set(m for m, r in placed)
    current_score = _score(placed)
    start_time = time.time()
    
    # Try adding unplaced moduli
    improved = True
    while improved:
        improved = False
        for m in all_moduli:
            if m in placed_set:
                continue
            ok, r = _try_place(placed, m)
            if ok:
                placed.append((m, r))
                placed_set.add(m)
                current_score += 1.0 / m
                improved = True
        if time.time() - start_time > time_limit:
            break
    
    # Try replacing a placed modulus with two unplaced ones that give better score
    # This is complex, so we'll do a simpler local search: try swapping
    for i, (m1, r1) in enumerate(placed):
        for m2 in all_moduli:
            if m2 in placed_set:
                continue
            # Try removing m1 and adding m2
            if 1.0 / m2 > 1.0 / m1:
                new_placed = [p for p in placed if p[0] != m1]
                ok, r = _try_place(new_placed, m2)
                if ok:
                    new_placed.append((m2, r))
                    new_score = _score(new_placed)
                    if new_score > current_score:
                        placed = new_placed
                        placed_set = set(m for m, r in placed)
                        current_score = new_score
                        improved = True
                        if time.time() - start_time > time_limit:
                            return placed
    return placed

def construct():
    start_time = time.time()
    
    # All moduli from 20 to 200
    all_moduli = list(range(20, 201))
    
    # Strategy: start with small moduli (higher 1/m), try to pack as many as possible
    # Greedy: sort by 1/m descending (i.e., m ascending), try to place each
    best_placed = []
    best_score = 0.0
    
    # Multiple greedy orders
    orders = [
        all_moduli,  # ascending
        sorted(all_moduli, key=lambda m: m),  # same as above
        # Try different starting points
    ]
    
    for order in orders:
        placed = _greedy_build(order)
        score = _score(placed)
        if score > best_score:
            best_score = score
            best_placed = placed
    
    # Local search to improve
    if time.time() - start_time < _DEADLINE - 5:
        best_placed = _local_search(best_placed, all_moduli, time_limit=_DEADLINE - 10)
        best_score = _score(best_placed)
    
    # Convert to the required format [[m, r], ...]
    result = [[m, r] for m, r in best_placed]
    
    return result
# EVOLVE-BLOCK-END
