import time
import math
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)

def _compat(assign, m, r):
    """Check if residue r for modulus m is compatible with all in assign."""
    for m2, r2 in assign.items():
        g = math.gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True

def _try_place_greedy(assign, m):
    """Try to place m with a compatible residue. Returns True on success."""
    # Try residues in order 0, 1, 2, ..., m-1
    for r in range(m):
        if _compat(assign, m, r):
            assign[m] = r
            return True
    return False

def _try_place_smart(assign, m):
    """Try to place m with a residue that maximizes future flexibility."""
    # For each candidate residue, count how many future moduli it would block
    # This is expensive, so use a heuristic: prefer residues that are 'unique' mod small gcds
    best_r = None
    best_score = -1
    for r in range(m):
        if not _compat(assign, m, r):
            continue
        # Heuristic: prefer residues that are 0 mod many small numbers (less constraining)
        # Actually, prefer residues that differ from existing ones mod their gcds
        score = 0
        for m2, r2 in assign.items():
            g = math.gcd(m, m2)
            if g > 1:
                # If r mod g is different from r2 mod g, that's good
                # Count how many other residues mod g are still available
                score += 1
        if score > best_score:
            best_score = score
            best_r = r
    if best_r is not None:
        assign[m] = best_r
        return True
    # Fallback: first compatible
    for r in range(m):
        if _compat(assign, m, r):
            assign[m] = r
            return True
    return False

def _compute_score(assign):
    """Compute sum of 1/m for all placed moduli."""
    return sum(1.0 / m for m in assign.keys())

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140.0
    
    best_assign = {}
    best_score = 0.0
    
    # Strategy 1: Greedy from smallest to largest modulus
    assign = {}
    for m in range(20, 201):
        if _remaining_time() < 1.0:
            break
        if _try_place_greedy(assign, m):
            pass  # placed
    score = _compute_score(assign)
    if score > best_score:
        best_score = score
        best_assign = dict(assign)
    
    # Strategy 2: Try with different residue ordering
    # For each modulus, try residues in a different order
    assign = {}
    for m in range(20, 201):
        if _remaining_time() < 1.0:
            break
        # Try residues in order: 0, m-1, 1, m-2, 2, m-3, ...
        tried = set()
        placed = False
        for offset in range(m // 2 + 1):
            for r in [offset, m - 1 - offset]:
                if r < 0 or r >= m or r in tried:
                    continue
                tried.add(r)
                if _compat(assign, m, r):
                    assign[m] = r
                    placed = True
                    break
            if placed:
                break
        if not placed:
            # Try all residues
            for r in range(m):
                if r not in tried and _compat(assign, m, r):
                    assign[m] = r
                    break
    score = _compute_score(assign)
    if score > best_score:
        best_score = score
        best_assign = dict(assign)
    
    # Strategy 3: Backtracking search for a subset
    # Try to find a better subset using limited backtracking
    if _remaining_time() > 10.0:
        # Use a priority queue approach: always try to add the next smallest modulus
        # but allow removal if it leads to a better overall solution
        # For now, try a local search: remove one modulus, try to add others
        assign = dict(best_assign)
        improved = True
        while improved and _remaining_time() > 5.0:
            improved = False
            # Try removing each modulus and see if we can add more
            for m_remove in list(assign.keys()):
                if _remaining_time() < 2.0:
                    break
                assign_copy = dict(assign)
                del assign_copy[m_remove]
                # Try to add moduli not in assign_copy, in order of 1/m (smallest m first)
                added_any = False
                for m_add in range(20, 201):
                    if m_add in assign_copy:
                        continue
                    if _try_place_greedy(assign_copy, m_add):
                        added_any = True
                if added_any:
                    new_score = _compute_score(assign_copy)
                    if new_score > best_score:
                        best_score = new_score
                        best_assign = dict(assign_copy)
                        assign = dict(assign_copy)
                        improved = True
                        break
    
    # Final: ensure we have at least the baseline
    if not best_assign:
        # Fallback to the original seed
        best_assign = {20*j: j-1 for j in range(1, 11)}
        best_score = _compute_score(best_assign)
    
    return [[m, best_assign[m]] for m in sorted(best_assign.keys())]
