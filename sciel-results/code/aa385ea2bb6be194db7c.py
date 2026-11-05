"""
Minimum maximal no-isosceles set using Greedy Set Cover approach.
"""
import time
import numpy as np
import random

N = 16
TOTAL = N * N

def _idx_point(i):
    return (i // N, i % N)

def _precompute_iso():
    """
    Precompute iso_mask[i, j] as a boolean array of length TOTAL.
    iso_mask[i, j][k] is True if {i, j, k} forms an isosceles triangle.
    """
    pts = [_idx_point(i) for i in range(TOTAL)]
    iso_mask = np.zeros((TOTAL, TOTAL, TOTAL), dtype=bool)
    for i in range(TOTAL):
        xi, yi = pts[i]
        for j in range(TOTAL):
            if i == j:
                continue
            xj, yj = pts[j]
            d2 = (xi - xj)**2 + (yi - yj)**2
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                xk, yk = pts[k]
                d_ik = (xi - xk)**2 + (yi - yk)**2
                d_jk = (xj - xk)**2 + (yj - yk)**2
                if d2 == d_ik or d2 == d_jk or d_ik == d_jk:
                    iso_mask[i, j, k] = True
    return iso_mask

_iso_mask = _precompute_iso()

def _compute_blocked_mask(chosen):
    """
    Compute the set of points blocked by the chosen set.
    A point p is blocked if there exist a, b in chosen such that {p, a, b} is isosceles.
    """
    if not chosen:
        return np.zeros(TOTAL, dtype=bool)
    
    blocked = np.zeros(TOTAL, dtype=bool)
    # For each pair in chosen, mark the third points as blocked
    # Optimization: iterate over chosen pairs
    c_list = list(chosen)
    for idx_a, a in enumerate(c_list):
        for b in c_list[idx_a+1:]:
            # All k such that {a, b, k} is isosceles are blocked
            blocked |= _iso_mask[a, b]
    
    # Points in chosen are also considered "occupied" and thus effectively blocked from being added
    blocked[chosen] = True
    return blocked

def _is_maximal(chosen, blocked):
    """
    Check if the set is maximal.
    It is maximal if every point not in chosen is blocked.
    """
    chosen_mask = np.zeros(TOTAL, dtype=bool)
    chosen_mask[chosen] = True
    unchosen = ~chosen_mask
    # If any unchosen point is not blocked, it can be added, so not maximal
    return not (blocked[unchosen] == False).any()

def _greedy_set_cover(deadline):
    """
    Greedy approach to find a maximal no-isosceles set.
    """
    chosen = []
    chosen_mask = np.zeros(TOTAL, dtype=bool)
    available = np.ones(TOTAL, dtype=bool)
    
    # Precompute for each point c, the list of points it would block given a set C?
    # That depends on C. So we calculate on the fly.
    
    while time.time() < deadline:
        if not available.any():
            break
            
        candidates = np.where(available)[0]
        best_c = -1
        best_block_count = -1
        best_blocked_set = None
        
        for c in candidates:
            # Check if c can be added: no isosceles triangle with any pair in chosen
            valid = True
            if chosen:
                # Check against all pairs in chosen?
                # Actually, we just need to check if c forms an isosceles triangle with any pair in chosen.
                # We can check: for each a in chosen, is there a b in chosen such that iso_mask[a, b, c] is True?
                # Or simpler: for each a in chosen, if iso_mask[a, c] has any intersection with chosen?
                # Let's check: for a in chosen, if _iso_mask[a, c][chosen_mask].any():
                #   Then there exists b in chosen such that {a, c, b} is isosceles.
                #   This would make C U {c} invalid.
                for a in chosen:
                    if _iso_mask[a, c][chosen_mask].any():
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Calculate how many available points would be blocked by adding c
            # p is blocked if exists a in chosen such that {p, c, a} is isosceles
            # i.e., _iso_mask[a, c, p] is True for some a in chosen.
            # Also, c itself is removed from available.
            
            new_blocked = np.zeros(TOTAL, dtype=bool)
            if chosen:
                for a in chosen:
                    new_blocked |= _iso_mask[a, c]
            
            # The points that become newly blocked are those in `available` (excluding c) that are in new_blocked
            # Note: new_blocked might contain points already blocked, but we only care about `available` ones.
            # Also, c is not in available (it is the candidate), so we don't count it.
            
            # Mask of available points (excluding c)
            avail_ex_c = available.copy()
            avail_ex_c[c] = False
            
            newly_blocked = new_blocked & avail_ex_c
            count = newly_blocked.sum()
            
            if count > best_block_count:
                best_block_count = count
                best_c = c
                best_blocked_set = newly_blocked
        
        if best_c == -1:
            # No valid candidate found? This shouldn't happen if we start empty and available is non-empty.
            # If chosen is empty, all candidates are valid.
            # If chosen is not empty, it's possible that all available points form isosceles triangles with pairs in chosen.
            # In that case, the set is already maximal?
            # If no point can be added, it is maximal.
            break
        
        # Add best_c to chosen
        chosen.append(best_c)
        chosen_mask[best_c] = True
        available[best_c] = False
        
        # Update available: remove newly blocked points
        if best_blocked_set is not None:
            available[best_blocked_set] = False
            
    return chosen

def _shrink_set(chosen, deadline):
    """
    Try to remove points from chosen while maintaining maximality.
    """
    chosen = chosen[:]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(chosen)):
            p = chosen[i]
            # Try removing p
            new_chosen = chosen[:i] + chosen[i+1:]
            
            # Compute blocked mask for new_chosen
            blocked = _compute_blocked_mask(new_chosen)
            
            # Check maximality
            if _is_maximal(new_chosen, blocked):
                chosen = new_chosen
                changed = True
                break # Restart loop as indices changed
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 320 # Safety margin
    
    # Run greedy set cover
    best_chosen = _greedy_set_cover(deadline)
    
    # Shrink
    best_chosen = _shrink_set(best_chosen, deadline)
    best_size = len(best_chosen)
    
    # Local Search / Perturbation
    # We can try to swap points or restart with different seeds
    rng = random.Random(42)
    
    # Try a few random restarts if time permits
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        
        # Modify the greedy process?
        # For now, let's try a simple local search on the current solution
        # Try swapping one point in best_chosen with one point not in it
        
        # This is complex to implement correctly with maximality constraint.
        # Let's stick to the greedy + shrink for now, but run greedy multiple times with slight variations?
        # The greedy is deterministic.
        
        # Variation: Randomize the tie-breaking in greedy.
        # We can re-implement greedy with a random factor.
        
        # Let's try a different heuristic: 
        # Instead of max blocking, use max blocking / cost (cost=1).
        # Or just run the same greedy but with a different initial state?
        # No, the state is empty.
        
        # Let's try "Greedy with Random Tie-Breaking"
        # We need a modified greedy function.
        
        pass # Placeholder for additional search if time permits

    # Convert to list of tuples
    return [_idx_point(i) for i in best_chosen]
