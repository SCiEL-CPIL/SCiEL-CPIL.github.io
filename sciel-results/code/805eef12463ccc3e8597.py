"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _get_all_progressions(n=128):
    """Precompute all nonconstant 4-term arithmetic progressions."""
    progs = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            e = a + 3 * d
            if e < n:
                progs.append((a, a + d, a + 2 * d, e))
    return progs

def _build_conflict_index(ap_list, n=128):
    """
    Build an index mapping each position to the list of AP indices containing it.
    Also precompute for each AP the indices of its 4 positions.
    """
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(i)
    return pos_to_aps

def _count_violations_fast(colors, ap_list, pos_to_aps, n=128):
    """
    Count violations efficiently. Returns (violation_count, first_violation_ap_idx).
    Uses numpy for vectorized color comparison.
    """
    colors_np = np.array(colors)
    # For each AP, check if < 3 distinct colors
    # We can do this by checking: all same OR exactly 2 distinct
    # Vectorized: for each AP row, compute min, max, and check patterns
    # Actually, let's just iterate but with early exit
    for i, ap in enumerate(ap_list):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        if c0 == c1 == c2 == c3:
            return 1, i
        # Check for exactly 2 distinct colors
        # This happens if the multiset has only 2 unique values
        # Quick check: if not all same, check if any two are same and the other two are same
        # Patterns: AABB, ABAB, ABBA, AABC won't work... 
        # Let's just use set for the first pass, it's fast enough
        if len({c0, c1, c2, c3}) < 3:
            return 1, i
    return 0, -1

def _try_fix_violation(colors, ap_idx, ap_list, pos_to_aps, k, n=128):
    """
    Try to fix a violation by recoloring one of the 4 positions.
    Returns True if fixed, False otherwise.
    """
    ap = ap_list[ap_idx]
    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
    
    # Determine the set of colors in this AP
    color_set = {c0, c1, c2, c3}
    
    # For each position in the AP, try recoloring to a color not in color_set
    # or to a color that breaks the 2-color pattern
    positions = [ap[0], ap[1], ap[2], ap[3]]
    random.shuffle(positions)
    
    for pos in positions:
        old_color = colors[pos]
        # Try colors in order: prefer colors not in the current AP's color set
        # Then try other colors
        candidate_colors = []
        for c in range(k):
            if c != old_color:
                if c not in color_set:
                    candidate_colors.insert(0, c)
                else:
                    candidate_colors.append(c)
        
        for new_color in candidate_colors:
            colors[pos] = new_color
            # Check if this AP is now fixed
            nc0, nc1, nc2, nc3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({nc0, nc1, nc2, nc3}) >= 3:
                # Check if we created new violations elsewhere
                # Quick check: only check APs that contain this position
                new_violations = 0
                for other_ap_idx in pos_to_aps[pos]:
                    if other_ap_idx == ap_idx:
                        continue
                    oap = ap_list[other_ap_idx]
                    oc0, oc1, oc2, oc3 = colors[oap[0]], colors[oap[1]], colors[oap[2]], colors[oap[3]]
                    if len({oc0, oc1, oc2, oc3}) < 3:
                        new_violations += 1
                        break
                if new_violations == 0:
                    return True
                # Revert
                colors[pos] = old_color
            else:
                # Revert
                colors[pos] = old_color
    
    return False

def _local_search_structural(k, deadline, ap_list, pos_to_aps, n=128, max_restarts=50):
    """
    Structural local search using conflict index for fast violation detection and fixing.
    """
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random coloring
        colors = [random.randint(0, k - 1) for _ in range(n)]
        
        # Track best state
        best_colors = None
        best_violations = float('inf')
        
        # Local search loop
        stuck_count = 0
        max_stuck = 1000
        
        while time.time() < deadline:
            # Find first violation
            v_count, v_idx = _count_violations_fast(colors, ap_list, pos_to_aps, n)
            
            if v_count == 0:
                # Valid!
                return colors
            
            # Update best
            if v_count < best_violations:
                best_violations = v_count
                best_colors = colors[:]
                stuck_count = 0
            else:
                stuck_count += 1
            
            if stuck_count > max_stuck:
                # Restart with current best or fresh
                if best_colors is not None and random.random() < 0.5:
                    colors = best_colors[:]
                else:
                    colors = [random.randint(0, k - 1) for _ in range(n)]
                stuck_count = 0
                best_violations = float('inf')
                best_colors = None
                continue
            
            # Try to fix the violation
            if v_idx >= 0:
                if _try_fix_violation(colors, v_idx, ap_list, pos_to_aps, k, n):
                    stuck_count = 0
                else:
                    # Random perturbation
                    for _ in range(random.randint(1, 3)):
                        pos = random.randint(0, n - 1)
                        colors[pos] = random.randint(0, k - 1)
        
        # End of this restart
        if best_colors is not None:
            # Check if best is valid
            v_count, _ = _count_violations_fast(best_colors, ap_list, pos_to_aps, n)
            if v_count == 0:
                return best_colors
    
    return None

def _greedy_construct(k, ap_list, pos_to_aps, n=128):
    """
    Greedy construction: assign colors one position at a time,
    choosing the smallest color that doesn't create a violation
    with already-assigned positions.
    """
    colors = [-1] * n  # -1 means unassigned
    
    for pos in range(n):
        # Find the smallest color that works
        found = False
        for c in range(k):
            colors[pos] = c
            # Check all APs containing this position
            valid = True
            for ap_idx in pos_to_aps[pos]:
                ap = ap_list[ap_idx]
                # Check if all positions are assigned
                if any(colors[p] == -1 for p in ap):
                    continue
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            if valid:
                found = True
                break
            colors[pos] = -1
        
        if not found:
            # No color works; this greedy approach failed
            return None
    
    # Verify full validity
    for ap_idx, ap in enumerate(ap_list):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        if len({c0, c1, c2, c3}) < 3:
            return None
    
    return colors

def _check_valid(colors, ap_list):
    """Check if all nonconstant 4-term progressions use at least 3 distinct colors."""
    for ap in ap_list:
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        if len({c0, c1, c2, c3}) < 3:
            return False
    return True

def _count_colors(colors):
    """Count distinct colors used."""
    return len(set(colors))

def construct():
    start_time = time.time()
    # Keep a strict safety margin to ensure return before 60s limit
    deadline = start_time + 45
    
    n = 128
    ap_list = _get_all_progressions(n)
    pos_to_aps = _build_conflict_index(ap_list, n)
    
    # Try increasing k from 4 upward
    # Use tighter time slices for each k
    for k in [4, 5, 6, 7, 8]:
        if time.time() > deadline - 15:
            break
        
        # First try greedy construction (fast, deterministic)
        if time.time() < deadline - 10:
            result = _greedy_construct(k, ap_list, pos_to_aps, n)
            if result is not None and _check_valid(result, ap_list):
                return result
        
        # Then try structural local search
        if time.time() < deadline - 10:
            # Allocate remaining time minus buffer to this k
            k_deadline = deadline - 10
            result = _local_search_structural(k, k_deadline, ap_list, pos_to_aps, n, max_restarts=50)
            if result is not None and _check_valid(result, ap_list):
                return result
    
    # Fallback: try higher k with less time
    for k in [9, 10, 12]:
        if time.time() > deadline - 5:
            break
        k_deadline = deadline - 5
        result = _local_search_structural(k, k_deadline, ap_list, pos_to_aps, n, max_restarts=20)
        if result is not None and _check_valid(result, ap_list):
            return result
    
    # Emergency: Try a known valid pattern if time is very tight
    # x % 128 is always valid (all distinct)
    # But we can try a smaller valid set if we have a few seconds
    if time.time() < deadline - 1:
        # Try k=128 with a quick random search
        result = _local_search_structural(128, deadline - 1, ap_list, pos_to_aps, n, max_restarts=5)
        if result is not None and _check_valid(result, ap_list):
            return result

    # Absolute fallback: unique colors (always valid, score is high but valid)
    return list(range(n))

# EVOLVE-BLOCK-END
