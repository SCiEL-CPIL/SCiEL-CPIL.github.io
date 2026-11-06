import time
import random

_DEADLINE = 90.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed() - 5.0

def _generate_progressions(n=128):
    progs = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            progs.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    return progs

_PROGRESSIONS = _generate_progressions(128)
_NUM_PROGS = len(_PROGRESSIONS)

# Map position to list of progression indices
_POS_TO_PROGS = [[] for _ in range(128)]
for idx, prog in enumerate(_PROGRESSIONS):
    for p in prog:
        _POS_TO_PROGS[p].append(idx)

def _count_distinct_colors(colors, prog):
    """Count distinct colors in a progression."""
    # colors is a list, prog is tuple of indices
    c0 = colors[prog[0]]
    c1 = colors[prog[1]]
    c2 = colors[prog[2]]
    c3 = colors[prog[3]]
    
    # Fast check for distinct count
    # If all same: 1
    if c0 == c1 == c2 == c3:
        return 1
    # If 3 same, 1 diff: 2
    # If 2 same, 2 same: 2
    # If 2 same, 1, 1: 3
    # If 1, 1, 1, 1: 4
    
    # Simple set conversion is fast enough for 4 elements
    return len({c0, c1, c2, c3})

def _is_violation(colors, prog):
    return _count_distinct_colors(colors, prog) < 3

def _min_conflicts(k, max_iters=20000, seed=None):
    """
    Try to find a valid coloring with k colors using MinConflicts.
    Returns coloring or None.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
        
    n = 128
    # Initialize random coloring
    colors = [rng.randint(0, k - 1) for _ in range(n)]
    
    # Count violations per position
    # viol_count[i] = number of APs involving i that are currently violated
    viol_count = [0] * n
    
    # Initial violation count
    for i in range(n):
        for p_idx in _POS_TO_PROGS[i]:
            if _is_violation(colors, _PROGRESSIONS[p_idx]):
                viol_count[i] += 1
                
    # If already valid
    if all(v == 0 for v in viol_count):
        return colors

    for it in range(max_iters):
        if _time_left() < 0.5:
            return None
            
        # Find a variable with violations
        # Pick one randomly from those with violations
        violating_vars = [i for i in range(n) if viol_count[i] > 0]
        if not violating_vars:
            return colors
            
        i = rng.choice(violating_vars)
        
        # Try changing color of i to minimize violations in its neighborhood
        old_color = colors[i]
        best_color = old_color
        min_violations = viol_count[i] # Current violations
        
        # Evaluate each possible new color
        for c_new in range(k):
            if c_new == old_color:
                continue
                
            # Temporarily assign c_new
            # We need to check which APs are affected
            # Only APs containing i
            new_violations = 0
            
            # We can optimize by checking only the APs containing i
            # But we need to know if they become valid or invalid
            # Let's just re-evaluate the APs containing i
            
            # Store old color to restore if we don't pick this
            # Actually, we just calculate the delta
            
            # For each AP containing i:
            # 1. Was it violated before?
            # 2. Is it violated now (with c_new)?
            
            # This requires checking the other 3 positions.
            
            # Optimization: Precompute the "state" of neighbors?
            # For simplicity, just iterate.
            
            temp_colors = colors # We will not modify, just simulate
            
            # We need a helper to check violation with a specific color at i
            # But _is_violation uses the global colors list.
            # Let's create a local check.
            
            # To avoid modifying colors list, we pass the hypothetical color
            # Or we can just modify, check, and restore if not best.
            # Since k is small (3 or 4), this is fast.
            
            colors[i] = c_new
            v = 0
            for p_idx in _POS_TO_PROGS[i]:
                if _is_violation(colors, _PROGRESSIONS[p_idx]):
                    v += 1
            colors[i] = old_color # Restore
            
            if v < min_violations:
                min_violations = v
                best_color = c_new
                
        # Apply best color
        if best_color != old_color:
            colors[i] = best_color
            # Update viol_count for i and its neighbors
            # Actually, updating viol_count for all neighbors is O(degree)
            # We can just recalculate viol_count for i and its neighbors?
            # Or simpler: recalculate viol_count for all positions? No, too slow.
            
            # Let's update viol_count for i and all j in APs with i.
            
            # First, clear the old contributions?
            # It's easier to just recalculate viol_count for the affected set.
            # Affected positions: i and all positions in APs containing i.
            
            affected = set([i])
            for p_idx in _POS_TO_PROGS[i]:
                for p in _PROGRESSIONS[p_idx]:
                    affected.add(p)
                    
            # Recalculate viol_count for affected positions
            # But we only need to update the counts.
            # Let's just recalculate viol_count for the whole board?
            # N=128, Progs~2600. Recalculating all is 2600 * 4 checks = 10k ops.
            # 20000 iters * 10k = 200M ops. Might be too slow in Python.
            
            # Better: Incremental update.
            # 1. For each AP containing i:
            #    - Determine if it was violated BEFORE change.
            #    - Determine if it is violated AFTER change.
            #    - Update viol_count for all 4 positions in the AP based on change.
            
            # Let's do incremental.
            
            # We need to know the state BEFORE change.
            # We have viol_count from previous step.
            # But viol_count[i] is the sum of violations of APs containing i.
            # When we change i, the violation status of APs containing i might change.
            
            # Let's store the violation status of each AP?
            # ap_violated[p_idx] = True/False
            
            # Let's restructure MinConflicts to track ap_violated.
            
            pass # This logic is getting complex to implement inline without a helper class.
            
    return None

# To make it robust and fast, I will implement a simpler but effective MinConflicts
# that recalculates violation counts for the whole board every few steps or uses a more efficient update.
# Given N=128, full recalculation is ~10k operations.
# 90 seconds / 10k ops = 9000 steps.
# This is low.
# We need incremental updates.

def _min_conflicts_efficient(k, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
        
    n = 128
    colors = [rng.randint(0, k - 1) for _ in range(n)]
    
    # ap_violated: list of bools, length _NUM_PROGS
    ap_violated = [False] * _NUM_PROGS
    # pos_viol_count: list of ints, length n
    pos_viol_count = [0] * n
    
    # Initialize
    for p_idx in range(_NUM_PROGS):
        if _is_violation(colors, _PROGRESSIONS[p_idx]):
            ap_violated[p_idx] = True
            for p in _PROGRESSIONS[p_idx]:
                pos_viol_count[p] += 1
                
    if all(v == 0 for v in pos_viol_count):
        return colors

    # Precompute for each position, the list of APs and the other 3 positions
    # pos_other_pos[p_idx][pos_idx_in_ap] = [other 3 positions]
    # Actually, we just need to iterate APs containing i.
    
    max_iters = 50000
    for it in range(max_iters):
        if _time_left() < 0.5:
            return None
            
        # Find a violated position
        # To avoid bias, we can pick a random violated position
        # Or the one with max violations?
        # Let's pick a random one from those with violations > 0
        
        # Optimization: maintain a list of violated positions?
        # Or just scan. N=128 is small.
        violated_positions = [i for i in range(n) if pos_viol_count[i] > 0]
        if not violated_positions:
            return colors
            
        i = rng.choice(violated_positions)
        
        # Try changing color of i
        old_color = colors[i]
        best_color = old_color
        best_cost = pos_viol_count[i]
        
        # Try all other colors
        for c_new in range(k):
            if c_new == old_color:
                continue
                
            # Calculate new violation count for i if we change to c_new
            # We need to check all APs containing i
            
            new_count = 0
            for p_idx in _POS_TO_PROGS[i]:
                prog = _PROGRESSIONS[p_idx]
                # Check if this AP would be violated with c_new at i
                # We can do this without modifying global colors
                # Get other colors
                others = [colors[p] for p in prog if p != i]
                # Check distinct count of {c_new} U others
                # If distinct count < 3, it's a violation
                # Fast check:
                # If c_new is in others, and others has <= 1 distinct color?
                # Let's just use the set logic, it's fast for 4 items
                if len({c_new} | set(others)) < 3:
                    new_count += 1
                    
            if new_count < best_cost:
                best_cost = new_count
                best_color = c_new
                
        if best_color != old_color:
            # Apply change
            colors[i] = best_color
            
            # Update ap_violated and pos_viol_count
            # Only APs containing i are affected
            
            for p_idx in _POS_TO_PROGS[i]:
                prog = _PROGRESSIONS[p_idx]
                
                # Determine new violation status
                # We already know the status for the APs we checked?
                # No, we only calculated the count for i.
                # We need to check if the AP is violated.
                # An AP is violated if distinct colors < 3.
                
                # Let's just re-evaluate the AP
                is_viol = _is_violation(colors, prog)
                
                # If status changed, update counts
                if ap_violated[p_idx] != is_viol:
                    # Update pos_viol_count for all 4 positions
                    delta = 1 if is_viol else -1
                    for p in prog:
                        pos_viol_count[p] += delta
                    ap_violated[p_idx] = is_viol

    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Try K=3 first
    # Use multiple restarts
    for seed in range(100):
        if _time_left() < 1.0:
            break
        colors = _min_conflicts_efficient(3, seed=seed)
        if colors is not None:
            return colors
            
    # If K=3 fails, try K=4
    for seed in range(100):
        if _time_left() < 1.0:
            break
        colors = _min_conflicts_efficient(4, seed=seed)
        if colors is not None:
            return colors
            
    # Fallback: Greedy with high K
    # This ensures we return *something* valid
    # Greedy with K=10 usually works
    def _greedy(k, seed=0):
        rng = random.Random(seed)
        n = 128
        colors = [-1] * n
        for i in range(n):
            avail = []
            for c in range(k):
                # Check validity
                valid = True
                for p_idx in _POS_TO_PROGS[i]:
                    prog = _PROGRESSIONS[p_idx]
                    others = [colors[p] for p in prog if p != i]
                    if -1 in others:
                        continue
                    if len({c} | set(others)) < 3:
                        valid = False
                        break
                if valid:
                    avail.append(c)
            if not avail:
                return None
            colors[i] = rng.choice(avail)
        return colors
        
    for k in range(5, 15):
        colors = _greedy(k, seed=0)
        if colors:
            return colors
            
    # Absolute fallback
    return list(range(128))

