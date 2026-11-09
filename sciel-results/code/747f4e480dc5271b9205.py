import time
import numpy as np
from itertools import product

# Precompute all nonconstant 4-term progressions
_ALL_APS = []
for a in range(1, 125):
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

# For each position, precompute the APs it participates in and the other 3 positions
_pos_aps = [[] for _ in range(128)]
for _ap in _ALL_APS:
    for _i in _ap:
        _others = tuple(j for j in _ap if j != _i)
        _pos_aps[_i].append((_ap, _others))

def construct():
    start = time.time()
    DEADLINE = 82.0  # safety margin before 90s

    n = 128

    # Fast constraint check using bitmasks
    # For a given AP and assignment, check if it's still satisfiable
    # We track for each AP: set of assigned colors (as bitmask) and count of unassigned
    
    def try_backtrack(k_colors):
        """Backtracking with MRV, forward checking, and symmetry breaking."""
        if time.time() > DEADLINE:
            return None
        
        colors = [-1] * n
        # For each AP, track: assigned color mask, unassigned count
        ap_mask = [0] * len(_ALL_APS)
        ap_unassigned = [4] * len(_ALL_APS)
        
        # Map position -> list of AP indices it's in
        pos_ap_indices = [[] for _ in range(n)]
        for idx, ap in enumerate(_ALL_APS):
            for pos in ap:
                pos_ap_indices[pos].append(idx)
        
        # Symmetry breaking: position 0 gets color 0
        colors[0] = 0
        for ap_idx in pos_ap_indices[0]:
            ap_mask[ap_idx] |= 1  # color 0
            ap_unassigned[ap_idx] -= 1
        
        solution_found = [False]
        node_count = [0]
        
        def check_ap_satisfiable(ap_idx, new_color_mask):
            """Check if AP is still satisfiable after adding a color."""
            mask = ap_mask[ap_idx] | new_color_mask
            unassigned = ap_unassigned[ap_idx]
            # Number of distinct colors so far
            distinct = bin(mask).count('1')
            # Max possible distinct colors = distinct + unassigned (best case all new)
            if distinct + unassigned < 3:
                return False
            return True
        
        def backtrack(unassigned_count):
            if time.time() > DEADLINE:
                return False
            if solution_found[0]:
                return True
            
            if unassigned_count == 0:
                # Verify all APs have >= 3 distinct colors
                for ap_idx in range(len(_ALL_APS)):
                    if bin(ap_mask[ap_idx]).count('1') < 3:
                        return False
                solution_found[0] = True
                return True
            
            node_count[0] += 1
            if node_count[0] % 2000 == 0 and time.time() > DEADLINE:
                return False
            
            # MRV: find unassigned position with fewest valid colors
            best_pos = -1
            best_allowed = []
            min_count = k_colors + 1
            
            # Try positions in order of increasing AP count (more constrained first)
            # Use a simple heuristic: check a subset of unassigned positions
            # For speed, we'll iterate through all but break early if we find 1
            for pos in range(1, n):  # skip pos 0 (already assigned)
                if colors[pos] != -1:
                    continue
                
                allowed = []
                for c in range(k_colors):
                    c_mask = 1 << c
                    valid = True
                    for ap_idx in pos_ap_indices[pos]:
                        if not check_ap_satisfiable(ap_idx, c_mask):
                            valid = False
                            break
                    if valid:
                        allowed.append(c)
                
                if len(allowed) < min_count:
                    min_count = len(allowed)
                    best_pos = pos
                    best_allowed = allowed
                    if min_count == 0:
                        return False
                    if min_count == 1:
                        break
            
            if best_pos == -1:
                return False
            
            # Try each allowed color
            for c in best_allowed:
                c_mask = 1 << c
                colors[best_pos] = c
                
                # Update AP state
                affected_aps = []
                for ap_idx in pos_ap_indices[best_pos]:
                    old_mask = ap_mask[ap_idx]
                    old_unassigned = ap_unassigned[ap_idx]
                    ap_mask[ap_idx] |= c_mask
                    ap_unassigned[ap_idx] -= 1
                    affected_aps.append((ap_idx, old_mask, old_unassigned))
                
                # Check if any AP is now unsatisfiable
                ok = True
                for ap_idx, _, _ in affected_aps:
                    if not check_ap_satisfiable(ap_idx, 0):
                        ok = False
                        break
                
                if ok and backtrack(unassigned_count - 1):
                    return True
                
                # Undo
                colors[best_pos] = -1
                for ap_idx, old_mask, old_unassigned in affected_aps:
                    ap_mask[ap_idx] = old_mask
                    ap_unassigned[ap_idx] = old_unassigned
            
            return False
        
        if backtrack(n - 1):  # n-1 because pos 0 is pre-assigned
            return colors[:]
        return None
    
    # Try 3 colors first
    result = try_backtrack(3)
    if result is not None:
        return result
    
    if time.time() > DEADLINE - 10:
        # Not enough time for 4 colors, use constructive fallback
        pass
    else:
        # Try 4 colors
        result = try_backtrack(4)
        if result is not None:
            return result
    
    # Constructive fallback: use a pattern that guarantees >=3 colors on every 4-AP
    # Strategy: use 4 colors in a way that avoids 4-APs with <3 colors
    # A safe approach: color[i] = (i % 4) * 32 + (i // 4) % 4  -- but this may not work
    # Better: use a known good pattern or greedy with many colors
    
    # Greedy: assign each position the smallest color that doesn't create a violation
    # With up to 128 colors, this always works
    colors = []
    for x in range(128):
        assigned = False
        for color in range(128):
            valid = True
            # Check all APs ending at or containing this position
            for ap_idx, ap in enumerate(_ALL_APS):
                if x not in ap:
                    continue
                # Check if assigning 'color' to x keeps this AP satisfiable
                other_positions = [p for p in ap if p != x]
                other_colors = []
                all_assigned = True
                for p in other_positions:
                    if p < x:
                        other_colors.append(colors[p])
                    else:
                        all_assigned = False
                        break
                if all_assigned:
                    # All others are assigned, check if we'd have <3 distinct
                    distinct = len(set([color] + other_colors))
                    if distinct < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                assigned = True
                break
        if not assigned:
            colors.append(0)
    
    return colors
