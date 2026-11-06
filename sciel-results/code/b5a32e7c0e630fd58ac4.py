import time
import random
from collections import defaultdict

def _build_ap_graph(n):
    """Build adjacency: for each position, which 4-APs it participates in."""
    ap_to_positions = {}
    pos_to_aps = defaultdict(list)
    count = 0
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n:
                break
            # All four positions are distinct for d > 0
            ap_key = (a, b, c, e)
            ap_to_positions[ap_key] = (a, b, c, e)
            for p in (a, b, c, e):
                pos_to_aps[p].append(count)
            count += 1
    return ap_to_positions, pos_to_aps, count

def _validate(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e >= n:
                break
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _backtrack_search(n, num_colors, ap_to_positions, pos_to_aps, max_aps, deadline):
    """
    Backtracking search with forward checking.
    Orders positions by degree (number of 4-APs they participate in).
    """
    # Order positions by degree (most constrained first)
    degrees = [len(pos_to_aps[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    colors = [-1] * n
    # Track for each AP: how many distinct colors are used so far
    ap_colors = [set() for _ in range(max_aps)]
    
    # Pre-compute: for each position, which AP indices it belongs to
    pos_ap_indices = {i: [] for i in range(n)}
    for idx, (a, b, c, e) in enumerate(ap_to_positions):
        for p in (a, b, c, e):
            pos_ap_indices[p].append(idx)
    
    def backtrack(idx_in_order):
        if time.time() > deadline:
            return False
        if idx_in_order == n:
            return True
        
        pos = order[idx_in_order]
        
        # Try each color
        for color in range(num_colors):
            # Forward checking: verify this assignment doesn't violate any AP
            valid = True
            for ap_idx in pos_ap_indices[pos]:
                a, b, c, e = ap_to_positions[ap_idx]
                # Check if this AP would have < 3 colors
                # Build the set of colors in this AP with current assignment
                ap_set = set()
                for p in (a, b, c, e):
                    if p == pos:
                        ap_set.add(color)
                    else:
                        if colors[p] != -1:
                            ap_set.add(colors[p])
                    # If AP already has all 4 positions assigned and < 3 colors, invalid
                # Check if any other position in this AP is unassigned
                all_assigned = True
                for p in (a, b, c, e):
                    if p != pos and colors[p] == -1:
                        all_assigned = False
                        break
                
                if all_assigned:
                    if len(ap_set) < 3:
                        valid = False
                        break
                else:
                    # Even if not all assigned, check if current partial set
                    # could potentially reach 3 colors
                    # If we already have all positions assigned except pos,
                    # and the other 3 positions use only 1 or 2 colors,
                    # we need to be able to add enough new colors
                    # Actually, just check: if the 3 other assigned positions
                    # use at most 1 distinct color, then even adding our color
                    # gives at most 2 colors -> invalid
                    other_colors = set()
                    for p in (a, b, c, e):
                        if p != pos and colors[p] != -1:
                            other_colors.add(colors[p])
                    # If other_colors has at most 1 distinct color, and pos is the
                    # last unassigned, then max distinct = len(other_colors) + 1
                    # which is at most 2 -> invalid
                    if len(other_colors) <= 1:
                        # Count how many positions are unassigned in this AP
                        unassigned_count = 0
                        for p in (a, b, c, e):
                            if p != pos and colors[p] == -1:
                                unassigned_count += 1
                        if unassigned_count == 0:
                            # All others assigned, only pos unassigned
                            if len(other_colors | {color}) < 3:
                                valid = False
                                break
            
            if not valid:
                continue
            
            # Assign
            colors[pos] = color
            for ap_idx in pos_ap_indices[pos]:
                ap_colors[ap_idx].add(color)
            
            if backtrack(idx_in_order + 1):
                return True
            
            # Undo
            colors[pos] = -1
            for ap_idx in pos_ap_indices[pos]:
                ap_colors[ap_idx].discard(color)
        
        return False
    
    return backtrack(0)

def _greedy_with_backtrack(n, num_colors, deadline):
    """
    Try to find a valid coloring with num_colors using backtracking.
    Returns the coloring or None.
    """
    ap_to_positions, pos_to_aps, max_aps = _build_ap_graph(n)
    
    colors = [-1] * n
    order = sorted(range(n), key=lambda i: -len(pos_to_aps[i]))
    
    pos_ap_indices = {i: [] for i in range(n)}
    for idx, (a, b, c, e) in enumerate(ap_to_positions):
        for p in (a, b, c, e):
            pos_ap_indices[p].append(idx)
    
    def backtrack(idx_in_order):
        if time.time() > deadline:
            return False
        if idx_in_order == n:
            return True
        
        pos = order[idx_in_order]
        
        # Try colors in order
        for color in range(num_colors):
            valid = True
            for ap_idx in pos_ap_indices[pos]:
                a, b, c, e = ap_to_positions[ap_idx]
                # Check if assigning color to pos creates a violation
                # in any AP where all positions are now assigned
                ap_set = set()
                all_assigned = True
                for p in (a, b, c, e):
                    if p == pos:
                        ap_set.add(color)
                    else:
                        if colors[p] != -1:
                            ap_set.add(colors[p])
                        else:
                            all_assigned = False
                
                if all_assigned and len(ap_set) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            colors[pos] = color
            if backtrack(idx_in_order + 1):
                return True
            colors[pos] = -1
        
        return False
    
    if backtrack(0):
        return colors
    return None

def _local_search_reduce(colors, n, deadline):
    """
    Try to reduce the number of colors used by local search.
    """
    current = list(colors)
    current_colors = sorted(set(current))
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        # Try to eliminate one color
        target_color = random.choice(current_colors)
        positions = [i for i in range(n) if current[i] == target_color]
        
        new_current = list(current)
        success = True
        for pos in positions:
            found = False
            for new_color in current_colors:
                if new_color == target_color:
                    continue
                new_current[pos] = new_color
                # Quick check: validate all APs involving pos
                valid = True
                for a in range(max(0, pos - 3*42), min(n, pos + 1)):
                    for d in range(1, 43):
                        b = a + d
                        c = a + 2*d
                        e = a + 3*d
                        if e >= n:
                            break
                        cols = {new_current[a], new_current[b], new_current[c], new_current[e]}
                        if len(cols) < 3:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    found = True
                    break
            if not found:
                success = False
                break
        
        if success:
            current = new_current
            current_colors = sorted(set(current))
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 128
    
    # Try backtracking search with 3 colors first
    best = None
    for num_colors in [3, 4, 5]:
        if time.time() > deadline - 5:
            break
        
        result = _greedy_with_backtrack(n, num_colors, deadline - 5)
        if result is not None:
            if _validate(result):
                best = result
                break
    
    if best is not None:
        # Try to reduce colors further with local search
        if time.time() < deadline - 10:
            improved = _local_search_reduce(best, n, deadline - 5)
            if _validate(improved) and _count_colors(improved) <= _count_colors(best):
                best = improved
    
    if best is not None:
        return best
    
    # Fallback: simple greedy with more colors
    colors = [0] * n
    for i in range(n):
        for color in range(128):
            colors[i] = color
            valid = True
            for d in range(1, (i) // 3 + 1):
                a = i - 3*d
                b = i - 2*d
                c = i - d
                if a < 0:
                    break
                cols = {colors[a], colors[b], colors[c], colors[i]}
                if len(cols) < 3:
                    valid = False
                    break
            if valid:
                break
    
    return colors
