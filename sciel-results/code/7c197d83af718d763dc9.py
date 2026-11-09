"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128

def _check_validity(colors):
    """Check if a coloring satisfies the constraint: every nonconstant 4-AP uses >=3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            # Four-term progression: a, a+d, a+2d, a+3d
            c0 = colors[a]
            c1 = colors[a + d]
            c2 = colors[a + 2 * d]
            c3 = colors[a + 3 * d]
            if len({c0, c1, c2, c3}) < 3:
                return False
    return True

def _count_bad(coloring, n=N):
    """Count the number of 4-APs that violate the constraint (use < 3 distinct colors)."""
    bad = 0
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            c0 = coloring[a]
            c1 = coloring[a + d]
            c2 = coloring[a + 2 * d]
            c3 = coloring[a + 3 * d]
            if len({c0, c1, c2, c3}) < 3:
                bad += 1
    return bad

def _local_search(coloring, max_iter=50000, deadline=None):
    """Local search: flip colors to reduce the number of bad 4-APs."""
    n = len(coloring)
    coloring = list(coloring)
    
    if deadline is not None:
        remaining = deadline - time.time()
        if remaining <= 0:
            return coloring
    
    # Precompute which 4-APs each position participates in
    # For position i, find all (a, d) such that i is one of a, a+d, a+2d, a+3d
    # We'll just recompute bad count efficiently by tracking per-AP counts
    
    # Better approach: for each flip, compute the change in bad count
    # A flip at position i affects all 4-APs containing i
    # Let's precompute for each position i, the list of (a, d) where i is in {a, a+d, a+2d, a+3d}
    
    # Actually, let's use a simpler but effective approach:
    # For each position, try each color, compute delta in bad count
    # We track bad_count globally
    
    bad_count = _count_bad(coloring)
    
    if bad_count == 0:
        return coloring
    
    # Precompute APs per position
    # pos_aps[i] = list of (a, d, role) where role is 0,1,2,3 (which position in the AP)
    pos_aps = [[] for _ in range(n)]
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            positions = [a, a+d, a+2*d, a+3*d]
            for role, p in enumerate(positions):
                pos_aps[p].append((a, d, role))
    
    # For each AP, track its current colors
    # We'll use a dictionary or just recompute on the fly for affected APs
    
    # For efficiency, let's track for each AP a count of distinct colors
    # APs are identified by (a, d)
    # Let's create a list of all APs
    all_aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            all_aps.append((a, d))
    
    # Map AP index
    ap_index = {}
    for idx, (a, d) in enumerate(all_aps):
        ap_index[(a, d)] = idx
    
    # For each AP, track color counts
    # ap_color_counts[ap_idx][color] = count of that color in the AP
    num_aps = len(all_aps)
    ap_color_counts = [[0]*128 for _ in range(num_aps)]
    for (a, d) in all_aps:
        for j in range(4):
            c = coloring[a + j*d]
            ap_color_counts[ap_index[(a,d)]][c] += 1
    
    # A AP is bad if it has < 3 distinct colors, i.e., it has at most 2 distinct colors
    # distinct count = number of colors with count > 0
    # We can track this: bad if sum(1 for c in range(128) if ap_color_counts[ap_idx][c] > 0) < 3
    
    # For speed, let's track distinct count per AP
    ap_distinct = [0] * num_aps
    for idx in range(num_aps):
        ap_distinct[idx] = sum(1 for c in range(128) if ap_color_counts[idx][c] > 0)
    
    # bad_count = number of APs with ap_distinct < 3
    # Verify
    bc = sum(1 for x in ap_distinct if x < 3)
    assert bc == bad_count, f"Mismatch: {bc} vs {bad_count}"
    
    # Now do local search
    colors_used = set(coloring)
    # We'll try colors from 0 to 127, but focus on those likely to help
    # Actually, let's just try all 128 colors for each position
    
    for iteration in range(max_iter):
        if deadline is not None:
            if time.time() > deadline:
                break
        
        improved = False
        # Try random positions
        for _ in range(20):  # Try 20 random positions per iteration
            if deadline is not None and time.time() > deadline:
                break
            
            i = random.randint(0, n-1)
            old_color = coloring[i]
            
            best_new_color = old_color
            best_delta = 0  # We want to minimize bad count, so delta < 0 is good
            
            # For each possible new color, compute the change in bad count
            # Flipping position i from old_color to new_color affects all APs containing i
            for new_color in range(128):
                if new_color == old_color:
                    continue
                
                # Compute delta in bad count
                delta = 0
                for (a, d, role) in pos_aps[i]:
                    ap_idx = ap_index[(a, d)]
                    # Current distinct count
                    old_distinct = ap_distinct[ap_idx]
                    
                    # Update color counts for this AP
                    ap_color_counts[ap_idx][old_color] -= 1
                    ap_color_counts[ap_idx][new_color] += 1
                    
                    # New distinct count
                    new_distinct = sum(1 for c in range(128) if ap_color_counts[ap_idx][c] > 0)
                    
                    # Change in bad status
                    old_bad = 1 if old_distinct < 3 else 0
                    new_bad = 1 if new_distinct < 3 else 0
                    delta += new_bad - old_bad
                    
                    # Revert
                    ap_color_counts[ap_idx][new_color] -= 1
                    ap_color_counts[ap_idx][old_color] += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_delta < 0 and best_new_color != old_color:
                # Apply the flip
                coloring[i] = best_new_color
                # Update AP data structures
                for (a, d, role) in pos_aps[i]:
                    ap_idx = ap_index[(a, d)]
                    old_distinct = ap_distinct[ap_idx]
                    ap_color_counts[ap_idx][old_color] -= 1
                    ap_color_counts[ap_idx][best_new_color] += 1
                    new_distinct = sum(1 for c in range(128) if ap_color_counts[ap_idx][c] > 0)
                    ap_distinct[ap_idx] = new_distinct
                    bad_count += (1 if new_distinct < 3 else 0) - (1 if old_distinct < 3 else 0)
                improved = True
        
        if not improved:
            break
    
    return coloring

def construct():
    deadline = time.time() + 85  # 85s deadline to be safe
    
    best_coloring = None
    best_num_colors = 128
    best_bad = float('inf')
    
    # Strategy: Try to find a coloring with few colors
    # Start with a greedy approach, then local search
    
    # First, try a simple construction: color by position mod k for various k
    # Then use local search to improve
    
    # Let's try different initial colorings
    attempts = [
        "greedy_3", "greedy_4", "greedy_5", "greedy_6",
        "mod3", "mod4", "mod5", "mod6", "mod7", "mod8",
    ]
    
    for attempt in attempts:
        if time.time() > deadline:
            break
        
        if attempt == "greedy_3":
            # Greedy with 3 colors
            coloring = [-1] * N
            for i in range(N):
                for c in range(3):
                    coloring[i] = c
                    # Check all 4-APs ending at or before i
                    valid = True
                    for a in range(i - 3*3 + 1, i + 1):
                        if a < 0:
                            continue
                        for d in range(1, (i - a) // 3 + 1):
                            if a + 3*d > i:
                                continue
                            if a + 3*d >= N:
                                continue
                            # Check if all four positions are assigned
                            if any(coloring[a + j*d] == -1 for j in range(4)):
                                continue
                            if len({coloring[a], coloring[a+d], coloring[a+2*d], coloring[a+3*d]}) < 3:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        break
                if coloring[i] == -1:
                    coloring[i] = 0  # Fallback
            # Local search
            result = _local_search(coloring, max_iter=10000, deadline=deadline)
            num_colors = len(set(result))
            bad = _count_bad(result)
            if num_colors < best_num_colors or (num_colors == best_num_colors and bad < best_bad):
                best_num_colors = num_colors
                best_bad = bad
                best_coloring = result
        
        elif attempt.startswith("mod"):
            k = int(attempt[3:])
            coloring = [i % k for i in range(N)]
            # Local search
            result = _local_search(coloring, max_iter=10000, deadline=deadline)
            num_colors = len(set(result))
            bad = _count_bad(result)
            if num_colors < best_num_colors or (num_colors == best_num_colors and bad < best_bad):
                best_num_colors = num_colors
                best_bad = bad
                best_coloring = result
        
        # Also try random initializations
        if attempt == "mod8":
            for trial in range(5):
                if time.time() > deadline:
                    break
                coloring = [random.randint(0, 5) for _ in range(N)]
                result = _local_search(coloring, max_iter=5000, deadline=deadline)
                num_colors = len(set(result))
                bad = _count_bad(result)
                if num_colors < best_num_colors or (num_colors == best_num_colors and bad < best_bad):
                    best_num_colors = num_colors
                    best_bad = bad
                    best_coloring = result
    
    if best_coloring is None:
        # Fallback: return the seed program's result
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color)
                    break
        return colors
    
    # Verify the best coloring
    if _check_validity(best_coloring):
        return best_coloring
    
    # If not valid, try to fix it or return a valid one
    # Let's do a final local search with more iterations
    result = _local_search(best_coloring, max_iter=20000, deadline=deadline)
    if _check_validity(result):
        return result
    
    # Fallback to seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors
# EVOLVE-BLOCK-END
