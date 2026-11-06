import time
import random
import numpy as np
from collections import defaultdict, Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    N = 128

    # Pre-compute all 4-term APs
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
                
    num_aps = len(aps)
    aps_np = np.array(aps, dtype=np.int32)

    # Run seed to get initial best
    def run_seed():
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors

    seed_colors = run_seed()
    best_colors = seed_colors[:]
    best_k = len(set(seed_colors))

    # Local search with incremental violation tracking
    def local_search(k, time_limit):
        """
        Attempts to find a valid coloring with at most k colors using local search.
        Returns list of colors or None.
        """
        if time.time() > time_limit:
            return None
        
        # Initialize with random colors
        colors = [random.randint(0, k-1) for _ in range(N)]
        
        # Pre-compute violation counts for each AP
        # violation[ap_idx] = 0 if valid, >0 if invalid
        # We track the number of distinct colors in each AP
        ap_distinct = np.zeros(num_aps, dtype=np.int8)
        
        # Compute initial distinct counts
        for ap_idx in range(num_aps):
            ap = aps_np[ap_idx]
            vals = [colors[i] for i in ap]
            ap_distinct[ap_idx] = len(set(vals))
        
        # Count violations (APs with < 3 distinct colors)
        violations = np.sum(ap_distinct < 3)
        
        if violations == 0:
            return colors[:]
        
        best_local_violations = violations
        best_local_colors = colors[:]
        
        # Local search loop
        while time.time() < time_limit:
            # Pick a random violated AP
            violated_aps = np.where(ap_distinct < 3)[0]
            if len(violated_aps) == 0:
                break
            
            ap_idx = violated_aps[random.randint(0, len(violated_aps)-1)]
            ap = aps_np[ap_idx]
            
            # Pick a random position in the AP to change
            pos_idx = random.randint(0, 3)
            var_idx = ap[pos_idx]
            
            # Try changing the color at var_idx
            old_color = colors[var_idx]
            new_color = random.randint(0, k-1)
            if new_color == old_color:
                continue
            
            # Update violations incrementally
            # Find all APs containing var_idx
            affected_aps = ap_indices[var_idx]
            
            # Save old distinct counts for affected APs
            old_counts = {}
            for a_idx in affected_aps:
                old_counts[a_idx] = ap_distinct[a_idx]
            
            # Update colors
            colors[var_idx] = new_color
            
            # Update distinct counts for affected APs
            for a_idx in affected_aps:
                ap = aps_np[a_idx]
                vals = [colors[i] for i in ap]
                ap_distinct[a_idx] = len(set(vals))
            
            # Check if we improved
            new_violations = np.sum(ap_distinct < 3)
            
            if new_violations < best_local_violations:
                best_local_violations = new_violations
                best_local_colors = colors[:]
                
            # Accept if improved or with some probability (simulated annealing-like)
            if new_violations <= old_violations:
                # Accept
                pass
            else:
                # Reject and revert
                colors[var_idx] = old_color
                for a_idx in affected_aps:
                    ap_distinct[a_idx] = old_counts[a_idx]
        
        if best_local_violations == 0:
            return best_local_colors
        return None

    # Main loop: Try to reduce k
    current_k = best_k
    
    # Try to find solution with fewer colors
    for k in range(current_k - 1, 2, -1):
        if time.time() > deadline - 5:
            break
            
        # Allow some time for local search
        time_limit = min(deadline - 2, time.time() + 15.0)
        
        # Try multiple restarts
        found = False
        for attempt in range(3):
            if time.time() > time_limit:
                break
            solution = local_search(k, time_limit)
            if solution is not None:
                best_colors = solution
                best_k = k
                found = True
                break
        
        if not found:
            # If we can't find a solution with k colors, try k+1 again with more time?
            # Or just break and try next
            break

    return best_colors
# EVOLVE-BLOCK-END
