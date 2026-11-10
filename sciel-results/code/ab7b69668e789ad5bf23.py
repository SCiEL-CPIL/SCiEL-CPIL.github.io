import time
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = 120
NUM_K4 = 1820

# Precompute edge indices
edge_to_idx = {}
idx_to_edge = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = idx
        idx_to_edge.append((i, j))
        idx += 1

# Precompute K4s and their edge indices
k4_edges = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                edges = [
                    edge_to_idx[(a, b)],
                    edge_to_idx[(a, c)],
                    edge_to_idx[(a, d)],
                    edge_to_idx[(b, c)],
                    edge_to_idx[(b, d)],
                    edge_to_idx[(c, d)]
                ]
                k4_edges.append(edges)

# Precompute for each edge, which K4s it belongs to
edge_to_k4s = [[] for _ in range(NUM_EDGES)]
for k4_idx, edges in enumerate(k4_edges):
    for e in edges:
        edge_to_k4s[e].append(k4_idx)

def count_distinct_colors(colors, edges):
    """Count distinct colors in a set of edges."""
    c = colors[edges]
    return len(np.unique(c))

def count_bad_k4s(colors):
    """Count total bad K4s."""
    bad_count = 0
    for k4_idx in range(NUM_K4):
        if count_distinct_colors(colors, k4_edges[k4_idx]) < 5:
            bad_count += 1
    return bad_count

def local_search(k, time_limit=20):
    """
    Try to find a valid coloring with k colors using simulated annealing.
    Returns colors if found, else None.
    """
    start_time = time.time()
    
    # Initialize
    colors = np.random.randint(0, k, size=NUM_EDGES)
    
    # Initialize bad status
    is_bad = np.zeros(NUM_K4, dtype=bool)
    bad_count = 0
    for k4_idx in range(NUM_K4):
        if count_distinct_colors(colors, k4_edges[k4_idx]) < 5:
            is_bad[k4_idx] = True
            bad_count += 1
            
    if bad_count == 0:
        return colors

    best_colors = colors.copy()
    best_bad = bad_count
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.9998
    min_T = 0.005
    
    # We will iterate until time limit
    while time.time() - start_time < time_limit:
        # Pick a random edge
        e_idx = np.random.randint(0, NUM_EDGES)
        old_color = colors[e_idx]
        new_color = np.random.randint(0, k)
        
        if new_color == old_color:
            continue
            
        # Calculate delta
        delta_bad = 0
        # Temporarily apply change
        colors[e_idx] = new_color
        
        # Store old states for potential revert
        old_states = []
        for k4_idx in edge_to_k4s[e_idx]:
            was_bad = is_bad[k4_idx]
            old_states.append((k4_idx, was_bad))
            is_now_bad = count_distinct_colors(colors, k4_edges[k4_idx]) < 5
            
            if was_bad and not is_now_bad:
                delta_bad -= 1
            elif not was_bad and is_now_bad:
                delta_bad += 1
            
            # Update state for this K4
            is_bad[k4_idx] = is_now_bad
        
        new_bad = bad_count + delta_bad
        
        # Acceptance
        if delta_bad < 0:
            bad_count = new_bad
            if bad_count < best_bad:
                best_bad = bad_count
                best_colors = colors.copy()
                if bad_count == 0:
                    return best_colors
        else:
            # Probabilistic acceptance
            if T > min_T and np.random.rand() < np.exp(-delta_bad / T):
                bad_count = new_bad
            else:
                # Revert
                colors[e_idx] = old_color
                # Revert is_bad states
                for k4_idx, was_bad in old_states:
                    is_bad[k4_idx] = was_bad
        
        T *= cooling

    if best_bad == 0:
        return best_colors
    return None

def try_color_elimination(colors, k):
    """
    Try to eliminate colors by reassigning edges of a color to other colors
    while maintaining validity.
    """
    colors = colors.copy()
    used_colors = set(colors.tolist())
    
    for target_color in sorted(used_colors):
        # Find all edges with this color
        edges_with_color = np.where(colors == target_color)[0]
        
        if len(edges_with_color) == 0:
            continue
        
        success = True
        for e_idx in edges_with_color:
            # Try to reassign this edge to another color
            found = False
            for new_color in range(k):
                if new_color == target_color:
                    continue
                    
                old_color = colors[e_idx]
                colors[e_idx] = new_color
                
                # Check if all K4s containing this edge are still valid
                valid = True
                for k4_idx in edge_to_k4s[e_idx]:
                    if count_distinct_colors(colors, k4_edges[k4_idx]) < 5:
                        valid = False
                        break
                
                if valid:
                    found = True
                    break
                else:
                    colors[e_idx] = old_color
            
            if not found:
                # Couldn't reassign this edge, so we can't eliminate this color
                success = False
                break
        
        if success:
            # Color eliminated
            pass
    
    # Count actual colors used
    actual_colors = len(set(colors.tolist()))
    return colors, actual_colors

def construct():
    total_start = time.time()
    deadline = 145.0
    
    # Try 13, 14, 15 colors
    best_solution = None
    best_color_count = 999
    
    for k in [13, 14, 15]:
        if time.time() - total_start > deadline:
            break
        # Allocate remaining time, keep 15s buffer
        remaining = deadline - (time.time() - total_start)
        if remaining < 15:
            break
        t_limit = min(remaining - 15, 35)
        
        # Try multiple restarts
        for restart in range(3):
            if time.time() - total_start > deadline:
                break
            t_limit_re = min(deadline - (time.time() - total_start) - 10, t_limit / 3)
            if t_limit_re < 5:
                break
            
            sol = local_search(k, t_limit_re)
            if sol is not None:
                # Try to eliminate colors
                sol, actual_colors = try_color_elimination(sol, k)
                if actual_colors < best_color_count:
                    best_color_count = actual_colors
                    best_solution = sol
                if actual_colors <= k:
                    return list(sol)
    
    if best_solution is not None:
        return list(best_solution)
    
    # Fallback: (i+j) % 17
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
