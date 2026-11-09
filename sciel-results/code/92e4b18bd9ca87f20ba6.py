import numpy as np
import time
import random
from itertools import combinations, permutations

def construct():
    # Target n=17
    n_target = 17
    deadline = time.time() + 110  # 10s safety margin
    
    # Precompute cycles for n_target
    # This takes a bit of time, but is done once.
    # Number of edges
    edges = list(combinations(range(n_target), 2))
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    num_edges = len(edges)
    
    # Generate all 6-cycles
    # A 6-cycle is a set of 6 vertices and a cyclic order.
    # We represent a cycle by the 6 edges in order.
    # To avoid duplicates, we fix the start vertex and direction?
    # Actually, for penalty counting, we just need a set of edges that *form* a cycle.
    # If we count every distinct 6-cycle (as a set of edges), we might count the same geometric cycle multiple times if we are not careful?
    # No, a 6-cycle is defined by its 6 edges.
    # In K_n, a set of 6 edges forms a C6 if and only if the graph of these edges is a single cycle of length 6.
    # It's easier to enumerate vertex tuples.
    # For each combination of 6 vertices, there are 60 distinct cycles.
    # We will generate all 60 edge-tuples for each 6-vertex set.
    
    cycle_edges_list = []
    for verts in combinations(range(n_target), 6):
        # Generate all 60 cycles on these 6 vertices
        # Fix v0 = verts[0]
        # Permute the other 5
        rest = verts[1:]
        for perm in permutations(rest):
            v = (verts[0],) + perm
            # Edges: (v0,v1), (v1,v2), ..., (v5,v0)
            cycle_e = []
            for i in range(6):
                u, w = v[i], v[(i+1)%6]
                if u > w: u, w = w, u
                cycle_e.append(edge_to_idx[(u, w)])
            # We might generate the same cycle in reverse or rotated.
            # To avoid duplicates, we can canonicalize the cycle.
            # A cycle is a set of 6 edges.
            # Let's just use a set of edge indices.
            cycle_set = frozenset(cycle_e)
            # We will add this to a set to remove duplicates
            # But we need the list of edges for the numpy array.
            # We can store the frozenset, then convert to list later.
            # Actually, just adding to a set of frozensets is fine.
            
            cycle_edges_list.append(cycle_set)
            
    # Remove duplicates
    unique_cycles = list(set(cycle_edges_list))
    
    # Convert to numpy array
    # If there are no cycles (n<6), return any coloring
    if len(unique_cycles) == 0:
        return {'n': n_target, 'colors': [0]*num_edges}
        
    cycle_array = np.array([list(c) for c in unique_cycles], dtype=np.int32)
    num_cycles = len(cycle_array)
    
    def get_penalty(colors):
        # colors is a numpy array of shape (num_edges,)
        # cycle_array is (num_cycles, 6)
        # We want to check if all 6 edges in a cycle have the same color.
        # c = colors[cycle_array] -> shape (num_cycles, 6)
        # is_mono = np.all(c == c[:, 0:1], axis=1)
        # return np.sum(is_mono)
        c = colors[cycle_array]
        # Optimization: check if min == max
        # Or just compare to first element
        return np.sum(np.all(c == c[:, 0:1], axis=1))

    # Simulated Annealing
    best_colors = None
    best_penalty = float('inf')
    
    # Initialize
    colors = np.random.randint(0, 4, num_edges)
    current_penalty = get_penalty(colors)
    
    if current_penalty == 0:
        best_colors = colors
        best_penalty = 0
    else:
        best_colors = None
        best_penalty = current_penalty
        
    temperature = 1.0
    cooling = 0.999
    steps = 0
    
    while time.time() < deadline:
        # Pick random edge
        idx = np.random.randint(0, num_edges)
        old_color = colors[idx]
        new_color = np.random.randint(0, 4)
        if new_color == old_color:
            continue
            
        colors[idx] = new_color
        steps += 1
        
        new_penalty = get_penalty(colors)
        
        if new_penalty < best_penalty:
            best_penalty = new_penalty
            best_colors = colors.copy()
            if new_penalty == 0:
                break
        
        # Acceptance criterion
        if new_penalty < current_penalty:
            current_penalty = new_penalty
        else:
            delta = new_penalty - current_penalty
            if random.random() < np.exp(-delta / temperature):
                current_penalty = new_penalty
            else:
                colors[idx] = old_color
                
        temperature *= cooling
        
    # If we found a valid coloring for n_target, return it
    if best_colors is not None and best_penalty == 0:
        return {'n': n_target, 'colors': best_colors.tolist()}
        
    # If not, try smaller n
    # We can just return a valid coloring for a smaller n.
    # For n <= 5, any coloring is valid.
    # For n=6, we can easily find one.
    # Let's just return n=5 to be safe and fast, or try to find a valid one for n=10 quickly?
    # The score is n. So n=5 is better than n=1.
    # But we can likely find n=17.
    # If we fail, let's try n=16 with remaining time?
    # Given the time pressure, returning a guaranteed valid small n is safer than risking a timeout.
    # However, n=17 is the "best known".
    # Let's try n=17 again with a different seed? No, time is up.
    
    # Fallback: Return a valid coloring for n=6.
    # In K_6, we can color edges such that no mono C6.
    # Actually, for n=6, a C6 uses all 6 edges of the cycle.
    # We can just use a random coloring and verify.
    # Or simpler: n=5.
    n_fallback = 5
    edges_f = list(combinations(range(n_fallback), 2))
    colors_f = [0] * len(edges_f)
    return {'n': n_fallback, 'colors': colors_f}
