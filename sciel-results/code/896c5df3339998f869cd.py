import random
import time
import math

N = 16
EDGES = []
edge_index = {}
idx = 0
for i in range(N):
    for j in range(i+1, N):
        EDGES.append((i, j))
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
        idx += 1

NUM_EDGES = len(EDGES)

# Precompute for each edge, the list of 4-cycles (as tuples of 4 edge indices) that contain it
# Each cycle is represented by its 4 edge indices.
edge_cycles = [[] for _ in range(NUM_EDGES)]

# We need to iterate over all 4-vertex subsets and the 3 cycles in each
# Or iterate over edges and find cycles containing them.
# Iterating over edges is easier for building the adjacency list for cycles.

for e_idx, (u, v) in enumerate(EDGES):
    others = [k for k in range(N) if k != u and k != v]
    for i in range(len(others)):
        for j in range(i+1, len(others)):
            x = others[i]
            y = others[j]
            # Cycle 1: u-x-y-v-u
            # Edges: (u,x), (x,y), (y,v), (v,u)
            e1 = edge_index[(u, x)]
            e2 = edge_index[(x, y)]
            e3 = edge_index[(y, v)]
            e4 = e_idx # (v,u) is (u,v)
            cycle1 = (e1, e2, e3, e4)
            
            # Cycle 2: u-y-x-v-u
            # Edges: (u,y), (y,x), (x,v), (v,u)
            e5 = edge_index[(u, y)]
            e6 = edge_index[(y, x)]
            e7 = edge_index[(x, v)]
            e8 = e_idx
            cycle2 = (e5, e6, e7, e8)
            
            edge_cycles[e_idx].append(cycle1)
            edge_cycles[e_idx].append(cycle2)

def is_rainbow(colors, cycle):
    c1, c2, c3, c4 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
    return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

def construct():
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    colors = [0] * NUM_EDGES
    freq = {0: NUM_EDGES}
    max_c = 0
    color_count = 1
    total_conflicts = 0
    
    best_colors = colors[:]
    best_count = 1
    
    T = 10.0
    cooling = 0.9995
    M = 1000 # Penalty for conflicts
    
    # Initial state has 0 conflicts
    
    while time.time() - start_time < time_limit:
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_c = colors[e_idx]
        
        # Choose new color
        # Bias towards new colors to maximize distinctness
        if random.random() < 0.6:
            new_c = max_c + 1
        else:
            new_c = random.randint(0, max_c)
            
        if new_c == old_c:
            continue
            
        # Calculate delta conflicts
        delta_conflicts = 0
        for cycle in edge_cycles[e_idx]:
            # Check if cycle was rainbow before
            was_rainbow = is_rainbow(colors, cycle)
            # Temporarily set new color to check after
            # Instead of modifying array, we can compute manually
            # But is_rainbow is fast. Let's just swap in memory for speed?
            # No, let's just compute the condition directly for the 4 edges.
            
            e1, e2, e3, e4 = cycle
            # One of these is e_idx.
            # We can just check the 4 colors with e_idx replaced by new_c.
            
            c1 = colors[e1] if e1 != e_idx else new_c
            c2 = colors[e2] if e2 != e_idx else new_c
            c3 = colors[e3] if e3 != e_idx else new_c
            c4 = colors[e4] if e4 != e_idx else new_c
            
            is_rainbow_now = (c1 != c2 and c1 != c3 and c1 != c4 and 
                              c2 != c3 and c2 != c4 and 
                              c3 != c4)
            
            if was_rainbow and not is_rainbow_now:
                delta_conflicts -= 1
            elif not was_rainbow and is_rainbow_now:
                delta_conflicts += 1
        
        # Calculate delta color count
        delta_count = 0
        if freq[old_c] == 1:
            # Old color disappears
            delta_count -= 1
        if new_c not in freq or freq.get(new_c, 0) == 0:
            # New color appears
            delta_count += 1
        
        delta_score = M * delta_conflicts - delta_count
        
        # Acceptance
        if delta_score < 0:
            accept = True
        else:
            # T might be too low, avoid overflow
            if T < 0.01:
                accept = False
            else:
                accept = random.random() < math.exp(-delta_score / T)
                
        if accept:
            # Update state
            colors[e_idx] = new_c
            freq[old_c] -= 1
            if freq[old_c] == 0:
                del freq[old_c]
            if new_c not in freq:
                freq[new_c] = 0
            freq[new_c] += 1
            
            color_count += delta_count
            total_conflicts += delta_conflicts
            
            if new_c > max_c:
                max_c = new_c
                
            if total_conflicts == 0 and color_count > best_count:
                best_count = color_count
                best_colors = colors[:]
                
        T *= cooling
        
    return best_colors
