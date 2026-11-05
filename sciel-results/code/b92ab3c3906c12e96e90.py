import random
import time
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

# Precompute edges
edges = []
edge_to_u_v = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edges.append((i, j))
        edge_to_u_v[idx] = (i, j)
        idx += 1

# Precompute 4-cycles
# A 4-cycle is a set of 4 vertices {a,b,c,d} and a specific pairing of edges.
# There are 3 distinct 4-cycles for each set of 4 vertices.
# We store the 4 edge indices for each cycle.
cycles = []
edge_to_cycles = [[] for _ in range(NUM_EDGES)]

# Map vertex pair to edge index
pair_to_edge = {}
for i, (u, v) in enumerate(edges):
    pair_to_edge[(u, v)] = i
    pair_to_edge[(v, u)] = i

for verts in combinations(range(N), 4):
    a, b, c, d = verts
    # 3 cycles:
    # 1: a-b, b-c, c-d, d-a
    e1 = pair_to_edge[(a, b)]
    e2 = pair_to_edge[(b, c)]
    e3 = pair_to_edge[(c, d)]
    e4 = pair_to_edge[(d, a)]
    cyc = (e1, e2, e3, e4)
    cycles.append(cyc)
    for e in cyc:
        edge_to_cycles[e].append(len(cycles) - 1)
        
    # 2: a-c, c-b, b-d, d-a  -> a-c, c-b, b-d, d-a
    # Edges: (a,c), (b,c), (b,d), (a,d)
    e1 = pair_to_edge[(a, c)]
    e2 = pair_to_edge[(b, c)]
    e3 = pair_to_edge[(b, d)]
    e4 = pair_to_edge[(a, d)]
    cyc = (e1, e2, e3, e4)
    cycles.append(cyc)
    for e in cyc:
        edge_to_cycles[e].append(len(cycles) - 1)
        
    # 3: a-b, b-d, d-c, c-a
    # Edges: (a,b), (b,d), (c,d), (a,c)
    e1 = pair_to_edge[(a, b)]
    e2 = pair_to_edge[(b, d)]
    e3 = pair_to_edge[(c, d)]
    e4 = pair_to_edge[(a, c)]
    cyc = (e1, e2, e3, e4)
    cycles.append(cyc)
    for e in cyc:
        edge_to_cycles[e].append(len(cycles) - 1)

def check_change(colors, edge_idx, new_color):
    """Check if changing edge_idx to new_color maintains validity."""
    old_color = colors[edge_idx]
    if old_color == new_color:
        return True
    
    for c_idx in edge_to_cycles[edge_idx]:
        cyc = cycles[c_idx]
        # Get the other 3 edges
        others = [e for e in cyc if e != edge_idx]
        c1 = colors[others[0]]
        c2 = colors[others[1]]
        c3 = colors[others[2]]
        
        # Check if this would create a rainbow 4-cycle
        if len({c1, c2, c3, new_color}) == 4:
            return False
            
    return True

def count_used_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    # Initial simple solution: triangle decomposition
    # Partition into 5 triangles + 1 vertex
    # T1: 0,1,2; T2: 3,4,5; T3: 6,7,8; T4: 9,10,11; T5: 12,13,14; V15: 15
    # Internal edges: unique colors 0-4
    # External edges: color 100
    
    trial_colors = [100] * NUM_EDGES
    tri_groups = [(0,1,2), (3,4,5), (6,7,8), (9,10,11), (12,13,14)]
    for i, (u,v,w) in enumerate(tri_groups):
        trial_colors[pair_to_edge[(u,v)]] = i
        trial_colors[pair_to_edge[(v,w)]] = i
        trial_colors[pair_to_edge[(u,w)]] = i
        
    best_solution = trial_colors[:]
    best_score = count_used_colors(best_solution)
    
    # Local search: try to introduce new colors by changing individual edges
    while time.time() < deadline:
        # Create a working copy
        working = best_solution[:]
        improved = False
        any_change = False
        
        # Try each edge in random order
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            if time.time() > deadline:
                break
                
            current_color = working[e_idx]
            
            # Try to change to a new color
            for new_color in range(MAX_COLORS):
                if new_color == current_color:
                    continue
                    
                if check_change(working, e_idx, new_color):
                    # Check if this introduces a new color
                    old_used = set(working)
                    working[e_idx] = new_color
                    new_used = set(working)
                    
                    if len(new_used) > len(old_used):
                        improved = True
                        any_change = True
                        break
                    else:
                        working[e_idx] = current_color
                        continue
                    
            if improved:
                best_solution = working[:]
                best_score = count_used_colors(best_solution)
                # Reset and continue trying to improve
                working = best_solution[:]
                improved = False
                
        # If no improvement in this pass, try some random perturbations
        if not any_change:
            # Try random single-edge changes that maintain validity
            for _ in range(200):
                if time.time() > deadline:
                    break
                    
                e_idx = random.randint(0, NUM_EDGES - 1)
                orig_color = best_solution[e_idx]
                new_color = random.randint(0, MAX_COLORS - 1)
                
                if new_color == orig_color:
                    continue
                    
                if check_change(best_solution, e_idx, new_color):
                    # Accept if it doesn't reduce color count
                    old_count = count_used_colors(best_solution)
                    best_solution[e_idx] = new_color
                    new_count = count_used_colors(best_solution)
                    
                    if new_count > best_score:
                        best_score = new_count
                    elif new_count < old_count:
                        # Revert
                        best_solution[e_idx] = orig_color
                        
    return best_solution
