import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices
edge_to_idx = {}
edges = []
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = len(edges)
        edges.append((i, j))

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return edge_to_idx[(i, j)]

# Precompute all 4-cycles as numpy array of shape (num_cycles, 4)
# For each 4-vertex set {a,b,c,d}, the 3 cycles are:
# (a,b,c,d): ab, bc, cd, da
# (a,b,d,c): ab, bd, dc, ca  
# (a,c,b,d): ac, cb, bd, da
all_cycles_list = []
for a, b, c, d in combinations(range(N), 4):
    # Cycle 1: a-b-c-d-a
    e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
    all_cycles_list.append(e1)
    # Cycle 2: a-b-d-c-a
    e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
    all_cycles_list.append(e2)
    # Cycle 3: a-c-b-d-a
    e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
    all_cycles_list.append(e3)

NUM_CYCLES = len(all_cycles_list)
cycles_arr = np.array(all_cycles_list, dtype=np.int32)  # shape (NUM_CYCLES, 4)

# For each edge, which cycles contain it (for fast constraint checking)
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for idx, cycle in enumerate(all_cycles_list):
    for e in cycle:
        edge_to_cycles[e].append(idx)

# Precompute: for each edge, the cycles it's in and the positions within those cycles
# This allows fast constraint checking when changing one edge's color
edge_cycle_positions = {}
for idx, cycle in enumerate(all_cycles_list):
    for pos, e in enumerate(cycle):
        if e not in edge_cycle_positions:
            edge_cycle_positions[e] = []
        edge_cycle_positions[e].append((idx, pos))

def count_colors(colors):
    return len(set(colors))

def is_valid(colors):
    """Check if no 4-cycle has 4 distinct colors using numpy."""
    c = np.array(colors, dtype=np.int32)
    # For each cycle, get the 4 colors
    cycle_colors = c[cycles_arr]  # shape (NUM_CYCLES, 4)
    # Check if any cycle has all 4 distinct colors
    # A cycle has 4 distinct colors if all pairwise differences are nonzero
    # More efficiently: sort and check if all unique
    for i in range(NUM_CYCLES):
        if len(set(cycle_colors[i].tolist())) == 4:
            return False
    return True

def check_all_valid(colors_arr):
    """Fast numpy-based validation. Returns True if valid."""
    # For each cycle, check if 4 colors are distinct
    # A set of 4 values is all distinct iff sum of pairwise absolute diffs > 0
    # More directly: use the fact that 4 values are distinct iff
    # (c0!=c1) & (c0!=c2) & (c0!=c3) & (c1!=c2) & (c1!=c3) & (c2!=c3)
    cc = colors_arr[cycles_arr]  # (NUM_CYCLES, 4)
    # Check all 6 pairwise inequalities
    valid = True
    for i in range(4):
        for j in range(i+1, 4):
            if np.any(cc[:, i] == cc[:, j]):
                # This pair is equal in some cycle, but that's fine
                # We need to check if ALL 4 are distinct
                pass
    # Better approach: for each cycle, count unique colors
    # Use a vectorized approach
    # For each cycle, the 4 colors are distinct iff:
    # min != max OR (they're equal but not all same)
    # Actually simplest: check if range > 0 AND not all same
    # Let's just do it with numpy unique per row - but that's not vectorized easily
    # Alternative: 4 distinct iff (c0-c1)*(c0-c2)*(c0-c3)*(c1-c2)*(c1-c3)*(c2-c3) != 0
    # But with integers, we can check:
    diffs = (cc[:, 0:1] != cc[:, 1:2]) & (cc[:, 0:1] != cc[:, 2:3]) & \
            (cc[:, 0:1] != cc[:, 3:4]) & (cc[:, 1:2] != cc[:, 2:3]) & \
            (cc[:, 1:2] != cc[:, 3:4]) & (cc[:, 2:3] != cc[:, 3:4])
    # All 6 must be True for a cycle to have 4 distinct colors
    all_distinct = np.all(diffs, axis=1)
    return not np.any(all_distinct)

def fast_check_edge_change(colors_arr, edge, new_color):
    """Check if setting edge to new_color would violate any constraint.
    colors_arr is a numpy array. Returns True if valid."""
    old_color = colors_arr[edge]
    if old_color == new_color:
        return True
    colors_arr[edge] = new_color
    # Only need to check cycles containing this edge
    valid = True
    for cycle_idx, pos in edge_cycle_positions[edge]:
        c0 = colors_arr[cycles_arr[cycle_idx, 0]]
        c1 = colors_arr[cycles_arr[cycle_idx, 1]]
        c2 = colors_arr[cycles_arr[cycle_idx, 2]]
        c3 = colors_arr[cycles_arr[cycle_idx, 3]]
        if len({c0, c1, c2, c3}) == 4:
            valid = False
            break
    colors_arr[edge] = old_color
    return valid

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    # Structured initial coloring:
    # Use a block-based approach to get more colors initially
    # Partition vertices into groups and use different color schemes
    
    # Start with a star-like coloring but more structured
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Strategy: assign colors based on vertex pairs
    # For edges (i,j) where i,j > 0, use color based on (i+j) mod something
    # For edges from 0, use distinct colors
    
    # Initial: star from vertex 0 with 15 distinct colors
    for j in range(1, N):
        idx = edge_idx(0, j)
        colors[idx] = j - 1  # colors 0..14
    
    # For remaining edges, use a pattern
    # Color based on the sum of endpoints mod 120 (but avoid conflicts)
    for idx in range(NUM_EDGES):
        if colors[idx] == 0 and edges[idx][0] != 0:
            i, j = edges[idx]
            # Use a hash-like function
            colors[idx] = (i * 7 + j * 13) % 120
    
    # Validate and fix
    if not check_all_valid(colors):
        # Fall back to simple star
        colors = np.full(NUM_EDGES, 15, dtype=np.int32)
        for j in range(1, N):
            idx = edge_idx(0, j)
            colors[idx] = j - 1
    
    best_count = count_colors(colors.tolist())
    best_colors = colors.copy()
    
    # Greedy: try to assign new distinct colors to edges
    used_colors = set(colors.tolist())
    available_colors = [c for c in range(120) if c not in used_colors]
    
    # Sort edges by number of cycles they participate in (fewer first)
    edge_order = sorted(range(NUM_EDGES), key=lambda e: len(edge_cycle_positions.get(e, [])))
    
    for idx in edge_order:
        if time.time() > DEADLINE - 10:
            break
        # Try to assign a new color to this edge
        for new_color in available_colors:
            if fast_check_edge_change(colors, idx, new_color):
                colors[idx] = new_color
                used_colors.add(new_color)
                available_colors.remove(new_color)
                break
    
    current_count = count_colors(colors.tolist())
    if current_count > best_count:
        best_count = current_count
        best_colors = colors.copy()
    
    # Simulated annealing / local search with numpy-accelerated checks
    if time.time() < DEADLINE - 5:
        random.seed(42)
        np.random.seed(42)
        current = best_colors.copy()
        current_count = best_count
        best = best_colors.copy()
        best_count_sa = best_count
        
        # Temperature for simulated annealing
        T = 1.0
        cooling = 0.9999
        
        for it in range(500000):
            if time.time() > DEADLINE - 2:
                break
            
            # Pick a random edge
            idx = np.random.randint(0, NUM_EDGES)
            old_color = current[idx]
            
            # Try a random new color
            new_color = np.random.randint(0, 120)
            
            if new_color == old_color:
                continue
            
            # Check if valid
            if fast_check_edge_change(current, idx, new_color):
                current[idx] = new_color
                new_count = count_colors(current.tolist())
                
                if new_count > current_count:
                    # Always accept improvement
                    current_count = new_count
                    if new_count > best_count_sa:
                        best_count_sa = new_count
                        best = current.copy()
                else:
                    # Accept with probability exp(-delta/T) for worsening
                    delta = current_count - new_count
                    if random.random() < np.exp(-delta / max(T, 0.001)):
                        current_count = new_count
                    else:
                        current[idx] = old_color
            
            T *= cooling
        
        if best_count_sa > best_count:
            best_count = best_count_sa
            best_colors = best.copy()
    
    # Final polish: try to find more colors with targeted search
    if time.time() < DEADLINE - 3:
        used_colors = set(best_colors.tolist())
        available_colors = [c for c in range(120) if c not in used_colors]
        random.shuffle(available_colors)
        
        # Try each available color on each edge
        for new_color in available_colors:
            if time.time() > DEADLINE - 2:
                break
            for idx in range(NUM_EDGES):
                if best_colors[idx] == new_color:
                    continue
                if fast_check_edge_change(best_colors, idx, new_color):
                    best_colors[idx] = new_color
                    break
    
    return best_colors.tolist()
