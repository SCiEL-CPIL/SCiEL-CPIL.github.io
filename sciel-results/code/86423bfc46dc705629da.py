import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

# Precompute edges in lexicographic order
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_IDX = {}
for idx, (i, j) in enumerate(EDGES):
    EDGE_IDX[(i, j)] = idx
    EDGE_IDX[(j, i)] = idx

# Precompute all 4-cycles and their edge indices
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, c)]
                e3 = EDGE_IDX[(c, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 2: a-b-d-c-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, d)]
                e3 = EDGE_IDX[(d, c)]
                e4 = EDGE_IDX[(c, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 3: a-c-b-d-a
                e1 = EDGE_IDX[(a, c)]
                e2 = EDGE_IDX[(c, b)]
                e3 = EDGE_IDX[(b, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))

NUM_CYCLES = len(CYCLES)  # 5460

# For each edge, which cycles does it belong to?
# OTHER_EDGES[e] = list of tuples (o1, o2, o3)
OTHER_EDGES = [[] for _ in range(NUM_EDGES)]
for cyc in CYCLES:
    e1, e2, e3, e4 = cyc
    OTHER_EDGES[e1].append((e2, e3, e4))
    OTHER_EDGES[e2].append((e1, e3, e4))
    OTHER_EDGES[e3].append((e1, e2, e4))
    OTHER_EDGES[e4].append((e1, e2, e3))

def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            violations += 1
    return violations

def check_edge_change(colors, edge_idx, new_color):
    """
    Check if changing colors[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if colors[edge_idx] == new_color:
        return True
    
    for o1, o2, o3 in OTHER_EDGES[edge_idx]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        return False
        
    return True

def check_swap(colors, e1, e2):
    """
    Check if swapping colors[e1] and colors[e2] creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    c1_old = colors[e1]
    c2_old = colors[e2]
    
    if c1_old == c2_old:
        return True

    # New colors
    c1_new = c2_old
    c2_new = c1_old

    # Check cycles involving e1
    for o1, o2, o3 in OTHER_EDGES[e1]:
        # Determine new colors for o1, o2, o3
        # If o_i is e2, its new color is c2_new (which is c1_old)
        # Otherwise, its new color is colors[o_i]
        
        nc1 = c1_new
        # Get new colors for neighbors
        if o1 == e2:
            nc_o1 = c2_new
        else:
            nc_o1 = colors[o1]
            
        if o2 == e2:
            nc_o2 = c2_new
        else:
            nc_o2 = colors[o2]
            
        if o3 == e2:
            nc_o3 = c2_new
        else:
            nc_o3 = colors[o3]
            
        # Check if {nc1, nc_o1, nc_o2, nc_o3} are distinct
        if nc1 != nc_o1 and nc1 != nc_o2 and nc1 != nc_o3 and nc_o1 != nc_o2 and nc_o1 != nc_o3 and nc_o2 != nc_o3:
            return False

    # Check cycles involving e2 (and not e1, to avoid redundant work? 
    # Actually, if a cycle has both e1 and e2, it was checked in the first loop.
    # If it has e2 but not e1, we need to check it here.
    # The list OTHER_EDGES[e2] contains all cycles with e2.
    # We can just iterate all of them. The ones with e1 were already checked.
    # Checking them again is safe (just redundant).
    for o1, o2, o3 in OTHER_EDGES[e2]:
        nc1 = c2_new
        if o1 == e1:
            nc_o1 = c1_new
        else:
            nc_o1 = colors[o1]
            
        if o2 == e1:
            nc_o2 = c1_new
        else:
            nc_o2 = colors[o2]
            
        if o3 == e1:
            nc_o3 = c1_new
        else:
            nc_o3 = colors[o3]
            
        if nc1 != nc_o1 and nc1 != nc_o2 and nc1 != nc_o3 and nc_o1 != nc_o2 and nc_o1 != nc_o3 and nc_o2 != nc_o3:
            return False

    return True

def _initial_algebraic_coloring():
    """
    Generate a valid coloring using a linear structure over F_2^4.
    Vertices are assigned 4-bit vectors. Edge color is a function of XOR.
    To ensure no rainbow 4-cycles, we map edge differences to a smaller set of colors.
    Actually, a simpler robust initialization:
    Assign each vertex a random 4-bit value.
    Edge (i,j) color = (v_i XOR v_j) mapped to 0..119.
    Since v_i XOR v_j XOR v_k XOR v_l = 0 for any 4-cycle i-j-k-l-i is NOT true for colors,
    but the vector sum is 0.
    
    Let's use a known valid construction:
    Color edge (i,j) based on the difference (j-i) mod 16? No, that's not enough.
    
    Let's try:
    v_i in {0,1,2,3}^4.
    Edge color = index of (v_i XOR v_j) in a fixed permutation of the 16 possible XOR values?
    This gives at most 16 colors.
    
    Better:
    Use a random assignment of 16 vertices to 16 distinct 4-bit vectors.
    There are 16 possible XOR results for any pair.
    If we map each of the 16 XOR results to a distinct color, we get 16 colors.
    Is it valid?
    Cycle i-j-k-l.
    Edges: (i,j), (j,k), (k,l), (l,i).
    Vectors: a, b, c, d.
    Edge vectors: a^b, b^c, c^d, d^a.
    Sum: (a^b) ^ (b^c) ^ (c^d) ^ (d^a) = 0.
    Do these 4 vectors lie in a 3-dim subspace? Yes, they are linearly dependent.
    However, "linearly dependent" does not mean they are equal.
    They are 4 distinct vectors in a 3-dim subspace?
    If they are distinct, we have 4 distinct colors -> VIOLATION.
    So this simple XOR mapping is NOT guaranteed valid.
    
    We need a mapping where any 4 linearly dependent vectors (summing to 0) must have at least two equal colors.
    This is equivalent to saying: no 4 distinct colors form a "cycle" in the color graph?
    
    Actually, the condition is: for any 4-cycle in the graph, the 4 edge colors must not be distinct.
    
    Let's try a different initialization:
    Random colors, but repair immediately.
    Or, use a greedy construction:
    Start with all 0.
    Try to change edge e to color c if it doesn't create a violation.
    Iterate edges, try colors 1, 2, 3...
    """
    colors = [0] * NUM_EDGES
    # Greedy expansion
    # For each edge, try to assign a new color if possible
    # To be faster, we can just run the local search from all-zeros with a strong bias to new colors.
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    best_colors = [0] * NUM_EDGES
    best_score = 1
    
    rng = random.Random(42)
    
    # State
    colors = [0] * NUM_EDGES
    color_counts = np.zeros(MAX_COLOR, dtype=np.int32)
    color_counts[0] = NUM_EDGES
    current_distinct = 1
    
    # Track best
    best_score = 1
    best_colors = list(colors)
    
    T = 1.0
    T_min = 1e-5
    cooling = 0.9995
    
    steps = 0
    no_improve_steps = 0
    
    # Precompute list of edges
    edge_list = list(range(NUM_EDGES))
    
    while time.time() < deadline:
        steps += 1
        
        # Cooling
        if steps % 5000 == 0:
            T *= cooling
            if T < T_min:
                T = T_min
            
            # If stuck for a long time, reheat and perturb
            if no_improve_steps > 20000 and T < 0.1:
                T = 0.5
                no_improve_steps = 0
                # Perturb: change 1-2 edges to random colors if valid
                for _ in range(2):
                    e = rng.randint(0, NUM_EDGES - 1)
                    new_c = rng.randint(0, MAX_COLOR - 1)
                    if check_edge_change(colors, e, new_c):
                        old_c = colors[e]
                        colors[e] = new_c
                        color_counts[old_c] -= 1
                        color_counts[new_c] += 1
                        if color_counts[old_c] == 0:
                            current_distinct -= 1
                        if color_counts[new_c] == 1:
                            current_distinct += 1

        # Move Type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Recolor
            e = rng.randint(0, NUM_EDGES - 1)
            
            # Bias towards unused colors
            unused_colors = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
            if unused_colors:
                # High chance to pick unused
                if rng.random() < 0.8:
                    c = rng.choice(unused_colors)
                else:
                    # Pick from low frequency
                    low_freq = [c for c in range(MAX_COLOR) if 0 < color_counts[c] <= 2]
                    if low_freq:
                        c = rng.choice(low_freq)
                    else:
                        c = rng.randint(0, MAX_COLOR - 1)
            else:
                c = rng.randint(0, MAX_COLOR - 1)
            
            if c == colors[e]:
                continue
                
            if check_edge_change(colors, e, c):
                old_c = colors[e]
                colors[e] = c
                
                # Update counts
                color_counts[old_c] -= 1
                color_counts[c] += 1
                
                # Update distinct
                if color_counts[old_c] == 0:
                    current_distinct -= 1
                if color_counts[c] == 1:
                    current_distinct += 1
                
                new_score = current_distinct
                delta = new_score - best_score
                
                if new_score > best_score:
                    best_score = new_score
                    best_colors = list(colors)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    
                    # Acceptance criterion
                    if delta < 0:
                        # Reject worsening moves mostly, unless T is high
                        if rng.random() > np.exp(delta / T):
                            # Reject
                            colors[e] = old_c
                            color_counts[c] -= 1
                            color_counts[old_c] += 1
                            if color_counts[c] == 0:
                                current_distinct -= 1
                            if color_counts[old_c] == 1:
                                current_distinct += 1
                    else:
                        # Delta is 0 (swap-like effect in count, but color changed)
                        # Actually delta is change in distinct count.
                        # If delta == 0, we might accept to explore.
                        if rng.random() > 0.5:
                            # Reject to keep best state? No, SA should accept.
                            # But we want to maximize distinct.
                            # If distinct didn't change, it's neutral.
                            pass # Accept
            else:
                no_improve_steps += 1
        else:
            # Swap
            e1 = rng.randint(0, NUM_EDGES - 1)
            e2 = rng.randint(0, NUM_EDGES - 1)
            if e1 == e2:
                continue
                
            if check_swap(colors, e1, e2):
                temp = colors[e1]
                colors[e1] = colors[e2]
                colors[e2] = temp
                # Distinct count unchanged
                no_improve_steps += 1
                # Always accept swap (neutral move)
            else:
                no_improve_steps += 1

    # Final safety check
    if count_violations(best_colors) != 0:
        best_colors = [0] * NUM_EDGES
        best_score = 1

    return list(best_colors)
