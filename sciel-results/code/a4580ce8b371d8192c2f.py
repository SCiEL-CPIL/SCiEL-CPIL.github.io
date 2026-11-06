"""Improved program using NumPy vectorized validation + simulated annealing."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles
def precompute_cycles():
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        e1 = (edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a))
        e2 = (edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a))
        e3 = (edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a))
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    return cycles

ALL_CYCLES = precompute_cycles()
NUM_CYCLES = len(ALL_CYCLES)

# Store cycles as numpy array for vectorized operations
CYCLES_ARRAY = np.array(ALL_CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)

# For each edge, which cycles contain it (as numpy array)
EDGE_TO_CYCLE_IDX = {}
for e in range(NUM_EDGES):
    mask = (CYCLES_ARRAY == e).any(axis=1)
    EDGE_TO_CYCLE_IDX[e] = np.where(mask)[0]

def count_violations(coloring):
    """Count number of 4-cycles with 4 distinct colors using NumPy."""
    c = CYCLES_ARRAY  # (NUM_CYCLES, 4)
    colors = np.array(coloring, dtype=np.int32)[c]  # (NUM_CYCLES, 4)
    # A cycle has 4 distinct colors iff all pairwise differences are nonzero
    # Check: c0!=c1, c0!=c2, c0!=c3, c1!=c2, c1!=c3, c2!=c3
    violations = (
        (colors[:, 0] != colors[:, 1]) &
        (colors[:, 0] != colors[:, 2]) &
        (colors[:, 0] != colors[:, 3]) &
        (colors[:, 1] != colors[:, 2]) &
        (colors[:, 1] != colors[:, 3]) &
        (colors[:, 2] != colors[:, 3])
    )
    return int(violations.sum())

def count_colors(coloring):
    return len(set(coloring))

def check_edge_change_valid(coloring, edge, new_color):
    """Check if changing coloring[edge] to new_color would create any 4-distinct-color cycle."""
    old_color = coloring[edge]
    if old_color == new_color:
        return True
    
    cycle_indices = EDGE_TO_CYCLE_IDX[edge]
    if len(cycle_indices) == 0:
        return True
    
    # Get the cycles containing this edge
    cycles_with_edge = CYCLES_ARRAY[cycle_indices]  # (num_cycles_for_edge, 4)
    
    # For each cycle, find the other 3 edges
    # The edge position in each cycle varies
    other_edges = np.zeros_like(cycles_with_edge)
    for pos in range(4):
        mask = cycles_with_edge[:, pos] == edge
        if mask.any():
            # Replace the edge with 0 (dummy), then extract others
            temp = cycles_with_edge[mask].copy()
            temp[:, pos] = -1  # mark as dummy
            other_edges[mask] = temp
            break
    
    # Get colors of other edges
    other_colors = np.array(coloring, dtype=np.int32)[other_edges]  # (num_cycles, 4)
    
    # For each cycle, check if the 3 other colors are all distinct and new_color is not among them
    # other_colors has one entry that's the dummy (-1 mapped to coloring[-1])
    # We need to handle this properly
    
    # Better approach: for each cycle, get the 3 non-edge positions
    for i, ci in enumerate(cycle_indices):
        e1, e2, e3, e4 = ALL_CYCLES[ci]
        other = [e for e in (e1, e2, e3, e4) if e != edge]
        oc = [coloring[e] for e in other]
        if len(set(oc)) == 3 and new_color not in oc:
            return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    random.seed(12345)
    
    # Strategy: Start with a structured coloring, then use simulated annealing
    # to maximize distinct colors while maintaining validity
    
    # Phase 1: Generate multiple initial structured colorings
    def generate_structured_coloring(seed):
        """Generate a valid coloring using a group structure."""
        rng = random.Random(seed)
        
        # Use a partition of vertices into groups
        # Edges within same group get color 0
        # Edges between groups get colors based on group pair
        num_groups = rng.randint(3, 6)
        
        # Assign vertices to groups
        groups = list(range(N))
        for i in range(N):
            groups[i] = i % num_groups
        
        # Assign colors to group pairs
        group_pair_colors = {}
        color = 1
        for g1 in range(num_groups):
            for g2 in range(g1 + 1, num_groups):
                group_pair_colors[(g1, g2)] = color
                color += 1
        
        coloring = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                g1, g2 = groups[i], groups[j]
                if g1 == g2:
                    coloring[edge_idx(i, j)] = 0
                else:
                    key = (min(g1, g2), max(g1, g2))
                    coloring[edge_idx(i, j)] = group_pair_colors.get(key, 0)
        
        return coloring
    
    # Generate several initial colorings and pick the best valid one
    best_initial = None
    best_initial_score = 0
    
    for seed in range(20):
        if time.time() > deadline - 60:
            break
        coloring = generate_structured_coloring(seed)
        if count_violations(coloring) == 0:
            score = count_colors(coloring)
            if score > best_initial_score:
                best_initial_score = score
                best_initial = coloring[:]
    
    if best_initial is None:
        # Fallback: all zeros
        best_initial = [0] * NUM_EDGES
        best_initial_score = 1
    
    # Phase 2: Simulated annealing to add new colors
    coloring = best_initial[:]
    color_counts = [0] * 120
    for c in coloring:
        color_counts[c] += 1
    current_colors = set(coloring)
    current_score = best_initial_score
    best_coloring = coloring[:]
    best_score = current_score
    
    # Temperature schedule
    initial_temp = 10.0
    final_temp = 0.1
    cooling_rate = 0.999
    
    max_iterations = 500000
    temp = initial_temp
    no_improve = 0
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Choose a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = coloring[e]
        
        # Choose a new color
        # With some probability, try a completely new color
        if random.random() < 0.3 and len(current_colors) < 119:
            # Try to add a new color
            available = [c for c in range(120) if color_counts[c] == 0]
            if available:
                new_color = random.choice(available)
            else:
                new_color = random.randint(0, 119)
        else:
            # Try an existing color or nearby color
            if current_colors:
                new_color = random.choice(list(current_colors))
            else:
                new_color = 0
        
        if new_color == old_color:
            continue
        
        # Check validity
        if not check_edge_change_valid(coloring, e, new_color):
            # Invalid move - reject
            continue
        
        # Apply the move
        coloring[e] = new_color
        
        # Update color counts
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        if color_counts[old_color] == 0:
            current_colors.discard(old_color)
        if color_counts[new_color] == 1:
            current_colors.add(new_color)
        
        new_score = len(current_colors)
        
        # Accept or reject based on simulated annealing
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_score = new_score
            no_improve = 0
        else:
            # Reject: revert
            coloring[e] = old_color
            color_counts[new_color] -= 1
            color_counts[old_color] += 1
            if color_counts[new_color] == 0:
                current_colors.discard(new_color)
            if color_counts[old_color] == 1:
                current_colors.add(old_color)
            current_score = len(current_colors)
            no_improve += 1
        
        if current_score > best_score:
            best_score = current_score
            best_coloring = coloring[:]
            no_improve = 0
        
        # Cool down
        temp *= cooling_rate
        if temp < final_temp:
            temp = final_temp
    
    # Phase 3: Greedy polish - try to add more colors
    coloring = best_coloring[:]
    color_counts = [0] * 120
    for c in coloring:
        color_counts[c] += 1
    current_colors = set(coloring)
    current_score = len(current_colors)
    
    for e in range(NUM_EDGES):
        if time.time() > deadline - 3:
            break
        
        old_color = coloring[e]
        available = [c for c in range(120) if color_counts[c] == 0]
        
        for new_color in available:
            if check_edge_change_valid(coloring, e, new_color):
                coloring[e] = new_color
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                if color_counts[old_color] == 0:
                    current_colors.discard(old_color)
                current_colors.add(new_color)
                current_score += 1
                if current_score > best_score:
                    best_score = current_score
                    best_coloring = coloring[:]
                break
    
    return best_coloring
# EVOLVE-BLOCK-END
