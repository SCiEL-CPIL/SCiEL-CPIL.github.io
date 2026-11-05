import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def get_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_edges():
    """Return list of (i, j) for all edges in lex order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def build_cycles():
    """
    For each 4-vertex subset, generate the 3 cycles.
    Each cycle is a list of 4 edge indices.
    """
    from itertools import combinations
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = [get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)]
        # Cycle 2: a-b-d-c-a
        e2 = [get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)]
        # Cycle 3: a-c-b-d-a
        e3 = [get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)]
        cycles.append((e1, e2, e3))
    return cycles

def build_edge_to_cycles():
    """For each edge index, list of (cycle_idx, position_in_cycle)."""
    cycles = build_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (c1, c2, c3) in enumerate(cycles):
        for pos, ei in enumerate(c1):
            edge_to_cycles[ei].append((ci, 0, pos))
        for pos, ei in enumerate(c2):
            edge_to_cycles[ei].append((ci, 1, pos))
        for pos, ei in enumerate(c3):
            edge_to_cycles[ei].append((ci, 2, pos))
    return cycles, edge_to_cycles

def check_cycle_valid(colors, edge_indices):
    """Check that 4 edges don't all have distinct colors."""
    c = [colors[e] for e in edge_indices]
    return len(set(c)) < 4

def is_valid(colors, cycles):
    """Check all cycles are valid."""
    for (c1, c2, c3) in cycles:
        if not check_cycle_valid(colors, c1):
            return False
        if not check_cycle_valid(colors, c2):
            return False
        if not check_cycle_valid(colors, c3):
            return False
    return True

def count_distinct(colors):
    return len(set(colors))

def try_assign_color(colors, edge_idx, new_color, cycles, edge_to_cycles):
    """
    Try to set colors[edge_idx] = new_color.
    Check if this would violate any cycle.
    Returns True if valid, False otherwise.
    """
    # Check all cycles containing this edge
    for (ci, which, pos) in edge_to_cycles[edge_idx]:
        c1, c2, c3 = cycles[ci]
        if which == 0:
            cycle = c1
        elif which == 1:
            cycle = c2
        else:
            cycle = c3
        # Simulate the change
        old_color = colors[edge_idx]
        colors[edge_idx] = new_color
        if not check_cycle_valid(colors, cycle):
            colors[edge_idx] = old_color
            return False
        colors[edge_idx] = old_color
    return True

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    random.seed(42)
    
    edges = build_edges()
    cycles, edge_to_cycles = build_edge_to_cycles()
    
    # Start with all zeros
    colors = [0] * NUM_EDGES
    used_colors = {0}
    
    # Greedy phase: try to introduce each new color
    for new_color in range(1, MAX_COLOR):
        if time.time() > deadline:
            break
        
        # Try to assign this color to some edge
        assigned = False
        # Try edges in random order
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for ei in edge_order:
            if time.time() > deadline:
                break
            if colors[ei] == new_color:
                continue
            if try_assign_color(colors, ei, new_color, cycles, edge_to_cycles):
                colors[ei] = new_color
                used_colors.add(new_color)
                assigned = True
                break
        
        if not assigned:
            # Try harder: maybe we need to change multiple edges
            # For now, just continue to next color
            continue
    
    # Local search: try to improve by changing colors
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        improved = False
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for ei in edge_order:
            if time.time() > deadline:
                break
            
            current_color = colors[ei]
            # Try to change to an unused color
            unused = [c for c in range(MAX_COLOR) if c not in used_colors]
            random.shuffle(unused)
            
            for new_color in unused:
                if try_assign_color(colors, ei, new_color, cycles, edge_to_cycles):
                    colors[ei] = new_color
                    used_colors.add(new_color)
                    improved = True
                    break
        
        if not improved:
            break
    
    return colors

# Pre-compute for efficiency
_cycles, _edge_to_cycles = build_edge_to_cycles()
