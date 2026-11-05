import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def get_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_cycles():
    """
    For each 4-vertex subset, generate the 3 cycles.
    Each cycle is a list of 4 edge indices.
    """
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
    """For each edge index, list of (cycle_idx, which_cycle, position_in_cycle)."""
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
    c0 = colors[edge_indices[0]]
    c1 = colors[edge_indices[1]]
    c2 = colors[edge_indices[2]]
    c3 = colors[edge_indices[3]]
    return not (c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3)

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
    old_color = colors[edge_idx]
    if old_color == new_color:
        return True
    colors[edge_idx] = new_color
    # Check all cycles containing this edge
    for (ci, which, pos) in edge_to_cycles[edge_idx]:
        c1, c2, c3 = cycles[ci]
        if which == 0:
            cycle = c1
        elif which == 1:
            cycle = c2
        else:
            cycle = c3
        if not check_cycle_valid(colors, cycle):
            colors[edge_idx] = old_color
            return False
    colors[edge_idx] = old_color
    return True

def greedy_fill(colors, cycles, edge_to_cycles, deadline):
    """
    Greedily introduce new colors by trying to assign them to edges.
    Returns the set of used colors.
    """
    used_colors = set(colors)
    
    for new_color in range(MAX_COLOR):
        if time.time() > deadline:
            break
        if new_color in used_colors:
            continue
        
        # Try to assign this color to some edge
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
                break
    
    return used_colors

def local_search(colors, cycles, edge_to_cycles, deadline):
    """
    Local search to find more unused colors.
    Tries to recolor edges to introduce new colors.
    Returns the set of used colors.
    """
    used_colors = set(colors)
    
    max_passes = 200
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        unused = [c for c in range(MAX_COLOR) if c not in used_colors]
        if not unused:
            break
        random.shuffle(unused)
        
        for ei in edge_order:
            if time.time() > deadline:
                break
            
            for new_color in unused:
                if try_assign_color(colors, ei, new_color, cycles, edge_to_cycles):
                    colors[ei] = new_color
                    used_colors.add(new_color)
                    unused.remove(new_color)
                    improved = True
                    break
        
        if not improved:
            break
    
    return used_colors

def branch_and_bound_coloring(cycles, edge_to_cycles, deadline):
    """
    Branch-and-bound on conflict hypergraph.
    
    Model: We want to maximize the number of distinct colors used.
    Each "node" in our search is a (edge, new_color) pair we're considering adding.
    
    Key constraint: If we assign color C to edge e, then for every 4-cycle containing e,
    the other 3 edges cannot all have distinct colors from C and from each other.
    
    Strategy: 
    1. Start with a base coloring (all 0s or simple pattern)
    2. For each unused color, find an edge where it can be safely placed
    3. Use backtracking when simple assignment fails
    4. Bitset acceleration: precompute which edges are in which cycles
    """
    colors = [0] * NUM_EDGES
    used_colors = set(colors)
    
    # Precompute: for each edge and each cycle containing it, 
    # store the other 3 edges in that cycle
    edge_cycle_neighbors = [[] for _ in range(NUM_EDGES)]
    for ei in range(NUM_EDGES):
        for (ci, which, pos) in edge_to_cycles[ei]:
            c1, c2, c3 = cycles[ci]
            if which == 0:
                cycle = c1
            elif which == 1:
                cycle = c2
            else:
                cycle = c3
            # Get the other 3 edges
            others = [cycle[p] for p in range(4) if p != pos]
            edge_cycle_neighbors[ei].append((ci, which, pos, others))
    
    # For each edge, precompute a bitmask of which edges share a cycle with it
    # Actually, let's use a different approach: for each edge e and color c,
    # determine which other edges would conflict
    
    # Build adjacency: for each pair of edges, do they share a 4-cycle?
    # If edges e1 and e2 are in the same 4-cycle, they're "adjacent" in our hypergraph
    
    # More efficient: for each edge e, list of (other_edge, cycle_id) pairs
    # where they appear together in a cycle
    edge_adjacency = [[] for _ in range(NUM_EDGES)]
    for ci, (c1, c2, c3) in enumerate(cycles):
        for cycle in (c1, c2, c3):
            for i in range(4):
                for j in range(i+1, 4):
                    e1, e2 = cycle[i], cycle[j]
                    edge_adjacency[e1].append((e2, ci))
                    edge_adjacency[e2].append((e1, ci))
    
    # Remove duplicates
    for i in range(NUM_EDGES):
        edge_adjacency[i] = list(set(edge_adjacency[i]))
    
    # Now the main search
    # We'll try to add colors one by one
    # For each new color, try all edges, with backtracking
    
    def can_place_color(colors, edge_idx, new_color, edge_cycle_neighbors):
        """Check if placing new_color on edge_idx violates any constraint."""
        for (ci, which, pos, others) in edge_cycle_neighbors[edge_idx]:
            # Check the cycle: new_color + colors of other 3 edges
            # They must not all be distinct
            c0 = new_color
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            # Check if all 4 are distinct
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True
    
    def try_add_color_with_backtrack(colors, new_color, edge_order, max_depth=3):
        """
        Try to add new_color to some edge, with limited backtracking.
        Returns True if successful, and updates colors in place.
        """
        # Simple case: try each edge
        for ei in edge_order:
            if colors[ei] == new_color:
                continue
            if can_place_color(colors, ei, new_color, edge_cycle_neighbors):
                colors[ei] = new_color
                return True
        
        # Backtracking: try changing one edge to free up space
        # For each edge, try changing it to a different color, then try new_color
        # This is expensive, so limit depth
        if max_depth <= 0:
            return False
        
        for ei in edge_order:
            if colors[ei] == new_color:
                continue
            old_color = colors[ei]
            # Try some alternative colors for ei
            for alt_color in range(min(10, MAX_COLOR)):
                if alt_color == old_color or alt_color == new_color:
                    continue
                if not can_place_color(colors, ei, alt_color, edge_cycle_neighbors):
                    continue
                colors[ei] = alt_color
                if can_place_color(colors, ei, new_color, edge_cycle_neighbors):
                    # Wait, we just set it to alt_color, now try new_color
                    # Actually we need to check if new_color can go on ei now
                    # But we changed ei to alt_color, so we need to check if 
                    # some OTHER edge can take new_color, or if ei can take new_color
                    # This gets complex. Let's try: can ei take new_color?
                    if can_place_color(colors, ei, new_color, edge_cycle_neighbors):
                        colors[ei] = new_color
                        return True
                    # Or can another edge take new_color?
                    for ej in edge_order:
                        if ej == ei:
                            continue
                        if colors[ej] == new_color:
                            continue
                        if can_place_color(colors, ej, new_color, edge_cycle_neighbors):
                            colors[ej] = new_color
                            return True
                colors[ei] = old_color
        
        return False
    
    # Main loop: try to add each unused color
    for new_color in range(MAX_COLOR):
        if time.time() > deadline:
            break
        if new_color in used_colors:
            continue
        
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        # First try direct placement
        placed = False
        for ei in edge_order:
            if colors[ei] == new_color:
                continue
            if can_place_color(colors, ei, new_color, edge_cycle_neighbors):
                colors[ei] = new_color
                used_colors.add(new_color)
                placed = True
                break
        
        if not placed:
            # Try with backtracking
            if try_add_color_with_backtrack(colors, new_color, edge_order, max_depth=2):
                used_colors.add(new_color)
    
    return colors, used_colors

def multi_restart_search(cycles, edge_to_cycles, deadline):
    """
    Multiple restarts with different initial colorings.
    Returns the best coloring found.
    """
    best_colors = None
    best_count = 0
    
    num_restarts = 8
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initial strategies
        if restart == 0:
            colors = [0] * NUM_EDGES
        elif restart == 1:
            colors = [0] * NUM_EDGES
            for i in range(0, NUM_EDGES, 7):
                colors[i] = 1
        elif restart == 2:
            colors = [random.randint(0, 2) for _ in range(NUM_EDGES)]
        elif restart == 3:
            colors = [0] * NUM_EDGES
            for i in range(0, NUM_EDGES, 4):
                colors[i] = (i // 4) % 3
        elif restart == 4:
            colors = [random.randint(0, 4) for _ in range(NUM_EDGES)]
        elif restart == 5:
            colors = [0] * NUM_EDGES
            for i in range(0, NUM_EDGES, 5):
                colors[i] = random.randint(1, 3)
        elif restart == 6:
            colors = [(i % 2) for i in range(NUM_EDGES)]
        else:
            colors = [(i % 3) for i in range(NUM_EDGES)]
        
        if not is_valid(colors, cycles):
            colors = [0] * NUM_EDGES
        
        # Use branch-and-bound approach
        bnb_colors, used_colors = branch_and_bound_coloring(cycles, edge_to_cycles, deadline)
        
        # Also try local search on top
        local_search(bnb_colors, cycles, edge_to_cycles, deadline)
        
        current_count = len(set(bnb_colors))
        if current_count > best_count:
            best_count = current_count
            best_colors = bnb_colors[:]
    
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
    
    return best_colors

def construct():
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    random.seed(42)
    
    cycles, edge_to_cycles = build_edge_to_cycles()
    
    # Use multi-restart search
    best_colors = multi_restart_search(cycles, edge_to_cycles, deadline)
    
    # Final verification - if invalid, fall back to safe coloring
    if not is_valid(best_colors, cycles):
        best_colors = [0] * NUM_EDGES
    
    return best_colors

# Pre-compute for efficiency
_cycles, _edge_to_cycles = build_edge_to_cycles()
