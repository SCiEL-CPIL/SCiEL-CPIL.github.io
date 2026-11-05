import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Try n from 19 down to 9
    for n in range(19, 8, -1):
        if time.time() > deadline:
            break
            
        try:
            result = solve_n(n, deadline)
            if result is not None:
                return result
        except Exception as e:
            print(f"Error for n={n}: {e}")
            continue
            
    # Fallback to seed n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, deadline):
    """
    Bit-parallel backtracking search that constructs the 4-coloring edge-by-edge,
    maintaining bitmask constraints to instantly reject assignments that would
    create a monochromatic C6.
    """
    # Precompute edge indices in lexicographic order
    edge_idx_map = {}
    edges_list = []
    for i in range(n):
        for j in range(i + 1, n):
            idx = len(edges_list)
            edge_idx_map[(i, j)] = idx
            edges_list.append((i, j))
            
    num_edges = len(edges_list)
    
    # Build adjacency structure
    adj = [[-1] * n for _ in range(n)]
    for (i, j), idx in edge_idx_map.items():
        adj[i][j] = idx
        adj[j][i] = idx
    
    # Precompute all C6 cycles as sets of edge indices
    # For each 6-tuple of vertices, enumerate all distinct 6-cycles
    cycles = []
    cycle_set = set()
    
    for verts in itertools.combinations(range(n), 6):
        v0, v1, v2, v3, v4, v5 = verts
        others = [v1, v2, v3, v4, v5]
        for perm in itertools.permutations(others):
            # Cycle: v0 - perm[0] - perm[1] - perm[2] - perm[3] - perm[4] - v0
            cycle_edges = []
            prev = v0
            for v in perm:
                a, b = min(prev, v), max(prev, v)
                cycle_edges.append(adj[a][b])
                prev = v
            a, b = min(prev, v0), max(prev, v0)
            cycle_edges.append(adj[a][b])
            
            # Canonical representation: sort the edge indices
            canon = tuple(sorted(cycle_edges))
            if canon not in cycle_set:
                cycle_set.add(canon)
                cycles.append(cycle_edges)
                
    num_cycles = len(cycles)
    
    # Build edge_to_cycles mapping: for each edge, list of cycle indices containing it
    edge_to_cycles = defaultdict(list)
    for c_idx, cycle in enumerate(cycles):
        for e_idx in cycle:
            edge_to_cycles[e_idx].append(c_idx)
    
    # For each edge and color, precompute which cycles would become monochromatic
    # if we assign this color to this edge, given that all other 5 edges in each
    # affected cycle already have that same color.
    # This is dynamic, so we'll compute it on-the-fly during backtracking.
    
    # State: colors assigned so far (list of -1 or 0-3)
    colors = [-1] * num_edges
    
    # For each color c and each cycle, track how many edges in the cycle
    # have been assigned color c. When this reaches 6, we have a violation.
    # Also track if any edge in the cycle has a different color (then it can't be mono-c).
    cycle_color_count = [[0] * 4 for _ in range(num_cycles)]
    cycle_has_other = [False] * num_cycles  # True if any edge in cycle has different color
    
    # For backtracking, we need to undo assignments efficiently.
    # We'll use a stack of (edge_idx, color) for undoing.
    
    def is_valid_assignment(e_idx, c):
        """Check if assigning color c to edge e_idx would create a monochromatic C6."""
        for c_idx in edge_to_cycles.get(e_idx, []):
            # If this cycle already has a non-c color, it can't become mono-c
            if cycle_has_other[c_idx]:
                continue
            # Check if all other 5 edges in this cycle have color c
            cycle = cycles[c_idx]
            for other_e in cycle:
                if other_e == e_idx:
                    continue
                if colors[other_e] != c:
                    break
            else:
                # All other edges are color c, so assigning c here creates mono-c
                return False
        return True
    
    def assign(e_idx, c):
        """Assign color c to edge e_idx and update cycle tracking."""
        colors[e_idx] = c
        for c_idx in edge_to_cycles.get(e_idx, []):
            cycle_color_count[c_idx][c] += 1
            # Check if this cycle now has a non-c color (it already does if count < 6)
            # Actually, we need to track if ANY edge has a different color
            # Simplification: if any edge in cycle has color != c, mark has_other
            # But this is per-color, so we need per-color tracking or a different approach
            # Let's just check during is_valid_assignment
    
    def unassign(e_idx, c):
        """Undo assignment of color c to edge e_idx."""
        colors[e_idx] = -1
        for c_idx in edge_to_cycles.get(e_idx, []):
            cycle_color_count[c_idx][c] -= 1
    
    # Better approach: maintain for each cycle a bitmask of colors used
    # cycle_color_mask[c_idx] = bitmask of colors present in cycle
    # If mask has only one bit set and count == 6, violation
    
    cycle_color_mask = [0] * num_cycles
    cycle_edge_colors = [[-1] * 6 for _ in range(num_cycles)]  # color of each edge in cycle
    
    def check_cycle(c_idx):
        """Check if cycle c_idx is monochromatic (all 6 edges same color)."""
        cycle = cycles[c_idx]
        first_color = -1
        for e_idx in cycle:
            c = colors[e_idx]
            if c == -1:
                return False  # Not all assigned yet
            if first_color == -1:
                first_color = c
            elif c != first_color:
                return False
        return True
    
    def would_create_mono_c6(e_idx, c):
        """Check if assigning color c to e_idx would create any monochromatic C6."""
        for c_idx in edge_to_cycles.get(e_idx, []):
            cycle = cycles[c_idx]
            # Check if all other 5 edges in this cycle have color c
            all_other_c = True
            for other_e in cycle:
                if other_e == e_idx:
                    continue
                if colors[other_e] != c:
                    all_other_c = False
                    break
            if all_other_c:
                return True
        return False
    
    # Backtracking search
    # Order edges to maximize pruning: edges in many cycles first
    edge_order = sorted(range(num_edges), key=lambda e: len(edge_to_cycles.get(e, [])), reverse=True)
    
    # Shuffle to break symmetry
    random.shuffle(edge_order)
    
    found = [False]
    
    def backtrack(pos):
        if time.time() > deadline - 2.0:
            return False
        if pos == len(edge_order):
            # All edges assigned, verify no monochromatic C6
            for c_idx in range(num_cycles):
                if check_cycle(c_idx):
                    return False
            return True
            
        e_idx = edge_order[pos]
        
        # Try each color
        for c in range(4):
            if would_create_mono_c6(e_idx, c):
                continue
            assign(e_idx, c)
            if backtrack(pos + 1):
                found[0] = True
                return True
            unassign(e_idx, c)
        
        return False
    
    # Run backtracking with multiple random restarts
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Reset state
        colors = [-1] * num_edges
        random.shuffle(edge_order)
        
        if backtrack(0):
            # Found a valid coloring
            return {'n': n, 'colors': colors}
    
    return None
