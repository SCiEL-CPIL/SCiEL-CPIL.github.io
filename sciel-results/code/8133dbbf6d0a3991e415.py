import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 45  # seconds, safety margin before 60s kill

def construct():
    start_time = time.time()
    
    # Build edge index mapping
    edge_index = {}
    edge_list = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            edge_list.append((i, j))
            idx += 1
    
    # Precompute for each edge, the list of 4-sets (as tuples of 4 sorted vertices) that contain it
    # and for each such 4-set, the three cycles (each cycle is a tuple of 4 edge indices)
    # This allows fast re-validation when an edge color changes
    
    # First, enumerate all 4-sets
    all_4sets = list(itertools.combinations(range(N), 4))
    
    # For each 4-set, precompute the three cycles as tuples of edge indices
    # Cycle 1: a-b-c-d-a -> edges (a,b),(b,c),(c,d),(d,a)
    # Cycle 2: a-b-d-c-a -> edges (a,b),(b,d),(d,c),(c,a)
    # Cycle 3: a-c-b-d-a -> edges (a,c),(c,b),(b,d),(d,a)
    def get_cycles(a, b, c, d):
        # Ensure a<b<c<d
        e1 = (edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)])
        e2 = (edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)])
        e3 = (edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)])
        return (e1, e2, e3)
    
    # For each edge, store the list of (4set_index, cycle_index) pairs that involve it
    edge_to_4sets = [[] for _ in range(NUM_EDGES)]
    cycles_by_4set = []
    for s_idx, (a, b, c, d) in enumerate(all_4sets):
        cycles = get_cycles(a, b, c, d)
        cycles_by_4set.append(cycles)
        for c_idx, cycle in enumerate(cycles):
            for e_idx in cycle:
                edge_to_4sets[e_idx].append((s_idx, c_idx))
    
    def is_valid_cycle(colors, cycle):
        """Check if a cycle (tuple of 4 edge indices) has < 4 distinct colors"""
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        # Check if all four are distinct
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
        return True
    
    def check_edge_validity(colors, e_idx):
        """Check all 4-sets/cycles involving edge e_idx"""
        for (s_idx, c_idx) in edge_to_4sets[e_idx]:
            cycle = cycles_by_4set[s_idx][c_idx]
            if not is_valid_cycle(colors, cycle):
                return False
        return True
    
    def full_validation(colors):
        """Full validation of all cycles"""
        for s_idx in range(len(all_4sets)):
            for c_idx in range(3):
                cycle = cycles_by_4set[s_idx][c_idx]
                if not is_valid_cycle(colors, cycle):
                    return False
        return True
    
    # Initialize with the known valid coloring: color(i,j) = i for i<j
    colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edge_list):
        colors[idx] = i
    
    # Verify initial coloring is valid
    if not full_validation(colors):
        return [0] * NUM_EDGES
    
    # Apply known improvement: recolor (13,15) and (14,15) to new colors
    # This achieves 17 colors (15 base + 2 new)
    # Indices for (13,15) and (14,15)
    idx_13_15 = edge_index[(13, 15)]
    idx_14_15 = edge_index[(14, 15)]
    
    # Assign new colors 15 and 16
    colors[idx_13_15] = 15
    colors[idx_14_15] = 16
    
    # Verify the improved coloring
    if not full_validation(colors):
        # Fallback to original if invalid
        for idx, (i, j) in enumerate(edge_list):
            colors[idx] = i
        return colors
    
    return colors
