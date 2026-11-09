import time
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # Pre-compute all 4-cycles: for each set of 4 vertices, there are 3 distinct 4-cycles
    # A 4-cycle on vertices {a,b,c,d} (sorted) is (a,b),(b,c),(c,d),(d,a)
    # The 3 cycles on {a,b,c,d} are:
    #   (a,b,c,d): edges ab, bc, cd, da
    #   (a,b,d,c): edges ab, bd, dc, ca  
    #   (a,c,b,d): edges ac, cb, bd, da
    
    vertices = list(range(N))
    
    # Map edge (i,j) with i<j to index 0..119
    edge_index = {}
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_index[(i,j)] = idx
            idx += 1
    
    # Generate all 4-vertex subsets and their 3 cycles
    cycles = []  # list of lists of 4 edge indices
    for quad in itertools.combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = [edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)]]
        # Cycle 2: a-b-d-c-a
        e2 = [edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)]]
        # Cycle 3: a-c-b-d-a
        e3 = [edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)]]
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    
    num_cycles = len(cycles)  # 1820 * 3 = 5460
    
    # Convert to numpy for fast access
    cycles_np = np.array(cycles, dtype=np.int32)  # shape (5460, 4)
    
    def is_valid(colors):
        """Check if no 4-cycle has 4 distinct colors."""
        # For each cycle, check if all 4 edges have distinct colors
        edge_colors = cycles_np  # (5460, 4)
        # Get colors for all edges in all cycles at once
        cycle_colors = np.array([colors[e] for e in cycles_np])  # This is slow; let's use indexing
        # Better: use numpy fancy indexing
        # colors is a list; convert once
        colors_arr = np.array(colors, dtype=np.int32)
        cycle_colors = colors_arr[cycles_np]  # (5460, 4)
        # For each row, check if all 4 values are distinct
        # Sort each row and check for duplicates
        sorted_colors = np.sort(cycle_colors, axis=1)
        # Check if any row has 4 distinct values: sorted_colors[:,0] != sorted_colors[:,1] 
        # and sorted_colors[:,1] != sorted_colors[:,2] and sorted_colors[:,2] != sorted_colors[:,3]
        invalid = (sorted_colors[:,0] != sorted_colors[:,1]) & \
                  (sorted_colors[:,1] != sorted_colors[:,2]) & \
                  (sorted_colors[:,2] != sorted_colors[:,3])
        return not np.any(invalid)
    
    def count_colors(colors):
        return len(set(colors))
    
    # Strategy: local search with simulated annealing or hill climbing
    # Start with a random assignment, then try single-edge recoloring
    
    best_score = -1
    best_colors = None
    
    # Initial: try a structured assignment
    # Use edge index as color mod 120
    init_colors = list(range(NUM_EDGES))
    if is_valid(init_colors):
        best_colors = init_colors[:]
        best_score = count_colors(init_colors)
    
    # Try random initializations
    random.seed(42)
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
        if is_valid(colors):
            score = count_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
    
    # Local search: try to improve by recoloring single edges
    if best_colors is None:
        best_colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
        best_score = count_colors(best_colors)
    
    current = best_colors[:]
    current_score = best_score
    current_colors_used = set(current)
    
    # Build a list of colors for each edge index for quick reference
    # For local search, we want to try changing one edge to a new color
    # and check if validity is maintained and score improves
    
    # Pre-compute which cycles each edge participates in
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(cycles_np):
        for ei in cyc:
            edge_to_cycles[ei].append(ci)
    
    # Convert to numpy arrays for speed
    edge_to_cycles_arr = [np.array(e2c, dtype=np.int32) if e2c else np.array([], dtype=np.int32) for e2c in edge_to_cycles]
    
    def check_local_validity(colors, edge_idx, new_color):
        """Check if changing edge_idx to new_color keeps all cycles valid."""
        # Only need to check cycles involving edge_idx
        for ci in edge_to_cycles_arr[edge_idx]:
            cyc = cycles_np[ci]
            # Get colors of all 4 edges in this cycle
            # One of them is edge_idx, which we're changing
            col_list = []
            for ei in cyc:
                if ei == edge_idx:
                    col_list.append(new_color)
                else:
                    col_list.append(colors[ei])
            # Check if all 4 are distinct
            if len(set(col_list)) == 4:
                return False
        return True
    
    # Hill climbing
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for ei in range(NUM_EDGES):
            if time.time() > deadline - 10:
                break
            # Try changing edge ei to a color not currently used, or a random color
            # First try to add a new color
            current_color = current[ei]
            # Try some random new colors
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                new_color = random.randint(0, 119)
                if new_color == current_color:
                    continue
                # Check validity
                if check_local_validity(current, ei, new_color):
                    # Tentatively apply
                    old_color = current[ei]
                    current[ei] = new_color
                    new_score = count_colors(current)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_colors = current[:]
                    else:
                        current[ei] = old_color
    
    # Try to maximize by filling in unused colors
    # This is a hard problem; the local search should have done most of the work
    
    return best_colors
