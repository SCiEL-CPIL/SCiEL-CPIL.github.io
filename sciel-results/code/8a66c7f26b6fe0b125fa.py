import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Precompute all 4-cycles (120 quads * 3 cycles = 360 cycles)
    all_cycles = []
    vertices = list(range(N))
    
    # Edge index helper for precomputation
    def get_idx(i, j):
        if i > j: i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        all_cycles.append((get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a)))
        # Cycle 2: a-b-d-c-a
        all_cycles.append((get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a)))
        # Cycle 3: a-c-b-d-a
        all_cycles.append((get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a)))
        
    num_cycles = len(all_cycles)
    cycles_arr = np.array(all_cycles, dtype=np.int32) # shape (360, 4)

    # 2. Precompute edge-to-cycle incidence for fast validity checks
    # edge_to_cycles[e] = list of (cycle_index, position)
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci in range(num_cycles):
        for pos in range(4):
            e = cycles_arr[ci, pos]
            edge_to_cycles[e].append((ci, pos))
            
    def is_valid_move(edge_idx, new_color, colors):
        """
        Returns True if setting colors[edge_idx] = new_color is valid.
        Optimized for speed: checks only the ~182 cycles containing edge_idx.
        """
        old_color = colors[edge_idx]
        if new_color == old_color:
            return True
            
        for ci, pos in edge_to_cycles[edge_idx]:
            # Get the other 3 edges in this cycle
            if pos == 0:
                e1, e2, e3 = cycles_arr[ci, 1], cycles_arr[ci, 2], cycles_arr[ci, 3]
            elif pos == 1:
                e1, e2, e3 = cycles_arr[ci, 0], cycles_arr[ci, 2], cycles_arr[ci, 3]
            elif pos == 2:
                e1, e2, e3 = cycles_arr[ci, 0], cycles_arr[ci, 1], cycles_arr[ci, 3]
            else:
                e1, e2, e3 = cycles_arr[ci, 0], cycles_arr[ci, 1], cycles_arr[ci, 2]
                
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            
            # Check for 4 distinct: new_color, c1, c2, c3
            if new_color == c1 or new_color == c2 or new_color == c3:
                continue
            if c1 == c2 or c1 == c3 or c2 == c3:
                continue
                
            return False
            
        return True

    def count_colors(colors):
        return len(np.unique(colors))

    def is_fully_valid(colors):
        c0 = colors[cycles_arr[:, 0]]
        c1 = colors[cycles_arr[:, 1]]
        c2 = colors[cycles_arr[:, 2]]
        c3 = colors[cycles_arr[:, 3]]
        distinct_mask = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        return not np.any(distinct_mask)

    # 3. Strategy: Structural Decomposition (2-color + 1-color)
    # We partition edges into G_A (unique colors) and G_B (single color).
    # G_A must be C4-free. G_B can be anything.
    # Total colors = |G_A| + 1.
    # We want to maximize |G_A|.
    
    # Randomized local search to find a large C4-free subgraph.
    # Start with a random subset of edges.
    # Move: Add an edge to G_A if it doesn't create a C4 in G_A.
    #       Remove an edge from G_A if it allows adding more edges?
    #       Or swap edges.
    
    # Better: Simulated Annealing on the subset of edges in G_A.
    # State: binary array in_G_A (1 if in G_A, 0 if in G_B).
    # Objective: Maximize sum(in_G_A) subject to G_A being C4-free.
    # Penalty: For each C4 in G_A, add a large penalty.
    
    # Since we need strict validity, we can use a repair-based approach or strict constraint satisfaction.
    # Given the small size (120 edges), we can use a greedy growth with random restarts.
    
    best_size = 0
    best_in_G_A = np.zeros(NUM_EDGES, dtype=bool)
    
    # Greedy growth:
    # 1. Start with empty G_A.
    # 2. Randomly pick an edge.
    # 3. Check if adding it creates a C4 in G_A.
    #    A C4 is created if there are two other edges in G_A that form a path of length 3 connecting the endpoints of the new edge.
    #    This check is O(N^2) or similar.
    #    Let's precompute for each edge e, the set of pairs of edges (e1, e2) such that e, e1, e2 form a C4? No, that's not right.
    #    e, e1, e2 form a C4 if they are part of a 4-cycle.
    #    Actually, adding e to G_A creates a C4 if there exists a 4-cycle in G_A U {e} that contains e.
    #    This means there are 3 other edges in G_A that complete a cycle with e.
    
    def creates_c4(edge_idx, in_G_A):
        # Check if adding edge_idx to in_G_A creates a C4.
        # Iterate over cycles containing edge_idx.
        # If the other 3 edges are in G_A, then a C4 is created.
        for ci, pos in edge_to_cycles[edge_idx]:
            if pos == 0:
                e1, e2, e3 = cycles_arr[ci, 1], cycles_arr[ci, 2], cycles_arr[ci, 3]
            elif pos == 1:
                e1, e2, e3 = cycles_arr[ci, 0], cycles_arr[ci, 2], cycles_arr[ci, 3]
            elif pos == 2:
                e1, e2, e3 = cycles_arr[ci, 0], cycles_arr[ci, 1], cycles_arr[ci, 3]
            else:
                e1, e2, e3 = cycles_arr[ci, 0], cycles_arr[ci, 1], cycles_arr[ci, 2]
                
            if in_G_A[e1] and in_G_A[e2] and in_G_A[e3]:
                return True
        return False

    rng = random.Random(42)
    
    # Run multiple random restarts of greedy growth
    for restart in range(500):
        if time.time() > deadline - 5.0:
            break
            
        in_G_A = np.zeros(NUM_EDGES, dtype=bool)
        # Random order of edges
        edges = list(range(NUM_EDGES))
        rng.shuffle(edges)
        
        for e in edges:
            if not creates_c4(e, in_G_A):
                in_G_A[e] = True
                
        size = np.sum(in_G_A)
        if size > best_size:
            best_size = size
            best_in_G_A = in_G_A.copy()
            
        # Also try a local improvement phase on the current best?
        # Or just rely on many restarts.
        
    # Now, construct the final coloring.
    # G_A edges get unique colors 0..best_size-1.
    # G_B edges get color best_size.
    
    final_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    color_idx = 0
    for e in range(NUM_EDGES):
        if best_in_G_A[e]:
            final_colors[e] = color_idx
            color_idx += 1
        else:
            final_colors[e] = best_size
            
    # Verify
    if not is_fully_valid(final_colors):
        # Fallback to all zeros
        return np.zeros(NUM_EDGES, dtype=np.int32).tolist()
        
    return final_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
