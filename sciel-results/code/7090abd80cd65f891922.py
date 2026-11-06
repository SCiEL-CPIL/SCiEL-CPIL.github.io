"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Edge index helper: map (i,j) with i<j to index
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)
    
    # Get all 4-cycles: for each 4-vertex subset, there are 3 cycles
    # A 4-cycle on vertices {a,b,c,d} (sorted) has three possible cycles:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d) - these are the three Hamiltonian cycles
    # Actually, the three cycles on 4 vertices are the three ways to pair them into a 4-cycle
    # Cycle 1: a-b-c-d-a (edges ab, bc, cd, da)
    # Cycle 2: a-b-d-c-a (edges ab, bd, dc, ca)
    # Cycle 3: a-c-b-d-a (edges ac, cb, bd, da)
    
    four_cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Three 4-cycles on {a,b,c,d}
                    four_cycles.append([(a,b), (b,c), (c,d), (d,a)])
                    four_cycles.append([(a,b), (b,d), (d,c), (c,a)])
                    four_cycles.append([(a,c), (c,b), (b,d), (d,a)])
    
    num_cycles = len(four_cycles)  # C(16,4) * 3 = 1820 * 3 = 5460
    
    # Initial construction: modular arithmetic
    # c(i,j) = (i*j + a*i + b*j + c) mod m
    # Try different parameters
    
    best_colors = None
    best_num_colors = 0
    
    # Try several modular constructions
    for m in [17, 19, 23, 29, 31]:
        if time.time() > deadline:
            break
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    if time.time() > deadline:
                        break
                    colors = [0] * NUM_EDGES
                    for i in range(N):
                        for j in range(i+1, N):
                            colors[edge_idx(i,j)] = (i*j + a*i + b*j + c) % m
                    
                    # Count distinct colors
                    distinct = len(set(colors))
                    if distinct <= best_num_colors:
                        continue
                    
                    # Check for rainbow 4-cycles
                    violations = 0
                    for cyc in four_cycles:
                        c1 = colors[edge_idx(cyc[0][0], cyc[0][1])]
                        c2 = colors[edge_idx(cyc[1][0], cyc[1][1])]
                        c3 = colors[edge_idx(cyc[2][0], cyc[2][1])]
                        c4 = colors[edge_idx(cyc[3][0], cyc[3][1])]
                        if len(set([c1, c2, c3, c4])) == 4:
                            violations += 1
                    
                    if violations == 0:
                        best_colors = colors[:]
                        best_num_colors = distinct
    
    # If no valid modular construction found, try local search from a base
    if best_colors is None:
        # Start with a simple construction
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[edge_idx(i,j)] = (i + j) % 15
        best_colors = colors
        best_num_colors = len(set(colors))
    
    # Local search: try to increase colors used while maintaining no rainbow cycles
    if time.time() < deadline:
        random.seed(42)
        colors = best_colors[:]
        current_num_colors = best_num_colors
        
        # Precompute which edges are in which cycles for fast violation checking
        # For each edge, store the list of cycles it participates in
        edge_to_cycles = [[] for _ in range(NUM_EDGES)]
        for ci, cyc in enumerate(four_cycles):
            for (u, v) in cyc:
                ei = edge_idx(u, v)
                edge_to_cycles[ei].append(ci)
        
        def count_violations(col):
            v = 0
            for cyc in four_cycles:
                c1 = col[edge_idx(cyc[0][0], cyc[0][1])]
                c2 = col[edge_idx(cyc[1][0], cyc[1][1])]
                c3 = col[edge_idx(cyc[2][0], cyc[2][1])]
                c4 = col[edge_idx(cyc[3][0], cyc[3][1])]
                if len(set([c1, c2, c3, c4])) == 4:
                    v += 1
            return v
        
        # Try to introduce new colors
        max_color_id = max(colors) if colors else 0
        new_color_pool = list(range(max_color_id + 1, 120))
        random.shuffle(new_color_pool)
        
        improvement_found = True
        while improvement_found and time.time() < deadline:
            improvement_found = False
            # Try assigning each edge a new color from the pool
            for new_c in new_color_pool[:20]:
                if time.time() > deadline:
                    break
                for ei in range(NUM_EDGES):
                    old_c = colors[ei]
                    if old_c == new_c:
                        continue
                    # Check if changing this edge creates violations
                    # Only need to check cycles containing this edge
                    creates_violation = False
                    for ci in edge_to_cycles[ei]:
                        cyc = four_cycles[ci]
                        c1 = colors[edge_idx(cyc[0][0], cyc[0][1])]
                        c2 = colors[edge_idx(cyc[1][0], cyc[1][1])]
                        c3 = colors[edge_idx(cyc[2][0], cyc[2][1])]
                        c4 = colors[edge_idx(cyc[3][0], cyc[3][1])]
                        # Replace the color of this edge in the cycle
                        for k, (u, v) in enumerate(cyc):
                            if edge_idx(u, v) == ei:
                                if k == 0: c1 = new_c
                                elif k == 1: c2 = new_c
                                elif k == 2: c3 = new_c
                                else: c4 = new_c
                                break
                        if len(set([c1, c2, c3, c4])) == 4:
                            creates_violation = True
                            break
                    if not creates_violation:
                        # Check if this actually increases distinct colors
                        if new_c not in set(colors):
                            colors[ei] = new_c
                            current_num_colors += 1
                            best_num_colors = current_num_colors
                            best_colors = colors[:]
                            improvement_found = True
                            new_color_pool.remove(new_c)
                            break
    
    return best_colors
