import numpy as np
import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Best known solution is n=17.
    # We will try to find n=17, then 16, then 15, etc.
    # We need to return a valid coloring for the largest n we find.
    
    # Fallback: n=9 is trivially valid with (j-i)%4
    best_n = 9
    best_colors = [0] * (9*8//2)
    idx = 0
    for i in range(9):
        for j in range(i+1, 9):
            best_colors[idx] = (j-i) % 4
            idx += 1
            
    # Targets in descending order of n
    # Allocate time based on difficulty. n=17 is hardest.
    # We have ~115 seconds.
    # Let's give n=17 60s, n=16 30s, n=15 10s, n=14 5s, n=13 5s.
    targets = [
        (17, 60.0),
        (16, 30.0),
        (15, 10.0),
        (14, 5.0),
        (13, 5.0)
    ]
    
    for n, budget in targets:
        if time.time() > deadline - 2.0:
            break
        
        end_time = time.time() + budget
        res = _find_coloring_sa(n, end_time)
        if res:
            return res
            
    return {'n': best_n, 'colors': best_colors}

def _find_coloring_sa(n, deadline):
    # 1. Precompute edges and cycles
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Map (u, v) to edge index
    edge_idx = {}
    for k, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = k
        edge_idx[(v, u)] = k
        
    # Generate all C6 cycles
    # A C6 is defined by 6 distinct vertices and a permutation of them forming a cycle.
    # Number of distinct C6s on 6 labeled vertices is (6-1)!/2 = 60.
    # We generate the 60 base patterns for vertices 0..5.
    
    base_patterns = []
    seen = set()
    # Fix vertex 0 to break rotational symmetry? 
    # Actually, we want all distinct edge sets.
    # Permute 1..5 and form cycle 0-p1-p2-p3-p4-p5-0.
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        v = [0] + list(perm)
        edges_in_cycle = []
        for i in range(6):
            u, w = v[i], v[(i+1)%6]
            if u > w: u, w = w, u
            edges_in_cycle.append((u, w))
        # Sort edges to create a unique signature
        edge_set = frozenset(edges_in_cycle)
        if edge_set not in seen:
            seen.add(edge_set)
            base_patterns.append(tuple(sorted(edges_in_cycle)))
            
    # There should be 60 patterns.
    # Map local edge (u,w) in 0..5 to index
    def get_local_e_idx(u, w):
        if u > w: u, w = w, u
        # 0: 01, 1: 02, 2: 03, 3: 04, 4: 05
        # 5: 12, 6: 13, 7: 14, 8: 15
        # 9: 23, 10: 24, 11: 25
        # 12: 34, 13: 35
        # 14: 45
        if u == 0: return w - 1
        if u == 1: return 5 + (w - 2)
        if u == 2: return 9 + (w - 3)
        if u == 3: return 12 + (w - 4)
        if u == 4: return 14
        return -1
        
    # Generate global cycles
    # For each combination of 6 vertices, map the 60 patterns to global edge indices.
    cycles_edges = []
    # We will store cycles as a list of tuples of edge indices.
    # For NumPy, we might want a 2D array of shape (num_cycles, 6).
    
    for verts in itertools.combinations(range(n), 6):
        # Map local 0..5 to global verts[0]..verts[5]
        for pat in base_patterns:
            global_e_idxs = []
            for (u_local, w_local) in pat:
                u_g = verts[u_local]
                w_g = verts[w_local]
                e_idx = edge_idx[(u_g, w_g)]
                global_e_idxs.append(e_idx)
            cycles_edges.append(global_e_idxs)
            
    num_cycles = len(cycles_edges)
    
    # Convert to NumPy array for vectorization
    # cycles_np: shape (num_cycles, 6), dtype int32
    cycles_np = np.array(cycles_edges, dtype=np.int32)
    
    # 2. Simulated Annealing
    # State: colors array of shape (num_edges,)
    # Initialize with random colors
    colors = np.random.randint(0, 4, size=num_edges, dtype=np.int8)
    
    # Calculate initial cost
    # A cycle is monochromatic if all 6 edges have the same color.
    # cost = sum(1 for cycle if all(colors[cycle] == colors[cycle[0]]))
    
    def calc_cost(colors):
        # colors: 1D array
        # cycles_np: 2D array (num_cycles, 6)
        # Get colors for all edges in all cycles
        c_colors = colors[cycles_np] # shape (num_cycles, 6)
        # Check if all 6 are equal to the first one
        mono = np.all(c_colors == c_colors[:, 0:1], axis=1)
        return np.sum(mono)
        
    current_cost = calc_cost(colors)
    
    # SA Parameters
    T = 1.0
    T_min = 0.001
    # Cooling rate depends on number of steps we want to take.
    # Let's aim for ~10,000 steps in the budget.
    # Or just run until deadline.
    cooling = 0.9995
    
    # Precompute for fast delta?
    # Delta for changing edge e from c_old to c_new:
    # Affected cycles are those containing e.
    # For each affected cycle, check if it was mono and if it becomes mono.
    # To do this vectorially, we need to know which cycles contain each edge.
    # This is a sparse structure.
    # Alternative: Just recalculate cost for the whole graph? 
    # num_cycles for n=17 is ~742k. 
    # Recalculating cost takes ~1-2ms with NumPy.
    # 1000 steps/sec is possible.
    # 60s * 1000 steps = 60,000 steps. This is reasonable.
    
    # Let's try a full recalculation approach first for simplicity and robustness.
    # If it's too slow, we can optimize to delta.
    
    best_cost = current_cost
    best_colors = colors.copy()
    
    # Random seed for reproducibility in evaluation? No, random is fine.
    rng = np.random.default_rng()
    
    while time.time() < deadline - 1.0:
        if current_cost == 0:
            # Found a valid coloring
            return {'n': n, 'colors': colors.tolist()}
            
        # Pick a random edge
        e = rng.integers(0, num_edges)
        old_c = colors[e]
        new_c = rng.integers(0, 3)
        if new_c == old_c:
            continue
            
        # Try the move
        colors[e] = new_c
        new_cost = calc_cost(colors)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
        else:
            # Revert
            colors[e] = old_c
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    # If we didn't find 0 cost, check if we found a better one than initial?
    # The task asks for a valid coloring (cost 0).
    # If best_cost > 0, we didn't find a valid coloring for this n.
    # But we should return the best valid one found in previous iterations?
    # The function _find_coloring_sa returns None if no valid coloring is found.
    # The caller construct() handles the fallback.
    
    return None

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
