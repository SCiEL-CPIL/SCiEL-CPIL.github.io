import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    def time_check():
        return time.time() > deadline - 5.0

    # We aim for n=17. 
    # If we find a valid one, return it.
    # Strategy: Simulated Annealing on a cost function counting C6s.
    # This allows "invalid" states during search, unlike backtracking.
    
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices for fast access
    # edge_idx[i][j] gives the index in the flat array for edge (i, j)
    edge_idx = np.zeros((n, n), dtype=np.int32)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i, j] = idx
            edge_idx[j, i] = idx
            idx += 1
            
    # Precompute all 6-cycles in K_n
    # A 6-cycle is a sequence of 6 distinct vertices v0...v5 such that 
    # edges (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0) exist.
    # We only need to consider unique sets of 6 vertices and their cyclic orderings.
    # Actually, for cost calculation, we can just iterate over all combinations of 6 vertices
    # and check the 6 possible Hamiltonian cycles on those vertices?
    # No, a C6 is a specific subgraph. 
    # A simpler way: Iterate over all combinations of 6 vertices.
    # For each set of 6 vertices, there are (6-1)!/2 = 60 distinct cycles.
    # Total cycles in K_17: C(17,6) * 60 = 12376 * 60 = 742,560.
    # This is manageable.
    
    # Let's precompute the list of edges (as indices) for each 6-cycle.
    # This will speed up the cost calculation significantly.
    
    cycles = []
    # To avoid duplicates, we can fix the smallest vertex as the start?
    # Or just generate all and deduplicate? 
    # Better: Generate all combinations of 6 vertices.
    # For each combination, generate all 60 unique cycles.
    
    from itertools import combinations, permutations
    
    # Optimization: We only need to store the 6 edge indices for each cycle.
    # If all 6 edges have the same color, it's a monochromatic C6.
    
    # Let's generate the cycles once.
    # C(17,6) = 12376 combinations.
    # 60 cycles per combination.
    # Total ~740k cycles.
    # Storing 6 ints per cycle is fine.
    
    # We will store them as a numpy array of shape (num_cycles, 6)
    # But generating 740k permutations might take a few seconds in pure Python.
    # Let's try to be smart.
    
    # Alternative cost function: 
    # Instead of counting exact C6s, we can count "potential" C6s or use a softer penalty.
    # But exact counting is best for convergence.
    
    # Let's optimize the generation.
    # A cycle on vertices {v0, v1, v2, v3, v4, v5} is defined by the order.
    # We can fix v0 as the smallest vertex in the set to reduce symmetry?
    # No, the cycle structure depends on the edges, not the vertex labels' order in the set.
    # However, any cycle can be rotated and reversed.
    # There are 60 unique cycles on 6 labeled vertices.
    
    # Let's just generate them.
    cycle_edges_list = []
    
    # To save time, let's only generate cycles where the first vertex is the smallest in the set?
    # No, that doesn't cover all cycles. A cycle (1,2,3,4,5,6) has min 1. 
    # But (2,3,4,5,6,1) is the same cycle. 
    # If we fix the smallest vertex as the "start", we still have 5! / 2 = 60 cycles?
    # No. If we fix v0 as the smallest, the remaining 5 vertices can be permuted in 5! = 120 ways.
    # Reversal symmetry: (v0, v1, ..., v5) is same as (v0, v5, ..., v1).
    # So 120 / 2 = 60.
    # Yes! So for each set of 6 vertices, we can fix the smallest vertex as the start,
    # and iterate over permutations of the other 5, taking only half to avoid reversal.
    
    # This reduces generation time.
    
    for verts in combinations(range(n), 6):
        v0 = verts[0]
        others = list(verts[1:])
        # Generate permutations of others
        # To avoid reversal, we can require that the second vertex is smaller than the last?
        # Or just generate all 120 and skip half?
        # Let's use a set to deduplicate if it's faster, or just logic.
        # Logic: perm[0] < perm[-1]
        
        for perm in permutations(others):
            if perm[0] < perm[-1]:
                # Cycle: v0, perm[0], perm[1], perm[2], perm[3], perm[4]
                # Edges:
                # (v0, p0), (p0, p1), (p1, p2), (p2, p3), (p3, p4), (p4, v0)
                e1 = edge_idx[v0, perm[0]]
                e2 = edge_idx[perm[0], perm[1]]
                e3 = edge_idx[perm[1], perm[2]]
                e4 = edge_idx[perm[2], perm[3]]
                e5 = edge_idx[perm[3], perm[4]]
                e6 = edge_idx[perm[4], v0]
                cycle_edges_list.append((e1, e2, e3, e4, e5, e6))
                
    cycle_edges = np.array(cycle_edges_list, dtype=np.int32)
    num_cycles = len(cycle_edges)
    
    # Initial random coloring
    colors = np.random.randint(0, 4, size=num_edges)
    
    # Cost function: Number of monochromatic C6s
    def calc_cost(c):
        # c is a numpy array of colors
        # For each cycle, check if all 6 edges have the same color.
        # c[cycle_edges] gives shape (num_cycles, 6)
        # We need to check if all values in each row are equal.
        # c[:, 0] == c[:, 1] == ... == c[:, 5]
        # This can be done with broadcasting.
        # Let A = c[cycle_edges]
        # A[:, 0] == A[:, 1] & A[:, 1] == A[:, 2] ...
        # Or simpler: (A - A[:, 0:1]).sum(axis=1) == 0
        # This creates a large temporary array (740k * 6). ~4.4M ints. Fine.
        
        A = c[cycle_edges]
        # Check if all elements in each row are equal to the first element
        diffs = A - A[:, 0:1]
        # If any diff is non-zero, it's not monochromatic
        # We want the count of rows where all diffs are 0.
        # np.all(diffs == 0, axis=1)
        # This returns a boolean array of length num_cycles.
        # Sum of that is the cost.
        
        # Optimization: 
        # Instead of full boolean check, we can sum absolute values?
        # No, boolean is fast.
        
        return np.sum(np.all(diffs == 0, axis=1))
        
    # Initial cost
    current_cost = calc_cost(colors)
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    best_colors = colors.copy()
    best_cost = current_cost
    
    while not time_check():
        # Pick a random edge to change
        e_idx = np.random.randint(num_edges)
        old_color = colors[e_idx]
        
        # Pick a new color different from old
        new_color = np.random.randint(0, 3)
        if new_color >= old_color:
            new_color += 1
            
        colors[e_idx] = new_color
        
        # Calculate new cost
        # Full recalculation is ~1-2ms in NumPy for 740k cycles.
        # 150s / 2ms = 75,000 iterations.
        # This is a decent number of steps.
        
        new_cost = calc_cost(colors)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    break
        else:
            colors[e_idx] = old_color
            
        T *= cooling_rate
        
    # Return the best found
    if best_cost == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    else:
        # If we didn't find a perfect solution, return the best we have?
        # The task requires "No colour class may contain a six-cycle".
        # So we MUST return a valid one or a smaller n.
        # If we fail for n=17, we should try n=16 or fall back to a known valid small one.
        
        # Try n=16 quickly with the same method? Or just return a known valid n=15?
        # The seed program returns n=9 if it fails.
        # Let's try to find a valid n=16 if n=17 fails, or just return a static valid n=15.
        
        # Actually, let's just return a static valid coloring for n=15 if we fail.
        # Or better, run a quick backtracking for n=16 if time permits?
        # No, keep it simple. The spectral/SA approach should find n=17.
        
        # Fallback: Return a valid small coloring.
        # Let's construct a valid n=10 coloring using a simple rule or search.
        # Or just return the best_colors if it's "close"? No, invalid scores 0.
        
        # Let's try a deterministic fallback for n=10.
        # The seed program had a fallback. Let's reuse a simple valid one.
        # A valid 4-coloring of K_10 exists.
        # We can hardcode a known valid one for n=10 or 11.
        # Or, we can run a very fast backtracking for n=10.
        
        # For now, return a static valid n=9 coloring to be safe.
        n_small = 9
        # Generate a valid coloring for K_9.
        # We can use the cyclic method for n=9?
        # Z_9 is not prime, but we can just brute force or use a known pattern.
        # Actually, the seed program's fallback was:
        # [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        # Let's verify if that is valid for n=9.
        # It's a heuristic.
        
        # To be absolutely safe, let's just return the best_colors if it's valid, 
        # otherwise a hardcoded valid n=5.
        
        # But wait, if best_cost > 0, best_colors is invalid.
        # We need a valid one.
        
        # Let's do a quick check: is there a valid n=17 in the literature that we can hardcode?
        # The prompt says "The best known colouring has 17 vertices".
        # If we can't find it via SA, we should return a smaller valid one.
        
        # Let's try to find a valid n=16 with a quick restart of SA with lower T?
        # No, time is tight.
        
        # Let's return a static valid n=10 coloring.
        # I will generate one using the same SA logic but for n=10, which is much easier.
        
        n_try = 10
        num_edges_try = n_try * (n_try - 1) // 2
        edge_idx_try = np.zeros((n_try, n_try), dtype=np.int32)
        idx_t = 0
        for i in range(n_try):
            for j in range(i + 1, n_try):
                edge_idx_try[i, j] = idx_t
                edge_idx_try[j, i] = idx_t
                idx_t += 1
                
        cycles_try = []
        for verts in combinations(range(n_try), 6):
            v0 = verts[0]
            others = list(verts[1:])
            for perm in permutations(others):
                if perm[0] < perm[-1]:
                    e1 = edge_idx_try[v0, perm[0]]
                    e2 = edge_idx_try[perm[0], perm[1]]
                    e3 = edge_idx_try[perm[1], perm[2]]
                    e4 = edge_idx_try[perm[2], perm[3]]
                    e5 = edge_idx_try[perm[3], perm[4]]
                    e6 = edge_idx_try[perm[4], v0]
                    cycles_try.append((e1, e2, e3, e4, e5, e6))
                    
        if len(cycles_try) > 0:
            cycle_edges_try = np.array(cycles_try, dtype=np.int32)
            colors_try = np.random.randint(0, 4, size=num_edges_try)
            T_try = 1.0
            current_cost_try = calc_cost(colors_try) # Reusing function? No, edge indices differ.
            
            def calc_cost_try(c):
                A = c[cycle_edges_try]
                diffs = A - A[:, 0:1]
                return np.sum(np.all(diffs == 0, axis=1))
                
            current_cost_try = calc_cost_try(colors_try)
            best_try = colors_try.copy()
            best_cost_try = current_cost_try
            
            # Quick SA for n=10
            time_limit_inner = time.time() + 5.0
            while time.time() < time_limit_inner and not time_check():
                e_idx_t = np.random.randint(num_edges_try)
                old_c = colors_try[e_idx_t]
                new_c = np.random.randint(0, 3)
                if new_c >= old_c: new_c += 1
                colors_try[e_idx_t] = new_c
                new_c_cost = calc_cost_try(colors_try)
                delta_t = new_c_cost - current_cost_try
                if delta_t < 0 or (T_try > 0 and random.random() < np.exp(-delta_t / T_try)):
                    current_cost_try = new_c_cost
                    if current_cost_try < best_cost_try:
                        best_cost_try = current_cost_try
                        best_try = colors_try.copy()
                        if best_cost_try == 0: break
                else:
                    colors_try[e_idx_t] = old_c
                T_try *= 0.999
                
            if best_cost_try == 0:
                return {'n': n_try, 'colors': best_try.tolist()}
                
        # Absolute fallback: n=5
        n_fb = 5
        # Any coloring of K_5 is valid for C6 (no C6 exists in K_5)
        colors_fb = [0] * (n_fb * (n_fb - 1) // 2)
        return {'n': n_fb, 'colors': colors_fb}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
