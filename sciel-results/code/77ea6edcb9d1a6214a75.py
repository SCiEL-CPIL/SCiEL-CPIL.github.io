import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit
    
    # Fallback solution (seed) - known valid K_9
    n_seed = 9
    seed_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_result = {'n': n_seed, 'colors': seed_colors}
    
    # Target n values to try, in ascending order of n (start with easy, go harder)
    # We know 17 is possible. 18 is the frontier.
    # Strategy: try 17 first (should be fast), then 18 if time permits.
    # If 17 fails for some reason, we fall back to 9.
    
    # Try n=17 first
    if time.time() < deadline - 20.0:
        result17 = _solve_n(17, deadline)
        if result17 is not None:
            best_result = result17
            
    # If we have a valid K17 and time, try to extend to K18
    if best_result['n'] == 17 and time.time() < deadline - 5.0:
        result18 = _extend_to_18(best_result, deadline)
        if result18 is not None:
            best_result = result18
            
    return best_result

def _build_cycles(n):
    """Precompute all C6 cycles and map edges to cycles."""
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_map[(i, j)] = idx
            idx += 1
            
    num_edges = idx
    edge_to_cycles = [[] for _ in range(num_edges)]
    cycles = []
    
    # Generate all C6s
    # For each 6-subset of vertices, there are 60 unique cycles (6! / (6*2) = 60)
    # But we need to be careful with canonicalization.
    # A cycle is a set of 6 edges.
    
    for subset in itertools.combinations(range(n), 6):
        # Generate all permutations of the 6 vertices
        # There are 720 permutations.
        # For a fixed set of 6 vertices, the number of unique Hamiltonian cycles is 5! / 2 = 60.
        # We will generate all permutations, canonicalize them, and deduplicate.
        
        seen = set()
        for perm in itertools.permutations(subset):
            # Canonicalize:
            # 1. Rotate so min vertex is first
            min_idx = perm.index(min(perm))
            rot = perm[min_idx:] + perm[:min_idx]
            # 2. If second vertex > last vertex, reverse
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            
            if rot in seen:
                continue
            seen.add(rot)
            
            edges = []
            for k in range(6):
                u = rot[k]
                v = rot[(k+1)%6]
                if u > v:
                    u, v = v, u
                edges.append(edge_map[(u, v)])
            
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(len(cycles)-1)
                
    return cycles, edge_to_cycles, edge_map

def _solve_n(n, deadline):
    """Solve for a specific n using Simulated Annealing."""
    if time.time() > deadline - 10.0:
        return None
        
    try:
        cycles, edge_to_cycles, edge_map = _build_cycles(n)
    except Exception:
        return None
        
    num_edges = n * (n - 1) // 2
    if num_edges == 0:
        return {'n': n, 'colors': []}
        
    # If n < 6, no cycles, any coloring works
    if not cycles:
        return {'n': n, 'colors': [0]*num_edges}
        
    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Precompute cycle counts
    # cycle_counts[c][color] = number of edges in cycle c with that color
    cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
    total_violations = 0
    
    for c_id, edges in enumerate(cycles):
        for e in edges:
            cycle_counts[c_id][colors[e]] += 1
        # A cycle is a violation if all 6 edges have the same color
        # Since sum of counts is 6, if any count is 6, it's a violation.
        # Actually, if one count is 6, others are 0.
        if max(cycle_counts[c_id]) == 6:
            total_violations += 1
            
    if total_violations == 0:
        return {'n': n, 'colors': colors}
        
    # Simulated Annealing
    T = 1.0
    min_T = 0.01
    cooling = 0.9999
    
    # Estimate steps:
    # Each step updates ~6 edges (avg degree in cycle graph for edges? 
    # Actually, an edge is in multiple cycles.
    # Number of cycles containing an edge e in K_n:
    # Choose 4 other vertices from n-2, and arrange them in a cycle with u,v.
    # It's roughly O(n^4). For n=17, 17^4 ~ 83k.
    # So each step touches ~83k cycles? No, that's high.
    # Let's check: edge (0,1). Cycles containing (0,1) are 0-1-a-b-c-d-0.
    # a,b,c,d are 4 distinct vertices from 1..n-1.
    # Number of ways to choose 4 is C(n-2, 4).
    # Number of cycles on these 6 vertices containing edge (0,1)?
    # Fix 0-1. The rest 4 vertices form a path from 1 to 0.
    # Permutations of 4 vertices: 4! = 24.
    # But direction matters? 0-1-a-b-c-d-0 is same as 0-d-c-b-a-1-0?
    # In undirected cycle, yes.
    # So C(n-2, 4) * 24.
    # For n=17, C(15,4) = 1365. 1365 * 24 = 32760.
    # So ~33k cycles per edge update.
    # This is too slow for Python if we do it naively.
    # We need to be careful.
    
    # Optimization:
    # Instead of iterating all affected cycles, we can just update the counts.
    # But we need to know which cycles are affected.
    # edge_to_cycles is precomputed.
    
    # Let's run SA until deadline.
    # We will stop if total_violations == 0.
    
    while time.time() < deadline - 2.0:
        if total_violations == 0:
            return {'n': n, 'colors': colors}
            
        # Pick random edge
        e = random.randint(0, num_edges - 1)
        old_color = colors[e]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
            
        # Calculate delta
        delta = 0
        affected = edge_to_cycles[e]
        
        for c_id in affected:
            cc = cycle_counts[c_id]
            old_count = cc[old_color]
            new_count = cc[new_color]
            
            old_violation = 1 if old_count == 6 else 0
            
            # Temporarily update
            cc[old_color] -= 1
            cc[new_color] += 1
            
            new_violation = 1 if cc[new_color] == 6 else 0
            
            delta += new_violation - old_violation
            
        # Acceptance criterion
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            # Accept
            colors[e] = new_color
            total_violations += delta
        else:
            # Reject: revert
            for c_id in affected:
                cc = cycle_counts[c_id]
                cc[old_color] += 1
                cc[new_color] -= 1
                
        T *= cooling
        if T < min_T:
            T = min_T
            
    if total_violations == 0:
        return {'n': n, 'colors': colors}
        
    return None

def _extend_to_18(coloring_17, deadline):
    """Try to extend a valid K17 coloring to K18."""
    n_old = 17
    n_new = 18
    colors_old = coloring_17['colors']
    
    # Rebuild structures for n=18
    if time.time() > deadline - 5.0:
        return None
        
    try:
        cycles, edge_to_cycles, edge_map = _build_cycles(n_new)
    except Exception:
        return None
        
    num_edges_new = n_new * (n_new - 1) // 2
    
    # Initialize colors for K18
    # Copy K17 colors
    colors = [0] * num_edges_new
    # Map old edges to new indices
    # Old edge (i,j) -> new index
    # New edges are (i, 17) for i in 0..16
    # And all old edges (i,j) with i,j < 17.
    
    # We need to map the old color array to the new one.
    # The lexicographic order is:
    # (0,1), (0,2), ..., (0,16), (0,17), (1,2), ..., (1,17), ..., (16,17)
    # Old order: (0,1), ..., (0,16), (1,2), ..., (16,17)
    
    # Let's just build the new array directly
    idx = 0
    for i in range(n_new):
        for j in range(i+1, n_new):
            if i < n_old and j < n_old:
                # Find index in old array
                # Old index for (i,j)
                # We can just use edge_map from _build_cycles(n_old) but we didn't return it.
                # Let's compute it or just store it in _solve_n and return?
                # Better: return edge_map from _solve_n or _build_cycles.
                # But _solve_n doesn't return it.
                # Let's just compute it.
                old_idx = 0
                for k in range(i):
                    old_idx += n_old - 1 - k
                old_idx += j - i - 1
                colors[idx] = colors_old[old_idx]
            else:
                # New edge, assign random color
                colors[idx] = random.randint(0, 3)
            idx += 1
            
    # Now run SA on K18
    # We can use the same logic as _solve_n but starting from this initial state.
    # To avoid code duplication, let's refactor _solve_n to accept an initial state?
    # Or just copy the SA loop here.
    # Given time, copying is safer.
    
    # Precompute cycle counts
    cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
    total_violations = 0
    
    for c_id, edges in enumerate(cycles):
        for e in edges:
            cycle_counts[c_id][colors[e]] += 1
        if max(cycle_counts[c_id]) == 6:
            total_violations += 1
            
    if total_violations == 0:
        return {'n': n_new, 'colors': colors}
        
    # SA
    T = 1.0
    min_T = 0.01
    cooling = 0.9999
    
    # Focus updates on the new edges first?
    # New edges are those involving vertex 17.
    new_edge_indices = []
    for i in range(n_new):
        for j in range(i+1, n_new):
            if i == n_old or j == n_old:
                # Compute index
                # Actually, we can just find them.
                pass
    
    # Let's just run standard SA.
    while time.time() < deadline - 1.0:
        if total_violations == 0:
            return {'n': n_new, 'colors': colors}
            
        e = random.randint(0, num_edges_new - 1)
        old_color = colors[e]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            continue
            
        delta = 0
        affected = edge_to_cycles[e]
        
        for c_id in affected:
            cc = cycle_counts[c_id]
            old_count = cc[old_color]
            
            cc[old_color] -= 1
            cc[new_color] += 1
            
            new_count = cc[new_color]
            
            old_violation = 1 if old_count == 6 else 0
            new_violation = 1 if new_count == 6 else 0
            
            delta += new_violation - old_violation
            
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            colors[e] = new_color
            total_violations += delta
        else:
            for c_id in affected:
                cc = cycle_counts[c_id]
                cc[old_color] += 1
                cc[new_color] -= 1
                
        T *= cooling
        if T < min_T:
            T = min_T
            
    if total_violations == 0:
        return {'n': n_new, 'colors': colors}
        
    return None

# EVOLVE-BLOCK-END
