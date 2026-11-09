import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    def get_edge_index(i, n):
        # Lexicographic edge index for (i, j) with i < j in K_n
        # Edges are ordered: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # Index of edge (i, j): i*(n-1) - i*(i+1)//2 + (j - i - 1)
        # Actually let me recompute:
        # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Wait: for vertex 0: n-1 edges, vertex 1: n-2 edges, ..., vertex i: n-1-i edges
        # Total before vertex i's edges: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # Within vertex i's edges, edge (i,j) is at position j-i-1
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def get_cycles_fast(n):
        """
        Precompute all C6 cycles as tuples of 6 edge indices.
        A cycle (v0,v1,v2,v3,v4,v5) has edges:
        (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
        We need to order vertices so that each edge is (min,max).
        """
        cycles = []
        for verts in itertools.combinations(range(n), 6):
            # For each set of 6 vertices, generate all distinct 6-cycles
            # Fix v0 as the smallest vertex to break rotation symmetry
            v0 = verts[0]
            rest = list(verts[1:])
            for perm in itertools.permutations(rest):
                # perm = (p1, p2, p3, p4, p5)
                # Cycle: v0, p1, p2, p3, p4, p5, back to v0
                # To avoid reversal duplicates, require p1 < p5
                if perm[0] < perm[-1]:
                    # Compute edge indices
                    e0 = get_edge_index(min(v0, perm[0]), max(v0, perm[0]), n)
                    e1 = get_edge_index(min(perm[0], perm[1]), max(perm[0], perm[1]), n)
                    e2 = get_edge_index(min(perm[1], perm[2]), max(perm[1], perm[2]), n)
                    e3 = get_edge_index(min(perm[2], perm[3]), max(perm[2], perm[3]), n)
                    e4 = get_edge_index(min(perm[3], perm[4]), max(perm[3], perm[4]), n)
                    e5 = get_edge_index(min(perm[4], perm[5]), max(perm[4], perm[5]), n)
                    e6 = get_edge_index(min(perm[5], v0), max(perm[5], v0), n)
                    cycles.append((e0, e1, e2, e3, e4, e5))
        return cycles
    
    def solve_local_search(n, time_budget):
        """
        Local search with incremental violation counting.
        Returns color list or None.
        """
        if time.time() > deadline:
            return None
        
        # Precompute cycles
        cycles = get_cycles_fast(n)
        num_cycles = len(cycles)
        if num_cycles == 0:
            return [0] * (n * (n - 1) // 2)
        
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        
        # Count violations for each cycle
        def compute_violations():
            viol = [0] * num_cycles
            for ci, cycle in enumerate(cycles):
                c = [colors[e] for e in cycle]
                if len(set(c)) == 1:
                    viol[ci] = 1
            return viol
        
        current_viol = compute_violations()
        total_viol = sum(current_viol)
        
        # Track which cycles are violated for quick access
        violated_cycles = set(i for i, v in enumerate(current_viol) if v)
        
        def is_violated(cycle):
            c = [colors[e] for e in cycle]
            return len(set(c)) == 1
        
        best_total = total_viol
        best_colors = colors[:]
        
        while time.time() < deadline - 2.0:
            if total_viol == 0:
                return colors
            
            if total_viol < best_total:
                best_total = total_viol
                best_colors = colors[:]
            
            # Pick a random violated cycle
            if not violated_cycles:
                # Recompute
                current_viol = compute_violations()
                total_viol = sum(current_viol)
                violated_cycles = set(i for i, v in enumerate(current_viol) if v)
                continue
            
            ci = random.choice(tuple(violated_cycles))
            cycle = cycles[ci]
            
            # Try changing each edge in the cycle to a different color
            best_delta = 0  # We want to minimize total violations
            best_edge = -1
            best_new_color = -1
            
            for ei, edge_idx in enumerate(cycle):
                old_color = colors[edge_idx]
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    # Check impact on all cycles containing this edge
                    # For simplicity, just check if this cycle becomes non-violated
                    # and how many other cycles are affected
                    # Quick check: does this make the current cycle non-violated?
                    new_c = [colors[cycle[j]] if j != ei else new_color for j in range(6)]
                    this_still_violated = len(set(new_c)) == 1
                    this_was_violated = True  # We picked it because it was violated
                    
                    delta = 0
                    if this_still_violated:
                        delta = 0  # No change for this cycle
                    else:
                        delta = -1  # Fixed one violation
                    
                    # This is a heuristic; for full accuracy we'd need to check all affected cycles
                    # Let's just try it and measure
                    if delta < best_delta:
                        best_delta = delta
                        best_edge = edge_idx
                        best_new_color = new_color
            
            if best_edge == -1:
                # Random move
                edge_idx = random.choice(cycle)
                new_color = random.choice([c for c in range(4) if c != colors[edge_idx]])
                best_edge = edge_idx
                best_new_color = new_color
            
            # Apply the move
            old_color = colors[best_edge]
            colors[best_edge] = best_new_color
            
            # Update violations for cycles containing best_edge
            # For efficiency, we just recompute the whole thing occasionally
            # But for now, let's do a quick update
            new_total = 0
            new_violated = set()
            for ci2, cyc in enumerate(cycles):
                if best_edge in cyc:
                    c = [colors[e] for e in cyc]
                    if len(set(c)) == 1:
                        new_violated.add(ci2)
                    new_total += 1 if ci2 in new_violated else 0
                else:
                    # Cycle not affected
                    if ci2 in violated_cycles:
                        new_violated.add(ci2)
                        new_total += 1
            
            total_viol = new_total
            violated_cycles = new_violated
        
        # Return best found
        if best_total == 0:
            return best_colors
        return None
    
    # Try increasing n from high to low
    for n in range(14, 7, -1):
        if time.time() > deadline - 5.0:
            break
        budget = min(20.0, deadline - time.time() - 5.0)
        if budget < 3.0:
            break
        
        sol = solve_local_search(n, budget)
        if sol is not None:
            # Verify
            colors = sol
            num_edges = n * (n - 1) // 2
            if len(colors) == num_edges:
                return {'n': n, 'colors': colors}
    
    # Fallback: try small n with more time
    for n in range(7, 3, -1):
        if time.time() > deadline - 2.0:
            break
        sol = solve_local_search(n, min(5.0, deadline - time.time() - 2.0))
        if sol is not None:
            num_edges = n * (n - 1) // 2
            if len(colors := sol) == num_edges:
                return {'n': n, 'colors': colors}
    
    # Last resort: n=1
    return {'n': 1, 'colors': []}
