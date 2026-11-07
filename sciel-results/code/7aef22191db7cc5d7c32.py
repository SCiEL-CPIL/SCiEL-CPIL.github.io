def construct():
    import itertools
    
    # We search for a 3-regular bipartite graph on 12 vertices (0-5, 6-11)
    # with no 4-cycles.
    
    # To speed up, we fix the first row.
    # A[0] connects to B[0], B[1], B[2] (renamed 6,7,8)
    
    # We will iterate over possible adjacency lists for A[1]..A[5].
    # This is a small CSP.
    
    # Vertices A: 0-5, B: 6-11
    # deg(A) = 3, deg(B) = 3
    
    # We represent the graph by the list of neighbors for each A vertex.
    # neighbors[i] is a set of 3 B vertices.
    
    # Fix neighbors[0] = {6, 7, 8}
    
    # We need to assign neighbors for 1,2,3,4,5 such that:
    # 1. Each B vertex appears exactly 3 times in total.
    # 2. No two A vertices share 2 or more B neighbors (no 4-cycles).
    
    # Let's use backtracking.
    
    def solve(idx, current_neigh, b_counts):
        if idx == 6:
            if all(c == 3 for c in b_counts.values()):
                return current_neigh
            return None
        
        # Try all subsets of B of size 3
        # Pruning: 
        # - Must not share >=2 neighbors with any previous A
        # - Must not make it impossible to satisfy remaining counts
        
        for combo in itertools.combinations(range(6, 12), 3):
            c_set = set(combo)
            
            # Check 4-cycle constraint with previous A's
            valid = True
            for prev_idx in range(idx):
                if len(c_set & current_neigh[prev_idx]) >= 2:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check count feasibility
            new_counts = b_counts.copy()
            feasible = True
            for b in c_set:
                new_counts[b] += 1
                if new_counts[b] > 3:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Optional: Check if remaining vertices can satisfy remaining counts
            # This is a small check.
            
            res = solve(idx + 1, current_neigh + [c_set], new_counts)
            if res:
                return res
        return None

    # Initial state
    # b_counts: 6->1, 7->1, 8->1, others 0
    init_counts = {b: 0 for b in range(6, 12)}
    init_neigh_0 = {6, 7, 8}
    for b in init_neigh_0:
        init_counts[b] += 1
        
    # Note: The solver above assumes we fill A[0]..A[5].
    # I fixed A[0] in the logic above but the function starts at idx=0.
    # Let's adjust: Fix A[0] outside, start solver at idx=1.
    
    # Actually, let's just let the solver handle it, but fix A[0] to break symmetry.
    # We can just call solve(1, [{6,7,8}], init_counts)
    
    solution = solve(1, [{6, 7, 8}], init_counts)
    
    if solution:
        edges = []
        for i, neigh in enumerate(solution):
            for b in neigh:
                edges.append([i, b])
        return edges
    else:
        # Fallback to current program's logic if search fails (unlikely)
        edges = set()
        for i in range(6):
            for j in range(3):
                a = i
                b = 6 + (i + j) % 6
                edges.add((min(a, b), max(a, b)))
        return [list(e) for e in sorted(edges)]
