"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a valid 2-coloring of 3-uniform hypergraph on n vertices
    with no red K_4^(3) and no blue K_5^(3).
    
    We use a deterministic local search (Min-Conflicts style) starting from
    a seeded algebraic initialization, with a fallback to smaller n.
    
    The seed construction (sum mod 3) works for n=9. For larger n, we use
    local search to fix conflicts.
    """
    import time
    import random
    from itertools import combinations
    
    deadline = time.time() + 110.0  # Leave 10s margin
    
    def count_violations(n, colors, edge_indices):
        """Count number of red K_4^(3) and blue K_5^(3)."""
        # Build lookup: triple (a,b,c) -> index in colors
        # For efficiency, precompute which triples are in a 4-set or 5-set
        
        # Red K4: 4 vertices {a,b,c,d}, all 4 triples red
        red_k4 = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Triples in this 4-set: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            t1 = (a, b, c)
            t2 = (a, b, d)
            t3 = (a, c, d)
            t4 = (b, c, d)
            i1 = edge_indices.get(t1)
            i2 = edge_indices.get(t2)
            i3 = edge_indices.get(t3)
            i4 = edge_indices.get(t4)
            if i1 is not None and i2 is not None and i3 is not None and i4 is not None:
                if colors[i1] == 0 and colors[i2] == 0 and colors[i3] == 0 and colors[i4] == 0:
                    red_k4 += 1
        
        # Blue K5: 5 vertices, all 10 triples blue
        blue_k5 = 0
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                idx = edge_indices.get(tri)
                if idx is None or colors[idx] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_k5 += 1
        
        return red_k4 + blue_k5
    
    def solve(n, time_limit):
        """Try to find a valid coloring for n vertices within time_limit seconds."""
        # Generate all triples in lexicographic order
        triples = list(combinations(range(n), 3))
        m = len(triples)
        
        # Build edge index
        edge_indices = {t: i for i, t in enumerate(triples)}
        
        # Initialize colors using algebraic seeding (sum mod 3)
        colors = bytearray([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples])
        
        # Compute initial violations
        violations = count_violations(n, colors, edge_indices)
        if violations == 0:
            return list(colors), n
        
        # Local search: Min-Conflicts
        rng = random.Random(42)
        
        # Precompute for each edge, which 4-sets (for red) and 5-sets (for blue) it participates in
        # Actually, let's just iterate and flip edges that reduce violations
        
        # For efficiency, track which edges are in red K4s and blue K5s
        # We'll use a simpler approach: repeatedly pick a random edge and flip if it helps
        
        # To speed up, we can compute the contribution of each edge to violations
        # Let's use a different approach: for each edge, compute how many violations it's involved in
        
        # Simpler: just do random flips and check if total violations decrease
        # This is O(m * check_cost) per iteration, which might be too slow
        
        # Better: maintain counts per vertex set
        # Let's precompute all 4-sets and their 4 edges, and all 5-sets and their 10 edges
        
        # Precompute 4-sets
        quads = list(combinations(range(n), 4))
        quad_edges = []
        for quad in quads:
            a, b, c, d = quad
            edges = [
                edge_indices[(a, b, c)],
                edge_indices[(a, b, d)],
                edge_indices[(a, c, d)],
                edge_indices[(b, c, d)]
            ]
            quad_edges.append(edges)
        
        # Precompute 5-sets
        quints = list(combinations(range(n), 5))
        quint_edges = []
        for quint in quints:
            edges = [edge_indices[tri] for tri in combinations(quint, 3)]
            quint_edges.append(edges)
        
        # Count violations using these precomputed structures
        def count_violations_fast(colors):
            red_k4 = 0
            for edges in quad_edges:
                if all(colors[e] == 0 for e in edges):
                    red_k4 += 1
            blue_k5 = 0
            for edges in quint_edges:
                if all(colors[e] == 1 for e in edges):
                    blue_k5 += 1
            return red_k4 + blue_k5
        
        violations = count_violations_fast(colors)
        
        if violations == 0:
            return list(colors), n
        
        # Track best solution found
        best_violations = violations
        best_colors = list(colors)
        
        # Local search
        # For each edge, compute its "conflict degree": how many violating 4-sets (if red) or 5-sets (if blue) it's in
        # When we flip an edge, the change in violations depends on neighboring structures
        
        # Let's use a simpler but effective approach: 
        # 1. Find a random violating 4-set or 5-set
        # 2. Flip one of its edges
        # 3. Repeat
        
        # To find violating structures efficiently, we can maintain a list
        # But for simplicity, let's just do random edge flips with acceptance if it reduces violations
        
        # Actually, let's do a more targeted approach:
        # Iterate through edges in random order, flip each if it reduces violations
        
        max_iters = 50000
        for iteration in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random edge
            edge_idx = rng.randint(0, m - 1)
            old_color = colors[edge_idx]
            new_color = 1 - old_color
            
            # Quick check: compute violations before and after
            # This is expensive, so let's use a heuristic
            
            # Alternative: just flip and check
            colors[edge_idx] = new_color
            new_violations = count_violations_fast(colors)
            
            if new_violations < violations:
                violations = new_violations
                if violations == 0:
                    return list(colors), n
                if violations < best_violations:
                    best_violations = violations
                    best_colors = list(colors)
            else:
                # Revert
                colors[edge_idx] = old_color
        
        # If we didn't find a solution, return the best we found (may be invalid)
        # But the task requires a valid coloring, so we should return a valid one
        # For now, return the best colors even if invalid (evaluator will check)
        return best_colors, n
    
    # Try n=34 first, then fall back to smaller values
    for n in [34, 33, 32, 31, 30, 28, 26, 24, 20, 15, 10, 9]:
        if time.time() > deadline:
            break
        
        try:
            colors, actual_n = solve(n, time.time() + min(30, deadline - time.time()))
            
            # Validate the solution
            triples = list(combinations(range(actual_n), 3))
            edge_indices = {t: i for i, t in enumerate(triples)}
            
            # Check red K4
            valid = True
            for quad in combinations(range(actual_n), 4):
                a, b, c, d = quad
                if (colors[edge_indices[(a, b, c)]] == 0 and 
                    colors[edge_indices[(a, b, d)]] == 0 and 
                    colors[edge_indices[(a, c, d)]] == 0 and 
                    colors[edge_indices[(b, c, d)]] == 0):
                    valid = False
                    break
            
            if valid:
                # Check blue K5
                for quint in combinations(range(actual_n), 5):
                    all_blue = True
                    for tri in combinations(quint, 3):
                        if colors[edge_indices[tri]] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        valid = False
                        break
            
            if valid:
                return {'n': actual_n, 'colors': colors}
        
        except Exception as e:
            continue
    
    # Fallback: return the seed solution for n=9
    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
