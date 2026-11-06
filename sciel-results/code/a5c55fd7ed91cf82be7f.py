import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    # Precompute all K4s and their edges
    # Vertices are 0..15
    # Edges are (i, j) with i < j
    # Map edge to index
    edge_idx = {}
    edges = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i, j)] = idx
            edges.append((i, j))
            idx += 1
            
    num_edges = len(edges) # 120
    
    # List of K4s
    k4s = []
    k4_edges = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e1 = edge_idx[(a, b)]
        e2 = edge_idx[(a, c)]
        e3 = edge_idx[(a, d)]
        e4 = edge_idx[(b, c)]
        e5 = edge_idx[(b, d)]
        e6 = edge_idx[(c, d)]
        k4s.append(quad)
        k4_edges.append((e1, e2, e3, e4, e5, e6))
        
    num_k4s = len(k4s)
    
    # Helper to check validity and count distinct colours
    def get_score(colours):
        # colours is a list of length num_edges
        # Returns (num_violations, distinct_colours)
        # For speed, we can just check violations first.
        # If valid, count distinct.
        violations = 0
        for e1, e2, e3, e4, e5, e6 in k4_edges:
            # Check distinct colours
            # A K4 is invalid if it has < 5 distinct colours.
            # This happens if:
            # - 1 colour (monochromatic)
            # - 2 colours
            # - 3 colours
            # - 4 colours
            # It is valid if 5 or 6 colours.
            # So invalid if max frequency of any colour is >= 2? No.
            # If 2 edges share a colour, and 4 are distinct, that's 5 colours. Valid.
            # If 2 pairs share a colour (matching), that's 4 colours. Invalid.
            # If 3 edges share a colour, that's 4 colours (at best). Invalid.
            # So, a K4 is invalid if there are >= 2 pairs of edges with the same colour,
            # OR if there is a set of 3 edges with the same colour.
            # Actually, simpler: just count distinct.
            # But counting distinct for 6 elements is fast.
            c1, c2, c3, c4, c5, c6 = colours[e1], colours[e2], colours[e3], colours[e4], colours[e5], colours[e6]
            # Use a set
            if len({c1, c2, c3, c4, c5, c6}) < 5:
                violations += 1
        if violations > 0:
            return violations, 0
        else:
            return 0, len(set(colours))

    # We try k = 13, 14, 15
    # For each k, we run a search.
    # If we find a valid config with <= k colours, we return it.
    
    best_result = None
    
    for k in [13, 14, 15]:
        if time.time() > deadline:
            break
            
        # Run multiple restarts
        restarts = 50
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            # Initialize random colours in 0..k-1
            colours = [random.randint(0, k-1) for _ in range(num_edges)]
            
            # Simulated Annealing / Local Search
            current_viol, _ = get_score(colours)
            best_viol = current_viol
            best_colours = colours[:]
            
            # We want to reach 0 violations.
            # If we reach 0, we check distinct colours.
            # If distinct <= k, we are done.
            
            temp = 1000.0
            cooling = 0.995
            steps = 5000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                old_c = colours[e_idx]
                new_c = random.randint(0, k-1)
                if new_c == old_c:
                    continue
                    
                colours[e_idx] = new_c
                new_viol, _ = get_score(colours)
                
                delta = new_viol - current_viol
                if delta < 0 or random.random() < pow(2.718, -delta / temp):
                    current_viol = new_viol
                    if current_viol < best_viol:
                        best_viol = current_viol
                        best_colours = colours[:]
                        if best_viol == 0:
                            # Check distinct colours
                            distinct = len(set(best_colours))
                            if distinct <= k:
                                # Found a solution!
                                # We can try to reduce distinct colours further?
                                # For now, return this.
                                # Pad to 120 ints? No, the return is just the list of 120 ints.
                                # The values are in 0..119.
                                # We used 0..k-1, which is within 0..119.
                                return best_colours
                else:
                    colours[e_idx] = old_c
                    
                temp *= cooling
                
            # End of SA run
            # If best_viol == 0, we might have missed it in the loop if we didn't check every step
            if best_viol == 0:
                distinct = len(set(best_colours))
                if distinct <= k:
                    return best_colours
                    
    # If no solution found with 13, 14, 15, fall back to seed (17 colours)
    # Or try k=16, 17 quickly?
    # The seed is 17. Let's just return the seed if we fail.
    # But we can try to improve the seed slightly?
    # The seed is (i+j) % 17.
    
    # Let's try one more thing: if we didn't find 13-15, try 16?
    # Actually, the seed is 17.
    # Let's just return the seed to be safe, or a slightly optimized version if time permits.
    # But the task is to minimize. If we can't beat 17, we return 17.
    
    # Re-implement seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
