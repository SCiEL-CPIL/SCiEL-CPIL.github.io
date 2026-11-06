def count_edges(points):
    edges = 0
    p_set = set(points)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    for p in p_set:
        for d in directions:
            if (p[0] + d[0], p[1] + d[1]) in p_set:
                edges += 1
    return edges // 2

def construct():
    # Directions in the triangular lattice (axial coordinates)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    # Start with the origin
    points = set([(0, 0)])
    # Candidates for the next point to add
    candidates = set()
    
    # Initialize candidates with neighbors of the origin
    for dq, dr in directions:
        candidates.add((dq, dr))
        
    # Greedy growth: always add the candidate that maximizes the number of
    # connections to existing points.
    while len(points) < 32:
        best_candidate = None
        max_connections = -1
        
        for c in candidates:
            # Count how many existing points are neighbors of c
            connections = 0
            for dq, dr in directions:
                if (c[0] + dq, c[1] + dr) in points:
                    connections += 1
            
            if connections > max_connections:
                max_connections = connections
                best_candidate = c
                
        # Add the best candidate
        if best_candidate is None:
            # Fallback if candidates are empty (should not happen for 32 pts)
            break
            
        points.add(best_candidate)
        candidates.remove(best_candidate)
        
        # Add neighbors of the new point to candidates
        for dq, dr in directions:
            new_cand = (best_candidate[0] + dq, best_candidate[1] + dr)
            if new_cand not in points:
                candidates.add(new_cand)
                
    # Local Search Refinement
    # Try to swap points to maximize edges
    current_edges = count_edges(points)
    improved = True
    while improved:
        improved = False
        # Sort points for determinism in iteration
        point_list = sorted(list(points))
        for i, p_in in enumerate(point_list):
            # Try swapping p_in with a candidate outside
            # To be efficient, only consider candidates near the current set
            # or just a random subset if the set is large. 
            # For 32 points, we can check all points in a bounding box.
            
            # Generate a set of potential swap candidates
            # Points within distance 2 of the current set
            swap_candidates = set()
            for p in points:
                for d1 in directions:
                    for d2 in directions + [(0,0)]:
                        c = (p[0] + d1[0] + d2[0], p[1] + d1[1] + d2[1])
                        if c not in points:
                            swap_candidates.add(c)
            
            for p_out in swap_candidates:
                # Check if swap is valid (distance constraint is always satisfied for lattice points? 
                # No, "mutual distance at least one". Lattice points are distinct integers, 
                # so min distance is 1. Any distinct lattice points have distance >= 1.
                # So we just need to ensure p_out is not in points.)
                
                # Calculate new edge count
                # Remove p_in, add p_out
                new_points = set(points)
                new_points.remove(p_in)
                new_points.add(p_out)
                
                new_edges = count_edges(new_points)
                
                if new_edges > current_edges:
                    points = new_points
                    current_edges = new_edges
                    improved = True
                    break # Restart loop with new set
            if improved:
                break

    result = sorted([list(p) for p in points])
    return result
