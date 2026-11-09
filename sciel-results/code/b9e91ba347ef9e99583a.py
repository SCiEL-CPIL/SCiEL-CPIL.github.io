import math
import time
import random

def _count_edges(points):
    """Count pairs at distance exactly 1 in the triangular lattice."""
    pts = set(points)
    count = 0
    for a, b in pts:
        # Six neighbors in triangular lattice
        for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            if (a+da, b+db) in pts:
                count += 1
    return count // 2

def _hex_spiral_points(n):
    """
    Generate n points in a hexagonal spiral order.
    This should achieve close to the Harborth optimum of 76 edges for n=32.
    
    The triangular lattice has basis (1,0) and omega=e^(i*pi/3).
    Neighbors of (a,b) are: (a±1,b), (a,b±1), (a+1,b-1), (a-1,b+1).
    
    We generate rings around the origin. Ring r contains all points at
    hex distance r from origin.
    """
    points = [(0, 0)]
    
    # Directions for the six sides of the hexagon
    # Going counterclockwise from the rightmost point
    # Side 1: go up-right (0, 1) for r steps
    # Side 2: go left (-1, 0) for r steps  
    # Side 3: go down-left (-1, 1) for r steps
    # Side 4: go down (0, -1) for r steps
    # Side 5: go right (1, 0) for r steps
    # Side 6: go up-right (1, -1) for r steps -- wait, this needs to close
    
    # Let me think about this differently.
    # Starting at (r, 0):
    # Move (0, 1) r times: (r, 0) -> (r, 1) -> ... -> (r, r)
    # Move (-1, 1) r times: (r, r) -> (r-1, r+1) -> ... -> (0, 2r) -- this is wrong
    
    # Let me use a known hex spiral pattern.
    # In axial coordinates for a hex grid, the spiral is well-defined.
    # Our coordinates (a, b) with omega = e^(i*pi/3) are related to axial coords.
    
    # Let me just try a simple approach: generate all points in a bounding box,
    # sort by hex distance, then by angle, and take the first n.
    
    max_r = 10
    candidates = []
    for a in range(-max_r, max_r + 1):
        for b in range(-max_r, max_r + 1):
            # Compute hex distance (minimum number of unit steps)
            # For our basis, the distance in unit steps is:
            # d = (|a| + |b| + |a+b|) / 2  -- but this might not be integer
            # Actually, let's compute it properly.
            # The three basis vectors are e1=(1,0), e2=(0,1), e3=(1,-1) in our coord system
            # Wait, the neighbors are (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
            # So the three independent directions are (1,0), (0,1), (1,-1)
            # The hex distance is the minimum of |x|+|y|+|z| where x+y+z=0 and
            # (a,b) = x*(1,0) + y*(0,1) + z*(1,-1)
            # So a = x + z, b = y - z
            # x = a - z, y = b + z
            # We need x+y+z = (a-z) + (b+z) + z = a + b + z = 0, so z = -(a+b)
            # x = a + a + b = 2a + b, y = b - a - b = -a
            # So hex distance = |2a+b| + |-a| + |-(a+b)| = |2a+b| + |a| + |a+b|
            # But this should be even, and the actual distance is half of this?
            # Let me check: for (1,0), distance should be 1.
            # |2(1)+0| + |1| + |1+0| = 2+1+1 = 4, half is 2. That's wrong.
            
            # Let me try a different approach. The hex distance in a triangular lattice
            # with the given neighbors can be computed as:
            # d = max(|a|, |b|, |a+b|) -- let me verify
            # (1,0): max(1,0,1) = 1 ✓
            # (0,1): max(0,1,1) = 1 ✓
            # (1,-1): max(1,1,0) = 1 ✓
            # (2,0): max(2,0,2) = 2 ✓
            # (1,1): max(1,1,2) = 2 ✓ (distance from (0,0) to (1,1) is 2: (0,0)->(1,0)->(1,1))
            # Great, so hex distance = max(|a|, |b|, |a+b|)
            
            hex_dist = max(abs(a), abs(b), abs(a + b))
            if hex_dist == 0:
                continue  # Already have origin
            
            # Angle for sorting within the same ring
            x = a + 0.5 * b
            y = (math.sqrt(3) / 2.0) * b
            angle = math.atan2(y, x)
            
            candidates.append((hex_dist, angle, a, b))
    
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Take the first n-1 candidates (we already have origin)
    for _, _, a, b in candidates:
        if len(points) >= n:
            break
        points.append((a, b))
    
    return points[:n]

def _generate_hex_cluster(n):
    """
    Generate a compact hexagonal cluster of n points.
    """
    return _hex_spiral_points(n)

def _local_search(points, max_time=10.0):
    """
    Perform local search to maximize edge count.
    Moves: Swap a point in the set with a point in the candidate neighborhood.
    """
    start_time = time.time()
    current_points = set(points)
    current_edges = _count_edges(current_points)
    best_points = set(current_points)
    best_edges = current_edges
    
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Bounding box of current set
        min_a = min(p[0] for p in current_points)
        max_a = max(p[0] for p in current_points)
        min_b = min(p[1] for p in current_points)
        max_b = max(p[1] for p in current_points)
        
        # Expand bounding box by 2 to find candidates
        cand_min_a = min_a - 2
        cand_max_a = max_a + 2
        cand_min_b = min_b - 2
        cand_max_b = max_b + 2
        
        candidates = set()
        for a in range(cand_min_a, cand_max_a + 1):
            for b in range(cand_min_b, cand_max_b + 1):
                candidates.add((a, b))
        
        candidates -= current_points
        
        # Try swapping each point in current_points with each candidate
        for p_out in list(current_points):
            for p_in in candidates:
                new_points = set(current_points)
                new_points.remove(p_out)
                new_points.add(p_in)
                
                new_edges = _count_edges(new_points)
                if new_edges > current_edges:
                    current_points = new_points
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_points = set(current_points)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try a random perturbation to escape local optimum
            if random.random() < 0.3 and candidates:
                p_out = random.choice(list(current_points))
                p_in = random.choice(list(candidates))
                new_points = set(current_points)
                new_points.remove(p_out)
                new_points.add(p_in)
                current_points = new_points
                current_edges = _count_edges(current_points)
            else:
                break
    
    return list(best_points)

def construct():
    deadline = time.time() + 145.0
    
    # Generate initial hex cluster using hex spiral
    points = _generate_hex_cluster(32)
    
    # Perform local search
    best_points = points
    best_edges = _count_edges(points)
    
    # If we already have 76 edges, return immediately
    if best_edges == 76:
        return [[a, b] for a, b in best_points]
    
    # Try local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        search_points = _local_search(points, max_time=min(remaining_time - 2.0, 10.0))
        search_edges = _count_edges(search_points)
        
        if search_edges > best_edges:
            best_edges = search_edges
            best_points = search_points
    
    # Safety margin: return before deadline
    if time.time() > deadline - 2.0:
        return [[a, b] for a, b in best_points]
    
    # Try a few more restarts if time permits
    restarts = 0
    while time.time() < deadline - 5.0 and restarts < 10:
        # Perturb the current best
        pts_set = set(best_points)
        pts_list = list(pts_set)
        for _ in range(2):
            if pts_list:
                p_out = pts_list.pop(random.randint(0, len(pts_list)-1))
                pts_set.remove(p_out)
                
                candidates = set()
                min_a = min(p[0] for p in pts_set)
                max_a = max(p[0] for p in pts_set)
                min_b = min(p[1] for p in pts_set)
                max_b = max(p[1] for p in pts_set)
                for a in range(min_a-2, max_a+3):
                    for b in range(min_b-2, max_b+3):
                        if (a,b) not in pts_set:
                            candidates.add((a,b))
                if candidates:
                    p_in = random.choice(list(candidates))
                    pts_set.add(p_in)
        
        # Run local search on perturbed set
        remaining = deadline - time.time() - 5.0
        if remaining > 1.0:
            search_points = _local_search(list(pts_set), max_time=min(remaining, 5.0))
            search_edges = _count_edges(search_points)
            
            if search_edges > best_edges:
                best_edges = search_edges
                best_points = search_points
            
            if best_edges == 76:
                break
        
        restarts += 1
    
    return [[a, b] for a, b in best_points]
