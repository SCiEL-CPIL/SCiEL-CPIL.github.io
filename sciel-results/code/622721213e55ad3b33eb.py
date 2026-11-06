"""Rewritten program: hexagonal spiral to attain Harborth bound of 76 unit distances."""
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Generate the first 32 points of a hexagonal spiral on the triangular lattice.
    # The hexagonal spiral starts at (0,0) and spirals outward in rings.
    # Ring 0: just (0,0)
    # Ring k (k>=1): 6k points on the hexagonal boundary at "hex distance" k.
    
    # In the axial coordinate system (a, b) for the triangular lattice where
    # omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2, the six neighbors of (a,b) are:
    # (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
    # These correspond to the six unit vectors in the triangular lattice.
    
    # The hexagonal spiral visits points in order of increasing "hex distance"
    # from origin, where hex distance in axial coords (a,b) is:
    # hex_dist = max(|a|, |b|, |a+b|) ... wait, let me reconsider.
    
    # Actually, for the triangular lattice with basis vectors (1,0) and omega,
    # the six nearest neighbors of (a,b) are:
    # (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
    
    # Let me use a BFS/spiral approach. The standard hexagonal spiral in axial
    # coordinates (q, r) for hex grids uses directions. But our lattice is triangular,
    # not hex. Let me think more carefully.
    
    # The triangular lattice point a + b*omega where omega = 1/2 + i*sqrt(3)/2.
    # The six nearest neighbors of point (a, b) are at:
    # (a+1, b)   -> vector (1, 0)
    # (a-1, b)   -> vector (-1, 0)
    # (a, b+1)   -> vector (0, 1) i.e., omega
    # (a, b-1)   -> vector (0, -1) i.e., -omega
    # (a+1, b-1) -> vector (1, -1) i.e., 1 - omega = 1/2 - i*sqrt(3)/2
    # (a-1, b+1) -> vector (-1, 1) i.e., -1 + omega = -1/2 + i*sqrt(3)/2
    
    # All six have squared distance 1. Good.
    
    # The hexagonal spiral: we can generate points by expanding rings.
    # Ring 0: {(0,0)}
    # Ring k: all points at hex-distance k from origin.
    
    # What is "hex-distance" on the triangular lattice? It's the minimum number
    # of unit steps (along the 6 directions) to reach the point. This is the
    # graph distance on the triangular lattice.
    
    # For a point (a, b) in axial coordinates, the graph distance from origin is:
    # d = max(|a|, |b|, |a+b|) ... let me verify.
    # 
    # Actually, let me just do a BFS to find the first 32 points in spiral order.
    
    # BFS from origin, adding neighbors in a consistent order to get a spiral.
    # To get a proper spiral (not just BFS order), I'll use the ring-based approach.
    
    # Let me use a simpler approach: generate all points in a bounding box,
    # sort by (hex_distance, angle) to get spiral order, take first 32.
    
    # Hex distance (graph distance) from origin for point (a,b):
    # The six step vectors are: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # This is equivalent to: given (a,b), we need to find min steps.
    # 
    # Let me think: if we use steps u=(1,0), v=(0,1), w=(1,-1), then any point
    # (a,b) can be written as a*u + b*v. But we also have w = u - v, so -v = w - u,
    # etc. The distance is the minimum of |x|+|y|+|z| where a = x - z, b = y + z
    # ... this is getting complex. Let me just do BFS.
    
    # Actually, for the triangular lattice, the distance from origin to (a,b) is:
    # d = max(|a|, |b|, |a+b|) if we define it properly... 
    # Let me verify with (1,0): max(1,0,1) = 1. Correct.
    # (0,1): max(0,1,1) = 1. Correct.
    # (1,-1): max(1,1,0) = 1. Correct.
    # (1,1): max(1,1,2) = 2. Is (1,1) at distance 2? 
    #   (1,1) = (1,0) + (0,1). Yes, 2 steps. Correct.
    # (2,0): max(2,0,2) = 2. (2,0) = 2*(1,0). 2 steps. Correct.
    # (1,2): max(1,2,3) = 3. (1,2) = (1,0)+(0,1)+(0,1). 3 steps. Correct.
    # (-1,1): max(1,1,0) = 1. (-1,1) is one of the 6 neighbors. Correct.
    # (2,-1): max(2,1,1) = 2. (2,-1) = (1,0)+(1,-1). 2 steps. Correct.
    
    # Great, so hex_dist(a,b) = max(|a|, |b|, |a+b|)
    
    # Now generate spiral: for each ring k=0,1,2,..., collect all points with
    # hex_dist = k, sort them by angle, and append to our list until we have 32.
    
    # For ring k, the points satisfy max(|a|,|b|,|a+b|) = k.
    # Let me enumerate them. For a given k, a ranges from -k to k, b from -k to k,
    # and we check if max(|a|,|b|,|a+b|) == k.
    
    points = []
    seen = set()
    
    # Ring 0
    points.append([0, 0])
    seen.add((0, 0))
    
    for k in range(1, 50):  # plenty of rings
        if len(points) >= 32:
            break
        if time.time() > deadline:
            break
        
        # Collect all points on ring k
        ring_points = []
        for a in range(-k, k + 1):
            for b in range(-k, k + 1):
                if max(abs(a), abs(b), abs(a + b)) == k:
                    if (a, b) not in seen:
                        ring_points.append((a, b))
        
        # Sort by angle for spiral ordering
        # Angle of point (a,b): the complex number is a + b*omega = a + b*(1/2 + i*sqrt(3)/2)
        # = (a + b/2) + i*(b*sqrt(3)/2)
        # angle = atan2(b*sqrt(3)/2, a + b/2)
        import math
        def angle_key(pt):
            a, b = pt
            x = a + b * 0.5
            y = b * math.sqrt(3) / 2
            return math.atan2(y, x)
        
        ring_points.sort(key=angle_key)
        
        for pt in ring_points:
            if len(points) >= 32:
                break
            if time.time() > deadline:
                break
            if pt not in seen:
                points.append([pt[0], pt[1]])
                seen.add(pt)
    
    # We need exactly 32 points
    if len(points) > 32:
        points = points[:32]
    
    # Verify: all distinct, |a|,|b| <= 64
    assert len(points) == 32, f"Got {len(points)} points"
    assert len(set(tuple(p) for p in points)) == 32, "Points not distinct"
    for p in points:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64, f"Point {p} out of bounds"
    
    # Count unit distances to verify
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da * da + db * db + da * db
    
    edge_count = 0
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if dist_sq(points[i], points[j]) == 1:
                edge_count += 1
    
    # The hexagonal spiral should give 76 edges for n=32
    # If not, try a different approach
    if edge_count < 76:
        # Fallback: try to optimize with local search
        if time.time() < deadline - 10:
            import random
            best_points = [list(p) for p in points]
            best_edges = edge_count
            
            for iteration in range(5000):
                if time.time() > deadline - 5:
                    break
                
                # Try swapping a point with a random nearby point
                idx = random.randint(0, n - 1)
                a, b = best_points[idx]
                # Random neighbor offset
                da = random.randint(-3, 3)
                db = random.randint(-3, 3)
                na, nb = a + da, b + db
                
                if abs(na) > 64 or abs(nb) > 64:
                    continue
                
                # Check if already in use
                if (na, nb) in [(p[0], p[1]) for p in best_points]:
                    continue
                
                # Check minimum distance constraint
                valid = True
                for j, p in enumerate(best_points):
                    if j == idx:
                        continue
                    if dist_sq([na, nb], p) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute new edge count
                old_edges = 0
                for j in range(n):
                    if j == idx:
                        continue
                    if dist_sq(best_points[idx], best_points[j]) == 1:
                        old_edges += 1
                
                new_edges = 0
                for j in range(n):
                    if j == idx:
                        continue
                    if dist_sq([na, nb], best_points[j]) == 1:
                        new_edges += 1
                
                total_new = best_edges - old_edges + new_edges
                if total_new > best_edges:
                    best_points[idx] = [na, nb]
                    best_edges = total_new
    
    return points
