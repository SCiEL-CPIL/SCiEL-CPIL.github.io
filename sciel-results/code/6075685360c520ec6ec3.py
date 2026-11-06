import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0

    # Distance squared on triangular lattice
    # Points are a + b*omega, omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # |z|^2 = da^2 + db^2 + da*db
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    # Generate hex spiral in proper order
    # The hex spiral starts at center and spirals outward
    # Ring 0: 1 point (center)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # Ring 4: 24 points
    # Total through ring 3: 1+6+12+18 = 37 points (more than 32)
    # Total through ring 2: 1+6+12 = 19 points (less than 32)
    # So we need center + ring1 + ring2 + first 13 of ring3
    
    # Hex spiral directions (6 neighbors in order)
    # Using axial coordinates (a, b) where point = a + b*omega
    # The 6 neighbors of (a,b) are:
    # (a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)
    
    def hex_spiral_order(max_ring):
        """Generate points in hex spiral order from center outward."""
        points = [[0, 0]]
        if max_ring == 0:
            return points
        
        # For each ring r, start at a specific corner and go around
        # Ring r has 6*r points
        # The corners of ring r are at specific positions
        # Corner 1: (r, 0), then go to (0, r), (-r, r), (-r, 0), (0, -r), (r, -r), back to (r, 0)
        # Actually in axial coords with our basis:
        # Let's use the standard hex spiral
        
        for r in range(1, max_ring + 1):
            # Start at corner (r, 0) in axial coords
            # The 6 sides of the hexagon at distance r:
            # Side 1: from (r, 0) to (r, -r) going in direction (0, -1) -- wait, need to be careful
            # 
            # In axial coordinates (q, r) with cube constraint, but we're using (a, b) directly.
            # Let me use a simpler approach: generate all points at distance r, sort by angle, take in order.
            
            ring_points = []
            for a in range(-r, r + 1):
                for b in range(-r, r + 1):
                    if hex_dist([a, b]) == r:
                        ring_points.append([a, b])
            
            # Sort by angle to create spiral
            ring_points.sort(key=lambda p: angle(p))
            points.extend(ring_points)
        
        return points
    
    # Generate spiral up to ring 4 (to have enough points)
    spiral = hex_spiral_order(4)
    
    # Take first 32
    result = [p[:] for p in spiral[:32]]
    
    # Verify edge count to be sure
    edges = count_edges(result)
    
    # If for some reason the spiral doesn't hit 76, do a quick local search.
    if edges < 76:
        # Local search improvement
        point_set = set((p[0], p[1]) for p in result)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(result)):
                if time.time() > deadline:
                    break
                ax, ay = result[i][0], result[i][1]
                # Try moving to nearby empty spots
                for da in range(-2, 3):
                    for db in range(-2, 3):
                        nx, ny = ax + da, ay + db
                        if (nx, ny) == (ax, ay) or (nx, ny) in point_set:
                            continue
                        if abs(nx) > 64 or abs(ny) > 64:
                            continue
                        
                        # Calculate delta
                        old_p = [ax, ay]
                        new_p = [nx, ny]
                        edges_lost = 0
                        edges_gained = 0
                        for j in range(len(result)):
                            if j == i: continue
                            if dist_sq(old_p, result[j]) == 1:
                                edges_lost += 1
                            if dist_sq(new_p, result[j]) == 1:
                                edges_gained += 1
                            
                        delta = edges_gained - edges_lost
                        if delta > 0:
                            result[i] = [nx, ny]
                            point_set.remove((ax, ay))
                            point_set.add((nx, ny))
                            edges += delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return result
