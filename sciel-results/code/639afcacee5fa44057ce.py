"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Generate a hexagonal/spiral arrangement of 32 points on triangular lattice
    # The triangular lattice has neighbors at offsets: (1,0), (0,1), (1,-1), (-1,0), (0,-1), (-1,1)
    # in (a,b) coordinates where point = a + b*omega
    
    # Build a hexagonal cluster: rings around origin
    # Ring 0: (0,0) - 1 point
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # Total: 1+6+12+18 = 37, we need 32
    
    # Let's build rings and take first 32
    points = []
    # Ring 0
    points.append((0, 0))
    
    # Ring r has 6*r points
    def ring_points(r):
        if r == 0:
            return [(0, 0)]
        pts = []
        # Start at (r, 0), go around
        # Directions in (a,b): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        pos = (r, 0)
        for d in dirs:
            for _ in range(r):
                pts.append(pos)
                pos = (pos[0] + d[0], pos[1] + d[1])
        return pts
    
    for r in range(1, 5):
        ring = ring_points(r)
        for p in ring:
            if len(points) < 32:
                points.append(p)
            else:
                break
        if len(points) >= 32:
            break
    
    points = points[:32]
    
    # Helper: squared distance between two lattice points
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        # |a + b*omega|^2 = a^2 + a*b + b^2 for omega = exp(i*pi/3)
        # Actually: (a + b*omega) - (a' + b'*omega) = da + db*omega
        # |da + db*omega|^2 = da^2 + da*db + db^2
        return da*da + da*db + db*db
    
    # Count edges (pairs at distance exactly 1, i.e., dist_sq == 1)
    def count_edges(pts):
        edges = 0
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(pts[i], pts[j]) == 1:
                    edges += 1
        return edges
    
    # Check validity: all distinct, all at mutual distance >= 1 (dist_sq >= 1)
    def is_valid(pts):
        if len(pts) != 32:
            return False
        seen = set()
        for p in pts:
            if p in seen:
                return False
            seen.add(p)
            if abs(p[0]) > 64 or abs(p[1]) > 64:
                return False
        for i in range(32):
            for j in range(i+1, 32):
                if dist_sq(pts[i], pts[j]) < 1:
                    return False
        return True
    
    # Local search: try swapping a point with a candidate
    def local_search(pts, max_time):
        pts = list(pts)
        best_edges = count_edges(pts)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try moving each point to a nearby empty position
            for i in range(32):
                if time.time() > max_time:
                    break
                old = pts[i]
                # Try all 6 neighbors of old, and also some random positions
                candidates = []
                # 6 neighbors
                for da, db in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                    cand = (old[0]+da, old[1]+db)
                    if abs(cand[0]) <= 64 and abs(cand[1]) <= 64:
                        candidates.append(cand)
                # Also try some positions further out
                for _ in range(5):
                    da = random.randint(-3, 3)
                    db = random.randint(-3, 3)
                    cand = (old[0]+da, old[1]+db)
                    if abs(cand[0]) <= 64 and abs(cand[1]) <= 64:
                        candidates.append(cand)
                
                for cand in candidates:
                    if cand in pts:
                        continue
                    # Check if cand is at distance >= 1 from all other points
                    valid = True
                    for j in range(32):
                        if j == i:
                            continue
                        if dist_sq(cand, pts[j]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Compute new edge count
                    # Remove edges involving old, add edges involving cand
                    old_edges = 0
                    for j in range(32):
                        if j == i:
                            continue
                        if dist_sq(old, pts[j]) == 1:
                            old_edges += 1
                    
                    new_edges = 0
                    for j in range(32):
                        if j == i:
                            continue
                        if dist_sq(cand, pts[j]) == 1:
                            new_edges += 1
                    
                    delta = new_edges - old_edges
                    if delta > 0:
                        pts[i] = cand
                        best_edges += delta
                        improved = True
                        break  # restart scan from beginning
        return pts, best_edges
    
    # Run local search
    max_time = min(deadline - 2.0, start_time + 130.0)
    result, best_edges = local_search(points, max_time)
    
    # Verify result
    if not is_valid(result):
        # Fallback to simple strip
        return [[a, b] for a in range(16) for b in range(2)]
    
    return [[a, b] for a, b in result]
# EVOLVE-BLOCK-END
