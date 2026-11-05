import time
import random

def construct():
    deadline = time.time() + 135
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + db * db + da * db
    
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count
    
    def is_valid(points):
        """Check all pairs have dist_sq >= 1"""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) < 1:
                    return False
        return True
    
    # Hexagon of radius R: |a|<=R, |b|<=R, |a+b|<=R
    def hex_points(R):
        pts = []
        for a in range(-R, R + 1):
            for b in range(-R, R + 1):
                if abs(a + b) <= R:
                    pts.append([a, b])
        return pts
    
    # Start with hexagon radius 3 (37 points), remove 5 to get 32
    # Remove corner points first (they have fewer edges)
    hex3 = hex_points(3)
    # Corners of hex radius 3: (3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3)
    # Actually corners are where two boundaries meet
    # Let's identify by degree
    def degree_in_set(pt, pts):
        count = 0
        for q in pts:
            if q != pt and dist_sq(pt, q) == 1:
                count += 1
        return count
    
    # Compute degrees and sort
    hex3_set = set(tuple(p) for p in hex3)
    degrees = []
    for p in hex3:
        d = sum(1 for q in hex3 if q != p and dist_sq(p, q) == 1)
        degrees.append((d, p))
    degrees.sort()  # lowest degree first
    
    # Remove 5 lowest-degree points
    to_remove = set()
    for d, p in degrees[:5]:
        to_remove.add(tuple(p))
    
    points = [p for p in hex3 if tuple(p) not in to_remove]
    
    best_edges = count_edges(points)
    best_points = [p[:] for p in points]
    
    # Local search
    # Try moving each point to nearby positions
    max_move_radius = 4
    neighbor_offsets = []
    for da in range(-max_move_radius, max_move_radius + 1):
        for db in range(-max_move_radius, max_move_radius + 1):
            if da == 0 and db == 0:
                continue
            neighbor_offsets.append((da, db))
    
    random.seed(42)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(neighbor_offsets)
        for i in range(len(points)):
            if time.time() > deadline:
                break
            orig = points[i][:]
            orig_edges = best_edges
            # Compute edges involving point i
            for da, db in neighbor_offsets:
                new_pos = [orig[0] + da, orig[1] + db]
                if abs(new_pos[0]) > 64 or abs(new_pos[1]) > 64:
                    continue
                # Check no collision
                if any(tuple(new_pos) == tuple(p) for p in points if p is not points[i]):
                    continue
                # Check distance constraint with all other points
                valid = True
                for j in range(len(points)):
                    if j == i:
                        continue
                    if dist_sq(new_pos, points[j]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                # Compute new edge count
                new_edges = 0
                for j in range(len(points)):
                    if j == i:
                        continue
                    if dist_sq(new_pos, points[j]) == 1:
                        new_edges += 1
                # Edges not involving i stay the same
                edges_not_i = 0
                for j in range(len(points)):
                    for k in range(j + 1, len(points)):
                        if j == i or k == i:
                            continue
                        if dist_sq(points[j], points[k]) == 1:
                            edges_not_i += 1
                total_new = new_edges + edges_not_i
                if total_new > best_edges:
                    points[i] = new_pos
                    best_edges = total_new
                    best_points = [p[:] for p in points]
                    improved = True
                    break  # restart from this point
    
    # Try swaps
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            i, j = random.sample(range(len(points)), 2)
            orig_i = points[i][:]
            orig_j = points[j][:]
            # Swap
            points[i], points[j] = orig_j[:], orig_i[:]
            if is_valid(points):
                new_e = count_edges(points)
                if new_e > best_edges:
                    best_edges = new_e
                    best_points = [p[:] for p in points]
                else:
                    points[i], points[j] = orig_i[:], orig_j[:]
            else:
                points[i], points[j] = orig_i[:], orig_j[:]
    
    return best_points
