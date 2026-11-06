import random
import math
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    # Lattice neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db

    def count_edges(points):
        edges = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges

    # Deterministic Spiral Construction
    # Generate points in order of distance from center, then angle.
    # We need 32 points.
    
    points = []
    # Add center
    points.append((0,0))
    
    # Add rings
    # Ring r has 6r points.
    # We need 31 more points.
    # Ring 1: 6 pts. Total 7.
    # Ring 2: 12 pts. Total 19.
    # Ring 3: 18 pts. We need 13 more. Total 32.
    
    # Generate Ring 1
    for step in range(6):
        # Direction vectors for the 6 sides of the hexagon
        # Side 1: (1,0) to (1,-1)? 
        # Let's just iterate through the perimeter of the hexagon of radius r.
        # Corners: (r,0), (0,r), (-r,r), (-r,0), (0,-r), (r,-r)
        # We can generate the perimeter points.
        pass # We'll do this in a helper or inline
        
    # Simpler: Generate all points in hexagon R=3, sort by distance, then take first 32?
    # Distance from center in lattice metric: max(|a|, |b|, |a+b|)
    # If we sort by this distance, we get rings.
    # Within a ring, order matters for edge count of the subset.
    # If we take the first 13 of Ring 3 in spiral order, we get a contiguous arc.
    
    hex_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) <= 3:
                hex_points.append((a,b))
                
    # Sort by distance from origin
    def lattice_dist(p):
        return max(abs(p[0]), abs(p[1]), abs(p[0]+p[1]))
        
    # Secondary sort: angle or spiral order
    # To get a contiguous arc, we can sort by angle?
    # Or just take the points in the order they appear in a spiral traversal.
    
    # Let's generate spiral order explicitly.
    # Start at (0,0).
    # Move to (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -- this is Ring 1?
    # No, (1,0) is dist 1. (0,1) is dist 1.
    # The 6 neighbors of (0,0) are Ring 1.
    # Then Ring 2.
    
    # Algorithm for spiral:
    # 1. (0,0)
    # 2. (1,0)
    # 3. (1,-1)
    # 4. (0,-1)
    # 5. (-1,0)
    # 6. (-1,1)
    # 7. (0,1)
    # This is Ring 1 (6 pts).
    # Then Ring 2 starts from (2,0)?
    # Actually, standard hex spiral:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-2), (-2,-2)? No.
    
    # Let's just use the sorted list and break ties by a deterministic angle.
    # Angle = atan2(b + a/2, a) ?
    # In triangular lattice, complex number is a + b*exp(i*pi/3).
    # x = a + b/2, y = b*sqrt(3)/2.
    # Angle = atan2(y, x).
    
    def get_angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3)/2.0
        return math.atan2(y, x)
        
    # Sort by distance, then angle
    hex_points.sort(key=lambda p: (lattice_dist(p), get_angle(p)))
    
    # Take first 32
    initial_points = hex_points[:32]
    
    # Local Search to ensure optimality
    best_points = initial_points[:]
    best_edges = count_edges(best_points)
    
    # Helper: check if a candidate point is valid (no collision with others)
    def is_valid_move(new_p, exclude_idx, pts):
        if abs(new_p[0]) > 64 or abs(new_p[1]) > 64:
            return False
        for j in range(32):
            if j == exclude_idx:
                continue
            if dist_sq(new_p, pts[j]) < 1:
                return False
        return True
    
    # Helper: count edges incident to point at index i
    def edges_at(i, pts):
        cnt = 0
        for j in range(32):
            if j == i:
                continue
            if dist_sq(pts[i], pts[j]) == 1:
                cnt += 1
        return cnt
    
    # Phase 1: Single-move local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges = count_edges(best_points)
        for i in range(32):
            if time.time() >= deadline:
                break
            a, b = best_points[i]
            for da, db in neighbors:
                new_p = (a + da, b + db)
                if not is_valid_move(new_p, i, best_points):
                    continue
                old_e = edges_at(i, best_points)
                new_e = 0
                for j in range(32):
                    if j == i:
                        continue
                    if dist_sq(new_p, best_points[j]) == 1:
                        new_e += 1
                delta = new_e - old_e
                if delta > 0:
                    best_points[i] = new_p
                    current_edges += delta
                    improved = True
                    break
            if improved:
                break
        if current_edges > best_edges:
            best_edges = current_edges
    
    # Phase 2: Two-point swap local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges = count_edges(best_points)
        for i in range(31):
            if time.time() >= deadline:
                break
            for j in range(i + 1, 32):
                if time.time() >= deadline:
                    break
                ai, bi = best_points[i]
                aj, bj = best_points[j]
                # Try moving i to a neighbor of j, and j to a neighbor of i
                # First: swap positions
                # Check if swapping is valid
                if is_valid_move((aj, bj), i, best_points) and is_valid_move((ai, bi), j, best_points):
                    # Compute edge change for swap
                    old_ei = edges_at(i, best_points)
                    old_ej = edges_at(j, best_points)
                    # After swap, point i is at (aj,bj) and point j is at (ai,bi)
                    new_ei = 0
                    for k in range(32):
                        if k == i:
                            continue
                        if k == j:
                            # j is now at (ai,bi)
                            if dist_sq((aj, bj), (ai, bi)) == 1:
                                new_ei += 1
                        else:
                            if dist_sq((aj, bj), best_points[k]) == 1:
                                new_ei += 1
                    new_ej = 0
                    for k in range(32):
                        if k == j:
                            continue
                        if k == i:
                            # i is now at (aj,bj)
                            if dist_sq((ai, bi), (aj, bj)) == 1:
                                new_ej += 1
                        else:
                            if dist_sq((ai, bi), best_points[k]) == 1:
                                new_ej += 1
                    delta = (new_ei + new_ej) - (old_ei + old_ej)
                    if delta > 0:
                        best_points[i] = (aj, bj)
                        best_points[j] = (ai, bi)
                        current_edges += delta
                        improved = True
                        break
                # Also try: move i to a neighbor of its current pos, and j to a neighbor of its current pos
                # (independent moves, but both in one step)
                for da_i, db_i in neighbors:
                    if time.time() >= deadline:
                        break
                    ni = (ai + da_i, bi + db_i)
                    if not is_valid_move(ni, i, best_points):
                        continue
                    for da_j, db_j in neighbors:
                        nj = (aj + da_j, bj + db_j)
                        if not is_valid_move(nj, j, best_points):
                            continue
                        # Check that ni and nj don't collide with each other
                        if dist_sq(ni, nj) < 1:
                            continue
                        old_ei = edges_at(i, best_points)
                        old_ej = edges_at(j, best_points)
                        new_ei = 0
                        for k in range(32):
                            if k == i:
                                continue
                            if k == j:
                                if dist_sq(ni, nj) == 1:
                                    new_ei += 1
                            else:
                                if dist_sq(ni, best_points[k]) == 1:
                                    new_ei += 1
                        new_ej = 0
                        for k in range(32):
                            if k == j:
                                continue
                            if k == i:
                                if dist_sq(nj, ni) == 1:
                                    new_ej += 1
                            else:
                                if dist_sq(nj, best_points[k]) == 1:
                                    new_ej += 1
                        delta = (new_ei + new_ej) - (old_ei + old_ej)
                        if delta > 0:
                            best_points[i] = ni
                            best_points[j] = nj
                            current_edges += delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if current_edges > best_edges:
            best_edges = current_edges
    
    return [[a, b] for a, b in best_points]
