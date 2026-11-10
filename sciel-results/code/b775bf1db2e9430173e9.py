"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 60  # Leave 60s buffer for evaluation overhead/safety
    
    best_score_estimate = -1
    best_points = None
    
    # Helper: Check if 4 points are collinear
    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        # Check collinearity using cross product
                        # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
                        def cross(a, b, c):
                            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
                        if cross(p1, p2, p3) == 0 and cross(p1, p2, p4) == 0:
                            return True
        return False

    # Helper: Count number of collinear triples (higher is better for minimizing independence number)
    def count_collinear_triples(points):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    def cross(a, b, c):
                        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
                    if cross(p1, p2, p3) == 0:
                        count += 1
        return count

    # Generate initial random valid set
    def generate_random_valid():
        while True:
            pts = []
            seen = set()
            for _ in range(20):
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if (x, y) not in seen:
                    seen.add((x, y))
                    pts.append([x, y])
            if not has_four_collinear(pts):
                return pts

    # Local Search: Swap a point with a random new point to improve triple count
    def local_search(points, time_limit):
        current = [p[:] for p in points]
        current_triples = count_collinear_triples(current)
        best_local = [p[:] for p in current]
        best_local_triples = current_triples
        
        while time.time() < time_limit:
            # Pick a random point to move
            idx = random.randint(0, 19)
            old_point = current[idx][:]
            
            # Try a few random moves
            improved = False
            for _ in range(10):
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
                if (new_x, new_y) in [tuple(p) for p in current if p != old_point]:
                    continue
                
                candidate = [p[:] for p in current]
                candidate[idx] = [new_x, new_y]
                
                if has_four_collinear(candidate):
                    continue
                
                new_triples = count_collinear_triples(candidate)
                if new_triples > current_triples:
                    current = candidate
                    current_triples = new_triples
                    improved = True
                    if current_triples > best_local_triples:
                        best_local = [p[:] for p in current]
                        best_local_triples = current_triples
                    break # Accept first improvement and restart loop
            
            if not improved:
                # Stagnation: restart or accept small penalty
                # Simple restart: randomize one point more aggressively
                idx = random.randint(0, 19)
                current[idx] = [random.randint(-50, 50), random.randint(-50, 50)]
                if has_four_collinear(current):
                    # Revert if invalid
                    current[idx] = old_point
                    # Try to escape local minimum by randomizing multiple points
                    for _ in range(5):
                        idx2 = random.randint(0, 19)
                        current[idx2] = [random.randint(-50, 50), random.randint(-50, 50)]
                    if has_four_collinear(current):
                        break # Failed to escape, stop local search

        return best_local, best_local_triples

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        init_pts = generate_random_valid()
        init_triples = count_collinear_triples(init_pts)
        
        if init_triples > best_score_estimate:
            best_score_estimate = init_triples
            best_points = [p[:] for p in init_pts]
        
        # Do local search from this start
        search_limit = min(deadline, time.time() + 10) # 10s per local search
        local_best, local_triples = local_search(init_pts, search_limit)
        
        if local_triples > best_score_estimate:
            best_score_estimate = local_triples
            best_points = local_best

    # Fallback: If no valid points found (unlikely), return a simple grid-like perturbation
    if best_points is None:
        # Simple deterministic fallback: 4x5 grid with slight perturbation to break 4-collinearity
        # But 4x5 grid has 4 in a column, which is invalid.
        # Use a known valid set: points on y=x^2 is valid but score 20.
        # Let's just return the parabola if search fails completely.
        return [[x, x * x] for x in range(-10, 10)]

    return best_points
# EVOLVE-BLOCK-END
