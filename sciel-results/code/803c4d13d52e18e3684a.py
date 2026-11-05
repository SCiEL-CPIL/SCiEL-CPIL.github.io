"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    def lattice_distance_sq(a1, b1, a2, b2):
        """Squared distance between two lattice points in the triangular lattice."""
        # omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
        # Point = a + b*omega = (a + b/2) + i*b*sqrt(3)/2
        # Distance squared = (da + db/2)^2 + (db*sqrt(3)/2)^2
        # = da^2 + da*db + db^2/4 + 3*db^2/4
        # = da^2 + da*db + db^2
        da = a1 - a2
        db = b1 - b2
        return da*da + da*db + db*db
    
    def is_unit_distance(a1, b1, a2, b2):
        return lattice_distance_sq(a1, b1, a2, b2) == 1
    
    def count_edges(points):
        """Count number of unit-distance pairs among the given points."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if is_unit_distance(points[i][0], points[i][1], points[j][0], points[j][1]):
                    count += 1
        return count
    
    def generate_hexagonal_cluster(n_points):
        """Generate a compact hexagonal cluster of n_points points centered at origin."""
        # Generate all points in a bounding box, sort by distance from origin
        # Use a hexagonal-shaped region for better compactness
        candidates = []
        max_r = 10
        for a in range(-max_r, max_r + 1):
            for b in range(-max_r, max_r + 1):
                # Distance in the triangular lattice metric
                # |a + b*omega| where omega = exp(i*pi/3)
                # |a + b/2 + i*b*sqrt(3)/2|^2 = (a+b/2)^2 + (b*sqrt(3)/2)^2
                # = a^2 + a*b + b^2/4 + 3b^2/4 = a^2 + a*b + b^2
                dist_sq = a*a + a*b + b*b
                candidates.append((dist_sq, a, b))
        
        # Sort by distance, with tie-breaking for compactness
        # For ties, prefer points that fill in the hexagonal shape
        candidates.sort(key=lambda x: (x[0], abs(x[1]), abs(x[2]), x[1], x[2]))
        
        # Take the first n_points
        cluster = [(a, b) for _, a, b in candidates[:n_points]]
        return cluster
    
    def local_optimization(points, max_iterations=10000):
        """Try to improve edge count by swapping points."""
        points = list(points)
        n = len(points)
        current_edges = count_edges(points)
        point_set = set(tuple(p) for p in points)
        
        # Generate candidate replacement points
        max_coord = 64
        all_candidates = []
        for a in range(-10, 11):
            for b in range(-10, 11):
                if (a, b) not in point_set:
                    all_candidates.append((a, b))
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > DEADLINE - 5.0:
                break
            
            for i in range(n):
                if time.time() > DEADLINE - 5.0:
                    break
                old_point = tuple(points[i])
                for new_point in all_candidates:
                    if new_point == old_point:
                        continue
                    # Try the swap
                    points[i] = list(new_point)
                    new_edges = count_edges(points)
                    if new_edges > current_edges:
                        current_edges = new_edges
                        point_set.remove(old_point)
                        point_set.add(tuple(new_point))
                        improved = True
                        break
                    else:
                        points[i] = list(old_point)
        
        return points, current_edges
    
    # Step 1: Generate initial hexagonal cluster
    points = generate_hexagonal_cluster(32)
    current_edges = count_edges(points)
    
    # Step 2: Local optimization
    if time.time() < DEADLINE - 10.0:
        points, current_edges = local_optimization(points)
    
    # Step 3: Try multiple random restarts if time permits
    best_points = points
    best_edges = current_edges
    
    random.seed(42)
    attempt = 0
    while time.time() < DEADLINE - 15.0 and attempt < 50:
        attempt += 1
        # Generate a perturbed cluster
        base = generate_hexagonal_cluster(32)
        # Randomly swap a few points
        pts = list(base)
        for _ in range(random.randint(1, 5)):
            i = random.randint(0, 31)
            a = random.randint(-8, 8)
            b = random.randint(-8, 8)
            if (a, b) not in [tuple(p) for p in pts]:
                pts[i] = [a, b]
        
        pts, edges = local_optimization(pts, max_iterations=5000)
        if edges > best_edges:
            best_points = pts
            best_edges = edges
    
    # Verify constraints
    result = [[p[0], p[1]] for p in best_points]
    
    # Check distinctness
    assert len(set(tuple(p) for p in result)) == 32, "Points must be distinct"
    # Check bounds
    for a, b in result:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    # Check minimum distance
    for i in range(32):
        for j in range(i+1, 32):
            d2 = lattice_distance_sq(result[i][0], result[i][1], result[j][0], result[j][1])
            assert d2 >= 1, f"Points {i} and {j} too close: d2={d2}"
    
    return result

# EVOLVE-BLOCK-END
