"""Improved program: hexagonal construction + local search for 32 points on triangular lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _dist2(a1, b1, a2, b2):
    """Squared distance on triangular lattice."""
    da = a1 - a2
    db = b1 - b2
    return da*da + da*db + db*db

def _count_edges(points):
    """Count unit-distance pairs among given (a,b) points."""
    point_set = set(tuple(p) for p in points)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    edge_count = 0
    for a, b in point_set:
        for da, db in neighbors:
            if (a+da, b+db) in point_set:
                edge_count += 1
    return edge_count // 2

def _hexagon_points(r):
    """All points with hex distance <= r from origin: max(|a|,|b|,|a+b|) <= r."""
    pts = []
    for a in range(-r, r+1):
        for b in range(-r, r+1):
            if max(abs(a), abs(b), abs(a+b)) <= r:
                pts.append((a, b))
    return pts

def _spiral_points(n):
    """Generate n points in a hexagonal spiral pattern starting from origin."""
    if n == 0:
        return []
    points = [(0, 0)]
    # Spiral directions: 6 directions, each ring adds 6*k points
    # Ring 0: 1 point (center)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # etc.
    ring = 1
    while len(points) < n:
        # Generate all points in this ring
        # The ring consists of 6 sides, each of length 'ring'
        # Start at (ring, 0) and go around
        # Direction vectors for the 6 sides
        dirs = [
            (0, 1),    # up
            (-1, 1),   # up-left
            (-1, 0),   # left
            (0, -1),   # down
            (1, -1),   # down-right
            (1, 0),    # right
        ]
        # Start point of this ring
        pos = (ring, 0)
        for d in range(6):
            da, db = dirs[d]
            for step in range(ring):
                points.append(pos)
                pos = (pos[0] + da, pos[1] + db)
        ring += 1
    return points[:n]

def _local_search(points, max_iters=50000, deadline=None):
    """Local search to improve edge count."""
    points = [list(p) for p in points]
    n = len(points)
    current_edges = _count_edges(points)
    
    # Precompute neighbor offsets
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    best_points = [list(p) for p in points]
    best_edges = current_edges
    
    start_time = time.time()
    iteration = 0
    
    while iteration < max_iters:
        if deadline and time.time() > deadline:
            break
        
        # Randomly pick a point to move
        i = random.randint(0, n-1)
        orig_a, orig_b = points[i][0], points[i][1]
        
        # Try moving to nearby positions
        best_move = None
        best_delta = 0
        
        # Generate candidate positions in a small neighborhood
        for da in range(-3, 4):
            for db in range(-3, 4):
                new_a = orig_a + da
                new_b = orig_b + db
                if abs(new_a) > 64 or abs(new_b) > 64:
                    continue
                
                # Check if position is already occupied
                occupied = False
                for j in range(n):
                    if j != i and points[j][0] == new_a and points[j][1] == new_b:
                        occupied = True
                        break
                if occupied:
                    continue
                
                # Check minimum distance constraint
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    d2 = _dist2(new_a, new_b, points[j][0], points[j][1])
                    if d2 < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Count edges for this move
                edges = 0
                for j in range(n):
                    if j == i:
                        continue
                    d2 = _dist2(new_a, new_b, points[j][0], points[j][1])
                    if d2 == 1:
                        edges += 1
                
                # Remove old edges from point i
                old_edges = 0
                for j in range(n):
                    if j == i:
                        continue
                    d2 = _dist2(orig_a, orig_b, points[j][0], points[j][1])
                    if d2 == 1:
                        old_edges += 1
                
                delta = edges - old_edges
                if delta > best_delta:
                    best_delta = delta
                    best_move = (new_a, new_b)
        
        if best_move:
            points[i][0] = best_move[0]
            points[i][1] = best_move[1]
            current_edges += best_delta
            if current_edges > best_edges:
                best_edges = current_edges
                best_points = [list(p) for p in points]
        
        iteration += 1
    
    return best_points, best_edges

def construct():
    """Construct 32 points on triangular lattice maximizing unit-distance pairs."""
    start_time = time.time()
    deadline = start_time + 120  # Leave margin before 150s kill
    
    # Start with spiral construction
    points = _spiral_points(32)
    current_edges = _count_edges(points)
    
    best_points = [list(p) for p in points]
    best_edges = current_edges
    
    # Try multiple random restarts with local search
    random.seed(42)
    restart_count = 0
    while time.time() < deadline - 5:
        # Create a perturbed version of the best solution
        perturbed = [list(p) for p in best_points]
        
        # Apply random perturbations
        for i in range(len(perturbed)):
            if random.random() < 0.3:
                da = random.randint(-2, 2)
                db = random.randint(-2, 2)
                new_a = perturbed[i][0] + da
                new_b = perturbed[i][1] + db
                if abs(new_a) <= 64 and abs(new_b) <= 64:
                    # Check validity
                    valid = True
                    for j in range(len(perturbed)):
                        if j == i:
                            continue
                        d2 = _dist2(new_a, new_b, perturbed[j][0], perturbed[j][1])
                        if d2 < 1:
                            valid = False
                            break
                    if valid:
                        # Check not occupied
                        occupied = False
                        for j in range(len(perturbed)):
                            if j != i and perturbed[j][0] == new_a and perturbed[j][1] == new_b:
                                occupied = True
                                break
                        if not occupied:
                            perturbed[i][0] = new_a
                            perturbed[i][1] = new_b
        
        # Run local search
        result_points, result_edges = _local_search(perturbed, max_iters=10000, deadline=deadline)
        
        if result_edges > best_edges:
            best_edges = result_edges
            best_points = result_points
        
        restart_count += 1
    
    # Final check
    final_edges = _count_edges(best_points)
    if final_edges > best_edges:
        best_edges = final_edges
    
    # Return as list of [a, b] pairs
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
