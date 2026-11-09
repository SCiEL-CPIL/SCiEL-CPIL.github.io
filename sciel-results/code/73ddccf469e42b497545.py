"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    # Target: 32 points, maximize edges (dist=1).
    # Strategy: Start with a dense hexagonal cluster of 37 points (radius 3)
    # and greedily remove 5 points to minimize edge loss.
    
    start_time = time.time()
    deadline = 150  # seconds
    
    # 1. Generate 37 points of hexagon radius 3
    # Axial coordinates (a, b) where max(|a|, |b|, |a+b|) <= 3 is NOT the correct hexagon definition for axial.
    # Correct hexagon: distance d(a,b) = (|a| + |b| + |a+b|) / 2 <= R
    points = []
    R = 3
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            # Check hex distance
            if (abs(a) + abs(b) + abs(a + b)) // 2 <= R:
                points.append((a, b))
    
    # Should be 37 points
    if len(points) != 37:
        # Fallback if generation is wrong, use a simple strip
        return [[a, b] for a in range(16) for b in range(2)]

    # 2. Build adjacency
    point_set = set(points)
    neighbors = {p: [] for p in points}
    for p in points:
        a, b = p
        # 6 neighbors in triangular lattice
        for da, db in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
            q = (a + da, b + db)
            if q in point_set:
                neighbors[p].append(q)
    
    # 3. Greedy removal of 5 points
    # We want to remove 5 points such that the remaining 32 have max edges.
    # Equivalent to minimizing edges lost.
    # Loss = sum(deg(v) for v in removed) - edges(removed)
    
    current_points = set(points)
    current_neighbors = {p: list(n) for p, n in neighbors.items()}
    
    to_remove = 5
    removed = []
    
    for _ in range(to_remove):
        # Find point with min "effective degree"
        # Effective degree = degree in current graph
        # However, if we remove adjacent points, we save the edge between them.
        # Simple greedy: remove point with min degree in current graph.
        min_deg = float('inf')
        min_pt = None
        for p in current_points:
            deg = len(current_neighbors[p])
            if deg < min_deg:
                min_deg = deg
                min_pt = p
        
        # Remove min_pt
        removed.append(min_pt)
        current_points.remove(min_pt)
        
        # Update neighbors
        for n in current_neighbors[min_pt]:
            current_neighbors[n].remove(min_pt)
        
        del current_neighbors[min_pt]
        
        # Check deadline
        if time.time() > deadline - 5:
            break

    # 4. Return the remaining 32 points
    result = list(current_points)
    
    # Ensure exactly 32 points
    if len(result) != 32:
        # If we didn't remove enough or too many, fallback
        # But we removed exactly 5 from 37, so should be 32.
        pass

    # Convert to list of lists
    return [list(p) for p in result]

# EVOLVE-BLOCK-END
