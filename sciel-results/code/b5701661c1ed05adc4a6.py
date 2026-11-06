import time
import math

def construct():
    """
    Constructs 32 points on the triangular lattice maximizing the number of unit-distance edges.
    Uses a hexagonal spiral (BFS from origin) which is known to achieve Harborth's bound.
    For n=32, the optimum is 76 edges.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n_points = 32
    
    # Neighbor offsets for the triangular lattice in axial coordinates (a, b)
    offsets = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, -1),
        (-1, 1)
    ]
    
    # BFS to generate points in order of increasing hex distance from origin
    visited = set()
    visited.add((0, 0))
    points = [(0, 0)]
    queue = [(0, 0)]
    
    while len(points) < n_points:
        next_queue = []
        for a, b in queue:
            for da, db in offsets:
                na, nb = a + da, b + db
                if (na, nb) not in visited:
                    visited.add((na, nb))
                    points.append((na, nb))
                    next_queue.append((na, nb))
                    if len(points) >= n_points:
                        break
            if len(points) >= n_points:
                break
        queue = next_queue
        if time.time() > deadline:
            break
    
    # Verify we have exactly 32 distinct points
    if len(points) != n_points:
        points = [(a, b) for a in range(6) for b in range(6)][:n_points]
    
    # Count unit-distance edges using adjacency map for efficiency
    def count_edges(pts):
        count = 0
        n = len(pts)
        pt_set = set(pts)
        for i in range(n):
            a1, b1 = pts[i]
            for da, db in offsets:
                neighbor = (a1 + da, b1 + db)
                if neighbor in pt_set:
                    count += 1
        return count // 2  # Each edge counted twice
    
    edges = count_edges(points)
    
    # If we already have 76, we're done
    if edges < 76 and time.time() < deadline - 5:
        used = set(points)
        
        # Generate candidate replacement points near the cluster
        max_coord = max(max(abs(a), abs(b)) for a, b in points) + 3
        candidates = []
        for a in range(-max_coord, max_coord + 1):
            for b in range(-max_coord, max_coord + 1):
                if (a, b) not in used and abs(a) <= 64 and abs(b) <= 64:
                    candidates.append((a, b))
        
        # Sort candidates by proximity to existing points (prefer closer ones)
        def dist_to_cluster(pt):
            min_d = float('inf')
            a, b = pt
            for pa, pb in points:
                d = abs(a - pa) + abs(b - pb)
                if d < min_d:
                    min_d = d
            return min_d
        
        candidates.sort(key=dist_to_cluster)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(n_points):
                if time.time() > deadline - 3:
                    break
                old_pt = points[i]
                for new_pt in candidates:
                    if new_pt in used:
                        continue
                    # Try replacing points[i] with new_pt
                    test_pts = points[:]
                    test_pts[i] = new_pt
                    new_edges = count_edges(test_pts)
                    if new_edges > edges:
                        points = test_pts
                        edges = new_edges
                        used.discard(old_pt)
                        used.add(new_pt)
                        # Update candidates list
                        candidates = [c for c in candidates if c != old_pt and c != new_pt]
                        candidates.append(old_pt)
                        improved = True
                        break
        
        # Second pass: try 2-point swaps for further improvement
        if edges < 76 and time.time() < deadline - 2:
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for i in range(n_points):
                    if time.time() > deadline - 1:
                        break
                    for j in range(i + 1, n_points):
                        if time.time() > deadline - 1:
                            break
                        old_i = points[i]
                        old_j = points[j]
                        for new_i in candidates:
                            if new_i in used or new_i == old_i:
                                continue
                            for new_j in candidates:
                                if new_j in used or new_j == old_j or new_j == new_i:
                                    continue
                                # Check distinctness
                                if new_i == new_j:
                                    continue
                                test_pts = points[:]
                                test_pts[i] = new_i
                                test_pts[j] = new_j
                                new_edges = count_edges(test_pts)
                                if new_edges > edges:
                                    points = test_pts
                                    edges = new_edges
                                    used.discard(old_i)
                                    used.discard(old_j)
                                    used.add(new_i)
                                    used.add(new_j)
                                    candidates = [c for c in candidates if c not in (old_i, old_j, new_i, new_j)]
                                    candidates.extend([old_i, old_j])
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
    
    # Convert to list of lists as required
    result = [[a, b] for a, b in points]
    
    return result
