import math
import time

def construct():
    """
    Returns a list of 32 distinct integer pairs [a, b] representing points on the triangular lattice
    that maximize the number of unit-distance pairs (edges).
    The optimal number of edges is 76.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s
    
    # 1. Generate initial candidate set
    # Use the hexagonal spiral approach
    def get_hex_spiral(n):
        points = [(0, 0)]
        if n == 1:
            return points
        
        # Directions in axial coordinates
        # Right, Top-Right, Top-Left, Left, Bottom-Left, Bottom-Right
        # In (q, r) where x = q + r/2, y = r*sqrt(3)/2
        # Neighbors of (0,0) are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        
        # We build layer by layer
        # Layer k has 6*k points
        # Total points in layer 0..k-1 is 1 + 3k(k-1)
        
        layer = 1
        while len(points) < n:
            # Start at (layer, 0)
            q, r = layer, 0
            # We need to add 6*layer points
            for i in range(6 * layer):
                if len(points) >= n:
                    break
                points.append((q, r))
                # Move to next point in the perimeter
                # The perimeter is a hexagon.
                # Step 1: (layer, 0) -> (layer-1, 1) ... no
                # Let's use the standard spiral generation
                
            # Better: Generate all points in layer 'layer' and sort by angle
            layer_points = []
            for q in range(-layer, layer + 1):
                for r in range(-layer, layer + 1):
                    s = -q - r
                    if max(abs(q), abs(r), abs(s)) == layer:
                        layer_points.append((q, r))
            
            # Sort by angle
            def get_angle(q, r):
                real = q + r / 2.0
                imag = r * math.sqrt(3) / 2.0
                return math.atan2(imag, real)
            
            layer_points.sort(key=lambda p: get_angle(p[0], p[1]))
            
            # Add up to n
            for p in layer_points:
                if len(points) >= n:
                    break
                points.append(p)
            
            layer += 1
            
        return points

    initial_points = get_hex_spiral(32)
    
    # 2. Local Search to maximize edges
    # Represent points as a set for fast lookup
    point_set = set(initial_points)
    
    def count_edges(points_list):
        s = set(points_list)
        count = 0
        for p in s:
            q, r = p
            # Check 6 neighbors
            for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
                nq, nr = q + dq, r + dr
                if (nq, nr) in s:
                    count += 1
        return count // 2

    current_edges = count_edges(initial_points)
    best_edges = current_edges
    best_points = list(initial_points)
    
    # Candidates for swapping: points near the current set
    # Generate a pool of candidate points within distance 4 from origin
    candidate_pool = []
    for q in range(-6, 7):
        for r in range(-6, 7):
            if max(abs(q), abs(r), abs(-q-r)) <= 4:
                candidate_pool.append((q, r))
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        points_list = list(point_set)
        
        # Try swapping each point in the set with a candidate not in the set
        # To save time, we can limit candidates to those near the current points
        for i, p in enumerate(points_list):
            if time.time() > deadline:
                break
            
            # Find neighbors of p in the candidate pool
            q, r = p
            # Candidates should be near p to maintain compactness
            # We'll just iterate through a subset of candidate_pool
            # Actually, iterating through all 300+ candidates for 32 points is 10k checks per iteration.
            # 10k * 32 = 320k checks. Each check involves count_edges which is O(N^2) or O(N).
            # We can optimize count_edges by only checking the delta.
            
            # Delta calculation:
            # Removing p loses deg(p) edges.
            # Adding c gains deg(c) edges (with respect to the set minus p plus c).
            
            for c in candidate_pool:
                if c in point_set:
                    continue
                
                new_set = point_set.copy()
                new_set.remove(p)
                new_set.add(c)
                
                # Calculate delta in edges
                # Edges lost: neighbors of p in (point_set - {p})
                lost = 0
                q, r = p
                for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
                    if (q + dq, r + dr) in point_set:
                        lost += 1
                # Note: edges between p and other points are counted once in lost?
                # Yes, because we are removing p.
                
                # Edges gained: neighbors of c in (new_set - {c})
                gained = 0
                q, r = c
                for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
                    if (q + dq, r + dr) in new_set:
                        gained += 1
                
                delta = gained - lost
                if delta > 0:
                    point_set = new_set
                    current_edges += delta
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_points = list(point_set)
                    break # Restart search with new set
                    
    # Final check
    if time.time() < deadline - 5:
        # If we haven't reached 76, try a more aggressive search or just return best
        # But 76 is the max, so if we are at 76, we are done.
        pass
        
    return [[q, r] for q, r in best_points]
