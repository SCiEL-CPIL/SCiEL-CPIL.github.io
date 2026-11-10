"""Search for 6-point rational-distance configurations in integer lattice."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Fallback: known 5-point configuration
    fallback = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def elapsed():
        return time.time() - start_time
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n
    
    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        # Cross product: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0
    
    def check_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        # Determinant of 4x4 matrix for concyclicity check
        # | x^2+y^2  x  y  1 |
        # All rows, determinant should be != 0 for no four concyclic
        a1 = x1*x1 + y1*y1
        a2 = x2*x2 + y2*y2
        a3 = x3*x3 + y3*y3
        a4 = x4*x4 + y4*y4
        # Compute determinant using expansion
        # Use 4x4 determinant formula
        def det4(m):
            return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                   - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                   + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                   - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
        m = [[a1, x1, y1, 1],
             [a2, x2, y2, 1],
             [a3, x3, y3, 1],
             [a4, x4, y4, 1]]
        return det4(m) == 0
    
    def validate(points):
        """Check no three collinear and no four concyclic."""
        n = len(points)
        # Check no three collinear
        for i, j, k in combinations(range(n), 3):
            if check_collinear(points[i], points[j], points[k]):
                return False
        # Check no four concyclic
        for i, j, k, l in combinations(range(n), 4):
            if check_concyclic(points[i], points[j], points[k], points[l]):
                return False
        return True
    
    # Search for 6-point clique in integer lattice
    # Try increasing radii
    for radius in range(30, 80):
        if elapsed() > deadline - 30:
            break
        
        # Generate lattice points
        points = []
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if x*x + y*y <= radius*radius:
                    points.append((x, y))
        
        n_pts = len(points)
        
        # Build adjacency: edge if squared distance is a perfect square
        adj = [set() for _ in range(n_pts)]
        for i in range(n_pts):
            xi, yi = points[i]
            for j in range(i + 1, n_pts):
                xj, yj = points[j]
                dx = xi - xj
                dy = yi - yj
                d2 = dx*dx + dy*dy
                if is_square(d2):
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Find 6-clique using backtracking
        # Order vertices by degree (descending) for efficiency
        deg_order = sorted(range(n_pts), key=lambda i: -len(adj[i]))
        
        found_6 = None
        
        def find_clique(candidates, clique, target_size):
            nonlocal found_6
            if len(clique) == target_size:
                found_6 = list(clique)
                return True
            if not candidates:
                return False
            if len(clique) + len(candidates) < target_size:
                return False
            
            # Try each candidate
            for idx in range(len(candidates) - 1, -1, -1):
                v = candidates[idx]
                new_clique = clique + [v]
                new_candidates = [c for c in candidates[:idx] if c in adj[v]]
                if find_clique(new_candidates, new_clique, target_size):
                    return True
            return False
        
        # Use a subset of high-degree vertices to limit search
        # Try to find 6-clique among top-degree vertices
        top_k = min(200, n_pts)
        search_set = deg_order[:top_k]
        search_set_set = set(search_set)
        
        # Filter candidates to search_set
        candidates = [v for v in search_set]
        
        if find_clique(candidates, [], 6):
            if found_6:
                six_pts = [points[i] for i in found_6]
                if validate(six_pts):
                    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in six_pts]}
        
        # If no 6-clique found, try 5-clique as fallback improvement
        if elapsed() > deadline - 20:
            break
    
    # Return fallback 5-point configuration
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in fallback]}
# EVOLVE-BLOCK-END
