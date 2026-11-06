import time
import math

def construct():
    """
    Construct 32 points on the triangular lattice maximizing unit-distance pairs.
    Optimum is 76 edges (Harborth's theorem for n=32).
    
    Strategy: Try all subsets of size 32 from a small candidate pool, or use
    a spiral/hexagon-based construction that is known to achieve the optimum.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # The optimal configuration for 32 points on triangular lattice with 76 edges
    # can be found by taking a hexagonal arrangement.
    # A hexagon of radius 4 has 1+6+12+18+24 = 61 points, but we need 32.
    # Actually, let's use the known result: for n=32, the optimum is achieved
    # by a specific subset.
    
    # Let's generate a candidate pool from a hexagonal region
    # and use a greedy/local search approach to maximize edges.
    
    # Distance squared in triangular lattice: a^2 + ab + b^2
    # Unit distance means a^2 + ab + b^2 == 1
    
    # Neighbor offsets for unit distance
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Generate candidate pool: points within radius ~5
    R = 6
    candidates = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if a * a + a * b + b * b <= R * R:
                candidates.append((a, b))
    
    n_candidates = len(candidates)
    
    # Build adjacency: which candidates are at unit distance
    adj = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        ai, bi = candidates[i]
        for dx, dy in neighbors:
            j = candidates.index((ai + dx, bi + dy)) if (ai + dx, bi + dy) in candidates else -1
            if j != -1:
                adj[i].add(j)
    
    # Greedy construction: start with a hexagonal spiral pattern
    # For 32 points, a good starting point is the center plus rings
    # Ring 0: 1 point
    # Ring 1: 6 points  
    # Ring 2: 12 points
    # Ring 3: 18 points (need only 13 more to reach 32)
    
    # Let's use local search: start with a reasonable configuration and improve
    # by swapping points.
    
    # Initial configuration: take the first 32 candidates in a hexagonal order
    # Sort by distance from origin, then angle
    def lattice_dist_sq(pt):
        a, b = pt
        return a * a + a * b + b * b
    
    candidates_sorted = sorted(candidates, key=lambda p: (lattice_dist_sq(p), p[0], p[1]))
    
    # Start with a subset
    current_set = set(candidates_sorted[:32])
    
    def count_edges(point_set):
        """Count number of unit-distance pairs in the set."""
        edges = 0
        pts = list(point_set)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a1, b1 = pts[i]
                a2, b2 = pts[j]
                da = a1 - a2
                db = b1 - b2
                if da * da + da * db + db * db == 1:
                    edges += 1
        return edges
    
    best_set = current_set
    best_edges = count_edges(best_set)
    
    # Local search: try swapping a point in the set with one outside
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        pts_in = list(best_set)
        pts_out = [c for c in candidates if c not in best_set]
        
        for i in range(len(pts_in)):
            if time.time() >= deadline - 5:
                break
            for j in range(len(pts_out)):
                new_set = best_set - {pts_in[i]} | {pts_out[j]}
                edges = count_edges(new_set)
                if edges > best_edges:
                    best_edges = edges
                    best_set = new_set
                    improved = True
                    break
            if improved:
                break
    
    # Convert to required format: list of [a, b] pairs
    result = [[a, b] for a, b in best_set]
    result.sort()
    
    return result
