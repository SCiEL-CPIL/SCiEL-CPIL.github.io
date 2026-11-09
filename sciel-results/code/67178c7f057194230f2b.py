import time
import math

# Pre-define neighbors for fast edge counting
# Triangular lattice neighbors for (a, b) are:
# (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def _hexdist(a, b):
    """Calculate hexagonal distance from origin (0,0)"""
    return (abs(a) + abs(b) + abs(a + b)) // 2

def _count_edges(points):
    """Count the number of pairs at distance exactly 1"""
    point_set = set(points)
    count = 0
    for p in points:
        a, b = p
        for da, db in _NEIGHBORS:
            if (a + da, b + db) in point_set:
                count += 1
    return count // 2  # Each edge is counted twice

def _generate_spiral(n):
    """
    Generate n points in a hexagonal spiral pattern.
    This creates a very compact shape that should be close to optimal.
    """
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    points = [(0, 0)]
    if n == 1:
        return points
    
    # Hexagonal spiral directions
    # We go around the hexagon: 
    # Right, Up-Right, Up-Left, Left, Down-Left, Down-Right
    # In (a,b) coordinates:
    # (1, 0) -> Right
    # (0, 1) -> Up-Right (actually up in the lattice sense)
    # (-1, 1) -> Up-Left
    # (-1, 0) -> Left
    # (0, -1) -> Down-Left
    # (1, -1) -> Down-Right
    
    # Let's use the standard hex spiral generation
    # Side length increases as: 1, 1, 2, 2, 3, 3, ...
    # For ring k (starting k=1), there are 6*k points
    
    ring = 1
    while len(points) < n:
        # Each ring has 6 sides, each of length 'ring'
        # Directions for the 6 sides
        # Side 0: (1, 0) * ring
        # Side 1: (0, 1) * ring
        # Side 2: (-1, 1) * ring
        # Side 3: (-1, 0) * ring
        # Side 4: (0, -1) * ring
        # Side 5: (1, -1) * ring
        
        # We need to track the current position
        # Start at the top-right of the previous ring?
        # Actually, let's just generate the ring points directly.
        
        # The ring of radius r consists of points with hexdist == r
        # We can iterate through them in order.
        
        # Let's use a simpler approach: generate all points with hexdist <= max_r
        # and sort by (hexdist, angle)
        # But that's what _generate_hexagon does.
        
        # Let's stick to the spiral generation logic properly.
        # Start point for ring 1 is (1, 0)
        # Let's define the starting point for ring k.
        # Ring 1 starts at (1, 0)
        # Ring 2 starts at (2, 0)
        # Ring k starts at (k, 0)
        
        start_a, start_b = ring, 0
        cur_a, cur_b = start_a, start_b
        
        # The 6 directions for the spiral
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        for side in range(6):
            da, db = dirs[side]
            for step in range(ring):
                if len(points) >= n:
                    break
                points.append((cur_a, cur_b))
                cur_a += da
                cur_b += db
            
            if len(points) >= n:
                break
        
        ring += 1
        
    return points[:n]

def _local_search(points, n, deadline):
    """
    Perform systematic local search to maximize edges.
    Tries all possible moves: remove point i, add point c.
    Uses best-improvement strategy.
    """
    current_points = list(points)
    current_set = set(current_points)
    current_edges = _count_edges(current_points)
    
    point_neighbors = {}
    for p in current_points:
        a, b = p
        nbrs = []
        for da, db in _NEIGHBORS:
            np = (a + da, b + db)
            if np in current_set:
                nbrs.append(np)
        point_neighbors[p] = nbrs

    pool = set()
    for a in range(-12, 13):
        for b in range(-12, 13):
            if _hexdist(a, b) <= 12:
                pool.add((a, b))
    pool = list(pool)
    
    best_points = current_points[:]
    best_edges = current_edges
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_i = -1
        best_c = None
        
        # Find the best move
        for i in range(n):
            if time.time() >= deadline:
                break
            p_i = current_points[i]
            
            edges_lost = len(point_neighbors[p_i])
            
            for p_c in pool:
                if p_c == p_i or p_c in current_set:
                    continue
                
                a_c, b_c = p_c
                edges_gained = 0
                for da, db in _NEIGHBORS:
                    np = (a_c + da, b_c + db)
                    if np in current_set and np != p_i:
                        edges_gained += 1
                
                gain = edges_gained - edges_lost
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_c = p_c
        
        if best_gain > 0:
            p_i = current_points[best_i]
            p_c = best_c
            
            new_edges = current_edges + best_gain
            
            # Update data structures
            for nbr in point_neighbors[p_i]:
                if p_i in point_neighbors[nbr]:
                    point_neighbors[nbr].remove(p_i)
            del point_neighbors[p_i]
            current_set.discard(p_i)
            
            current_points[best_i] = p_c
            current_set.add(p_c)
            
            point_neighbors[p_c] = []
            a_c, b_c = p_c
            for da, db in _NEIGHBORS:
                np = (a_c + da, b_c + db)
                if np in current_set:
                    point_neighbors[p_c].append(np)
                    if p_c not in point_neighbors[np]:
                        point_neighbors[np].append(p_c)
            
            current_edges = new_edges
            improved = True
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_points = current_points[:]
            
            if current_edges >= 76:
                return current_points

    return best_points

def construct():
    deadline = time.time() + 140  # Safety margin
    n = 32
    
    # 1. Generate a high-quality initial guess using hexagonal spiral
    initial_points = _generate_spiral(n)
    
    # Check if already optimal
    edges = _count_edges(initial_points)
    if edges >= 76:
        return [[a, b] for a, b in initial_points]
    
    # 2. Refine with local search
    refined_points = _local_search(initial_points, n, deadline)
    
    final_edges = _count_edges(refined_points)
    
    # 3. If not optimal, try a few perturbed restarts
    if final_edges < 76 and time.time() < deadline - 10:
        import random
        rng = random.Random(42)
        best_points = refined_points
        best_edges = final_edges
        
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            pts = list(best_points)
            pset = set(pts)
            
            # Perturb: move 1-2 random points
            for _ in range(2):
                idx = rng.randint(0, n - 1)
                pool = [(a,b) for a in range(-15, 16) for b in range(-15, 16) if _hexdist(a,b) <= 15]
                candidates = [p for p in pool if p not in pset]
                if candidates:
                    new_p = rng.choice(candidates)
                    old_p = pts[idx]
                    pset.remove(old_p)
                    pset.add(new_p)
                    pts[idx] = new_p
            
            refined = _local_search(pts, n, deadline - 5)
            e = _count_edges(refined)
            
            if e > best_edges:
                best_edges = e
                best_points = refined
                if e >= 76:
                    break

    return [[a, b] for a, b in best_points]
