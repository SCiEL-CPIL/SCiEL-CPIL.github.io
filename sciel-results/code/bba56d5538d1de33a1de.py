import time
import math
import random

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

def _generate_hexagon(radius):
    """Generate all points in a hexagon of given radius"""
    points = []
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            if _hexdist(a, b) <= radius:
                points.append((a, b))
    return points

def _prune_hexagon(points_to_remove):
    """
    Greedily remove points from a hexagon to maximize remaining edges.
    We remove points with the lowest current degree first.
    """
    current_points = set(points_to_remove) # Wait, input is the full set
    # Let's rename input to full_set
    pass

def _construct_by_pruning():
    """
    Construct the optimal 32-point cluster by starting with a 37-point hexagon
    and greedily removing 5 points with the lowest degrees.
    """
    # Hexagon of radius 3 has 37 points
    hex37 = _generate_hexagon(3)
    
    # We need to remove 5 points to get 32
    points_to_remove = 5
    current_set = set(hex37)
    
    for _ in range(points_to_remove):
        # Calculate degrees for all points in current_set
        degrees = {}
        for p in current_set:
            a, b = p
            deg = 0
            for da, db in _NEIGHBORS:
                if (a + da, b + db) in current_set:
                    deg += 1
            degrees[p] = deg
        
        # Find point with minimum degree
        min_deg = min(degrees.values())
        candidates = [p for p, d in degrees.items() if d == min_deg]
        
        # If multiple, pick one arbitrarily (or randomly for diversity, but deterministic is fine)
        # To ensure we get a "good" shape, picking any min-degree point is usually fine.
        # However, to be robust, we can try all candidates if the count is small?
        # For now, pick the first one.
        victim = candidates[0]
        current_set.remove(victim)
        
    return list(current_set)

def construct():
    deadline = time.time() + 100
    n = 32
    
    # Strategy 1: Hexagon Pruning
    points_pruned = _construct_by_pruning()
    edges_pruned = _count_edges(points_pruned)
    
    if edges_pruned >= 76:
        return [[a, b] for a, b in points_pruned]
        
    # Strategy 2: Fallback to Spiral + SA if pruning fails (unlikely)
    # (Keep the existing SA logic as a fallback if needed, but pruning should work)
    
    # For safety, let's also try the spiral method if pruning didn't hit 76
    # But we can just return the pruned result as it's likely optimal or near-optimal.
    # If it's not 76, we might need to refine.
    
    if edges_pruned < 76:
        # Simple local search refinement on the pruned set
        best_points = list(points_pruned)
        best_edges = edges_pruned
        current_set = set(best_points)
        
        # Try to improve by swapping points
        # Generate a pool of candidate points nearby
        pool = set()
        for p in best_points:
            a, b = p
            for da in range(-3, 4):
                for db in range(-3, 4):
                    if _hexdist(a+da, b+db) <= 6:
                        pool.add((a+da, b+db))
        pool = list(pool)
        
        rng = random.Random(42)
        T = 1.0
        for _ in range(10000):
            if time.time() > deadline:
                break
            
            i = rng.randrange(n)
            p_i = best_points[i]
            c_idx = rng.randrange(len(pool))
            p_c = pool[c_idx]
            
            if p_c == p_i or p_c in current_set:
                continue
            
            # Calculate delta
            lost = 0
            for da, db in _NEIGHBORS:
                if (p_i[0] + da, p_i[1] + db) in current_set:
                    lost += 1
            
            gained = 0
            for da, db in _NEIGHBORS:
                nb = (p_c[0] + da, p_c[1] + db)
                if nb in current_set and nb != p_i:
                    gained += 1
            
            delta = gained - lost
            new_edges = best_edges + delta
            
            if new_edges > best_edges:
                best_points[i] = p_c
                current_set.discard(p_i)
                current_set.add(p_c)
                best_edges = new_edges
                if best_edges >= 76:
                    return [[a, b] for a, b in best_points]
            elif delta == 0:
                # Accept sideways moves to explore
                if rng.random() < 0.5:
                    best_points[i] = p_c
                    current_set.discard(p_i)
                    current_set.add(p_c)
            else:
                # Accept worsening moves with low probability
                if rng.random() < 0.1:
                    best_points[i] = p_c
                    current_set.discard(p_i)
                    current_set.add(p_c)
                    best_edges = new_edges

        return [[a, b] for a, b in best_points]
        
    return [[a, b] for a, b in points_pruned]
