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

def _generate_spiral(n, radius_limit=10):
    """
    Generate a list of n points in a spiral order around the origin.
    This creates a dense, compact shape.
    """
    candidates = []
    for a in range(-radius_limit, radius_limit + 1):
        for b in range(-radius_limit, radius_limit + 1):
            if _hexdist(a, b) <= radius_limit:
                # Calculate angle for sorting
                # Convert axial to cartesian for angle
                x = a + 0.5 * b
                y = math.sqrt(3) / 2 * b
                angle = math.atan2(y, x)
                candidates.append((_hexdist(a, b), angle, a, b))
    
    # Sort by distance first, then angle
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Take the first n points
    result = [(a, b) for _, _, a, b in candidates[:n]]
    return result

def _local_search(points, n, deadline, rng):
    """
    Perform hill-climbing local search to maximize edges.
    Uses incremental edge counting for speed.
    """
    current_points = list(points)
    current_set = set(current_points)
    
    # Initial edge count
    current_edges = _count_edges(current_points)
    
    # Build neighbor adjacency for faster incremental updates
    # For each point, store its neighbors that are in the set
    point_neighbors = {}
    for p in current_points:
        a, b = p
        nbrs = []
        for da, db in _NEIGHBORS:
            np = (a + da, b + db)
            if np in current_set:
                nbrs.append(np)
        point_neighbors[p] = nbrs
    
    # Build a pool of candidate points
    # Points close to the current shape
    pool = set()
    for p in current_points:
        a, b = p
        # Add neighbors and next-nearest neighbors to the pool
        for da in range(-4, 5):
            for db in range(-4, 5):
                if _hexdist(a+da, b+db) <= 10:
                    pool.add((a+da, b+db))
    pool = list(pool)
    
    best_points = current_points[:]
    best_edges = current_edges
    
    while time.time() < deadline:
        improved = False
        # Try several random swaps per iteration to accelerate convergence
        for _ in range(200):
            i = rng.randrange(n)
            p_i = current_points[i]
            
            if not pool: continue
            
            c_idx = rng.randrange(len(pool))
            p_c = pool[c_idx]
            
            if p_c == p_i or p_c in current_set:
                continue
            
            # Evaluate swap incrementally
            # Remove p_i: we lose edges between p_i and its neighbors
            # Add p_c: we gain edges between p_c and its neighbors
            # But we must be careful about double counting and edges between p_i and p_c's neighbors
            
            # Calculate delta in edges
            # Edges lost: edges incident to p_i
            edges_lost = len(point_neighbors[p_i])
            
            # Edges gained: edges from p_c to existing points (excluding p_i since it's being removed)
            edges_gained = 0
            a_c, b_c = p_c
            for da, db in _NEIGHBORS:
                np = (a_c + da, b_c + db)
                if np in current_set and np != p_i:
                    edges_gained += 1
            
            # Also need to check if p_i is a neighbor of p_c (it's being removed, so if it was a neighbor, 
            # that edge is lost, but we already counted it in edges_lost if p_c was in point_neighbors[p_i])
            # Actually, let's just recalculate the full edge count for simplicity if the delta is close,
            # or use a more careful incremental approach.
            # Given n=32, full recount is fast enough (32*6 operations).
            
            new_edges = current_edges - edges_lost + edges_gained
            
            if new_edges > current_edges:
                # Perform the swap
                # Update data structures
                # 1. Remove p_i
                # Remove p_i from all neighbors' adjacency lists
                for nbr in point_neighbors[p_i]:
                    if p_i in point_neighbors[nbr]:
                        point_neighbors[nbr].remove(p_i)
                del point_neighbors[p_i]
                
                current_set.discard(p_i)
                current_points[i] = p_c
                current_set.add(p_c)
                
                # 2. Add p_c
                point_neighbors[p_c] = []
                a_c, b_c = p_c
                for da, db in _NEIGHBORS:
                    np = (a_c + da, b_c + db)
                    if np in current_set:
                        point_neighbors[p_c].append(np)
                        # Add p_c to np's neighbors
                        if p_c not in point_neighbors[np]:
                            point_neighbors[np].append(p_c)
                
                current_edges = new_edges
                improved = True
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = current_points[:]
                    
                # If we hit the theoretical maximum, we can stop early
                # Harborth bound for n=32 is 76
                if current_edges >= 76:
                    return current_points
        
        # If no improvement in a full batch, we might be stuck.
        # We could try a kick (random move) but for this problem, 
        # the landscape is likely smooth enough.
        if not improved:
            # Try a few random restarts or kicks? 
            # For now, just continue. The randomization in the pool should help.
            pass

    return best_points

def construct():
    deadline = time.time() + 100  # Safety margin
    n = 32
    rng = random.Random(42)  # Deterministic seed for reproducibility
    
    # 1. Generate a high-quality initial guess using a spiral
    initial_points = _generate_spiral(n, radius_limit=10)
    
    # Check if the initial guess is already optimal
    edges = _count_edges(initial_points)
    if edges >= 76:
        return [[a, b] for a, b in initial_points]
    
    # 2. Refine with local search
    # The spiral is usually very close to optimal. Local search should fix any minor imperfections.
    refined_points = _local_search(initial_points, n, deadline, rng)
    
    # Verify the result
    final_edges = _count_edges(refined_points)
    
    # If for some reason it's still not 76 (unlikely), try a few more restarts with different seeds
    if final_edges < 76 and time.time() < deadline - 5:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 5:
                break
            rng.seed(seed)
            # Perturb the initial solution slightly or regenerate?
            # Let's just run local search from the previous best
            refined_points = _local_search(refined_points, n, deadline - 5, rng)
            if _count_edges(refined_points) >= 76:
                break
    
    return [[a, b] for a, b in refined_points]
