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

def _greedy_construct(n, radius_limit=8):
    """
    Greedily build a set of n points maximizing local edge density.
    Start from center, add points that maximize new edges while maintaining distance >= 1.
    """
    selected = set()
    selected_list = []
    
    # Pre-compute all candidate points within radius
    candidates = []
    for a in range(-radius_limit, radius_limit + 1):
        for b in range(-radius_limit, radius_limit + 1):
            if _hexdist(a, b) <= radius_limit:
                candidates.append((a, b))
    
    # Start with origin
    selected.add((0, 0))
    selected_list.append((0, 0))
    
    for _ in range(n - 1):
        best_point = None
        best_edges = -1
        best_dist = float('inf')
        
        for cand in candidates:
            if cand in selected:
                continue
            a, b = cand
            
            # Check distance constraint: must be at distance >= 1 from all selected
            valid = True
            for sa, sb in selected:
                da, db = a - sa, b - sb
                # Distance squared in triangular lattice: da^2 + db^2 + da*db
                dist_sq = da*da + db*db + da*db
                if dist_sq < 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Count new edges this point would add
            new_edges = 0
            for da, db in _NEIGHBORS:
                if (a + da, b + db) in selected:
                    new_edges += 1
            
            # Prefer points with more new edges; break ties by smaller hex distance from origin
            hex_d = _hexdist(a, b)
            
            if new_edges > best_edges or (new_edges == best_edges and hex_d < best_dist):
                best_edges = new_edges
                best_dist = hex_d
                best_point = cand
        
        if best_point is None:
            # Shouldn't happen with radius_limit=8 and n=32
            break
        
        selected.add(best_point)
        selected_list.append(best_point)
    
    return selected_list

def _local_search(points, n, deadline, rng):
    """
    Perform hill-climbing local search to maximize edges.
    Uses incremental edge counting for speed.
    """
    current_points = list(points)
    current_set = set(current_points)
    current_edges = _count_edges(current_points)
    
    # Build a pool of candidate points
    pool = set()
    for p in current_points:
        a, b = p
        for da in range(-4, 5):
            for db in range(-4, 5):
                if _hexdist(a+da, b+db) <= 10:
                    pool.add((a+da, b+db))
    pool = list(pool)
    
    max_edges_found = current_edges
    best_points = current_points[:]
    
    # Track per-point edge contributions for incremental updates
    # edge_count[p] = number of neighbors of p that are in current_set
    edge_count = {}
    for p in current_points:
        a, b = p
        cnt = 0
        for da, db in _NEIGHBORS:
            if (a + da, b + db) in current_set:
                cnt += 1
        edge_count[p] = cnt
    
    stall_count = 0
    max_stall = 50
    
    while time.time() < deadline:
        improved = False
        
        # Try random swaps
        for _ in range(200):
            i = rng.randrange(n)
            p_i = current_points[i]
            
            if not pool:
                continue
            
            c_idx = rng.randrange(len(pool))
            p_c = pool[c_idx]
            
            if p_c == p_i or p_c in current_set:
                continue
            
            # Incremental edge calculation
            # Remove p_i: lose edge_count[p_i] edges
            # Add p_c: gain edges from p_c's neighbors in (current_set - {p_i} + {p_c})
            # = neighbors of p_c in current_set, minus 1 if p_i is neighbor of p_c
            
            new_edges = current_edges - edge_count[p_i]
            
            # Count edges p_c would have
            a_c, b_c = p_c
            cnt_c = 0
            for da, db in _NEIGHBORS:
                if (a_c + da, b_c + db) in current_set:
                    cnt_c += 1
            # If p_i is a neighbor of p_c, we already counted it, but p_i is being removed
            # So if p_i is neighbor of p_c, subtract 1
            a_i, b_i = p_i
            if (a_c - a_i, b_c - b_i) in _NEIGHBORS or (a_i - a_c, b_i - b_c) in _NEIGHBORS:
                cnt_c -= 1
            
            new_edges += cnt_c
            
            if new_edges > current_edges:
                # Accept the move
                current_points[i] = p_c
                current_set.discard(p_i)
                current_set.add(p_c)
                current_edges = new_edges
                edge_count[p_c] = cnt_c
                del edge_count[p_i]
                improved = True
                
                if current_edges > max_edges_found:
                    max_edges_found = current_edges
                    best_points = current_points[:]
                
                if current_edges >= 76:
                    return current_points
                break
        
        if not improved:
            stall_count += 1
            if stall_count >= max_stall:
                # Try a kick: randomly move a few points
                for _ in range(3):
                    i = rng.randrange(n)
                    p_i = current_points[i]
                    # Find a random valid replacement
                    for _attempt in range(50):
                        c_idx = rng.randrange(len(pool))
                        p_c = pool[c_idx]
                        if p_c == p_i or p_c in current_set:
                            continue
                        # Check distance constraint
                        valid = True
                        for other in current_points:
                            if other == p_i:
                                continue
                            da, db = p_c[0] - other[0], p_c[1] - other[1]
                            dist_sq = da*da + db*db + da*db
                            if dist_sq < 1:
                                valid = False
                                break
                        if valid:
                            current_points[i] = p_c
                            current_set.discard(p_i)
                            current_set.add(p_c)
                            # Recalculate edge counts
                            edge_count = {}
                            for p in current_points:
                                a, b = p
                                cnt = 0
                                for da, db in _NEIGHBORS:
                                    if (a + da, b + db) in current_set:
                                        cnt += 1
                                edge_count[p] = cnt
                            current_edges = _count_edges(current_points)
                            break
                stall_count = 0
        else:
            stall_count = 0
    
    return best_points

def construct():
    deadline = time.time() + 100  # Safety margin
    n = 32
    rng = random.Random(42)  # Deterministic seed for reproducibility
    
    # 1. Generate initial guesses from multiple strategies
    initial_candidates = []
    
    # Spiral-based
    spiral_points = _generate_spiral(n, radius_limit=10)
    initial_candidates.append(spiral_points)
    
    # Greedy construction
    greedy_points = _greedy_construct(n, radius_limit=8)
    if len(greedy_points) == n:
        initial_candidates.append(greedy_points)
    
    # 2. Pick the best initial candidate
    best_initial = None
    best_initial_edges = -1
    for cand in initial_candidates:
        e = _count_edges(cand)
        if e > best_initial_edges:
            best_initial_edges = e
            best_initial = cand
    
    if best_initial_edges >= 76:
        return [[a, b] for a, b in best_initial]
    
    # 3. Refine with local search
    refined_points = _local_search(best_initial, n, deadline, rng)
    final_edges = _count_edges(refined_points)
    
    if final_edges >= 76:
        return [[a, b] for a, b in refined_points]
    
    # 4. Try local search from other initial candidates if time permits
    if time.time() < deadline - 10:
        for cand in initial_candidates:
            if cand is best_initial:
                continue
            if time.time() > deadline - 5:
                break
            candidate_refined = _local_search(cand, n, deadline - 5, rng)
            candidate_edges = _count_edges(candidate_refined)
            if candidate_edges > final_edges:
                refined_points = candidate_refined
                final_edges = candidate_edges
                if final_edges >= 76:
                    break
    
    return [[a, b] for a, b in refined_points]

if __name__ == "__main__":
    # Simple self-test
    pts = construct()
    edges = _count_edges(pts)
    print(f"Points: {pts}")
    print(f"Edges: {edges}")
    # Expected 76
