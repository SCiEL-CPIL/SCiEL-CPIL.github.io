"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

N = 64
TIME_LIMIT = 280.0

# EVOLVE-BLOCK-START

def _can_add_point_fast(point_set, dist_to_point, new_pt):
    """
    Fast check if adding new_pt creates an isosceles triangle.
    point_set: set of (x,y) tuples
    dist_to_point: dict mapping (x,y) -> set of squared distances from that point to others
    new_pt: (x, y) tuple
    Returns True if safe to add, False otherwise.
    """
    n = len(point_set)
    if n < 2:
        return True
    
    nx, ny = new_pt
    
    # Check 1: No two existing points at same distance from new_pt
    seen_dists = set()
    for (px, py) in point_set:
        dx = nx - px
        dy = ny - py
        d2 = dx*dx + dy*dy
        if d2 in seen_dists:
            return False
        seen_dists.add(d2)
    
    # Check 2: For each existing point i, dist(new_pt, i) must not equal dist(i, j) for any j != i
    for (px, py) in point_set:
        dx = nx - px
        dy = ny - py
        d2 = dx*dx + dy*dy
        if d2 in dist_to_point.get((px, py), set()):
            return False
    
    return True


def _greedy_construct_fast(seed, order_type='random', deadline=None):
    """
    Fast greedy construction using distance tracking.
    """
    rng = random.Random(seed)
    points = []
    point_set = set()
    dist_to_point = defaultdict(set)  # (x,y) -> set of squared distances to other points
    
    # Generate candidate positions
    if order_type == 'random':
        candidates = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(candidates)
    elif order_type == 'spiral':
        cx, cy = N // 2, N // 2
        candidates = []
        seen = set()
        for r in range(N):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < N and 0 <= y < N and (x, y) not in seen:
                            seen.add((x, y))
                            candidates.append((x, y))
    elif order_type == 'grid':
        candidates = [(x, y) for x in range(N) for y in range(N)]
    else:
        candidates = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(candidates)
    
    for pt in candidates:
        if deadline is not None and time.time() > deadline:
            break
        if _can_add_point_fast(point_set, dist_to_point, pt):
            point_set.add(pt)
            points.append(pt)
            # Update distance sets
            for (px, py) in point_set:
                if pt == (px, py):
                    continue
                dx = pt[0] - px
                dy = pt[1] - py
                d2 = dx*dx + dy*dy
                dist_to_point[pt].add(d2)
                dist_to_point[(px, py)].add(d2)
    
    return points


def _local_search_fast(points, seed, max_iters=20, deadline=None):
    """
    Local search: try removing one point and adding another to maintain or improve size.
    """
    rng = random.Random(seed)
    current = list(points)
    current_set = set(current)
    
    # Build dist_to_point for current
    dist_to_point = defaultdict(set)
    for i in range(len(current)):
        for j in range(i+1, len(current)):
            dx = current[i][0] - current[j][0]
            dy = current[i][1] - current[j][1]
            d2 = dx*dx + dy*dy
            dist_to_point[current[i]].add(d2)
            dist_to_point[current[j]].add(d2)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if len(current) < 2:
            break
        
        idx = rng.randrange(len(current))
        removed = current.pop(idx)
        current_set.remove(removed)
        
        # Remove distances involving removed point
        for (px, py) in current_set:
            dx = removed[0] - px
            dy = removed[1] - py
            d2 = dx*dx + dy*dy
            if d2 in dist_to_point.get((px, py), set()):
                dist_to_point[(px, py)].discard(d2)
        if removed in dist_to_point:
            del dist_to_point[removed]
        
        # Try to find a replacement point
        added_any = False
        test_candidates = [(x, y) for x in range(N) for y in range(N)]
        rng.shuffle(test_candidates)
        test_candidates = test_candidates[:500]  # Limit for speed
        
        for pt in test_candidates:
            if pt in current_set:
                continue
            if _can_add_point_fast(current_set, dist_to_point, pt):
                current.append(pt)
                current_set.add(pt)
                # Add distances
                for (px, py) in current_set:
                    if pt == (px, py):
                        continue
                    dx = pt[0] - px
                    dy = pt[1] - py
                    d2 = dx*dx + dy*dy
                    dist_to_point[pt].add(d2)
                    dist_to_point[(px, py)].add(d2)
                added_any = True
                break
        
        if not added_any:
            # Put back the removed point
            current.append(removed)
            current_set.add(removed)
            for (px, py) in current_set:
                if removed == (px, py):
                    continue
                dx = removed[0] - px
                dy = removed[1] - py
                d2 = dx*dx + dy*dy
                dist_to_point[removed].add(d2)
                dist_to_point[(px, py)].add(d2)
    
    return current


def _hypergraph_deletion(points, deadline=None):
    """
    Vectorized greedy deletion algorithm for maximum independent set in 3-uniform hypergraph.
    Hyperedges represent triples of points forming an isosceles triangle.
    Iteratively removes the vertex with the highest hyperedge degree until no hyperedges remain.
    
    This is a structural change: instead of greedy addition, we start with all points
    and iteratively remove the most "conflicting" point.
    """
    pts = list(points)
    n = len(pts)
    if n < 3:
        return pts
    
    # Compute all pairwise squared distances
    # dist[i][j] = squared distance between pts[i] and pts[j]
    dist = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i+1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx*dx + dy*dy
            dist[i][j] = d2
            dist[j][i] = d2
    
    # For each point i, compute the multiset of squared distances to other points
    # A triple (i, j, k) forms an isosceles triangle if:
    #   dist[i][j] == dist[i][k] (isosceles with apex at i)
    #   dist[i][j] == dist[j][k] (isosceles with apex at j)
    #   dist[i][k] == dist[j][k] (isosceles with apex at k)
    
    # For each point, count how many isosceles triangles it participates in
    # We'll use a degree array
    degree = np.zeros(n, dtype=np.int32)
    
    # For each point i, count duplicates in distances from i to others
    # This gives us the number of isosceles triangles with apex at i
    for i in range(n):
        # Get all distances from i to other points
        d_from_i = dist[i].copy()
        d_from_i[i] = -1  # Exclude self
        # Count frequency of each distance
        # A distance value d2 appearing c times means C(c, 2) pairs at that distance
        # Each such pair forms an isosceles triangle with apex at i
        from collections import Counter
        cnt = Counter(d_from_i[d_from_i >= 0])
        for d2, c in cnt.items():
            if c >= 2:
                degree[i] += c * (c - 1) // 2
    
    # Now we need to also count isosceles triangles where i is not the apex
    # For a triple (i, j, k), if dist[i][j] == dist[j][k], then j is the apex
    # This is already counted in degree[j]
    # If dist[i][j] == dist[i][k], then i is the apex - counted in degree[i]
    # If dist[i][k] == dist[j][k], then k is the apex - counted in degree[k]
    
    # So the total number of isosceles triangles involving point i is:
    #   (triangles with apex at i) + (triangles where i is a base vertex)
    # For base vertex counting, we need to count for each pair (j, k) with j != i, k != i,
    # whether dist[i][j] == dist[j][k] or dist[i][k] == dist[j][k]
    
    # This is O(n^3) which is too slow for n=4096. Let's use a different approach.
    # Instead, let's just use the apex degree as a proxy and iteratively remove the highest degree point.
    
    # Iterative deletion
    active = list(range(n))
    active_set = set(active)
    
    # We'll maintain a dynamic degree that gets updated as we remove points
    # For efficiency, we'll recompute degrees periodically
    
    max_iters = 1000
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        if len(active) < 3:
            break
        
        # Recompute degrees for active points
        active_arr = np.array(active, dtype=np.int32)
        n_active = len(active_arr)
        
        if n_active < 3:
            break
        
        # Get submatrix of distances
        sub_dist = dist[np.ix_(active_arr, active_arr)]
        
        # Compute apex degrees
        apex_degree = np.zeros(n_active, dtype=np.int32)
        for i in range(n_active):
            d_from_i = sub_dist[i].copy()
            d_from_i[i] = -1
            from collections import Counter
            cnt = Counter(d_from_i[d_from_i >= 0])
            for d2, c in cnt.items():
                if c >= 2:
                    apex_degree[i] += c * (c - 1) // 2
        
        # Find the point with highest degree
        max_idx = np.argmax(apex_degree)
        
        if apex_degree[max_idx] == 0:
            # No more isosceles triangles
            break
        
        # Remove this point
        removed_global_idx = active_arr[max_idx]
        active.remove(removed_global_idx)
        active_set.discard(removed_global_idx)
    
    return [pts[i] for i in active]


def _hybrid_construct(seed, deadline=None):
    """
    Hybrid approach:
    1. Start with a greedy construction to get a good initial set
    2. Apply hypergraph deletion to remove conflicting points
    3. Apply local search to try to add back points
    """
    # Step 1: Greedy construction
    greedy_pts = _greedy_construct_fast(seed, 'random', deadline)
    
    if len(greedy_pts) < 3:
        return greedy_pts
    
    # Step 2: Hypergraph deletion (only if we have many points)
    if len(greedy_pts) > 50:
        deleted_pts = _hypergraph_deletion(greedy_pts, deadline)
        if len(deleted_pts) > len(greedy_pts) * 0.8:
            greedy_pts = deleted_pts
    
    # Step 3: Local search
    if deadline is not None and time.time() < deadline:
        greedy_pts = _local_search_fast(greedy_pts, seed, max_iters=15, deadline=deadline)
    
    return greedy_pts


def construct():
    start_time = time.time()
    best_points = []
    best_size = 0
    deadline = start_time + TIME_LIMIT - 20  # Safety margin
    
    # Quick greedy constructions with different orders
    for order_type in ['random', 'spiral', 'grid']:
        if time.time() > deadline:
            break
        pts = _greedy_construct_fast(42, order_type, deadline)
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
    
    # Multiple random restarts with local search
    seed = 0
    while time.time() < deadline:
        # Use hybrid approach for some iterations
        if seed % 5 == 0:
            pts = _hybrid_construct(seed, deadline)
        else:
            pts = _greedy_construct_fast(seed, 'random', deadline)
            if time.time() < deadline:
                pts = _local_search_fast(pts, seed, max_iters=10, deadline=deadline)
        
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
        
        seed += 1
        # If we've done enough restarts, try different orderings
        if seed % 10 == 0:
            for order_type in ['spiral', 'grid']:
                if time.time() > deadline:
                    break
                pts = _greedy_construct_fast(seed, order_type, deadline)
                if len(pts) > best_size:
                    best_size = len(pts)
                    best_points = list(pts)
    
    return best_points

# EVOLVE-BLOCK-END
