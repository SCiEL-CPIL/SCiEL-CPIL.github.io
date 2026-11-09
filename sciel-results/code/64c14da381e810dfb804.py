"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

_NEIGHBORS = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]

def _dist_sq(p1, p2):
    """Squared distance in triangular lattice between two points."""
    da = p1[0] - p2[0]
    db = p1[1] - p2[1]
    return da*da + da*db + db*db

def _count_edges(points):
    """Count pairs at distance exactly 1 in the triangular lattice."""
    pts = set(points)
    count = 0
    for a, b in pts:
        for da, db in _NEIGHBORS:
            if (a+da, b+db) in pts:
                count += 1
    return count // 2

def _greedy_construct(n=32, max_coord=12, seed=0):
    """Build a set of n points greedily, always choosing the point that
    maximizes the number of new edges formed with already-placed points.
    Uses a random tie-breaking to get diverse configurations."""
    rng = random.Random(seed)
    
    # Candidate pool: all lattice points in a reasonable bounding box
    candidates = []
    for a in range(-max_coord, max_coord+1):
        for b in range(-max_coord, max_coord+1):
            candidates.append((a, b))
    
    # Shuffle for diversity
    rng.shuffle(candidates)
    
    placed = []
    placed_set = set()
    
    for _ in range(n):
        best_pt = None
        best_edges = -1
        best_tie = -1
        for pt in candidates:
            if pt in placed_set:
                continue
            # Check distance >= 1 from all placed points
            valid = True
            edge_count = 0
            for pp in placed:
                d = _dist_sq(pt, pp)
                if d < 1:
                    valid = False
                    break
                if d == 1:
                    edge_count += 1
            if not valid:
                continue
            # Use edge_count as primary, random tie-break for diversity
            tie = rng.random()
            if edge_count > best_edges or (edge_count == best_edges and tie > best_tie):
                best_edges = edge_count
                best_tie = tie
                best_pt = pt
        if best_pt is None:
            # Fallback: pick any valid point
            for pt in candidates:
                if pt in placed_set:
                    continue
                valid = True
                for pp in placed:
                    if _dist_sq(pt, pp) < 1:
                        valid = False
                        break
                if valid:
                    best_pt = pt
                    break
            if best_pt is None:
                break
        placed.append(best_pt)
        placed_set.add(best_pt)
    
    return placed

def _local_search(points, deadline, max_coord=14):
    """Fast local search using incremental edge count updates."""
    pts = list(points)
    n = len(pts)
    pts_set = set(pts)
    current_count = _count_edges(pts)
    
    # Precompute neighbor relationships
    # For each point, track which other points are at distance 1
    adj = {}
    for i in range(n):
        adj[i] = set()
    for i in range(n):
        for j in range(i+1, n):
            if _dist_sq(pts[i], pts[j]) == 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Candidate replacement points: all points in bounding box not in pts_set
    candidates = []
    for a in range(-max_coord, max_coord+1):
        for b in range(-max_coord, max_coord+1):
            if (a, b) not in pts_set:
                candidates.append((a, b))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            old_pt = pts[i]
            # Compute edges lost by removing old_pt
            edges_lost = len(adj[i])
            for new_pt in candidates:
                if new_pt == old_pt:
                    continue
                if new_pt in pts_set:
                    continue
                # Check validity: distance >= 1 from all other points
                valid = True
                edges_gained = 0
                for j in range(n):
                    if i == j:
                        continue
                    d = _dist_sq(new_pt, pts[j])
                    if d < 1:
                        valid = False
                        break
                    if d == 1:
                        edges_gained += 1
                if not valid:
                    continue
                delta = edges_gained - edges_lost
                if delta > 0:
                    # Accept the move
                    pts_set.remove(old_pt)
                    pts_set.add(new_pt)
                    pts[i] = new_pt
                    current_count += delta
                    # Update adjacency
                    adj[i] = set()
                    for j in range(n):
                        if i == j:
                            continue
                        if _dist_sq(new_pt, pts[j]) == 1:
                            adj[i].add(j)
                            adj[j].add(i)
                    else:
                        pass
                    # Fix adjacency for all j that had i
                    for j in range(n):
                        if i in adj[j] and _dist_sq(new_pt, pts[j]) != 1:
                            adj[j].discard(i)
                        if i != j and _dist_sq(new_pt, pts[j]) == 1 and i not in adj[j]:
                            adj[j].add(i)
                    improved = True
                    break
    return pts, current_count

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_count = -1
    
    # Try multiple greedy constructions with different seeds
    num_restarts = 20
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        initial = _greedy_construct(32, max_coord=12, seed=seed)
        if len(initial) < 32:
            continue
        improved_pts, count = _local_search(initial, deadline, max_coord=14)
        if count > best_count:
            best_count = count
            best_points = improved_pts
        # Early exit if we hit the theoretical optimum
        if best_count >= 76:
            break
    
    if best_points is None:
        # Fallback: hexagonal cluster
        best_points = _greedy_construct(32, max_coord=12, seed=0)
    
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
