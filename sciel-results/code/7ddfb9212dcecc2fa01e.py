"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def _dist_sq(p1, p2):
    """Squared distance in triangular lattice between two points."""
    da = p1[0] - p2[0]
    db = p1[1] - p2[1]
    return da * da + da * db + db * db

def _count_edges(points):
    """Count pairs at distance exactly 1 in the triangular lattice."""
    pts = set(points)
    count = 0
    for a, b in pts:
        for da, db in _NEIGHBORS:
            if (a + da, b + db) in pts:
                count += 1
    return count // 2

def _hex_spiral_points(n=32):
    """
    Generate n points in hexagonal spiral order on the triangular lattice.
    The hexagonal spiral fills the lattice in rings of increasing hexagonal
    distance from the origin, producing a compact cluster that maximizes
    nearest-neighbor contacts (Harborth optimal).
    
    Hexagonal distance for point (a,b) in basis {1, omega} where omega=e^{i*pi/3}:
    The Euclidean distance squared is a^2 + a*b + b^2.
    We sort by this distance, with tie-breaking by angle for spiral ordering.
    """
    # For n=32, we need points up to hex distance 3 (ring 0: 1 pt, ring 1: 6, ring 2: 12, ring 3: 18 => total 37)
    # So we need coords in range [-4, 4] to be safe
    max_coord = 6
    
    # Generate all candidate points and sort by hex distance
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            dist_sq = a * a + a * b + b * b
            # For tie-breaking, use angle in the complex plane
            # Point is a + b*omega = a + b*(1/2 + i*sqrt(3)/2)
            # Real part: a + b/2, Imag part: b*sqrt(3)/2
            x = a + b * 0.5
            y = b * math.sqrt(3) / 2.0
            angle = math.atan2(y, x) if (x != 0 or y != 0) else 0.0
            candidates.append((dist_sq, angle, a, b))
    
    # Sort by distance squared, then by angle
    candidates.sort(key=lambda c: (c[0], c[1]))
    
    # Take the first n points
    result = [(a, b) for _, _, a, b in candidates[:n]]
    return result

def _local_search(points, deadline, max_coord=10):
    """Fast local search using incremental edge count updates."""
    pts = list(points)
    n = len(pts)
    pts_set = set(pts)
    current_count = _count_edges(pts)
    
    # Precompute adjacency
    adj = {}
    for i in range(n):
        adj[i] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if _dist_sq(pts[i], pts[j]) == 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Candidate replacement points
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            if (a, b) not in pts_set:
                candidates.append((a, b))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            old_pt = pts[i]
            edges_lost = len(adj[i])
            for new_pt in candidates:
                if new_pt == old_pt or new_pt in pts_set:
                    continue
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
    
    # Primary strategy: direct hexagonal spiral construction (Harborth optimal)
    best_points = _hex_spiral_points(32)
    best_count = _count_edges(best_points)
    
    # If we already hit the theoretical optimum, return immediately
    if best_count >= 76:
        return [[a, b] for a, b in best_points]
    
    # Otherwise, try local search to improve
    if time.time() < deadline - 5:
        improved_pts, count = _local_search(best_points, deadline, max_coord=10)
        if count > best_count:
            best_count = count
            best_points = improved_pts
    
    # Fallback: if something went wrong, try greedy
    if best_count < 70:
        rng = random.Random(0)
        candidates = []
        for a in range(-10, 11):
            for b in range(-10, 11):
                candidates.append((a, b))
        rng.shuffle(candidates)
        
        placed = []
        placed_set = set()
        for pt in candidates:
            if len(placed) >= 32:
                break
            valid = True
            for pp in placed:
                if _dist_sq(pt, pp) < 1:
                    valid = False
                    break
            if valid:
                placed.append(pt)
                placed_set.add(pt)
        
        if len(placed) == 32:
            count = _count_edges(placed)
            if count > best_count:
                best_points = placed
                best_count = count
    
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
