import time
import random

def _dist_sq(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return da * da + da * db + db * db

def _neighbors(p):
    a, b = p
    return [
        (a + 1, b), (a - 1, b),
        (a, b + 1), (a, b - 1),
        (a + 1, b - 1), (a - 1, b + 1)
    ]

def _edge_count(points):
    pset = set(points)
    cnt = 0
    for p in points:
        for n in _neighbors(p):
            if n in pset:
                cnt += 1
    return cnt // 2

def _is_valid(points):
    if len(points) != 32:
        return False
    pset = set(points)
    if len(pset) != 32:
        return False
    for p in points:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _dist_sq(points[i], points[j]) < 1:
                return False
    return True

def _build_hexagonal_cluster():
    """Build a compact hexagonal cluster of 32 points from the 37-point hex of radius 3."""
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Generate all points in hex of radius 3 (37 points)
    all_pts = [(0, 0)]
    for ring in range(1, 4):
        pos = (ring, 0)
        for i in range(6):
            d = dirs[i]
            for step in range(ring):
                all_pts.append(pos)
                pos = (pos[0] + d[0], pos[1] + d[1])
    
    # We have 37 points, need to remove 5 to get 32
    # Strategy: remove points that have the fewest neighbors within the set
    # to minimize edge loss
    best_subset = None
    best_edges = -1
    
    # Try removing combinations of 5 points
    # To keep it fast, try removing points from the outer ring preferentially
    outer_ring = []
    for p in all_pts:
        hex_dist = max(abs(p[0]), abs(p[1]), abs(-p[0] - p[1]))
        if hex_dist == 3:
            outer_ring.append(p)
    
    # Try removing 5 from outer ring (there are 18 outer points)
    import itertools
    # Sample a subset of combinations to keep it fast
    if len(outer_ring) <= 20:
        for combo in itertools.combinations(range(len(outer_ring)), 5):
            remove_set = set(outer_ring[i] for i in combo)
            subset = [p for p in all_pts if p not in remove_set]
            if len(subset) == 32:
                edges = _edge_count(subset)
                if edges > best_edges:
                    best_edges = edges
                    best_subset = subset
    
    if best_subset is None:
        # Fallback: just take first 32
        best_subset = all_pts[:32]
    
    return best_subset

def _local_search(pts, deadline, max_iter=5000):
    """Local search to maximize edge count with targeted moves."""
    pts = pts[:]
    current_score = _edge_count(pts)
    pset = set(pts)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        random.shuffle(pts)
        
        for i in range(len(pts)):
            if time.time() > deadline:
                break
            
            p = pts[i]
            # Try moving to each neighbor position
            for n in _neighbors(p):
                if abs(n[0]) > 64 or abs(n[1]) > 64:
                    continue
                if n in pset:
                    continue
                
                # Check validity: distance to all other points
                valid = True
                for j in range(len(pts)):
                    if i == j:
                        continue
                    if _dist_sq(n, pts[j]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute new score efficiently
                # Remove edges involving p, add edges involving n
                old_edges_from_p = sum(1 for nb in _neighbors(p) if nb in pset)
                new_edges_from_n = sum(1 for nb in _neighbors(n) if nb in pset)
                
                # But we need to be careful: some neighbors of p might also be neighbors of n
                # For simplicity, just compute full edge count
                new_pts = pts[:]
                new_pts[i] = n
                new_score = _edge_count(new_pts)
                
                if new_score > current_score:
                    pts = new_pts
                    current_score = new_score
                    pset = set(pts)
                    improved = True
                    break
        
        if not improved:
            # Try a random perturbation to escape local optimum
            i = random.randint(0, len(pts) - 1)
            p = pts[i]
            candidates = []
            for n in _neighbors(p):
                if abs(n[0]) > 64 or abs(n[1]) > 64:
                    continue
                if n in pset:
                    continue
                valid = True
                for j in range(len(pts)):
                    if i == j:
                        continue
                    if _dist_sq(n, pts[j]) < 1:
                        valid = False
                        break
                if valid:
                    candidates.append(n)
            
            if candidates:
                n = random.choice(candidates)
                new_pts = pts[:]
                new_pts[i] = n
                new_score = _edge_count(new_pts)
                
                # Accept with probability based on temperature
                if new_score >= current_score or random.random() < 0.15:
                    pts = new_pts
                    current_score = new_score
                    pset = set(pts)
    
    return pts, current_score

def _simulated_annealing(pts, deadline, max_iter=10000):
    """Simulated annealing to escape local optima."""
    pts = pts[:]
    current_score = _edge_count(pts)
    pset = set(pts)
    best_score = current_score
    best_pts = pts[:]
    
    temperature = 1.0
    cooling = 0.9995
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        i = random.randint(0, len(pts) - 1)
        p = pts[i]
        
        # Try moving to a random neighbor
        candidates = []
        for n in _neighbors(p):
            if abs(n[0]) > 64 or abs(n[1]) > 64:
                continue
            if n in pset:
                continue
            valid = True
            for j in range(len(pts)):
                if i == j:
                    continue
                if _dist_sq(n, pts[j]) < 1:
                    valid = False
                    break
            if valid:
                candidates.append(n)
        
        if not candidates:
            continue
        
        n = random.choice(candidates)
        new_pts = pts[:]
        new_pts[i] = n
        new_score = _edge_count(new_pts)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < pow(2.718, delta / max(temperature, 0.01)):
            pts = new_pts
            current_score = new_score
            pset = set(pts)
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts[:]
        
        temperature *= cooling
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Phase 1: Start with hexagonal cluster
    pts = _build_hexagonal_cluster()
    if not _is_valid(pts):
        # Fallback: simple grid
        pts = [(a, b) for a in range(8) for b in range(4)]
    
    # Local search
    pts, score = _local_search(pts, deadline, max_iter=3000)
    best_score = score
    best_points = pts[:]
    
    if best_score >= 76:
        return [[p[0], p[1]] for p in best_points]
    
    # Phase 2: Simulated annealing
    pts, score = _simulated_annealing(best_points, deadline, max_iter=8000)
    if score > best_score:
        best_score = score
        best_points = pts[:]
    
    if best_score >= 76:
        return [[p[0], p[1]] for p in best_points]
    
    # Phase 3: Multiple restarts with different initial configurations
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Create perturbed version of best solution
        pts = best_points[:]
        for _ in range(5):
            i = random.randint(0, 31)
            p = pts[i]
            n = random.choice(_neighbors(p))
            if abs(n[0]) <= 64 and abs(n[1]) <= 64 and n not in pts:
                valid = True
                for j in range(32):
                    if i != j and _dist_sq(n, pts[j]) < 1:
                        valid = False
                        break
                if valid:
                    pts[i] = n
        
        # Local search from perturbed state
        pts, score = _local_search(pts, deadline, max_iter=1500)
        if score > best_score:
            best_score = score
            best_points = pts[:]
        
        if best_score >= 76:
            break
    
    if best_points is None:
        best_points = [(a, b) for a in range(16) for b in range(2)]
    
    return [[p[0], p[1]] for p in best_points]
