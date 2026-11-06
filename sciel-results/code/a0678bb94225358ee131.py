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
            if n in pset and p < n:
                cnt += 1
    return cnt

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

def _build_hexagonal_spiral():
    """Build a compact hexagonal cluster of 32 points."""
    pts = [(0, 0)]
    # Hexagonal spiral: add rings
    # Ring 0: 1 point
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # Total: 1 + 6 + 12 + 18 = 37 points
    # We need 32, so we'll take the first 32 in spiral order
    
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    current = [(0, 0)]
    
    ring = 1
    while len(pts) < 32:
        # Start of ring
        # The starting point of ring r is at distance r from center
        # In the hexagonal lattice, ring r starts at (r, 0)
        start = (ring, 0)
        
        # Generate all points in this ring
        ring_points = []
        pos = list(start)
        for i in range(6):
            # Move along direction i for 'ring' steps
            d = dirs[i]
            for step in range(ring):
                ring_points.append(tuple(pos))
                pos = (pos[0] + d[0], pos[1] + d[1])
        
        # Add points from this ring until we have 32
        for p in ring_points:
            if len(pts) >= 32:
                break
            if p not in pts:
                pts.append(p)
        
        ring += 1
    
    return pts[:32]

def _local_search(pts, deadline, max_iter=5000):
    """Local search to maximize edge count."""
    pts = pts[:]
    current_score = _edge_count(pts)
    pset = set(pts)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        random.shuffle(pts)
        
        for i in range(len(pts)):
            if time.time() > deadline:
                break
            
            p = pts[i]
            # Try moving to a random neighbor
            candidates = []
            for n in _neighbors(p):
                if abs(n[0]) > 64 or abs(n[1]) > 64:
                    continue
                if n in pset and n != p:
                    continue
                
                # Check validity: distance to all other points
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
            
            # Pick a random candidate
            n = random.choice(candidates)
            
            # Compute new score
            new_pts = pts[:]
            new_pts[i] = n
            new_score = _edge_count(new_pts)
            
            if new_score >= current_score:
                pts = new_pts
                current_score = new_score
                improved = True
                pset = set(pts)
                break
        
        if not improved:
            # Try a random perturbation to escape local optimum
            i = random.randint(0, len(pts) - 1)
            p = pts[i]
            candidates = []
            for n in _neighbors(p):
                if abs(n[0]) > 64 or abs(n[1]) > 64:
                    continue
                if n in pset and n != p:
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
                if new_score >= current_score or random.random() < 0.1:
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
    cooling = 0.999
    
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
            if n in pset and n != p:
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
        if delta > 0 or random.random() < pow(2.718, delta / temperature):
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
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Phase 1: Start with hexagonal spiral
    pts = _build_hexagonal_spiral()
    if not _is_valid(pts):
        pts = [[a, b] for a in range(8) for b in range(4)]
    
    # Local search
    pts, score = _local_search(pts, deadline, max_iter=2000)
    best_score = score
    best_points = pts[:]
    
    if best_score >= 76:
        return [[p[0], p[1]] for p in best_points]
    
    # Phase 2: Simulated annealing
    pts, score = _simulated_annealing(best_points, deadline, max_iter=5000)
    if score > best_score:
        best_score = score
        best_points = pts[:]
    
    if best_score >= 76:
        return [[p[0], p[1]] for p in best_points]
    
    # Phase 3: Multiple restarts with different initial configurations
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Create perturbed version of best solution
        pts = best_points[:]
        for _ in range(3):
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
        pts, score = _local_search(pts, deadline, max_iter=1000)
        if score > best_score:
            best_score = score
            best_points = pts[:]
        
        if best_score >= 76:
            break
    
    if best_points is None:
        best_points = [[a, b] for a in range(16) for b in range(2)]
    
    return [[p[0], p[1]] for p in best_points]
