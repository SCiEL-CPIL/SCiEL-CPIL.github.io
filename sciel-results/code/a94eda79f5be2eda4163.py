"""Triangular lattice packing: maximize edges for 32 points."""
import time
import random

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _dist_sq(p, q):
    """Squared distance between two lattice points a+b*omega."""
    da = p[0] - q[0]
    db = p[1] - q[1]
    # |a + b*omega|^2 = a^2 + ab + b^2 for omega = exp(i*pi/3)
    return da * da + da * db + db * db

def _edge_count(Sset):
    """Count pairs at distance exactly 1."""
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    """Check validity: 32 distinct points, |a|,|b| <= 64, min distance >= 1."""
    if len(Sset) != 32:
        return False
    if len(set(Sset)) != 32:
        return False
    for p in Sset:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    for i, p in enumerate(Sset):
        for q in Sset[i+1:]:
            if _dist_sq(p, q) < 1:
                return False
    return True

def _spiral_solution():
    """Generate a spiral arrangement of 32 points on triangular lattice."""
    # The optimal configuration is a hexagonal spiral (Harborth spiral)
    # Points are placed in concentric hexagonal rings
    points = [(0, 0)]
    
    # Hexagonal ring generation
    # Ring k has 6*k points
    # Ring 0: 1 point (center)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # Total after ring 2: 1 + 6 + 12 = 19
    # Total after ring 3: 1 + 6 + 12 + 18 = 37 (too many)
    
    # So we need 32 points: center + ring 1 (6) + ring 2 (12) = 19, then 13 more from ring 3
    # Actually, let's use a different approach: place points greedily in a spiral
    
    # Better approach: use the known optimal configuration
    # For n=32, the Harborth spiral gives 76 edges
    
    # Let's construct it by placing points in order of distance from origin
    # using a spiral pattern
    
    # The triangular lattice spiral:
    # Step directions in order: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    directions = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    points = [(0, 0)]
    visited = {(0, 0)}
    
    # Spiral: at each "layer", walk 6 steps in each direction
    step = 1
    while len(points) < 32:
        for d in directions:
            da, db = d
            for _ in range(step):
                if len(points) >= 32:
                    break
                new_a = points[-1][0] + da
                new_b = points[-1][1] + db
                pt = (new_a, new_b)
                if pt not in visited:
                    # Check minimum distance constraint
                    valid = True
                    for existing in points:
                        if _dist_sq(pt, existing) < 1:
                            valid = False
                            break
                    if valid:
                        points.append(pt)
                        visited.add(pt)
            if len(points) >= 32:
                break
        step += 1
    
    return [[a, b] for a, b in points]

def _local_search(points, time_limit=10.0):
    """Improve the solution via local search (swap moves)."""
    start_time = time.time()
    Sset = set(tuple(p) for p in points)
    best_edges = _edge_count(Sset)
    best_points = list(Sset)
    
    # Precompute neighbor relationships
    all_neighbors = {}
    for p in Sset:
        all_neighbors[p] = set()
        for q in _neighbors(p):
            if q in Sset:
                all_neighbors[p].add(q)
    
    # Candidate positions to consider for swapping
    # Generate a set of candidate positions near the current points
    candidates = set()
    for p in Sset:
        for r in range(1, 4):
            for da in range(-r, r+1):
                for db in range(-r, r+1):
                    if da == 0 and db == 0:
                        continue
                    if abs(da) > 64 or abs(db) > 64:
                        continue
                    pt = (p[0] + da, p[1] + db)
                    if pt not in Sset:
                        candidates.add(pt)
    
    # Convert to list for random access
    candidate_list = list(candidates)
    random.shuffle(candidate_list)
    
    current_points = list(Sset)
    current_edges = best_edges
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, len(current_points) - 1)
        old_pt = current_points[idx]
        
        # Pick a random candidate position
        new_pt = random.choice(candidate_list)
        
        # Check if new position is valid (minimum distance)
        valid = True
        for other in current_points:
            if other == old_pt:
                continue
            if _dist_sq(new_pt, other) < 1:
                valid = False
                break
        
        if not valid:
            continue
        
        # Compute edge change
        # Remove old_pt, add new_pt
        # Edges involving old_pt that are lost
        lost = 0
        for q in _neighbors(old_pt):
            if q in Sset and q != old_pt:
                lost += 1
        
        # Edges involving new_pt that are gained
        gained = 0
        for q in _neighbors(new_pt):
            if q in Sset and q != old_pt:
                gained += 1
        
        delta = gained - lost
        
        if delta > 0:
            # Accept the move
            Sset.remove(old_pt)
            Sset.add(new_pt)
            current_points[idx] = new_pt
            current_edges += delta
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_points = list(Sset)
        
        # Occasionally do a random restart or perturbation
        if random.random() < 0.01:
            # Perturb: move a random point to a random valid position
            idx2 = random.randint(0, len(current_points) - 1)
            pt2 = current_points[idx2]
            for _ in range(50):
                cand = random.choice(candidate_list)
                valid2 = True
                for other in current_points:
                    if other == pt2:
                        continue
                    if _dist_sq(cand, other) < 1:
                        valid2 = False
                        break
                if valid2:
                    Sset.remove(pt2)
                    Sset.add(cand)
                    current_points[idx2] = cand
                    # Recompute edges
                    current_edges = _edge_count(Sset)
                    break
    
    return best_points, best_edges

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with the spiral solution
    points = _spiral_solution()
    
    if not _valid(points):
        # Fallback: simple grid
        points = [[a, b] for a in range(8) for b in range(4)]
    
    best_points = points
    best_edges = _edge_count(set(tuple(p) for p in points))
    
    # Try local search with multiple random seeds
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed version
        random.seed(seed)
        perturbed = list(best_points)
        for _ in range(5):
            idx = random.randint(0, len(perturbed) - 1)
            # Small random perturbation
            da = random.randint(-2, 2)
            db = random.randint(-2, 2)
            new_pt = [perturbed[idx][0] + da, perturbed[idx][1] + db]
            if abs(new_pt[0]) <= 64 and abs(new_pt[1]) <= 64:
                # Check validity
                valid = True
                for j, other in enumerate(perturbed):
                    if j == idx:
                        continue
                    if _dist_sq(tuple(new_pt), tuple(other)) < 1:
                        valid = False
                        break
                if valid and new_pt not in perturbed:
                    perturbed[idx] = new_pt
        
        # Run local search
        result_points, result_edges = _local_search(perturbed, time_limit=5.0)
        
        if result_edges > best_edges:
            best_edges = result_edges
            best_points = result_points
    
    return [[a, b] for a, b in best_points]
