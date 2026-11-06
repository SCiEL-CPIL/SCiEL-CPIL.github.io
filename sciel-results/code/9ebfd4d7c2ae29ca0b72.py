import time
import random
import math

# EVOLVE-BLOCK-START

def _is_valid(points):
    """
    Check if the configuration is valid:
    1. 20 distinct points.
    2. |x|, |y| <= 100.
    3. No 4 collinear.
    """
    n = len(points)
    if n != 20:
        return False
    
    # Check bounds and distinctness
    seen = set()
    for p in points:
        x, y = p
        if not (-100 <= x <= 100 and -100 <= y <= 100):
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
        
    # Check no 4 collinear
    # For each point i, check slopes to others.
    # If any slope count >= 3, then i + 3 others are collinear (4 total).
    for i in range(n):
        xi, yi = points[i]
        slopes = {}
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize slope
            if dx == 0:
                slope = (0, 1)
            elif dy == 0:
                slope = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
                slope = (dx, dy)
            slopes[slope] = slopes.get(slope, 0) + 1
        for count in slopes.values():
            if count >= 3:
                return False
    return True

def _build_hypergraph(points):
    """
    Build the list of collinear triples.
    Returns list of (i, j, k) indices.
    """
    n = len(points)
    triples = []
    # For each pair (i, j), find k such that i, j, k are collinear
    # To avoid duplicates, only add (i, j, k) if i < j < k
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            for k in range(j + 1, n):
                xk, yk = points[k]
                # Cross product check
                if dx * (yk - yi) == dy * (xk - xi):
                    triples.append((i, j, k))
    return triples

def _score(points):
    """
    Calculate the size of the Maximum Independent Set (MIS) in the 3-uniform hypergraph.
    """
    n = len(points)
    triples = _build_hypergraph(points)
    
    if not triples:
        return n
        
    # Build adjacency for fast conflict checking
    # For each vertex v, store list of (u, w) pairs such that (v, u, w) is a triple
    # Note: We need to be careful with ordering in the triple list.
    # Let's just store all partners.
    # Actually, for B&B, we process vertices in a specific order.
    
    # Precompute for each vertex v, the list of triples containing v.
    # triple_masks is a list of bitmasks for each triple.
    # point_triples[v] is a list of indices into triples list? 
    # Or better: for each v, list of pairs (u, w) that form a triple with v.
    
    v_pairs = [[] for _ in range(n)]
    for i, j, k in triples:
        v_pairs[i].append((j, k))
        v_pairs[j].append((i, k))
        v_pairs[k].append((i, j))
        
    # Degree
    degrees = [len(p) for p in v_pairs]
    
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Remap indices to order? No, just iterate through 'order' in B&B.
    # But we need to check conflicts with vertices already in mask.
    
    best_size = 0
    start_time = time.time()
    time_limit = 0.1 # 100ms per score evaluation is plenty for N=20
    
    def bnb(idx, cur_mask, cur_size):
        nonlocal best_size
        if time.time() - start_time > time_limit:
            return
            
        # Pruning 1: Upper bound
        if cur_size + (n - idx) <= best_size:
            return
            
        if idx == n:
            if cur_size > best_size:
                best_size = cur_size
            return
            
        v = order[idx]
        
        # Option 1: Include v
        # Check if v conflicts with any triple where other two are in cur_mask
        valid = True
        for u, w in v_pairs[v]:
            if (cur_mask & (1 << u)) and (cur_mask & (1 << w)):
                valid = False
                break
                
        if valid:
            bnb(idx + 1, cur_mask | (1 << v), cur_size + 1)
            
        # Option 2: Exclude v
        bnb(idx + 1, cur_mask, cur_size)

    bnb(0, 0, 0)
    return best_size

def _initial_config():
    """
    Generate a dense initial configuration.
    """
    random.seed(42)
    points = []
    # Use a small grid to encourage collinearities
    # 5x4 grid
    for i in range(5):
        for j in range(4):
            points.append([i * 2, j * 2])
            
    # Perturb to break 4-collinearities while keeping 3-collinearities
    # The 5x4 grid has many 4-collinearities (rows, cols, diagonals).
    # We need to repair.
    points = _repair(points)
    return points

def _repair(points):
    """
    Move points to remove 4-collinearities.
    """
    random.seed(123)
    for _ in range(1000):
        if _is_valid(points):
            return points
            
        # Find a violation
        violated = False
        for i in range(20):
            xi, yi = points[i]
            slopes = {}
            slope_counts = {}
            for j in range(20):
                if i == j: continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx //= g
                    dy //= g
                    if dx < 0:
                        dx = -dx
                        dy = -dy
                    slope = (dx, dy)
                if slope not in slope_counts:
                    slope_counts[slope] = []
                slope_counts[slope].append(j)
                
            for slope, pts in slope_counts.items():
                if len(pts) >= 3:
                    violated = True
                    # Move one of the points in pts
                    idx_to_move = random.choice(pts)
                    new_x = random.randint(-10, 10)
                    new_y = random.randint(-10, 10)
                    while [new_x, new_y] in points:
                        new_x = random.randint(-10, 10)
                        new_y = random.randint(-10, 10)
                    points[idx_to_move] = [new_x, new_y]
                    break
            if violated:
                break
                
    if not _is_valid(points):
        # Fallback: random points
        points = []
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] not in points:
                points.append([x, y])
        # This fallback might be invalid, but _is_valid is called in construct loop?
        # Better to ensure validity here.
        # If still invalid, just return and let SA fix it? No, SA checks validity.
        # Let's try a few more repairs.
        for _ in range(5000):
             if _is_valid(points): return points
             # Random move
             i = random.randint(0, 19)
             points[i] = [random.randint(-100, 100), random.randint(-100, 100)]
             
    return points

def construct():
    deadline = time.time() + 115
    random.seed(123)
    
    points = _initial_config()
    if not _is_valid(points):
        # Force valid
        points = _repair(points)
        
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    
    current_points = [p[:] for p in points]
    current_score = best_score
    
    T = 5.0
    T_min = 0.1
    alpha = 0.99
    
    while time.time() < deadline:
        i = random.randint(0, 19)
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        
        if [new_x, new_y] in current_points:
            continue
            
        candidate = [p[:] for p in current_points]
        candidate[i] = [new_x, new_y]
        
        if not _is_valid(candidate):
            continue
            
        new_score = _score(candidate)
        
        if new_score < current_score:
            current_points = candidate
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = candidate
        else:
            delta = new_score - current_score
            if delta == 0:
                if random.random() < 0.5:
                    current_points = candidate
                    current_score = new_score
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_points = candidate
                    current_score = new_score
                    
        T = max(T_min, T * alpha)
        
    return best_points

# EVOLVE-BLOCK-END
