"""Combined program: Dense initialization + Simulated Annealing with Exact B&B scoring."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _collinear(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _no_four_collinear(points):
    n = len(points)
    # Precompute all pairs to speed up? No, N=20 is small.
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            pi = points[i]
            pj = points[j]
            for k in range(n):
                if k != i and k != j:
                    if _collinear(pi, pj, points[k]):
                        count += 1
                        if count >= 4:
                            return False
    return True

def _precompute_triples(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _max_indep_bnb(n, triples, time_limit=0.5):
    """
    Exact Branch and Bound for Maximum Independent Set in a 3-uniform hypergraph.
    Vertices: 0..n-1
    Edges: triples of vertices.
    """
    if not triples:
        return n
        
    start_time = time.time()
    best_size = 0
    
    # Build adjacency: for each point, which triples does it belong to?
    point_triples = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        point_triples[i].append(idx)
        point_triples[j].append(idx)
        point_triples[k].append(idx)
        
    # Degree: number of triples each point is in
    degree = [len(pt) for pt in point_triples]
    
    # Order points by degree descending (high degree first prunes better)
    order = sorted(range(n), key=lambda x: -degree[x])
    
    # Precompute triple masks for fast intersection checks
    # mask for triple t: bit i, j, k set
    triple_masks = [0] * len(triples)
    for idx, (i, j, k) in enumerate(triples):
        triple_masks[idx] = (1 << i) | (1 << j) | (1 << k)
        
    # For each point in 'order', precompute the list of triple masks it touches
    # This allows fast check: if any touched triple is fully contained in current_mask, invalid.
    point_triple_masks = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        point_triple_masks[i].append(triple_masks[idx])
        point_triple_masks[j].append(triple_masks[idx])
        point_triple_masks[k].append(triple_masks[idx])

    def bnb(idx, cur_mask, cur_size):
        nonlocal best_size
        if time.time() - start_time > time_limit:
            return
            
        # Pruning 1: Upper bound
        # Max possible size is cur_size + (n - idx)
        if cur_size + (n - idx) <= best_size:
            return
            
        if idx == n:
            if cur_size > best_size:
                best_size = cur_size
            return
            
        p = order[idx]
        
        # Option 1: Include p
        new_mask = cur_mask | (1 << p)
        valid = True
        for tm in point_triple_masks[p]:
            if (new_mask & tm) == tm:
                valid = False
                break
                
        if valid:
            bnb(idx + 1, new_mask, cur_size + 1)
            
        # Option 2: Exclude p
        bnb(idx + 1, cur_mask, cur_size)

    bnb(0, 0, 0)
    return best_size

def _score(points):
    n = len(points)
    triples = _precompute_triples(points)
    return _max_indep_bnb(n, triples, time_limit=0.5)

def _is_valid_config(points):
    # Check distinctness
    if len(set(map(tuple, points))) != len(points):
        return False
    # Check bounds
    for p in points:
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
    # Check no 4 collinear
    if not _no_four_collinear(points):
        return False
    return True

def _initial_config():
    """
    Generate a dense random configuration and repair it to satisfy 'no 4 collinear'.
    Dense config promotes more collinear triples, lowering the MIS.
    """
    random.seed(42)
    points = []
    # Start with points in a small box to maximize density
    # Box size ~ 20x20 allows many lines
    while len(points) < 20:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        p = [x, y]
        if p in points:
            continue
        # Tentatively add
        points.append(p)
        # Check if adding this point creates a 4-collinear set with any existing 3
        # Optimization: just check full set at the end? No, incremental is safer.
        # But N=20 is small, just check full set after generating all 20?
        # If invalid, we restart or repair.
        
    # The above loop might generate invalid sets. Let's do a repair step.
    # Actually, let's generate 20 distinct points in a small box, then repair.
    points = []
    attempts = 0
    while len(points) < 20 and attempts < 10000:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        p = [x, y]
        if p in points:
            attempts += 1
            continue
        points.append(p)
        attempts += 1
        
    # If we didn't get 20 (unlikely with 441 spots), fill randomly
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        p = [x, y]
        if p not in points:
            points.append(p)
            
    # Repair: Ensure no 4 collinear
    # Strategy: While invalid, pick a point involved in a 4-collinear set and move it randomly.
    for _ in range(1000):
        if _no_four_collinear(points):
            return points
            
        # Find a violation
        violated = False
        for i in range(20):
            for j in range(i + 1, 20):
                count = 2
                pi = points[i]
                pj = points[j]
                collinear_pts = []
                for k in range(20):
                    if k != i and k != j:
                        if _collinear(pi, pj, points[k]):
                            count += 1
                            collinear_pts.append(k)
                if count >= 4:
                    violated = True
                    # Move one of the collinear points (randomly choose from i, j, or collinear_pts)
                    candidates = [i, j] + collinear_pts
                    idx_to_move = random.choice(candidates)
                    # Move to a random location
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    # Ensure distinct
                    while [new_x, new_y] in points:
                        new_x = random.randint(-100, 100)
                        new_y = random.randint(-100, 100)
                    points[idx_to_move] = [new_x, new_y]
                    break
            if violated:
                break
        
    # If still invalid (highly unlikely), fallback to grid
    if not _no_four_collinear(points):
        points = [[i, i*i] for i in range(-10, 10)]
        
    return points

def construct():
    deadline = time.time() + 115  # Safety margin
    random.seed(123)
    
    points = _initial_config()
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    
    current_points = [p[:] for p in points]
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.1
    alpha = 0.99
    
    steps = 0
    no_improve_steps = 0
    
    while time.time() < deadline:
        steps += 1
        # Generate neighbor
        # Strategy: Move one point to a random location
        i = random.randint(0, 19)
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        
        # Ensure distinct
        if [new_x, new_y] in current_points:
            continue
            
        candidate = [p[:] for p in current_points]
        candidate[i] = [new_x, new_y]
        
        # Check validity (No 4 collinear)
        if not _no_four_collinear(candidate):
            continue
            
        # Calculate score
        new_score = _score(candidate)
        
        # Acceptance criterion
        if new_score < current_score:
            current_points = candidate
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = candidate
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            # Accept with probability exp(-(delta)/T)
            delta = new_score - current_score
            if delta == 0:
                # Tie, accept with 0.5 prob to avoid getting stuck in local minima of same score
                if random.random() < 0.5:
                    current_points = candidate
                    current_score = new_score
                no_improve_steps += 1
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_points = candidate
                    current_score = new_score
                no_improve_steps += 1
                
        # Cool down
        T = max(T_min, T * alpha)
        
        # Optional: If stuck for too long, restart from best? 
        # Or just keep going. SA should handle it.
        
    return best_points

# EVOLVE-BLOCK-END
