import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave margin for 120s limit
    
    # Helper: Check if 4 points are collinear
    def has_four_collinear(points):
        n = len(points)
        # Map from slope/intercept to count? 
        # Simpler: for each pair, check if 2 more points are on the line.
        # O(N^3) is 20^3 = 8000 ops, very fast.
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Check other points
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    # Cross product check
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        count += 1
                        if count >= 4:
                            return True
        return False

    # Helper: Get all collinear triples
    def get_collinear_triples(points):
        triples = []
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Find k such that k, i, j are collinear
                # To avoid duplicates, we can just collect all (i,j,k) with i<j<k
                # But checking all pairs and adding k is O(N^3).
                # Let's just iterate i<j<k
                pass
        
        # Simpler O(N^3)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        triples.append((i, j, k))
        return triples

    # Helper: Min Vertex Cover on 3-uniform hypergraph
    # Returns the minimum number of vertices to remove to break all triples
    def min_vertex_cover(triples):
        if not triples:
            return 0
        
        # Convert to set of sets for faster removal
        # We use a recursive B&B
        # State: remaining_triples (list of tuples), current_size
        # We want to minimize current_size + min_vc(remaining)
        
        # Optimization: Use a set of triples for O(1) removal check?
        # Since N=20, triples count is small (max ~ C(20,3) = 1140, but usually much less).
        
        best_vc = 20 # Upper bound
        
        def solve(remaining_triples, current_size):
            nonlocal best_vc
            if not remaining_triples:
                if current_size < best_vc:
                    best_vc = current_size
                return
            
            # Pruning
            if current_size >= best_vc:
                return
            
            # Pick a triple
            # Heuristic: Pick a triple that shares vertices with many other triples?
            # For simplicity, pick the first one.
            t = remaining_triples[0]
            
            # Branch on removing u, v, or w
            u, v, w = t
            
            for vertex in [u, v, w]:
                # Remove all triples containing vertex
                new_remaining = [rt for rt in remaining_triples if vertex not in rt]
                solve(new_remaining, current_size + 1)
                
        solve(triples, 0)
        return best_vc

    def score(points):
        triples = get_collinear_triples(points)
        vc = min_vertex_cover(triples)
        return 20 - vc

    # Initialize points
    # Try a dense grid configuration
    # 4x5 grid: x in [-2, 3], y in [-2, 3] -> 5x5=25 points. Pick 20.
    # Or 4x4 grid = 16 points, add 4 more.
    
    # Strategy: Start with a 4x5 grid subset, perturb to avoid 4-collinear.
    # Grid points:
    base_points = []
    for x in range(-2, 4): # -2, -1, 0, 1, 2, 3 (6 values)
        for y in range(-2, 3): # -2, -1, 0, 1, 2 (5 values)
            base_points.append([x, y])
            
    # We have 30 points. We need 20.
    # Let's just take the first 20? No, that might have 4-collinear.
    # Let's use a random subset of a larger grid and then local search.
    
    def generate_random_config():
        # Pick 20 points from a 10x10 grid [-5, 4]
        grid = [[x, y] for x in range(-5, 5) for y in range(-5, 5)]
        random.shuffle(grid)
        return grid[:20]

    best_score = 20
    best_points = None
    
    # Initial attempts
    configs = []
    # 1. Random grid
    for _ in range(10):
        p = generate_random_config()
        if not has_four_collinear(p):
            configs.append(p)
            
    # 2. Structured: 4x5 grid with perturbations
    for _ in range(10):
        p = []
        for x in range(-2, 2): # 4 cols
            for y in range(-2, 3): # 5 rows
                # Perturb
                px = x + random.randint(-1, 1)
                py = y + random.randint(-1, 1)
                if -100 <= px <= 100 and -100 <= py <= 100:
                    p.append([px, py])
        # Check distinctness
        if len(set(tuple(pt) for pt in p)) == 20:
            if not has_four_collinear(p):
                configs.append(p)
                
    if not configs:
        # Fallback: Parabola (score 20)
        return [[x, x * x] for x in range(-10, 10)]

    # Local Search
    for p in configs:
        current_score = score(p)
        if current_score < best_score:
            best_score = current_score
            best_points = [pt[:] for pt in p]
            
    # If we already found a very good score (e.g., <= 10), we might be done.
    # But let's run local search to improve.
    
    # We will perform local search on the best config found so far, 
    # and also on other good configs.
    
    search_configs = [best_points] if best_points else [configs[0]]
    # Add some random valid configs to diversify
    for _ in range(5):
        p = generate_random_config()
        if not has_four_collinear(p):
            search_configs.append(p)
            
    for i, p in enumerate(search_configs):
        current_score = score(p)
        
        while time.time() < deadline:
            # Pick a random point to move
            idx = random.randint(0, 19)
            old_pt = p[idx][:]
            
            # Move to nearby position
            # Step size 1 or 2
            step = random.choice([1, 1, 1, 2])
            new_x = old_pt[0] + random.randint(-step, step)
            new_y = old_pt[1] + random.randint(-step, step)
            
            if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
                continue
                
            new_pt = [new_x, new_y]
            
            # Check distinctness
            if any(pt == new_pt for j, pt in enumerate(p) if j != idx):
                continue
                
            p[idx] = new_pt
            
            # Check 4-collinear
            if has_four_collinear(p):
                p[idx] = old_pt
                continue
                
            # Calculate score
            new_score = score(p)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [pt[:] for pt in p]
            else:
                # Revert
                p[idx] = old_pt
                
            # If we hit a very low score, we can break early
            if best_score <= 8:
                break

    if best_points:
        return best_points
    else:
        return [[x, x * x] for x in range(-10, 10)]
