import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    # Precompute: for each pair, the line they define (normalized direction)
    # We'll work with points in [-100, 100]^2
    
    def line_key(p1, p2):
        """Return a canonical key for the line through p1 and p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
        dx //= g
        dy //= g
        # Canonicalize direction
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # For the line itself, we need a unique identifier
        # Use (dx, dy, c) where c is determined by the line equation
        # Line: dy*x - dx*y = c
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)
    
    def count_collinearities(points):
        """Count number of 3-collinear and 4-collinear triples/sets."""
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
        # Count how many lines have exactly 3 points, exactly 4, etc.
        three_count = 0
        four_count = 0
        for key, cnt in line_counts.items():
            if cnt == 3:
                three_count += 1
            elif cnt == 4:
                four_count += 1
            elif cnt > 4:
                four_count += 1  # any line with 4+ is invalid
        return three_count, four_count
    
    def is_valid(points):
        """Check no four collinear and all distinct."""
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
        for key, cnt in line_counts.items():
            if cnt >= 4:
                return False
        # Check distinctness
        if len(set(tuple(p) for p in points)) != n:
            return False
        return True
    
    def compute_score(points):
        """Compute the independence number of the 3-collinearity hypergraph.
        Returns the largest subset with no three collinear."""
        n = len(points)
        # Build the hypergraph: edges are triples that are collinear
        # For efficiency, build for each pair the line, then find all collinear triples
        
        # For each point, build a set of lines through it with other points
        # Then for each line with 3+ points, all triples on that line are "bad"
        
        # Build adjacency: for each pair (i,j), which line do they share?
        # A triple (i,j,k) is collinear if line(i,j) == line(i,k) == line(j,k)
        
        # More efficient: for each point i, group other points by line through i
        point_lines = [{} for _ in range(n)]  # point_lines[i][line_key] = set of points on that line through i
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                key = line_key(points[i], points[j])
                if key not in point_lines[i]:
                    point_lines[i][key] = set()
                point_lines[i][key].add(j)
        
        # Now for each line with 3+ points, all subsets of size 3 are forbidden
        # We need the maximum subset with no 3 collinear
        
        # Build a set of forbidden triples
        forbidden = set()
        for i in range(n):
            for key, pts in point_lines[i].items():
                if len(pts) >= 3:
                    # All triples from {i} + pts are collinear
                    pt_list = list(pts)
                    for a in range(len(pt_list)):
                        for b in range(a + 1, len(pt_list)):
                            forbidden.add((i, pt_list[a], pt_list[b]))
        
        # Now use branch and bound to find max subset with no forbidden triple
        # This is the 3-uniform hypergraph independence number
        
        best = 0
        # Sort points by degree (number of forbidden triples involving them)
        degrees = [0] * n
        for (a, b, c) in forbidden:
            degrees[a] += 1
            degrees[b] += 1
            degrees[c] += 1
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        # Backtracking with pruning
        # State: current set, index in order
        # Upper bound: len(current) + remaining points
        # Better upper bound: current + max possible from remaining given constraints
        
        selected = [False] * n
        
        def backtrack(idx, current_size):
            nonlocal best
            if current_size > best:
                best = current_size
            
            if idx == n:
                return
            
            # Upper bound: current_size + (n - idx)
            if current_size + (n - idx) <= best:
                return
            
            p = order[idx]
            
            # Try including p
            can_include = True
            # Check if adding p creates a forbidden triple with two already selected
            for q in range(idx):
                if not selected[order[q]]:
                    continue
                # Check if (p, order[q], r) is forbidden for any selected r
                # This is expensive; instead precompute for each pair (p,q) the forbidden r's
                pass
            
            # Simpler: just check all pairs in selected
            sel_list = [order[k] for k in range(idx) if selected[order[k]]]
            valid = True
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    a, b = sel_list[i], sel_list[j]
                    triple = tuple(sorted((p, a, b)))
                    if triple in forbidden:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                selected[p] = True
                backtrack(idx + 1, current_size + 1)
                selected[p] = False
            
            # Try excluding p
            backtrack(idx + 1, current_size)
        
        backtrack(0, 0)
        return best
    
    # Simulated Annealing to find good configurations
    # Objective: maximize 3-collinearities while keeping 4-collinearities = 0
    
    # Start from a random valid configuration
    def random_valid_config():
        while True:
            pts = []
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                pts.append([x, y])
            if is_valid(pts):
                return pts
    
    # Initial configuration: use a structured approach
    # Start with points on a curve and perturb
    def initial_config():
        # Points on parabola y = x^2, but with some perturbations to create 3-collinearities
        pts = []
        for i in range(-10, 10):
            pts.append([i, i * i])
        return pts
    
    # Let's try a different approach: use known constructions
    # Or just do SA from a random valid config
    
    # For SA, we need a fast objective function
    def objective(points):
        """Higher is better: more 3-collinearities, penalize 4-collinearities heavily."""
        three, four = count_collinearities(points)
        if four > 0:
            return -1000000 + three - four * 1000
        return three
    
    # SA
    current = random_valid_config()
    current_obj = objective(current)
    best_config = [p[:] for p in current]
    best_obj = current_obj
    best_score = None  # lazily computed
    
    T = 100.0
    T_min = 0.1
    alpha = 0.999
    
    def neighbor(pts):
        """Generate a neighbor by moving one point slightly."""
        new_pts = [p[:] for p in pts]
        idx = random.randint(0, 19)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            dx = 1
        new_pts[idx][0] += dx
        new_pts[idx][1] += dy
        # Clamp
        new_pts[idx][0] = max(-100, min(100, new_pts[idx][0]))
        new_pts[idx][1] = max(-100, min(100, new_pts[idx][1]))
        return new_pts
    
    elapsed = time.time() - start_time
    while T > T_min and elapsed < deadline - 20:  # leave 20s for final scoring
        new_pts = neighbor(current)
        if not is_valid(new_pts):
            T *= alpha
            elapsed = time.time() - start_time
            continue
        
        new_obj = objective(new_pts)
        delta = new_obj - current_obj
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current = new_pts
            current_obj = new_obj
            if current_obj > best_obj:
                best_obj = current_obj
                best_config = [p[:] for p in current]
        
        T *= alpha
        elapsed = time.time() - start_time
    
    # Also try some structured configurations
    # Configuration 1: Grid-like with perturbations
    structured_configs = []
    
    # Try a few hand-crafted or semi-structured configs
    # Config: points near a circle with some collinear triples
    for trial in range(5):
        if time.time() - start_time > deadline - 15:
            break
        pts = []
        for i in range(20):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            pts.append([x, y])
        if is_valid(pts):
            obj = objective(pts)
            if obj > best_obj:
                best_obj = obj
                best_config = [p[:] for p in pts]
    
    # Now compute the exact score for the best config
    # Use branch and bound
    score = compute_score(best_config)
    
    # Try to improve with more SA iterations if time allows
    if time.time() - start_time < deadline - 30:
        # Try local search from best_config
        for _ in range(100):
            if time.time() - start_time > deadline - 25:
                break
            new_pts = neighbor(best_config)
            if not is_valid(new_pts):
                continue
            new_obj = objective(new_pts)
            if new_obj >= best_obj:
                # Check if it gives a better score
                # Only compute score occasionally (expensive)
                if random.random() < 0.1:
                    new_score = compute_score(new_pts)
                    if new_score < score:
                        score = new_score
                        best_config = new_pts
                        best_obj = new_obj
    
    return best_config
# EVOLVE-BLOCK-END
