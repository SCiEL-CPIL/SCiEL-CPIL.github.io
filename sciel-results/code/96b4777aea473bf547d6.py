import random
import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds safety margin (150s limit)
    
    n = 20
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    def get_collinear_triples(points):
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples
    
    def has_4_collinear(points):
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 4:
                    return True
        return False
    
    def min_vertex_cover_bnb(triples, n_points):
        """Branch and bound for minimum vertex cover of 3-uniform hypergraph."""
        if not triples:
            return 0
        
        best_cover = n_points
        
        # Greedy upper bound
        remaining = set(triples)
        cover = 0
        while remaining:
            counts = {}
            for t in remaining:
                for p in t:
                    counts[p] = counts.get(p, 0) + 1
            if not counts:
                break
            best_point = max(counts, key=counts.get)
            cover += 1
            remaining = {t for t in remaining if best_point not in t}
        best_cover = min(best_cover, cover)
        
        def backtrack(remaining_triples, current_cover):
            nonlocal best_cover
            if not remaining_triples:
                if current_cover < best_cover:
                    best_cover = current_cover
                return
            if current_cover >= best_cover:
                return
            
            # Pick triple with highest total degree
            deg = {}
            for t in remaining_triples:
                for p in t:
                    deg[p] = deg.get(p, 0) + 1
            
            max_sum = -1
            target = None
            for t in remaining_triples:
                s = sum(deg.get(p, 0) for p in t)
                if s > max_sum:
                    max_sum = s
                    target = t
            
            if target is None:
                return
            
            for v in target:
                new_remaining = {rt for rt in remaining_triples if v not in rt}
                backtrack(new_remaining, current_cover + 1)
        
        backtrack(set(triples), 0)
        return best_cover
    
    def calculate_score(points):
        if has_4_collinear(points):
            return None
        triples = get_collinear_triples(points)
        cover_size = min_vertex_cover_bnb(triples, n)
        return n - cover_size
    
    def ilp_find_good_config(seed_points, time_budget=2.0):
        """Use ILP to find a configuration near seed_points that minimizes independence number."""
        # We'll fix the point positions and just optimize which points to "select" 
        # Actually, for this problem, we need to search over configurations.
        # Instead, let's use ILP as a scoring oracle with local search.
        # The ILP approach: for a given set of 20 points, we can compute the score exactly.
        # For searching, we use local search guided by the score.
        return None  # Placeholder - we'll use local search instead
    
    def random_points(coords):
        pts = []
        while len(pts) < n:
            x = random.choice(coords)
            y = random.choice(coords)
            p = (x, y)
            if p not in pts:
                pts.append(p)
        return pts
    
    def local_search(pts, coords, time_limit):
        """Local search to minimize score."""
        current_pts = [list(p) for p in pts]
        current_score = calculate_score(current_pts)
        if current_score is None:
            return None, None
        
        best_score = current_score
        best_pts = current_pts[:]
        
        while time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                original = current_pts[i][:]
                # Try nearby moves
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx = original[0] + dx
                        ny = original[1] + dy
                        if nx < -100 or nx > 100 or ny < -100 or ny > 100:
                            continue
                        new_pts = [p[:] for p in current_pts]
                        new_pts[i] = [nx, ny]
                        
                        # Check distinct
                        if len(set(tuple(p) for p in new_pts)) != n:
                            continue
                        
                        score = calculate_score(new_pts)
                        if score is not None and score < current_score:
                            current_pts = new_pts
                            current_score = score
                            if score < best_score:
                                best_score = score
                                best_pts = current_pts[:]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try random restarts if stuck
                if best_score <= 14:
                    break
                # Perturb randomly
                for _ in range(5):
                    idx = random.randint(0, n - 1)
                    new_x = random.choice(coords)
                    new_y = random.choice(coords)
                    trial = [p[:] for p in current_pts]
                    trial[idx] = [new_x, new_y]
                    if len(set(tuple(p) for p in trial)) == n:
                        score = calculate_score(trial)
                        if score is not None and score <= current_score:
                            current_pts = trial
                            current_score = score
                            if score < best_score:
                                best_score = score
                                best_pts = current_pts[:]
                            break
        
        return best_score, best_pts
    
    # Search strategy:
    # 1. Try several initial configurations
    # 2. For each, run local search
    # 3. Keep the best
    
    coords = list(range(-15, 16))  # Slightly larger grid for diversity
    
    best_score = 21
    best_pts = None
    
    # Initial configurations to try
    initial_configs = []
    
    # Grid-based
    for offset in range(-5, 6, 2):
        pts = []
        for i in range(4):
            for j in range(5):
                x = offset + i * 4
                y = offset + j * 3
                if -100 <= x <= 100 and -100 <= y <= 100:
                    pts.append([x, y])
        if len(pts) == 20:
            initial_configs.append(pts)
    
    # Random initial configs
    for _ in range(10):
        pts = random_points(coords)
        initial_configs.append([list(p) for p in pts])
    
    # Try each initial config with local search
    for init_pts in initial_configs:
        if time.time() > deadline - 10:
            break
        
        # Check validity first
        if has_4_collinear(init_pts):
            # Try to fix
            fixed = False
            for _ in range(50):
                idx = random.randint(0, n - 1)
                new_x = random.choice(coords)
                new_y = random.choice(coords)
                trial = [p[:] for p in init_pts]
                trial[idx] = [new_x, new_y]
                if len(set(tuple(p) for p in trial)) == n and not has_4_collinear(trial):
                    init_pts = trial
                    fixed = True
                    break
            if not fixed:
                continue
        
        score, pts = local_search(init_pts, coords, deadline - 5)
        if score is not None and score < best_score:
            best_score = score
            best_pts = pts
            if best_score <= 12:
                break
    
    # If no valid config found, use a known valid one
    if best_pts is None:
        # Parabola: y = x^2 for x in range
        best_pts = [[x, x * x] for x in range(-10, 10)]
        # Check validity
        if not has_4_collinear(best_pts):
            pass
        else:
            # Fallback to random valid
            for _ in range(1000):
                pts = random_points(coords)
                if not has_4_collinear(pts):
                    best_pts = pts
                    break
    
    return [list(p) for p in best_pts]
