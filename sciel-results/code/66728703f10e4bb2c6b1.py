import time
import random
import numpy as np
from collections import Counter

def _score_fast(points):
    """
    Fast scoring using NumPy.
    points: list of tuples (x, y)
    Returns: max multiplicity of a non-zero doubled area.
    """
    P = np.array(points, dtype=np.int32)
    n = len(P)
    
    if n < 3:
        return 0
    
    # Max possible determinant: coords in [-30,30], diffs up to 60
    # det = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # Max |x| = 30, max |y diff| = 60, so max term = 1800, sum of 3 = 5400
    # But actually max is when points are at corners: (30,30),(-30,30),(-30,-30)
    # det = 30*(30-(-30)) + (-30)*((-30)-30) + (-30)*(30-30) = 30*60 + (-30)*(-60) + 0 = 3600
    # So max doubled area is around 3600. Use 4000 to be safe.
    hist_size = 4001
    counts = np.zeros(hist_size, dtype=np.int32)
    
    # Vectorized computation for all triples
    # For each i, compute det with all j,k > i
    for i in range(n - 2):
        pi = P[i]
        # Points after i
        Pj = P[i+1:]
        m = len(Pj)
        if m < 2:
            continue
        
        # All pairs (j, k) with j < k in Pj
        j_idx, k_idx = np.triu_indices(m, k=1)
        
        v1 = Pj[j_idx] - pi
        v2 = Pj[k_idx] - pi
        
        dets = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        # Filter non-zero and within bounds
        mask = (dets > 0) & (dets < hist_size)
        dets = dets[mask]
        
        if len(dets) > 0:
            # Use bincount for fast accumulation
            temp = np.bincount(dets, minlength=hist_size)
            counts += temp
    
    return int(np.max(counts)) if n >= 3 else 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s limit
    
    best_score = 0
    best_points = None
    
    def _try_config(points):
        nonlocal best_score, best_points
        score = _score_fast(points)
        if score > best_score:
            best_score = score
            best_points = points[:]
        return score
    
    # 1. Systematic Grid Search - try various grid configurations
    factors = [(1, 30), (2, 15), (3, 10), (5, 6), (6, 5), (10, 3), (15, 2), (30, 1)]
    
    for W, H in factors:
        max_sx = 60 // (W - 1) if W > 1 else 60
        max_sy = 60 // (H - 1) if H > 1 else 60
        
        for sx in range(1, max_sx + 1):
            for sy in range(1, max_sy + 1):
                grid_width = (W - 1) * sx
                grid_height = (H - 1) * sy
                
                if grid_width <= 60 and grid_height <= 60:
                    points = [(x * sx, y * sy) for x in range(W) for y in range(H)]
                    _try_config(points)
                    
                    # Also try centered grids
                    if W > 1 and H > 1:
                        cx = -grid_width // 2
                        cy = -grid_height // 2
                        points_c = [(x * sx + cx, y * sy + cy) for x in range(W) for y in range(H)]
                        if all(-30 <= px <= 30 and -30 <= py <= 30 for px, py in points_c):
                            _try_config(points_c)
    
    if best_points is None:
        best_points = [(i, 0) for i in range(30)]
        best_score = _score_fast(best_points)
    
    # 2. Targeted search: try to maximize count of a specific small area value
    # Common small doubled areas: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
    # For area A, we want many triples with |det| = A
    
    # Try configurations that naturally produce many small-area triangles
    # e.g., points on a line with small spacing produce area 0 (excluded)
    # Points in a narrow band produce small areas
    
    # Strategy: pick a target area A, then greedily select points that form
    # many triangles of area A with already-selected points
    
    def _greedy_for_area(A, n_points=30, max_iter=5000):
        """
        Greedily select points to maximize count of triangles with doubled area A.
        Returns (points, count_of_area_A)
        """
        if time.time() > deadline - 3:
            return None, 0
        
        # All candidate points in [-30,30]^2
        # But that's 61*61 = 3721 points. Too many to consider all pairs.
        # Instead, start with a small set and grow.
        
        # Start with a few seed points
        random.seed(42)
        seeds = []
        for _ in range(5):
            seeds.append((random.randint(-30, 30), random.randint(-30, 30)))
        
        selected = list(set(seeds))[:5]
        
        # Precompute area contributions
        # For a new point p, the number of new triangles with area A
        # is the count of pairs (i,j) in selected such that |det(p, si, sj)| = A
        
        def _count_new_area(p, selected_pts, target_A):
            """Count pairs in selected_pts that form triangle of area target_A with p."""
            count = 0
            for i in range(len(selected_pts)):
                for j in range(i+1, len(selected_pts)):
                    x1, y1 = p
                    x2, y2 = selected_pts[i]
                    x3, y3 = selected_pts[j]
                    det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if det == target_A:
                        count += 1
            return count
        
        # Greedy: at each step, pick the point that maximizes new area-A triangles
        # To speed up, only consider points near existing ones or in a structured way
        
        # Generate candidate pool: points in a reasonable range
        # Focus on points that can form area A with existing points
        # For area A, if we have points (x1,y1), (x2,y2), the locus of points (x,y)
        # with |det| = A is two lines parallel to the line through (x1,y1),(x2,y2)
        
        # Simpler: just try a limited set of candidate points
        # Use a grid of candidates around the current selected points
        
        for step in range(n_points - len(selected)):
            if time.time() > deadline - 3:
                break
            
            # Generate candidates: points in [-30,30]^2 that are not already selected
            # But that's 3721 points. Let's limit to a subset.
            
            # Candidate strategy: 
            # 1. Points on lines that create area A with pairs in selected
            # 2. Nearby points to existing selected points
            
            candidates = set()
            
            # Add points near existing selected points (within radius 10)
            for (sx, sy) in selected:
                for dx in range(-10, 11):
                    for dy in range(-10, 11):
                        nx, ny = sx + dx, sy + dy
                        if -30 <= nx <= 30 and -30 <= ny <= 30:
                            candidates.add((nx, ny))
            
            # Also add some random candidates
            for _ in range(200):
                candidates.add((random.randint(-30, 30), random.randint(-30, 30)))
            
            # Remove already selected
            selected_set = set(selected)
            candidates = [c for c in candidates if c not in selected_set]
            
            if not candidates:
                break
            
            # Score each candidate by how many new area-A triangles it creates
            best_cand = None
            best_count = -1
            best_total = -1
            
            for cand in candidates:
                new_count = _count_new_area(cand, selected, A)
                # Also consider total score (all areas) as tiebreaker
                # But that's expensive. Just use new_count.
                if new_count > best_count:
                    best_count = new_count
                    best_cand = cand
                    best_total = new_count
            
            if best_cand is None or best_count <= 0:
                # No improvement, pick a random candidate
                best_cand = random.choice(candidates)
            
            selected.append(best_cand)
        
        # Compute actual score
        score = _score_fast(selected)
        return selected, score
    
    # Try greedy for several target areas
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    
    for A in target_areas:
        if time.time() > deadline - 5:
            break
        points, score = _greedy_for_area(A)
        if points is not None:
            _try_config(points)
    
    # 3. Local search / hill climbing on best configuration
    if time.time() > deadline - 5:
        return best_points
    
    current_points = best_points[:]
    current_score = best_score
    n = 30
    
    # Hill climbing: try moving one point at a time
    for round_num in range(20):  # Up to 20 rounds
        if time.time() > deadline - 2:
            break
        
        improved = False
        # Try each point in random order
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            
            ox, oy = current_points[idx]
            
            # Try neighbors in a small window
            # Use a spiral pattern for efficiency
            for r in range(1, 4):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        if abs(dx) != r and abs(dy) != r:
                            continue  # Only outer ring
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx = ox + dx
                        ny = oy + dy
                        
                        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                            continue
                        
                        # Check collision
                        if (nx, ny) in current_points:
                            continue
                        
                        # Apply move
                        current_points[idx] = (nx, ny)
                        new_score = _score_fast(current_points)
                        
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points = current_points[:]
                        else:
                            # Revert
                            current_points[idx] = (ox, oy)
            
            if improved:
                # Restart the round since the configuration changed
                break
        
        if not improved:
            break
    
    return best_points
