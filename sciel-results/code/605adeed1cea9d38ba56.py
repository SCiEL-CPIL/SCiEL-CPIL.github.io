"""Improved program: dense lattice subset with fast scoring and targeted local search."""
import random
import time
import numpy as np
from collections import Counter
from itertools import combinations

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples."""
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    if n < 3:
        return 0
    
    # Vectorized computation of all triple determinants
    # For each pair (i,j), compute cross product with all k
    # This is O(n^3) but vectorized
    
    # Method: for each i, compute vectors from i to all other points
    # Then for each pair (j,k) where j,k != i, compute det
    # Use broadcasting: for fixed i, we have (n-1) vectors
    # The determinant for (i,j,k) = v_j x v_k where v_j = pts[j] - pts[i]
    
    # More efficient: precompute all pairwise differences
    # diff[i,j] = pts[j] - pts[i]
    # det(i,j,k) = diff[i,j][0] * diff[i,k][1] - diff[i,j][1] * diff[i,k][0]
    
    # Use the formula: det = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    # Group by first index i
    
    max_count = 0
    for i in range(n):
        # Vectors from point i to all other points
        others = pts[list(range(i)) + list(range(i+1, n))]
        m = len(others)
        if m < 2:
            continue
        
        # All pairwise cross products among vectors from i
        # cross[j,k] = others[j][0]*others[k][1] - others[j][1]*others[k][0]
        x = others[:, 0]
        y = others[:, 1]
        
        # Cross product matrix
        cross = np.outer(x, y) - np.outer(y, x)
        
        # Only consider upper triangle (j < k) to avoid double counting and self
        j_idx, k_idx = np.triu_indices(m, k=1)
        cross_vals = cross[j_idx, k_idx]
        
        # Filter non-zero
        cross_vals = cross_vals[cross_vals != 0]
        
        if len(cross_vals) == 0:
            continue
        
        values, counts = np.unique(cross_vals, return_counts=True)
        local_max = int(np.max(counts))
        if local_max > max_count:
            max_count = local_max
    
    return max_count

def _make_dense_lattice_subset(width, height, n_points):
    """Create a dense rectangular grid subset of n_points points."""
    # Fill a grid row by row
    pts = []
    for y in range(height):
        for x in range(width):
            pts.append((x, y))
            if len(pts) >= n_points:
                return pts
    return pts[:n_points]

def _make_hexagonal_lattice(n_points):
    """Create points on a hexagonal/triangular lattice pattern."""
    pts = []
    # Triangular lattice: (a, b) where a >= 0, b >= 0, a + b <= some limit
    # Or use the standard triangular lattice: (a + b/2, b*sqrt(3)/2)
    # For integer coordinates, use: (2a+b, b) for a,b >= 0
    # This gives a denser packing
    
    for b in range(20):
        for a in range(20):
            x = 2*a + b
            y = b
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
            if len(pts) >= n_points:
                return pts
    return pts[:n_points]

def _make_arithmetic_progression_2d(n_points):
    """Create points along arithmetic progressions in 2D."""
    pts = []
    # Use a set of base directions and create APs
    directions = [(1,0), (0,1), (1,1), (1,-1), (2,1), (1,2)]
    bases = [(0,0), (3,0), (0,3), (3,3), (6,0), (0,6)]
    
    for base in bases:
        for d in directions:
            for t in range(6):
                x = base[0] + d[0]*t
                y = base[1] + d[1]*t
                if -30 <= x <= 30 and -30 <= y <= 30:
                    if (x,y) not in pts:
                        pts.append((x,y))
            if len(pts) >= n_points:
                break
        if len(pts) >= n_points:
            break
    
    # Pad if needed
    if len(pts) < n_points:
        for x in range(-30, 31):
            for y in range(-30, 31):
                if (x,y) not in pts:
                    pts.append((x,y))
                if len(pts) >= n_points:
                    break
            if len(pts) >= n_points:
                break
    
    return pts[:n_points]

def construct():
    deadline = time.time() + 50  # Safety margin: finish well before 60s
    
    rng = random.Random(42)
    
    best_score = 0
    best_points = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Dense rectangular grids of various sizes
    for w in range(4, 10):
        h = (30 + w - 1) // w  # Ceiling to get at least 30
        if h <= 10:
            config = _make_dense_lattice_subset(w, h, 30)
            if len(config) == 30:
                initial_configs.append(config)
    
    # 2. Hexagonal/triangular lattice subsets
    config = _make_hexagonal_lattice(30)
    if len(config) == 30:
        initial_configs.append(config)
    
    # 3. Arithmetic progression based
    config = _make_arithmetic_progression_2d(30)
    if len(config) == 30:
        initial_configs.append(config)
    
    # 4. Various shifted/scaled grids
    for scale in [1, 2, 3]:
        for shift in [(0,0), (5,0), (0,5), (5,5), (10,0), (0,10)]:
            pts = []
            for y in range(6):
                for x in range(6):
                    px = x * scale + shift[0]
                    py = y * scale + shift[1]
                    if -30 <= px <= 30 and -30 <= py <= 30:
                        pts.append((px, py))
                    if len(pts) >= 30:
                        break
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                initial_configs.append(pts)
    
    # 5. Random dense clusters
    for _ in range(10):
        cx = rng.randint(-10, 10)
        cy = rng.randint(-10, 10)
        pts = []
        seen = set()
        attempts = 0
        while len(pts) < 30 and attempts < 1000:
            x = cx + rng.randint(-8, 8)
            y = cy + rng.randint(-8, 8)
            if -30 <= x <= 30 and -30 <= y <= 30 and (x,y) not in seen:
                seen.add((x,y))
                pts.append((x,y))
            attempts += 1
        if len(pts) == 30:
            initial_configs.append(pts)
    
    # Evaluate all initial configurations
    for config in initial_configs:
        if time.time() > deadline - 35:
            break
        s = _score(config)
        if s > best_score:
            best_score = s
            best_points = config[:]
    
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _score(best_points)
    
    # Local search: move one point at a time, try to improve
    current_points = best_points[:]
    current_score = best_score
    
    def _valid(points):
        """Check all points are distinct and in range."""
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
            if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
                return False
        return True
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 3:
        # Pick a random point to move
        idx = rng.randint(0, 29)
        old_pt = current_points[idx]
        
        # Generate a small set of candidate new positions (radius 1-2 only)
        candidates = []
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if -30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30:
                    if new_pt not in current_points:
                        candidates.append(new_pt)
        
        # One random distant position
        new_pt = (rng.randint(-30, 30), rng.randint(-30, 30))
        if new_pt not in current_points:
            candidates.append(new_pt)
        
        # Try candidates, keep the best one that improves or is equal
        best_candidate = None
        best_candidate_score = current_score - 1
        
        for new_pt in candidates:
            current_points[idx] = new_pt
            
            if not _valid(current_points):
                current_points[idx] = old_pt
                continue
            
            new_score = _score(current_points)
            
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = new_pt
        
        if best_candidate is not None and best_candidate_score >= current_score:
            current_points[idx] = best_candidate
            current_score = best_candidate_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            current_points[idx] = old_pt
            no_improve_count += 1
        
        # If stuck for too long, restart from best with perturbation
        if no_improve_count > max_no_improve:
            current_points = best_points[:]
            current_score = best_score
            no_improve_count = 0
            
            # Perturb a few random points
            for _ in range(3):
                idx2 = rng.randint(0, 29)
                old_pt2 = current_points[idx2]
                new_pt2 = (rng.randint(max(-30, old_pt2[0]-5), min(30, old_pt2[0]+5)),
                         rng.randint(max(-30, old_pt2[1]-5), min(30, old_pt2[1]+5)))
                if new_pt2 not in current_points and _valid(current_points):
                    current_points[idx2] = new_pt2
                    current_score = _score(current_points)
    
    return best_points
