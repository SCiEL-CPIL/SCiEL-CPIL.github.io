"""Rewritten program: search for 30 integer points in [-30,30]^2 maximizing
the largest multiplicity of a single positive doubled triangle area."""
import time
import numpy as np
from itertools import combinations

_DEADLINE = time.time() + 80  # safety margin before 90s kill

def _doubled_areas(pts):
    """Return array of absolute doubled areas for all triples (excluding zero)."""
    n = len(pts)
    P = np.array(pts, dtype=np.int64)
    # For each pair (i,j), compute cross product with all k
    # area2(i,j,k) = |x_i*(y_j-y_k) + x_j*(y_k-y_i) + x_k*(y_i-y_j)|
    # Vectorize: for each (i,j), compute with all k
    # Build index arrays for all triples i<j<k
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    # Only keep i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    xi = P[i_idx, 0]; yi = P[i_idx, 1]
    xj = P[j_idx, 0]; yj = P[j_idx, 1]
    xk = P[k_idx, 0]; yk = P[k_idx, 1]
    
    det = xi*(yj - yk) + xj*(yk - yi) + xk*(yi - yj)
    abs_det = np.abs(det)
    # Exclude zero-area
    abs_det = abs_det[abs_det > 0]
    return abs_det

def _score(pts):
    """Return the largest multiplicity of a single positive doubled area."""
    areas = _doubled_areas(pts)
    if len(areas) == 0:
        return 0
    vals, counts = np.unique(areas, return_counts=True)
    return int(counts.max())

def _valid_points(pts):
    """Check that pts has exactly 30 distinct integer points in [-30,30]^2."""
    if len(pts) != 30:
        return False
    seen = set()
    for x, y in pts:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _neighborhood_move(pts, rng):
    """Generate a neighboring configuration by moving one point slightly."""
    pts = list(pts)
    idx = rng.integers(0, 30)
    dx = rng.integers(-3, 4)
    dy = rng.integers(-3, 4)
    if dx == 0 and dy == 0:
        return None
    nx = pts[idx][0] + dx
    ny = pts[idx][1] + dy
    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
        return None
    new_pts = pts[:]
    new_pts[idx] = (nx, ny)
    # Check distinctness
    seen = set()
    for p in new_pts:
        if p in seen:
            return None
        seen.add(p)
    return new_pts

def _large_move(pts, rng):
    """Generate a configuration by randomly reassigning one or more points."""
    pts = list(pts)
    n_moves = rng.integers(1, 4)
    for _ in range(n_moves):
        idx = rng.integers(0, 30)
        nx = rng.integers(-30, 31)
        ny = rng.integers(-30, 31)
        new_pts = pts[:]
        new_pts[idx] = (nx, ny)
        seen = set()
        valid = True
        for p in new_pts:
            if p in seen:
                valid = False
                break
            seen.add(p)
        if valid:
            pts = new_pts
    return pts

def _initial_configurations(rng):
    """Generate diverse initial configurations."""
    configs = []
    
    # Grid-like
    for start in range(-30, 0, 3):
        pts = []
        for x in range(start, start + 6):
            for y in range(start, start + 5):
                pts.append((x, y))
        if len(pts) == 30 and _valid_points(pts):
            configs.append(pts)
    
    # Diagonal strip
    for start in range(-20, 0, 2):
        pts = []
        for i in range(30):
            x = start + i
            y = start + (i % 6)
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
        if len(pts) == 30 and _valid_points(pts):
            configs.append(pts)
    
    # Parabolic
    for a in range(1, 5):
        pts = []
        for i in range(30):
            x = -15 + i
            y = a * (x * x) // 10 - 15
            y = max(-30, min(30, y))
            pts.append((x, y))
        if len(pts) == 30 and _valid_points(pts):
            configs.append(pts)
    
    # Random
    for _ in range(10):
        pts = []
        while len(pts) < 30:
            x = rng.integers(-30, 31)
            y = rng.integers(-30, 31)
            if (x, y) not in pts:
                pts.append((x, y))
        if _valid_points(pts):
            configs.append(pts)
    
    # Structured: rows with offsets
    for base in range(-20, 0, 2):
        pts = []
        for row in range(5):
            for col in range(6):
                x = base + col
                y = base + row * 2
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        if len(pts) == 30 and _valid_points(pts):
            configs.append(pts)
    
    return configs

def construct():
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    configs = _initial_configurations(rng)
    
    # Score all initial configs
    best_score = 0
    best_pts = None
    for cfg in configs:
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg
    
    if best_pts is None:
        # Fallback
        best_pts = [(x, y) for x in range(6) for y in range(5)]
        best_score = _score(best_pts)
    
    # Simulated annealing
    current_pts = best_pts[:]
    current_score = best_score
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < _DEADLINE:
        # Try a move
        if rng.random() < 0.7:
            new_pts = _neighborhood_move(current_pts, rng)
        else:
            new_pts = _large_move(current_pts, rng)
        
        if new_pts is None:
            T *= cooling
            continue
        
        new_score = _score(new_pts)
        
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = new_pts[:]
        else:
            # Accept with probability exp((new - curr) / T)
            if T > T_min:
                prob = np.exp((new_score - current_score) / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
            T *= cooling
        
        # Periodically restart from best if stuck
        if rng.random() < 0.001:
            current_pts = best_pts[:]
            current_score = best_score
            T = 3.0
    
    # Final local polish: try all single-point moves
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        for i in range(30):
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx = best_pts[i][0] + dx
                    ny = best_pts[i][1] + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    new_pts = best_pts[:]
                    new_pts[i] = (nx, ny)
                    seen = set()
                    valid = True
                    for p in new_pts:
                        if p in seen:
                            valid = False
                            break
                        seen.add(p)
                    if not valid:
                        continue
                    s = _score(new_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = new_pts
                        improved = True
    
    return best_pts

# EVOLVE-BLOCK-START
# (construct is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
