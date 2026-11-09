import time
import itertools
import numpy as np
from collections import Counter
import random

# Precompute indices for all triples (i, j, k) with i < j < k for n=30
_N = 30
_triple_indices = None

def _get_triple_indices():
    global _triple_indices
    if _triple_indices is None:
        indices = []
        for i in range(_N):
            for j in range(i + 1, _N):
                for k in range(j + 1, _N):
                    indices.append((i, j, k))
        _triple_indices = np.array(indices, dtype=np.int32)
    return _triple_indices

def _valid(points):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _compute_areas(points):
    """
    Compute all doubled areas for the given points.
    Returns a list of positive areas.
    """
    pts = np.array(points, dtype=np.int64)
    idx = _get_triple_indices()
    
    i_idx = idx[:, 0]
    j_idx = idx[:, 1]
    k_idx = idx[:, 2]
    
    xi = pts[i_idx, 0]
    yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]
    yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]
    yk = pts[k_idx, 1]
    
    dx1 = xj - xi
    dy1 = yj - yi
    dx2 = xk - xi
    dy2 = yk - yi
    
    areas = np.abs(dx1 * dy2 - dy1 * dx2)
    
    # Filter out zero areas
    positive_areas = areas[areas > 0]
    
    return positive_areas

def _score(points):
    """
    Compute the maximum multiplicity of a single positive doubled area.
    """
    if not _valid(points):
        return 0
    
    a_pos = _compute_areas(points)
    
    if len(a_pos) == 0:
        return 0
        
    # Count frequencies
    # np.unique is efficient for this size
    unique_areas, counts = np.unique(a_pos, return_counts=True)
    max_mult = np.max(counts)
    
    return int(max_mult)

def _fast_score(points):
    """
    Faster scoring assuming points are already a numpy array or list.
    Skips validity check for speed in tight loops.
    """
    pts = np.array(points, dtype=np.int64)
    idx = _get_triple_indices()
    
    i_idx = idx[:, 0]
    j_idx = idx[:, 1]
    k_idx = idx[:, 2]
    
    xi = pts[i_idx, 0]
    yi = pts[i_idx, 1]
    xj = pts[j_idx, 0]
    yj = pts[j_idx, 1]
    xk = pts[k_idx, 0]
    yk = pts[k_idx, 1]
    
    dx1 = xj - xi
    dy1 = yj - yi
    dx2 = xk - xi
    dy2 = yk - yi
    
    areas = np.abs(dx1 * dy2 - dy1 * dx2)
    
    positive_areas = areas[areas > 0]
    
    if len(positive_areas) == 0:
        return 0
        
    unique_areas, counts = np.unique(positive_areas, return_counts=True)
    return int(np.max(counts))

def _generate_candidate(type_id, params=None):
    """
    Generate structured candidates known to have high symmetry.
    """
    if params is None:
        params = {}
        
    points = []
    
    if type_id == 'square_grid':
        # Standard 5x6 grid, centered
        # x: -10, -6, -2, 2, 6, 10
        # y: -8, -4, 0, 4, 8
        xs = [-10, -6, -2, 2, 6, 10]
        ys = [-8, -4, 0, 4, 8]
        for x in xs:
            for y in ys:
                points.append((x, y))
                
    elif type_id == 'hex_grid':
        # Hexagonal lattice subset
        pts = []
        for r in range(5):
            y = (r - 2) * 6 # -12, -6, 0, 6, 12
            for c in range(6):
                x = (c - 2.5) * 4 # -10, -6, -2, 2, 6, 10
                if r % 2 == 1:
                    x += 2
                pts.append((int(x), y))
        points = pts

    elif type_id == 'circle_points':
        # Pick 30 points on circles
        pts = []
        pts.append((0,0))
        pts.extend([(5,0), (0,5), (-5,0), (0,-5), (3,4), (4,3), (-3,-4), (-4,-3)])
        pts.extend([(10,0), (0,10), (-10,0), (0,-10), (6,8), (8,6), (-6,-8), (-8,-6)])
        pts.extend([(15,0), (0,15), (-15,0), (0,-15), (9,12), (12,9), (-9,-12), (-12,-9)])
        pts.extend([(20,0), (0,20), (-20,0), (0,-20), (12,16), (16,12), (-12,-16)])
        points = pts

    elif type_id == 'random':
        pts = []
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        points = pts

    return points

def _local_search(points, time_limit=10.0):
    """
    Hill Climbing with Simulated Annealing style restarts.
    Uses _fast_score for speed.
    """
    start_time = time.time()
    best_points = list(points)
    best_score = _fast_score(best_points)
    
    if best_score == 0:
        return best_points
    
    occupied = set(best_points)
    n = len(best_points)
    
    no_improve_count = 0
    max_no_improve = 50 # Reduced to allow more restarts
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try moving a random point
        if random.random() < 0.7:
            idx = random.randint(0, n - 1)
            px, py = best_points[idx]
            
            # Try a few candidate moves for this point
            for _ in range(5):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                nx, ny = px + dx, py + dy
                
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                new_point = (nx, ny)
                if new_point in occupied:
                    continue
                
                old_point = best_points[idx]
                best_points[idx] = new_point
                occupied.discard(old_point)
                occupied.add(new_point)
                
                s = _fast_score(best_points)
                
                if s > best_score:
                    best_score = s
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    best_points[idx] = old_point
                    occupied.discard(new_point)
                    occupied.add(old_point)
                    
        # 2. Try swapping two random points
        else:
            idx1 = random.randint(0, n - 1)
            idx2 = random.randint(0, n - 1)
            if idx1 == idx2:
                continue
                
            p1 = best_points[idx1]
            p2 = best_points[idx2]
            
            best_points[idx1] = p2
            best_points[idx2] = p1
            
            s = _fast_score(best_points)
            
            if s > best_score:
                best_score = s
                improved = True
                no_improve_count = 0
            else:
                best_points[idx1] = p1
                best_points[idx2] = p2
                
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart with a random perturbation
                for _ in range(3):
                    idx = random.randint(0, n - 1)
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    new_point = (nx, ny)
                    if new_point not in occupied:
                        old_point = best_points[idx]
                        best_points[idx] = new_point
                        occupied.discard(old_point)
                        occupied.add(new_point)
                        
                s = _fast_score(best_points)
                if s > best_score:
                    best_score = s
                no_improve_count = 0

    return best_points

def construct():
    """
    Construct 30 points maximizing the multiplicity of a single doubled area.
    """
    start_time = time.time()
    deadline = start_time + 50 # Strict deadline to ensure < 60s
    
    candidates = []
    
    # Generate structured candidates
    candidates.append(_generate_candidate('square_grid'))
    candidates.append(_generate_candidate('hex_grid'))
    candidates.append(_generate_candidate('circle_points'))
    
    # Add a few random candidates
    for _ in range(3):
        candidates.append(_generate_candidate('random'))
        
    best_points = None
    best_score = 0
    
    # Score all candidates
    for cand in candidates:
        if not _valid(cand):
            continue
        s = _fast_score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            
    if best_points is None:
        # Fallback
        return [(x, y) for x in range(6) for y in range(5)]
        
    # Run local search on the best candidate
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        search_time = min(remaining_time - 2, 40) # Cap search time
        best_points = _local_search(best_points, time_limit=search_time)
        best_score = _fast_score(best_points)
    
    # If we have significant time left, try local search on other top candidates
    if deadline - time.time() > 10:
        # Sort candidates by score
        scored_cands = []
        for cand in candidates:
            if not _valid(cand):
                continue
            s = _fast_score(cand)
            scored_cands.append((s, cand))
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        
        # Try the next best few
        for s, cand in scored_cands[1:4]:
            if time.time() > deadline - 5:
                break
            if cand == best_points:
                continue
            # Only search if it's close to the best
            if s >= best_score - 5:
                remaining = deadline - time.time() - 3
                if remaining > 5:
                    cand_search = _local_search(cand, time_limit=min(remaining, 10))
                    s2 = _fast_score(cand_search)
                    if s2 > best_score:
                        best_score = s2
                        best_points = cand_search
    
    # Final check
    if not _valid(best_points):
        return [(x, y) for x in range(6) for y in range(5)]
        
    return best_points

# EVOLVE-BLOCK-END
