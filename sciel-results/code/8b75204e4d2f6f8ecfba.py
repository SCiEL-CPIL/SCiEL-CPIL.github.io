import time
import random
import itertools
from collections import defaultdict

def _score(points):
    """Compute the maximum multiplicity of a positive doubled area."""
    counts = {}
    n = len(points)
    for i in range(n - 2):
        x1, y1 = points[i]
        for j in range(i + 1, n - 1):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if d > 0:
                    counts[d] = counts.get(d, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def _generate_grid(w, h, offset_x=0, offset_y=0):
    pts = []
    for x in range(w):
        for y in range(h):
            pts.append((x + offset_x, y + offset_y))
    return pts

def _is_valid(pts):
    if len(pts) != 30:
        return False
    if len(set(pts)) != 30:
        return False
    return all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts)

def _greedy_target_area(target_area, points_so_far, all_points_set, start_points, time_limit):
    """
    Greedily add points to maximize triangles with target_area.
    This is a heuristic to find good initial configurations.
    """
    start_time = time.time()
    pts = list(start_points)
    
    # Precompute areas between new point and existing points
    # We want to pick p such that many triples involving p have area A.
    
    # For efficiency, we'll just iterate candidates.
    # Candidate pool: all valid points not in pts
    # To speed up, we can filter candidates that are "likely" to work?
    # Or just random sampling?
    
    # Let's try a simple hill climbing on the count of target_area triangles.
    # Objective: count of triples with area == target_area.
    
    def _count_target(pts):
        cnt = 0
        n = len(pts)
        for i in range(n - 2):
            x1, y1 = pts[i]
            for j in range(i + 1, n - 1):
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if d == target_area:
                        cnt += 1
        return cnt

    current_count = _count_target(pts)
    
    # Candidate pool
    candidates = [p for p in all_points_set if p not in pts]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try to swap a point in pts with a point in candidates
        # Or just add/remove? We need exactly 30 points.
        # So we must swap.
        
        # Pick a random index in pts
        idx = random.randint(0, len(pts) - 1)
        old_pt = pts[idx]
        
        # Try a few random candidates
        for _ in range(20):
            if not candidates:
                break
            new_pt = random.choice(candidates)
            if new_pt == old_pt:
                continue
            
            # Swap
            pts[idx] = new_pt
            new_count = _count_target(pts)
            
            if new_count > current_count:
                current_count = new_count
                improved = True
                # Move new_pt to candidates, old_pt back?
                # Actually, we just swapped. The set of pts changed.
                # We don't strictly need to maintain the candidate list perfectly,
                # as long as we don't pick a point already in pts.
                # But my loop `new_pt in pts` check is missing in the simple swap above.
                # Let's add a check.
                pass
            else:
                # Revert
                pts[idx] = old_pt
                
        if improved:
            # Rebuild candidate list?
            # This is expensive.
            # Instead, just keep the current candidate list and rely on the `new_pt == old_pt` check
            # and the fact that we are swapping.
            # Wait, if I swap A->B, B is now in pts. If I pick B again from candidates,
            # it's a no-op swap effectively if I don't check membership.
            # Let's check membership in the swap logic.
            
            # Let's restart the candidate list to be safe, but limit frequency.
            if random.random() < 0.1:
                pts_set = set(pts)
                candidates = [p for p in all_points_set if p not in pts_set]
                random.shuffle(candidates)
                
    return pts, current_count

def construct():
    start_time = time.time()
    deadline = 55.0  # 55 seconds to be safe
    
    # Initial candidates
    candidates = [
        _generate_grid(5, 6, -2, -2),
        _generate_grid(6, 5, -2, -2),
        _generate_grid(3, 10, -1, -4),
        _generate_grid(10, 3, -4, -1),
        _generate_grid(2, 15, 0, -7),
        _generate_grid(15, 2, -7, 0),
    ]
    
    valid_candidates = [c for c in candidates if _is_valid(c)]
    if not valid_candidates:
        return [(x, y) for x in range(6) for y in range(5)]
        
    # Evaluate initial candidates
    best_score = -1
    best_points = None
    best_area = None
    for c in valid_candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c[:]
            # Find the area that achieved this score
            # Re-compute counts to find the mode
            counts = {}
            n = len(c)
            for i in range(n - 2):
                x1, y1 = c[i]
                for j in range(i + 1, n - 1):
                    x2, y2 = c[j]
                    for k in range(j + 1, n):
                        x3, y3 = c[k]
                        d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                        if d > 0:
                            counts[d] = counts.get(d, 0) + 1
            if counts:
                best_area = max(counts, key=counts.get)
                
    current_points = best_points[:]
    current_score = best_score
    
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    all_points_set = set(all_points)
    
    # Strategy 1: Targeted Area Optimization
    # Try to optimize for the current best_area
    if best_area is not None:
        # Give 10 seconds for this phase
        time_limit = 10.0
        opt_pts, opt_count = _greedy_target_area(best_area, current_points, all_points_set, current_points, time_limit)
        
        # Evaluate the optimized points with the full score
        new_score = _score(opt_pts)
        if new_score > best_score:
            best_score = new_score
            best_points = opt_pts[:]
            current_points = opt_pts[:]
            current_score = new_score
            
    # Strategy 2: Standard Local Search
    # Continue with the random local search from the current best
    while time.time() - start_time < deadline:
        idx = random.randint(0, 29)
        old_pt = current_points[idx]
        
        new_pt = None
        # Rejection sampling
        for _ in range(10):
            candidate_pt = random.choice(all_points)
            if candidate_pt == old_pt:
                continue
            if candidate_pt in current_points:
                continue
            new_pt = candidate_pt
            break
        
        if new_pt is None:
            continue
            
        current_points[idx] = new_pt
        new_score = _score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        elif new_score == current_score:
            if random.random() < 0.5:
                pass
            else:
                current_points[idx] = old_pt
        else:
            current_points[idx] = old_pt
            
    return best_points
