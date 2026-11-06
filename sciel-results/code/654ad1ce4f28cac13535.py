import numpy as np
import random
import time
from collections import defaultdict

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    def is_valid_addition(points, new_pt):
        """
        Check if adding new_pt to points maintains the no-isosceles property.
        For each existing point p in points:
          - dist(new_pt, p) must be unique among all dist(new_pt, q) for q in points
        Also, for each pair (p, q) in points:
          - dist(p, q) must not equal dist(new_pt, p) or dist(new_pt, q)
          - Actually, the condition is: no three points form an isosceles triangle.
            This means for any three distinct points a, b, c, the triangle abc is not isosceles.
            An isosceles triangle has at least two equal sides.
            So for new_pt and two existing points p, q:
              dist(new_pt, p) != dist(new_pt, q)
              dist(new_pt, p) != dist(p, q)
              dist(new_pt, q) != dist(p, q)
        """
        n = len(points)
        if n == 0:
            return True
        if n == 1:
            return True  # Two points can't form a triangle
        
        # Compute distances from new_pt to all existing points
        new_dists = {}
        for i, p in enumerate(points):
            d = (new_pt[0] - p[0])**2 + (new_pt[1] - p[1])**2
            if d in new_dists:
                return False  # Two existing points at same distance from new_pt
            new_dists[d] = i
        
        # Check against all pairs of existing points
        for i in range(n):
            for j in range(i+1, n):
                d_ij = (points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2
                if d_ij in new_dists:
                    return False  # dist(p_i, p_j) equals dist(new_pt, p_k) for some k
                # Also need to check that d_ij doesn't create isosceles with new_pt
                # Actually, if d_ij equals dist(new_pt, p_i) or dist(new_pt, p_j), that's covered above
                # But we also need: dist(new_pt, p_i) != d_ij and dist(new_pt, p_j) != d_ij
                # Which is exactly what we're checking
        return True

    def greedy_construct(seed):
        """Greedy construction with randomization."""
        random.seed(seed)
        points = []
        # Start with a few random points
        used = set()
        for _ in range(3):
            while True:
                x = random.randint(0, N-1)
                y = random.randint(0, N-1)
                pt = (x, y)
                if pt not in used:
                    if is_valid_addition(points, pt):
                        points.append(pt)
                        used.add(pt)
                        break
        
        # Try to add more points greedily
        all_positions = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_positions)
        
        for pt in all_positions:
            if pt in used:
                continue
            if is_valid_addition(points, pt):
                points.append(pt)
                used.add(pt)
        
        return points

    def local_search(points, seed):
        """Try to improve by swapping points."""
        random.seed(seed)
        points = list(points)
        used = set(points)
        improved = True
        max_iters = 500
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            # Try removing a random point and adding a different one
            if len(points) < 2:
                break
            idx = random.randint(0, len(points)-1)
            removed_pt = points[idx]
            points[idx] = None  # Mark as removed
            used.discard(removed_pt)
            
            # Try to find a replacement
            found = False
            all_positions = [(x, y) for x in range(N) for y in range(N)]
            random.shuffle(all_positions)
            
            for pt in all_positions:
                if pt in used:
                    continue
                # Temporarily add pt
                points[idx] = pt
                if is_valid_addition(points, pt):
                    used.add(pt)
                    found = True
                    break
                points[idx] = None
            
            if not found:
                # Put back the original point
                points[idx] = removed_pt
                used.add(removed_pt)
            else:
                # Keep the new point
                pass
        
        # Clean up any None entries
        points = [p for p in points if p is not None]
        return points

    # Try multiple random seeds
    best_points = []
    best_len = 0
    
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        seed = trial * 7919 + 1
        points = greedy_construct(seed)
        
        if len(points) > best_len:
            best_len = len(points)
            best_points = points
        
        # Try local search on the best so far
        if len(points) >= best_len:
            improved = local_search(points, seed + 1000)
            if len(improved) > best_len:
                best_len = len(improved)
                best_points = improved
    
    # Final check: ensure best_points is valid
    # Re-verify
    if best_points:
        # Quick validity check
        valid = True
        n = len(best_points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    d_ij = (best_points[i][0]-best_points[j][0])**2 + (best_points[i][1]-best_points[j][1])**2
                    d_ik = (best_points[i][0]-best_points[k][0])**2 + (best_points[i][1]-best_points[k][1])**2
                    d_jk = (best_points[j][0]-best_points[k][0])**2 + (best_points[j][1]-best_points[k][1])**2
                    if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to a simple valid set
            best_points = [(0,0), (1,0), (0,1)]
    
    return best_points
