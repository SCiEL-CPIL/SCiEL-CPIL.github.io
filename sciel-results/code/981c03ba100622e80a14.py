import time
import random
import math

N = 64

def construct():
    deadline = time.time() + 315
    random.seed(123)
    
    def dist2(p, q):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx * dx + dy * dy
    
    def can_add(pts, p):
        """Check if adding p to pts keeps the set valid (no isosceles triangle)."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = dist2(pts[i], pts[j])
                d_pi = dist2(p, pts[i])
                d_pj = dist2(p, pts[j])
                if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                    return False
        return True
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    point_set = set(all_points)
    
    def greedy_init():
        pts = []
        remaining = all_points[:]
        random.shuffle(remaining)
        for p in remaining:
            if time.time() > deadline - 5:
                break
            if can_add(pts, p):
                pts.append(p)
        return pts
    
    def local_search(pts, max_iters=100000):
        best = list(pts)
        best_size = len(best)
        no_improve = 0
        max_no_improve = 5000
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            if no_improve > max_no_improve:
                break
            
            move_type = random.random()
            new_pts = list(best)
            accepted = False
            
            if move_type < 0.5:
                # Try to add a point
                p = random.choice(all_points)
                if p not in best:
                    if can_add(best, p):
                        new_pts.append(p)
                        if len(new_pts) > best_size:
                            best = new_pts
                            best_size = len(new_pts)
                            no_improve = 0
                            accepted = True
            elif move_type < 0.8:
                # Try swap: remove one, add one
                if len(best) >= 2:
                    idx = random.randint(0, len(best) - 1)
                    removed = best[idx]
                    new_pts.pop(idx)
                    # Try adding a random point
                    p = random.choice(all_points)
                    if p != removed and p not in best:
                        if can_add(new_pts, p):
                            new_pts.append(p)
                            if len(new_pts) > best_size:
                                best = new_pts
                                best_size = len(new_pts)
                                no_improve = 0
                                accepted = True
            else:
                # Try double swap: remove two, add two
                if len(best) >= 4:
                    idx1 = random.randint(0, len(best) - 1)
                    idx2 = random.randint(0, len(best) - 1)
                    if idx1 == idx2:
                        idx2 = (idx2 + 1) % len(best)
                    removed1 = best[idx1]
                    removed2 = best[idx2]
                    new_pts.pop(max(idx1, idx2))
                    new_pts.pop(min(idx1, idx2))
                    p1 = random.choice(all_points)
                    p2 = random.choice(all_points)
                    if p1 != p2 and p1 not in best and p2 not in best:
                        if can_add(new_pts, p1) and can_add(new_pts + [p1], p2):
                            new_pts.append(p1)
                            new_pts.append(p2)
                            if len(new_pts) > best_size:
                                best = new_pts
                                best_size = len(new_pts)
                                no_improve = 0
                                accepted = True
            
            if not accepted:
                no_improve += 1
        
        return best
    
    best_overall = []
    best_overall_size = 0
    
    # Multiple greedy starts
    for start in range(30):
        if time.time() > deadline - 40:
            break
        
        pts = greedy_init()
        if len(pts) > best_overall_size:
            best_overall = list(pts)
            best_overall_size = len(pts)
        
        improved = local_search(pts, max_iters=50000)
        if len(improved) > best_overall_size:
            best_overall = improved
            best_overall_size = len(improved)
    
    return list(best_overall)
