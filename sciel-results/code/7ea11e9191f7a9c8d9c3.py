"""No-isosceles set construction with efficient greedy and local search."""
import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # Safety margin before 330s kill
    
    def is_valid_point(p, points, dist_sets):
        """Check if point p can be added without creating isosceles triangle."""
        px, py = p
        n = len(points)
        if n == 0:
            return True
        if n == 1:
            return True
        
        # Condition 1: All distances from p to existing points must be distinct
        dists_from_p = []
        for ax, ay in points:
            d = (px - ax) ** 2 + (py - ay) ** 2
            dists_from_p.append(d)
        
        if len(set(dists_from_p)) != n:
            return False  # Two points equidistant from p -> p is apex
        
        # Condition 2: For each existing point A, d(p,A) must not equal d(A,B) for any other B
        for i, (ax, ay) in enumerate(points):
            d_pa = dists_from_p[i]
            # Check if d_pa is in dist_sets[i] (distances from A to other points)
            if d_pa in dist_sets[i]:
                return False
        
        return True
    
    def add_point(p, points, dist_sets):
        """Add point p to the set, updating distance sets."""
        px, py = p
        points.append(p)
        dist_sets.append(set())
        idx = len(points) - 1
        for j in range(idx):
            ax, ay = points[j]
            d = (px - ax) ** 2 + (py - ay) ** 2
            dist_sets[idx].add(d)
            dist_sets[j].add(d)
    
    def try_greedy(order, points, dist_sets):
        """Try to add points in given order, return number added."""
        for p in order:
            if time.time() > deadline:
                break
            if is_valid_point(p, points, dist_sets):
                add_point(p, points, dist_sets)
    
    def local_search(points, dist_sets, max_iters=5000):
        """Try to improve by removing one point and adding two others."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(points)
            for i in range(n):
                if time.time() > deadline:
                    break
                # Try removing point i
                removed = points.pop(i)
                # Rebuild dist_sets (expensive but correct)
                new_dist_sets = [set() for _ in range(n - 1)]
                for a in range(n - 1):
                    for b in range(a + 1, n - 1):
                        d = (points[a][0] - points[b][0]) ** 2 + (points[a][1] - points[b][1]) ** 2
                        new_dist_sets[a].add(d)
                        new_dist_sets[b].add(d)
                
                # Try to add back removed point or find better replacement
                best_additions = []
                # Try adding the removed point back
                if is_valid_point(removed, points, new_dist_sets):
                    add_point(removed, points, new_dist_sets)
                    # Now try to add one more point
                    for p in all_candidates:
                        if p in points:
                            continue
                        if is_valid_point(p, points, new_dist_sets):
                            add_point(p, points, new_dist_sets)
                            improved = True
                            break
                    if improved:
                        break
                else:
                    # Try to find a different point to add
                    for p in all_candidates:
                        if p in points:
                            continue
                        if is_valid_point(p, points, new_dist_sets):
                            add_point(p, points, new_dist_sets)
                            improved = True
                            break
                    if improved:
                        break
                    else:
                        # Restore
                        points.insert(i, removed)
                        # Rebuild
                        dist_sets = [set() for _ in range(len(points))]
                        for a in range(len(points)):
                            for b in range(a + 1, len(points)):
                                d = (points[a][0] - points[b][0]) ** 2 + (points[a][1] - points[b][1]) ** 2
                                dist_sets[a].add(d)
                                dist_sets[b].add(d)
        
        return points, dist_sets
    
    all_candidates = [(x, y) for y in range(N) for x in range(N)]
    
    best_points = []
    best_size = 0
    
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 13)
        points = []
        dist_sets = []
        
        # Random order for greedy
        order = all_candidates[:]
        random.shuffle(order)
        
        try_greedy(order, points, dist_sets)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points[:]
        
        # Try local search if we have time
        if len(points) > best_size * 0.8 and time.time() < deadline - 60:
            points, dist_sets = local_search(points, dist_sets)
            if len(points) > best_size:
                best_size = len(points)
                best_points = points[:]
    
    return best_points
# EVOLVE-BLOCK-END
