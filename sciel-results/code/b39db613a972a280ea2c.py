import random
import time
import math

N = 64

def construct():
    start_time = time.time()
    timeout = 290  # Leave some margin
    
    best_points = []
    
    # Generate all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def try_greedy(order):
        points = []
        # dists[a] is a set of squared distances from a to other points in points
        dists = {}
        
        for p in order:
            px, py = p
            valid = True
            dist_to_p = []
            
            # Check against existing points
            for i, a in enumerate(points):
                ax, ay = a
                d2 = (px - ax)**2 + (py - ay)**2
                dist_to_p.append((d2, i))
                
            # Condition 1: All distances from p to existing points must be distinct
            d2_vals = [d for d, _ in dist_to_p]
            if len(set(d2_vals)) != len(d2_vals):
                valid = False
            
            if valid:
                # Condition 2 & 3: d(p, a) must not be in dists[a]
                for d2, i in dist_to_p:
                    a = points[i]
                    if d2 in dists[a]:
                        valid = False
                        break
            
            if valid:
                # Add p
                points.append(p)
                new_idx = len(points) - 1
                dists[p] = set(d2_vals)
                
                # Update dists for existing points
                for d2, i in dist_to_p:
                    a = points[i]
                    dists[a].add(d2)
                    
        return points

    # We can try multiple random orders
    # Or a specific heuristic order
    # Random shuffling is a good start
    
    best_count = 0
    trials = 0
    
    while time.time() - start_time < timeout:
        random.shuffle(all_points)
        pts = try_greedy(all_points)
        if len(pts) > best_count:
            best_count = len(pts)
            best_points = pts
        trials += 1
        # If we get stuck or time is running out, break
        if best_count > 1000: # Heuristic: if we find a very good set, maybe stop? 
             # But we can keep trying to improve
             pass

    return best_points
