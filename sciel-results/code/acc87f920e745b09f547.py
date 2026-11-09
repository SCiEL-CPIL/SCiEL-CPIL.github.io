"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64
TIME_LIMIT = 280  # Safety margin below 330s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_pts = []
    best_len = 0
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def dist2(a, b):
        return (a[0] - b[0])**2 + (a[1] - b[1])**2
    
    def is_valid_addition(p, pts):
        """Check if adding p to pts creates any isosceles triangle."""
        k = len(pts)
        if k == 0:
            return True
        if k == 1:
            return True
        # For each pair of existing points, check if p forms an isosceles triangle
        for i in range(k):
            for j in range(i + 1, k):
                d_ij = dist2(pts[i], pts[j])
                d_ip = dist2(pts[i], p)
                d_jp = dist2(pts[j], p)
                # Check all three ways to form isosceles
                if d_ij == d_ip or d_ij == d_jp or d_ip == d_jp:
                    return False
        return True
    
    def greedy_construct(pts, time_limit):
        """Continue building pts greedily until time runs out or no more can be added."""
        remaining = [p for p in all_points if p not in pts]
        rng = random.Random(42)
        
        while remaining:
            if time.time() - start_time > time_limit:
                break
            
            # Try to find a point that can be added
            added = False
            for _ in range(min(len(remaining), 100)):
                idx = rng.randint(0, len(remaining) - 1)
                p = remaining[idx]
                if is_valid_addition(p, pts):
                    pts.append(p)
                    remaining.pop(idx)
                    added = True
                    break
            
            if not added:
                break
        
        return pts
    
    def simulated_annealing(pts, time_limit):
        """Simulated annealing to improve the set by accepting temporary reductions."""
        pts = list(pts)
        current_len = len(pts)
        pts_set = set(pts)
        
        # Temperature schedule
        T = 1.0
        cooling = 0.95
        
        while time.time() - start_time < time_limit:
            T *= cooling
            if T < 0.01:
                break
            
            rng = random.Random()
            
            # Try random moves: add, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try to add a random point
                p = rng.choice(all_points)
                if p not in pts_set:
                    if is_valid_addition(p, pts):
                        pts.append(p)
                        pts_set.add(p)
                        current_len = len(pts)
                    else:
                        # Accept with probability based on temperature (rarely for additions)
                        pass
            elif move_type < 0.6:
                # Try to remove a random point
                if current_len > 1:
                    idx = rng.randint(0, current_len - 1)
                    p = pts[idx]
                    pts.pop(idx)
                    pts_set.remove(p)
                    current_len = len(pts)
            else:
                # Try to swap: remove one, add another
                if current_len > 1:
                    idx = rng.randint(0, current_len - 1)
                    p_remove = pts[idx]
                    p_add = rng.choice(all_points)
                    if p_add not in pts_set:
                        # Temporarily remove
                        pts.pop(idx)
                        pts_set.remove(p_remove)
                        # Try to add
                        if is_valid_addition(p_add, pts):
                            pts.append(p_add)
                            pts_set.add(p_add)
                            current_len = len(pts)
                        else:
                            # Revert
                            pts.append(p_remove)
                            pts_set.add(p_remove)
        
        return pts
    
    # Run multiple attempts with different random seeds
    for seed in range(1000):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        rng = random.Random(seed)
        pts = []
        
        # Start with a random point
        if all_points:
            pts.append(rng.choice(all_points))
        
        # Greedily add points
        pts = greedy_construct(pts, TIME_LIMIT)
        
        # Apply simulated annealing to improve
        if time.time() - start_time < TIME_LIMIT - 30:
            pts = simulated_annealing(pts, TIME_LIMIT - 10)
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts[:]
    
    return best_pts
# EVOLVE-BLOCK-END
