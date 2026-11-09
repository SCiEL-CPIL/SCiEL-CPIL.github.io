import random
import time
import numpy as np

N = 64
M = N * N

def construct():
    deadline = time.time() + 290
    
    xs = [i % N for i in range(M)]
    ys = [i // N for i in range(M)]
    
    def sq_dist_fast(i, j):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        return dx*dx + dy*dy

    all_indices = list(range(M))
    
    def try_add(p, points, dist_sets):
        n = len(points)
        if n == 0:
            return True, [p], [set()]
            
        new_dists = []
        for q in points:
            d = sq_dist_fast(p, q)
            new_dists.append(d)
            
        if len(set(new_dists)) != n:
            return False, None, None
            
        for i, q in enumerate(points):
            if new_dists[i] in dist_sets[i]:
                return False, None, None
                
        # Success
        new_points = points + [p]
        # Copy sets
        new_dist_sets = [s.copy() for s in dist_sets]
        for i in range(n):
            new_dist_sets[i].add(new_dists[i])
        new_dist_sets.append(set(new_dists))
        
        return True, new_points, new_dist_sets

    def try_remove(p, points, dist_sets):
        # Find index
        idx = -1
        for i, q in enumerate(points):
            if q == p:
                idx = i
                break
        if idx == -1:
            return points, dist_sets
            
        new_points = points[:idx] + points[idx+1:]
        new_dist_sets = [s.copy() for s in dist_sets]
        
        # Remove distances from p to others
        for i, q in enumerate(points):
            if i == idx:
                continue
            d = sq_dist_fast(p, q)
            if d in new_dist_sets[i]:
                new_dist_sets[i].remove(d)
                
        new_dist_sets.pop(idx)
        
        return new_points, new_dist_sets

    def greedy_init(seed):
        random.seed(seed)
        points = []
        dist_sets = []
        candidates = all_indices[:]
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline:
                break
            success, new_points, new_dist_sets = try_add(p, points, dist_sets)
            if success:
                points = new_points
                dist_sets = new_dist_sets
                
        return points, dist_sets

    def local_search(points, dist_sets, time_limit):
        current_points = points[:]
        current_dist_sets = [s.copy() for s in dist_sets]
        current_size = len(current_points)
        
        best_points = current_points[:]
        best_dist_sets = [s.copy() for s in current_dist_sets]
        best_size = current_size
        
        T = 10.0
        cooling = 0.9995
        min_T = 0.1
        
        while time.time() < time_limit:
            if random.random() < 0.5:
                # Add
                p = random.choice(all_indices)
                if p in current_points:
                    continue
                success, new_points, new_dist_sets = try_add(p, current_points, current_dist_sets)
                if success:
                    current_points = new_points
                    current_dist_sets = new_dist_sets
                    current_size = len(current_points)
                    if current_size > best_size:
                        best_size = current_size
                        best_points = current_points[:]
                        best_dist_sets = [s.copy() for s in current_dist_sets]
            else:
                # Swap
                if not current_points:
                    continue
                p_remove = random.choice(current_points)
                new_points, new_dist_sets = try_remove(p_remove, current_points, current_dist_sets)
                
                p_add = random.choice(all_indices)
                # Ensure p_add is not in new_points
                # Since new_points is smaller, it's likely not there, but check
                while p_add in new_points:
                    p_add = random.choice(all_indices)
                    
                success, final_points, final_dist_sets = try_add(p_add, new_points, new_dist_sets)
                if success:
                    new_size = len(final_points)
                    delta = new_size - current_size
                    
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = random.random() < 0.5
                    else:
                        accept = random.random() < np.exp(-delta / T)
                        
                    if accept:
                        current_points = final_points
                        current_dist_sets = final_dist_sets
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_points = current_points[:]
                            best_dist_sets = [s.copy() for s in current_dist_sets]
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return best_points, best_dist_sets

    best_global_points = []
    best_global_size = 0
    
    # Run multiple restarts
    # Estimate time per restart
    # Greedy is fast. SA is the main cost.
    # Let's do 20 restarts.
    
    for i in range(20):
        if time.time() > deadline - 10:
            break
        
        p, ds = greedy_init(i)
        
        # Run SA for a portion of time
        # If we have 290s, and 20 restarts, that's ~14s each.
        # But we want to be safe.
        # Let's run SA until deadline - 5, but break if we have done enough restarts?
        # No, just run SA for a fixed slice.
        
        # Better: Run SA for at most 10 seconds per restart.
        ls_deadline = min(time.time() + 10.0, deadline - 5)
        
        bp, bds = local_search(p, ds, ls_deadline)
        
        if len(bp) > best_global_size:
            best_global_size = len(bp)
            best_global_points = bp[:]
            
    result = [(i % N, i // N) for i in best_global_points]
    
    return result
