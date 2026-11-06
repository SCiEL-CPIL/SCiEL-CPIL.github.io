import numpy as np
import random
import time

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin before 330s kill

    # Generate all grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = N * N

    # Map point to index for fast lookup
    def get_idx(x, y):
        return x * N + y

    # Pre-compute squared distances
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)

    # Compute distance matrix
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = dx**2 + dy**2

    best_set = []
    best_size = 0

    # Greedy construction with incremental distance tracking
    def greedy_random_restart():
        indices = list(range(num_points))
        random.shuffle(indices)
        
        selected = []
        # dist_sets[i] is a set of squared distances from selected[i] to other selected points
        dist_sets = []
        
        for idx in indices:
            if time.time() > deadline - 10:
                break
                
            if not selected:
                selected.append(idx)
                dist_sets.append(set())
                continue
                
            # Calculate squared distances from idx to all selected points
            d_to_selected = dist_sq[idx, selected]
            
            # Check if any distance is duplicated among the new edges
            if len(set(d_to_selected.tolist())) < len(d_to_selected):
                continue
                
            # Check against existing distance sets
            valid = True
            for i, sel_idx in enumerate(selected):
                d = d_to_selected[i]
                if d in dist_sets[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Accept idx
            selected.append(idx)
            dist_sets.append(set(d_to_selected.tolist()))
            
            # Update existing distance sets
            for i in range(len(selected) - 1):
                dist_sets[i].add(d_to_selected[i])
                
        return [points[i] for i in selected]

    # Local search refinement
    def local_search(pts, max_iters=5000):
        # Convert to indices
        selected = [get_idx(p[0], p[1]) for p in pts]
        S_set = set(selected)
        
        # Rebuild dist_sets
        dist_sets = {}
        for i in selected:
            ds = set()
            for j in selected:
                if i != j:
                    ds.add(dist_sq[i, j])
            dist_sets[i] = ds
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            
            # Try random swaps
            if len(selected) == 0:
                break
                
            # Remove a random point
            rem_idx = random.choice(selected)
            selected.remove(rem_idx)
            S_set.remove(rem_idx)
            
            # Update dist_sets for remaining points
            for i in selected:
                dist_sets[i].remove(dist_sq[i, rem_idx])
            del dist_sets[rem_idx]
            
            # Try to add a random point not in S
            # Pick from candidates - try multiple random candidates
            added = False
            for _ in range(20):  # Try 20 random candidates
                new_idx = random.randint(0, num_points - 1)
                if new_idx in S_set:
                    continue
                    
                # Check validity
                d_to_S = dist_sq[new_idx, selected]
                if len(d_to_S) > 0:
                    if len(set(d_to_S.tolist())) != len(d_to_S):
                        continue
                
                valid = True
                for i, sel_idx in enumerate(selected):
                    d = d_to_S[i]
                    if d in dist_sets[sel_idx]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Accept
                selected.append(new_idx)
                S_set.add(new_idx)
                dist_sets[new_idx] = set(d_to_S.tolist())
                for i, sel_idx in enumerate(selected[:-1]):
                    dist_sets[sel_idx].add(d_to_S[i])
                added = True
                improved = True
                break
                
            if not added:
                # Restore removed point
                selected.append(rem_idx)
                S_set.add(rem_idx)
                dist_sets[rem_idx] = set()
                for i in selected[:-1]:
                    d = dist_sq[rem_idx, i]
                    dist_sets[rem_idx].add(d)
                    dist_sets[i].add(d)
                    
        return [points[i] for i in selected]

    # Main loop with multiple restarts
    num_restarts = 0
    while time.time() < deadline - 20:
        num_restarts += 1
        random.seed(num_restarts)
        
        pts = greedy_random_restart()
        
        if len(pts) > best_size:
            best_size = len(pts)
            best_set = pts
            
            # Try to improve with local search if we have time
            if best_size > 50 and time.time() < deadline - 60:
                improved_pts = local_search(best_set)
                if len(improved_pts) > best_size:
                    best_size = len(improved_pts)
                    best_set = improved_pts

    return [(int(x), int(y)) for x, y in best_set]
