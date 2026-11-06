"""Robust program: Hexagonal spiral + local search for max edges on triangular lattice."""
import time
import random

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _edge_count(Sset):
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _build_hexagonal_spiral(n):
    """Build a hexagonal spiral configuration with n points."""
    points = [(0, 0)]
    if n <= 1:
        return points
    
    # Hexagonal spiral layers
    # Layer k has 6k points
    layer = 1
    while len(points) < n:
        # Add points in layer 'layer'
        # The 6 directions for layer k:
        # Starting from (k, 0) and going around
        start = (layer, 0)
        # 6 edges of the hexagon
        directions = [
            (0, 1),    # up
            (-1, 1),   # up-left
            (-1, 0),   # left
            (0, -1),   # down
            (1, -1),   # down-right
            (1, 0)     # right
        ]
        
        pos = list(start)
        for d_idx in range(6):
            dx, dy = directions[d_idx]
            for _ in range(layer):
                if len(points) >= n:
                    break
                points.append(tuple(pos))
                pos[0] += dx
                pos[1] += dy
            if len(points) >= n:
                break
        layer += 1
    
    return points[:n]

def _local_search(Sset, pool, deadline, max_iter=5000):
    """Local search to maximize edge count."""
    current = set(Sset)
    best_score = _edge_count(current)
    best_set = set(current)
    pool = set(pool)
    
    improved = True
    iter_count = 0
    while improved and time.time() < deadline and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        # Try all possible swaps
        current_list = list(current)
        pool_list = list(pool)
        
        # Shuffle for diversity
        random.shuffle(current_list)
        random.shuffle(pool_list)
        
        best_swap_score = best_score
        best_swap_out = None
        best_swap_in = None
        
        for p_out in current_list:
            for p_in in pool_list:
                if p_in in current:
                    continue
                new_set = set(current)
                new_set.remove(p_out)
                new_set.add(p_in)
                score = _edge_count(new_set)
                if score > best_swap_score:
                    best_swap_score = score
                    best_swap_out = p_out
                    best_swap_in = p_in
                    if best_swap_score >= 76:  # Theoretical max
                        break
            if best_swap_score >= 76:
                break
        
        if best_swap_out is not None:
            current.remove(best_swap_out)
            current.add(best_swap_in)
            pool.remove(best_swap_in)
            pool.add(best_swap_out)
            best_score = best_swap_score
            best_set = set(current)
            improved = True
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s limit
    
    # Build initial hexagonal spiral configuration
    initial_points = _build_hexagonal_spiral(32)
    initial_set = set(initial_points)
    
    # Validate initial configuration
    if len(initial_set) != 32:
        # Fallback to a simpler configuration
        initial_set = set((a, b) for a in range(8) for b in range(4))
    
    # Build candidate pool: points within reasonable distance
    pool = set()
    for a in range(-8, 9):
        for b in range(-8, 9):
            if (a, b) not in initial_set:
                pool.add((a, b))
    
    # Run local search
    best_set, best_score = _local_search(initial_set, pool, deadline)
    
    # If we haven't reached 76, try a few more restarts with different initial configs
    if best_score < 76 and time.time() < deadline - 10:
        random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Perturb the current best solution
            perturbed = set(best_set)
            for _ in range(3):
                if perturbed:
                    p_out = random.choice(list(perturbed))
                    perturbed.remove(p_out)
                    if pool:
                        p_in = random.choice(list(pool))
                        perturbed.add(p_in)
                        pool.remove(p_in)
                        pool.add(p_out)
            
            new_set, new_score = _local_search(perturbed, pool, deadline - 5)
            if new_score > best_score:
                best_set = new_set
                best_score = new_score
                if best_score >= 76:
                    break
    
    # Final validation
    result = [[a, b] for a, b in best_set]
    if len(result) != 32:
        # Emergency fallback
        result = [[a, b] for a in range(8) for b in range(4)]
    
    return result
