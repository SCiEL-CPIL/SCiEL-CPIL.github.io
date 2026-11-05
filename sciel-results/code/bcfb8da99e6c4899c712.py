"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 280
    num_points = N * N
    xs = np.arange(N, dtype=np.int32)
    ys = np.arange(N, dtype=np.int32)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    xs_flat = X.flatten()
    ys_flat = Y.flatten()
    point_coords = [(int(xs_flat[i]), int(ys_flat[i])) for i in range(num_points)]
    
    dx = xs_flat[:, None] - xs_flat[None, :]
    dy = ys_flat[:, None] - ys_flat[None, :]
    dist2 = (dx*dx + dy*dy)
    
    def can_add(point, S_set, dist_map):
        if point in S_set:
            return False
        seen = set()
        for idx_Q in S_set:
            d = int(dist2[point, idx_Q])
            if d in seen:
                return False
            seen.add(d)
            qd = dist_map.get(idx_Q)
            if qd is not None and d in qd:
                return False
        return True
    
    def add_point(point, S, S_set, dist_map):
        S.append(point)
        S_set.add(point)
        new_dists = set()
        for idx_Q in S_set:
            if idx_Q == point:
                continue
            d = int(dist2[point, idx_Q])
            new_dists.add(d)
            if idx_Q in dist_map:
                dist_map[idx_Q].add(d)
        dist_map[point] = new_dists
    
    def remove_point(point, S, S_set, dist_map):
        S.remove(point)
        S_set.discard(point)
        for idx_Q in S_set:
            d = int(dist2[point, idx_Q])
            if idx_Q in dist_map:
                dist_map[idx_Q].discard(d)
        if point in dist_map:
            del dist_map[point]
    
    best_S = []
    best_size = 0
    best_S_set = set()
    best_dist_map = {}
    
    def update_best(S, S_set, dist_map):
        nonlocal best_S, best_size, best_S_set, best_dist_map
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            best_S_set = S_set.copy()
            best_dist_map = {k: set(v) for k, v in dist_map.items()}
    
    # Phase 1: Multiple greedy constructions with different strategies
    strategies = [
        'random', 'random', 'random', 'random', 'random',
        'corners_first', 'center_first', 'sparse', 'sparse', 'sparse',
        'random', 'random', 'random', 'random', 'random',
        'random', 'random', 'random', 'random', 'random',
    ]
    
    for trial in range(len(strategies)):
        if time.time() > deadline - 60:
            break
        rng = random.Random(5000 + trial * 31)
        strat = strategies[trial]
        
        if strat == 'random':
            order = list(range(num_points))
            rng.shuffle(order)
        elif strat == 'corners_first':
            # Start with corners, then random
            corners = [0, N-1, (N-1)*N, (N-1)*N + N-1]
            rest = list(range(num_points))
            rest = [x for x in rest if x not in set(corners)]
            rng.shuffle(rest)
            order = corners + rest
        elif strat == 'center_first':
            c = N // 2
            center = c * N + c
            rest = list(range(num_points))
            rest = [x for x in rest if x != center]
            rng.shuffle(rest)
            order = [center] + rest
        elif strat == 'sparse':
            # Try points that are far apart
            step = max(1, N // 8)
            candidates = []
            for i in range(0, N, step):
                for j in range(0, N, step):
                    candidates.append(i * N + j)
            rest = list(range(num_points))
            rest = [x for x in rest if x not in set(candidates)]
            rng.shuffle(rest)
            order = candidates + rest
        else:
            order = list(range(num_points))
            rng.shuffle(order)
        
        S = []
        S_set = set()
        dist_map = {}
        
        for idx_P in order:
            if time.time() > deadline - 60:
                break
            if idx_P in S_set:
                continue
            if can_add(idx_P, S_set, dist_map):
                add_point(idx_P, S, S_set, dist_map)
        
        update_best(S, S_set, dist_map)
    
    # Phase 2: Aggressive local search
    if best_size > 0:
        S = best_S[:]
        S_set = best_S_set.copy()
        dist_map = {k: set(v) for k, v in best_dist_map.items()}
        rng = random.Random(31337)
        
        stall = 0
        while time.time() < deadline - 3:
            # Try to add points
            improved = False
            for _ in range(500):
                if time.time() > deadline - 3:
                    break
                p = rng.randint(0, num_points - 1)
                if p in S_set:
                    continue
                if can_add(p, S_set, dist_map):
                    add_point(p, S, S_set, dist_map)
                    improved = True
                    update_best(S, S_set, dist_map)
                    stall = 0
            
            if not improved:
                stall += 1
                if stall > 50 and len(S) > 3:
                    # Perturbation: remove 1-3 points
                    n_rem = min(3, len(S) - 2)
                    for _ in range(n_rem):
                        if S:
                            p = rng.choice(S)
                            remove_point(p, S, S_set, dist_map)
                    # Refill greedily
                    remaining = list(range(num_points))
                    rng.shuffle(remaining)
                    for idx_P in remaining:
                        if time.time() > deadline - 3:
                            break
                        if can_add(idx_P, S_set, dist_map):
                            add_point(idx_P, S, S_set, dist_map)
                    update_best(S, S_set, dist_map)
                    stall = 0
    
    return [point_coords[i] for i in best_S]
# EVOLVE-BLOCK-END
