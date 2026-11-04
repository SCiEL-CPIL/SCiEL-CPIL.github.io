import numpy as np
import time
import random

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    # Generate candidate positions: grid + random + corner/boundary emphasis
    rng = np.random.default_rng(42)
    
    # Grid points
    grid_size = 20
    grid = np.array([(i / (grid_size - 1), j / (grid_size - 1)) for i in range(grid_size) for j in range(grid_size)])
    
    # Random points
    rand_points = rng.random((500, 2))
    
    # Combine and deduplicate (approximately)
    candidates = np.vstack([grid, rand_points])
    
    # Keep only points that allow a minimum radius (e.g., > 0.02)
    min_r_allowed = 0.02
    valid_mask = (
        (candidates[:, 0] >= min_r_allowed) & 
        (candidates[:, 0] <= 1 - min_r_allowed) & 
        (candidates[:, 1] >= min_r_allowed) & 
        (candidates[:, 1] <= 1 - min_r_allowed)
    )
    candidates = candidates[valid_mask]
    
    n_candidates = len(candidates)
    
    def max_radius_at(pos, other_circles):
        """Compute max radius for a circle at pos given other circles (x,y,r)."""
        x, y = pos
        # Boundary constraints
        r_max = min(x, 1 - x, y, 1 - y)
        if r_max <= 0:
            return 0.0
        # Overlap constraints
        if len(other_circles) > 0:
            dx = other_circles[:, 0] - x
            dy = other_circles[:, 1] - y
            dists = np.sqrt(dx**2 + dy**2)
            # r_i + r_j <= dist -> r_i <= dist - r_j
            r_from_others = dists - other_circles[:, 2]
            r_max = min(r_max, np.min(r_from_others))
        return max(0.0, r_max)

    def compute_radii(assignment):
        """Given assignment of N circles to candidate indices, compute radii greedily (largest first)."""
        # We want to maximize sum of radii. Greedy: place largest possible first?
        # Actually, the radii are coupled. A better approach: 
        # For a fixed assignment of centers, the max sum of radii is a convex optimization problem.
        # But for speed, let's use a simple iterative relaxation or greedy.
        # Greedy: sort by potential radius (distance to boundaries/others) descending, assign max.
        # This is not optimal but fast.
        
        # Better: Use a simple "grow" phase. Start with r=0, grow all simultaneously?
        # Or just compute max radius for each independent center, then resolve overlaps?
        
        # Let's use a simple heuristic:
        # 1. Compute initial r_i = min distance to boundary.
        # 2. Iteratively: for each i, r_i = min(r_i, min_j (dist_ij - r_j)).
        # This is like a message passing.
        
        centers = candidates[assignment]
        radii = np.zeros(N)
        
        # Initial radii from boundaries
        for i in range(N):
            radii[i] = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        
        # Relaxation
        for _ in range(50):
            changed = False
            for i in range(N):
                max_r = radii[i]
                for j in range(N):
                    if i == j:
                        continue
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    constraint_r = dist - radii[j]
                    if constraint_r < max_r:
                        max_r = constraint_r
                if max_r < 0:
                    max_r = 0.0
                if abs(max_r - radii[i]) > 1e-8:
                    radii[i] = max_r
                    changed = True
            if not changed:
                break
        
        return radii

    def evaluate(assignment):
        """Return sum of radii for a given assignment."""
        radii = compute_radii(assignment)
        return np.sum(radii)

    # Initial assignments
    # 1. First N candidates
    init1 = np.arange(N)
    
    # 2. Random assignment
    init2 = rng.choice(n_candidates, N, replace=False)
    
    # 3. Greedy packing init
    # Place circles one by one at the position that allows the largest radius
    init3 = np.zeros(N, dtype=int)
    used = set()
    current_circles = [] # list of (idx, x, y, r)
    for i in range(N):
        best_idx = -1
        best_r = -1
        for idx in range(n_candidates):
            if idx in used:
                continue
            # Compute max radius if placed here
            other_centers = candidates[[j for j in range(N) if j < i and init3[j] != -1]] if i > 0 else np.empty((0,2))
            other_radii = np.array([max_radius_at(candidates[init3[j]], np.empty((0,3))) for j in range(i)]) if i > 0 else np.empty(0)
            # This is getting complex. Let's just use a simpler random or grid.
            pass
        # Fallback: just pick random unused
        available = [i for i in range(n_candidates) if i not in used]
        if available:
            idx = rng.choice(available)
            init3[i] = idx
            used.add(idx)
        else:
            break
    
    # Let's stick to init1 and init2 for simplicity and speed.
    inits = [init1, init2]
    
    best_score = -np.inf
    best_assignment = None
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current_assignment = init.copy()
        current_score = evaluate(current_assignment)
        
        if current_score > best_score:
            best_score = current_score
            best_assignment = current_assignment.copy()
        
        # Simulated Annealing
        T = 0.1
        T_min = 1e-4
        cooling = 0.999
        
        while T > T_min and time.time() < deadline:
            # Generate a move
            # 1. Swap two circles' positions
            # 2. Move one circle to a random new position
            move_type = rng.random()
            
            new_assignment = current_assignment.copy()
            
            if move_type < 0.5:
                # Swap
                i, j = rng.choice(N, 2, replace=False)
                new_assignment[i], new_assignment[j] = current_assignment[j], current_assignment[i]
            else:
                # Move one to random
                i = rng.choice(N)
                new_assignment[i] = rng.choice(n_candidates)
            
            new_score = evaluate(new_assignment)
            
            # Acceptance
            delta = new_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_assignment = new_assignment
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_assignment = current_assignment.copy()
            
            T *= cooling
    
    # Convert best assignment to result
    if best_assignment is None:
        best_assignment = init1
    
    centers = candidates[best_assignment]
    radii = compute_radii(best_assignment)
    
    # Final check: ensure no overlaps due to numerical errors
    # If there are tiny overlaps, shrink radii slightly
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            sum_r = radii[i] + radii[j]
            if sum_r > dist:
                # Shrink both proportionally
                scale = dist / sum_r * 0.999
                radii[i] *= scale
                radii[j] *= scale
    
    # Ensure boundary constraints
    for i in range(N):
        radii[i] = min(radii[i], centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
    
    result = np.column_stack([centers[:, 0], centers[:, 1], radii])
    
    return result
