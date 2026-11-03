"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
import random

def _max_radius_for_positions(pos):
    """
    Vectorized: given positions (N,2), compute the maximum radius each circle
    can have such that all circles are disjoint and inside the unit square.
    r_i = min(boundary[i], min_{j != i} dist(i,j)/2)
    """
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)

def _objective(pos):
    """Sum of max radii for given positions."""
    return float(np.sum(_max_radius_for_positions(pos)))

def _local_search_grid(pos, deadline, seed=42):
    """
    Dense grid-based best-fit local search: for each circle, search a fine grid
    around its current position to find the best new location.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)
    n = len(current_pos)
    
    # Adaptive grid resolution and step
    base_step = 0.02
    grid_res = 4  # (2*grid_res+1)^2 candidates per circle
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5: # Leave 5s for final checks
        improved_any = False
        order = rng.permutation(n)
        
        # Adaptive step: shrink as we make progress
        step_scale = max(0.3, 1.0 - no_improve_count / max_no_improve)
        step = base_step * step_scale
        
        for i in order:
            if time.time() > deadline - 5:
                break
            x_i, y_i = current_pos[i]
            
            # Generate dense grid of candidate positions
            best_new_pos = None
            best_new_obj = current_obj

            for di in range(-grid_res, grid_res + 1):
                for dj in range(-grid_res, grid_res + 1):
                    if di == 0 and dj == 0:
                        continue
                    cx = x_i + di * step
                    cy = y_i + dj * step
                    if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                        continue
                    new_pos = current_pos.copy()
                    new_pos[i] = [cx, cy]
                    new_obj = _objective(new_pos)
                    if new_obj > best_new_obj + 1e-12:
                        best_new_obj = new_obj
                        best_new_pos = new_pos

            # Also try some random perturbations
            for _ in range(5):
                angle = rng.uniform(0, 2 * np.pi)
                mag = step * rng.uniform(0.5, 1.5)
                cx = np.clip(x_i + mag * np.cos(angle), 0.0, 1.0)
                cy = np.clip(y_i + mag * np.sin(angle), 0.0, 1.0)
                new_pos = current_pos.copy()
                new_pos[i] = [cx, cy]
                new_obj = _objective(new_pos)
                if new_obj > best_new_obj + 1e-12:
                    best_new_obj = new_obj
                    best_new_pos = new_pos

            if best_new_pos is not None:
                current_pos = best_new_pos
                current_obj = best_new_obj
                improved_any = True
                no_improve_count = 0

        if not improved_any:
            no_improve_count += 1
            # Reduce step size
            base_step *= 0.8
            if base_step < 0.001:
                break
            if no_improve_count > max_no_improve:
                break

    return current_pos, current_obj

def _simulated_annealing(pos, deadline, seed=42):
    """
    Simulated annealing to maximize sum of radii.
    """
    rng = np.random.RandomState(seed)
    current_pos = pos.copy()
    current_obj = _objective(current_pos)

    best_pos = current_pos.copy()
    best_obj = current_obj

    n = len(current_pos)
    t_start = 0.3
    t_end = 0.001
    # Estimate iterations based on time budget, but cap it
    # We don't know how long each iter takes, so we just run until deadline
    it = 0
    
    while time.time() < deadline - 10: # Leave 10s for local search
        t = t_start * (t_end / t_start) ** (it / 10000.0) # Normalize over 10k iters
        if t < t_end:
            t = t_end
            
        i = rng.randint(n)
        # Random move: Gaussian perturbation
        scale = 0.05 * (1.0 + t) # Scale moves with temperature
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        
        new_pos = current_pos.copy()
        new_pos[i, 0] = np.clip(current_pos[i, 0] + dx, 0.0, 1.0)
        new_pos[i, 1] = np.clip(current_pos[i, 1] + dy, 0.0, 1.0)
        
        new_obj = _objective(new_pos)
        
        delta = new_obj - current_obj
        if delta > 0 or rng.uniform(0, 1) < np.exp(delta / t):
            current_pos = new_pos
            current_obj = new_obj
            if current_obj > best_obj:
                best_obj = current_obj
                best_pos = current_pos.copy()
        
        it += 1

    return best_pos, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # 1. Initialization
    # Use a grid as a strong starting point
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) >= N:
                break
            # Add slight jitter to break symmetry
            pos.append([(i + 0.5) / side + random.uniform(-0.01, 0.01), 
                        (j + 0.5) / side + random.uniform(-0.01, 0.01)])
    pos = np.array(pos[:N])
    
    best_pos = pos
    best_obj = _objective(pos)
    
    # 2. Simulated Annealing
    # Run SA for a chunk of time, then local search
    # We can alternate or just do SA then LS
    sa_deadline = start_time + 60 # 60s for SA
    sa_pos, sa_obj = _simulated_annealing(pos, sa_deadline, seed=42)
    
    if sa_obj > best_obj:
        best_obj = sa_obj
        best_pos = sa_pos
    
    # 3. Local Search Polish
    # Run local search on the best result from SA
    ls_deadline = deadline - 5
    ls_pos, ls_obj = _local_search_grid(best_pos, ls_deadline, seed=43)
    
    if ls_obj > best_obj:
        best_obj = ls_obj
        best_pos = ls_pos
        
    # 4. Optional: Try a second restart with random init if time permits
    if time.time() < deadline - 30:
        rng = np.random.RandomState(123)
        pos2 = rng.uniform(0.1, 0.9, size=(N, 2))
        # Ensure minimum separation in init to avoid immediate collapse
        # Simple rejection sampling
        valid_pos = [pos2[0]]
        for i in range(1, N):
            p = pos2[i]
            if all(np.hypot(p[0]-v[0], p[1]-v[1]) > 0.05 for v in valid_pos):
                valid_pos.append(p)
        while len(valid_pos) < N:
            valid_pos.append(rng.uniform(0.1, 0.9))
        pos2 = np.array(valid_pos)
        
        sa2_deadline = deadline - 20
        sa2_pos, sa2_obj = _simulated_annealing(pos2, sa2_deadline, seed=124)
        
        if sa2_obj > best_obj:
            best_obj = sa2_obj
            best_pos = sa2_pos
            
        # Local search on the second best
        if time.time() < deadline - 5:
            ls2_pos, ls2_obj = _local_search_grid(best_pos, deadline - 5, seed=125)
            if ls2_obj > best_obj:
                best_obj = ls2_obj
                best_pos = ls2_pos

    # Final feasibility check and radius calculation
    radii = _max_radius_for_positions(best_pos)
    
    # Ensure strict validity (epsilon shrink)
    eps = 1e-9
    final_circles = []
    for i in range(N):
        x, y = best_pos[i]
        r = radii[i]
        # Shrink radius slightly to ensure disjoint interiors and boundary containment
        # The max_radius calc gives r such that dist >= 2r, so dist - 2r >= 0.
        # We need dist > 2r for disjoint interiors? No, "disjoint interiors" means dist >= r1+r2 is sufficient for closed disks to touch, 
        # but usually "disjoint interiors" allows touching. 
        # However, to be safe against floating point errors in evaluator:
        r = r * (1.0 - 1e-6)
        
        # Check bounds
        if x - r < 0: r = x
        if x + r > 1: r = 1 - x
        if y - r < 0: r = y
        if y + r > 1: r = 1 - y
        r = max(0, r)
        
        final_circles.append([x, y, r])
        
    return np.array(final_circles)

# EVOLVE-BLOCK-END
