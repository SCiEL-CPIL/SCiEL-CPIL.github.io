"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time


def _evaluate(w, centers, radii, lam=100.0):
    """Compute objective: sum(r) - penalty for overlaps and boundary violations."""
    h = 2.0 - w
    if w <= 0 or h <= 0:
        return -1e10
    
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    
    # Boundary penalty: circles must be inside [0,w] x [0,h]
    bound_pen = 0.0
    for i in range(N):
        if x[i] < r[i]:
            bound_pen += (r[i] - x[i]) ** 2
        if x[i] + r[i] > w:
            bound_pen += (x[i] + r[i] - w) ** 2
        if y[i] < r[i]:
            bound_pen += (r[i] - y[i]) ** 2
        if y[i] + r[i] > h:
            bound_pen += (y[i] + r[i] - h) ** 2
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist) ** 2
    
    return np.sum(r) - lam * (bound_pen + overlap_pen)


def _sa_optimize(w0, centers0, radii0, time_budget=10.0, seed=0):
    """Simulated annealing to optimize the packing."""
    rng = np.random.default_rng(seed)
    
    w = w0
    centers = centers0.copy()
    radii = radii0.copy()
    
    current_obj = _evaluate(w, centers, radii)
    best_obj = current_obj
    best_w = w
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    t_start = time.time()
    temp = 0.5
    cool_rate = 0.999
    min_temp = 1e-6
    
    n_iter = 0
    while time.time() - t_start < time_budget:
        n_iter += 1
        temp *= cool_rate
        if temp < min_temp:
            break
        
        # Random perturbation
        new_w = w + rng.normal(0, 0.05)
        new_centers = centers.copy()
        new_radii = radii.copy()
        
        # Perturb a random subset
        idx = rng.choice(N, size=rng.integers(1, 4), replace=False)
        for i in idx:
            new_centers[i, 0] += rng.normal(0, 0.1)
            new_centers[i, 1] += rng.normal(0, 0.1)
            new_radii[i] += rng.normal(0, 0.02)
        
        # Clip to reasonable bounds
        new_w = np.clip(new_w, 0.5, 1.5)
        h = 2.0 - new_w
        new_radii = np.clip(new_radii, 0.01, 0.5)
        
        # Keep centers in bounds
        new_centers[:, 0] = np.clip(new_centers[:, 0], 0.0, new_w)
        new_centers[:, 1] = np.clip(new_centers[:, 1], 0.0, h)
        
        new_obj = _evaluate(new_w, new_centers, new_radii)
        
        # Accept or reject
        delta = new_obj - current_obj
        if delta > 0 or rng.random() < np.exp(delta / temp):
            w, centers, radii = new_w, new_centers, new_radii
            current_obj = new_obj
            if current_obj > best_obj:
                best_obj = current_obj
                best_w = w
                best_centers = centers.copy()
                best_radii = radii.copy()
    
    return best_w, best_centers, best_radii, best_obj


def _grid_init(w, h, seed=0):
    """Initialize circles in a grid pattern."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(N)))
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    
    for i in range(N):
        col = i % side
        row = i // side
        cx = (col + 0.5) * w / side + rng.normal(0, 0.01)
        cy = (row + 0.5) * h / side + rng.normal(0, 0.01)
        centers[i] = [cx, cy]
        radii[i] = 0.5 * min(w, h) / side * 0.9
    
    return centers, radii


def _random_init(w, h, seed=0):
    """Initialize circles at random positions."""
    rng = np.random.default_rng(seed)
    centers = np.column_stack([
        rng.uniform(0.1, w - 0.1, N),
        rng.uniform(0.1, h - 0.1, N)
    ])
    radii = rng.uniform(0.03, 0.1, N)
    return centers, radii


def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin before 150s limit
    
    best_result = None
    best_score = -1e10
    
    # Try multiple aspect ratios and initialization strategies
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]
    
    for w in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
        h = 2.0 - w
        
        # Try different initializations
        for init_type in ['grid', 'random']:
            if time.time() > deadline - 10.0:
                break
            
            if init_type == 'grid':
                centers, radii = _grid_init(w, h, seed=42)
            else:
                centers, radii = _random_init(w, h, seed=42)
            
            # Run SA with multiple seeds
            for seed in [0, 1, 2]:
                if time.time() > deadline - 5.0:
                    break
                
                time_budget = min(8.0, deadline - time.time() - 3.0)
                if time_budget < 2.0:
                    break
                
                bw, bc, br, score = _sa_optimize(w, centers, radii, time_budget=time_budget, seed=seed)
                
                if score > best_score:
                    best_score = score
                    # Check validity: no overlaps, all in bounds
                    x = bc[:, 0]
                    y = bc[:, 1]
                    r = br
                    valid = True
                    for i in range(N):
                        if x[i] < r[i] - 1e-6 or x[i] + r[i] > bw + 1e-6:
                            valid = False
                            break
                        if y[i] < r[i] - 1e-6 or y[i] + r[i] > (2.0 - bw) + 1e-6:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i + 1, N):
                                dx = x[i] - x[j]
                                dy = y[i] - y[j]
                                dist = np.sqrt(dx * dx + dy * dy)
                                if dist < r[i] + r[j] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                    
                    if valid:
                        best_result = np.column_stack([bc, br])
        
        # Also try to refine with a smaller SA
        if best_result is not None and time.time() < deadline - 20.0:
            bw = best_result[:, 0]
            bc = best_result[:, :2]
            br = best_result[:, 2]
            time_budget = min(10.0, deadline - time.time() - 5.0)
            if time_budget > 3.0:
                bw2, bc2, br2, score2 = _sa_optimize(bw, bc, br, time_budget=time_budget, seed=100)
                if score2 > best_score:
                    x = bc2[:, 0]
                    y = bc2[:, 1]
                    r = br2
                    valid = True
                    for i in range(N):
                        if x[i] < r[i] - 1e-6 or x[i] + r[i] > bw2 + 1e-6:
                            valid = False
                            break
                        if y[i] < r[i] - 1e-6 or y[i] + r[i] > (2.0 - bw2) + 1e-6:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i + 1, N):
                                dx = x[i] - x[j]
                                dy = y[i] - y[j]
                                dist = np.sqrt(dx * dx + dy * dy)
                                if dist < r[i] + r[j] - 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid:
                        best_score = score2
                        best_result = np.column_stack([bc2, br2])
    
    if best_result is None:
        # Fallback to seed solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    # Final validity check and return
    x = best_result[:, 0]
    y = best_result[:, 1]
    r = best_result[:, 2]
    
    # Ensure all radii are positive
    r = np.maximum(r, 1e-4)
    best_result[:, 2] = r
    
    return best_result
# EVOLVE-BLOCK-END
