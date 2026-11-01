"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(centers):
    """Compute the packing radius: min(min_pair_dist/2, min_wall_dist)."""
    min_wall = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq))
    return min(min_pair / 2.0, min_wall)


def _local_refinement(centers, rng, time_budget=5.0):
    """
    Perform local refinement using scipy L-BFGS-B with a barrier approach.
    We maximize the minimum distance by minimizing the negative of a smooth approximation.
    """
    start = time.time()
    centers = centers.copy()
    
    def objective(x):
        c = x.reshape(-1, 2)
        # Compute all pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        
        # For pair distances, use the minimum
        min_pair_sq = np.min(dists_sq)
        min_pair = np.sqrt(min_pair_sq)
        
        # The score is min(min_pair/2, min_wall)
        # We want to maximize this, so minimize -score
        # But we need a smooth function. Let's use a barrier/penalty approach.
        # If either constraint is violated, heavily penalize.
        
        score = min(min_pair / 2.0, min_wall)
        return -score
    
    # Use Nelder-Mead for local refinement as it doesn't require gradients
    # and is robust for this type of problem
    result = minimize(
        objective, 
        centers.flatten(), 
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
    )
    
    if result.fun < -_compute_score(centers):
        return result.x.reshape(-1, 2)
    
    return centers


def _sa_run(centers_init, rng, time_budget=20.0):
    """
    Run simulated annealing on centers.
    """
    start_time = time.time()
    centers = centers_init.copy()
    
    # Precompute initial score
    current_score = _compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # SA parameters
    n_iters = 500000
    T_start = 0.02
    T_end = 1e-10
    alpha = (T_end / T_start) ** (1.0 / n_iters)
    
    # Adaptive step size
    step = 0.03
    base_step = 0.03
    no_improve = 0
    max_no_improve = 5000
    
    for it in range(n_iters):
        if it % 100 == 0:
            if time.time() - start_time > time_budget:
                break
        
        T = T_start * (alpha ** it)
        
        # Pick random circle to move
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Generate new position
        new_pos = old_pos + rng.normal(0, step, 2)
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        
        # Compute new score if we move circle i
        old_centers = centers.copy()
        centers[i] = new_pos
        new_score = _compute_score(centers)
        
        delta = new_score - current_score
        
        if delta >= 0:
            # Accept
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Probabilistically accept
            if rng.random() < np.exp(delta / T):
                current_score = new_score
                no_improve += 1
            else:
                centers[i] = old_pos
                no_improve += 1
        
        # Adaptive step size
        if no_improve > max_no_improve:
            step *= 0.7
            no_improve = 0
        elif it % 5000 == 0 and it > 0:
            step *= 0.999
        
        # Periodically increase step to escape local minima
        if it % 10000 == 0 and it > 0:
            step = base_step * (0.8 + 0.4 * rng.random())
            no_improve = 0
    
    return best_centers, best_score


def construct():
    """
    Multi-start simulated annealing with local refinement to maximize packing radius.
    """
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    max_time = 110.0
    
    best_centers = None
    best_score = -np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Grid-based with jitter
    for rows in [5, 6, 7, 8]:
        for cols in [6, 7, 8, 9, 10]:
            if rows * cols < N:
                continue
            x_ticks = np.linspace(-0.48, 0.48, cols)
            y_ticks = np.linspace(-0.48, 0.48, rows)
            centers = np.zeros((N, 2))
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx < N:
                        centers[idx] = [x_ticks[c], y_ticks[r]]
                        idx += 1
                    else:
                        break
                if idx >= N:
                    break
            if idx == N:
                centers += rng.normal(0, 0.005, (N, 2))
                centers = np.clip(centers, -0.4999, 0.4999)
                initializations.append(centers)
    
    # 2. Hexagonal packing
    hex_spacing = 0.19
    hex_row_h = hex_spacing * np.sqrt(3) / 2
    hex_centers = np.zeros((N, 2))
    idx = 0
    row = 0
    while idx < N and row < 15:
        offset = 0 if row % 2 == 0 else hex_spacing / 2
        col = 0
        while idx < N and col < 12:
            x = -0.48 + offset + col * hex_spacing
            y = -0.48 + row * hex_row_h
            if -0.4999 <= x <= 0.4999 and -0.4999 <= y <= 0.4999:
                hex_centers[idx] = [x, y]
                idx += 1
            col += 1
        row += 1
    if idx == N:
        hex_centers += rng.normal(0, 0.005, (N, 2))
        hex_centers = np.clip(hex_centers, -0.4999, 0.4999)
        initializations.append(hex_centers)
    
    # 3. Random initializations
    for _ in range(3):
        initializations.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Structured patterns: concentric rings
    ring_centers = np.zeros((N, 2))
    # Center
    ring_centers[0] = [0, 0]
    # Ring 1: 6 circles
    angle_step = 2 * np.pi / 6
    for j in range(1, 7):
        angle = j * angle_step
        ring_centers[j] = [0.15 * np.cos(angle), 0.15 * np.sin(angle)]
    # Ring 2: 12 circles
    angle_step = 2 * np.pi / 12
    for j in range(7, 19):
        angle = j * angle_step
        ring_centers[j] = [0.30 * np.cos(angle), 0.30 * np.sin(angle)]
    # Ring 3: 12 circles
    angle_step = 2 * np.pi / 12
    for j in range(19, 31):
        angle = j * angle_step
        ring_centers[j] = [0.42 * np.cos(angle), 0.42 * np.sin(angle)]
    # Remaining 19: random
    ring_centers[31:] = rng.uniform(-0.45, 0.45, (N - 31, 2))
    ring_centers = np.clip(ring_centers, -0.4999, 0.4999)
    initializations.append(ring_centers)
    
    # Run SA from each initialization
    n_inits = len(initializations)
    for init_idx, init_centers in enumerate(initializations):
        remaining = max_time - (time.time() - start_time)
        if remaining < 8.0:
            break
        
        # Allocate time budget
        time_budget = min(remaining, max(15.0, remaining / (n_inits - init_idx)))
        
        # Run SA
        sa_centers, sa_score = _sa_run(init_centers, rng, time_budget)
        
        # Local refinement
        if time.time() - start_time < max_time - 3.0:
            refined_centers = _local_refinement(sa_centers, rng, time_budget=3.0)
            refined_score = _compute_score(refined_centers)
            if refined_score > sa_score:
                sa_score = refined_score
                sa_centers = refined_centers
        
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers
    
    # Fallback
    if best_centers is None:
        x_ticks = np.linspace(-0.48, 0.48, 8)
        y_ticks = np.linspace(-0.48, 0.48, 7)
        best_centers = np.zeros((N, 2))
        idx = 0
        for r in range(7):
            for c in range(8):
                if idx < N:
                    best_centers[idx] = [x_ticks[c], y_ticks[r]]
                    idx += 1
                else:
                    break
            if idx >= N:
                break
    
    # Final polish: run one more local refinement on the best
    if time.time() - start_time < max_time - 2.0:
        final_centers = _local_refinement(best_centers, rng, time_budget=min(3.0, max_time - (time.time() - start_time) - 1.0))
        final_score = _compute_score(final_centers)
        if final_score > best_score:
            best_centers = final_centers
    
    return best_centers
# EVOLVE-BLOCK-END
