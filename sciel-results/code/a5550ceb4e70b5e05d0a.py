"""
Strategy: Multi-start Simulated Annealing with Nelder-Mead refinement.
Instead of binary searching for a feasible radius using a smooth gradient-based
feasibility check (which can get stuck in local minima of the violation surface),
we directly maximize the minimum distance using a stochastic global search method
(Simulated Annealing) from multiple diverse starting points.

Key differences:
1. Direct objective maximization (min distance) rather than feasibility check.
2. Stochastic search (SA) to escape local minima, rather than deterministic gradient descent.
3. Multiple restarts with different random seeds to explore the configuration space.
4. Nelder-Mead local refinement for high-precision polishing after SA converges.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _objective(x):
    """
    Objective function for optimization: negative of the minimum distance.
    We minimize this to maximize the minimum distance.
    """
    c = x.reshape(N, 2)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1])))
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_min = np.min(dists) / 2.0
    return -min(wall_min, pair_min)

def _hex_initial():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        y_spacing = r * np.sqrt(3.0)
        total_y_span = (n_rows - 1) * y_spacing
        y_start = -0.5 + r + total_y_span / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * y_spacing
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = r if (row_idx % 2 == 1) else 0.0
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _simulated_annealing(x0, time_limit=10.0):
    """
    Perform Simulated Annealing to maximize the minimum distance.
    """
    start_time = time.time()
    x = x0.copy()
    best_x = x.copy()
    best_score = _score(x.reshape(N, 2))
    current_score = best_score
    
    # Initial temperature
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.9995
    steps_per_iter = 1
    
    while time.time() - start_time < time_limit:
        # Random perturbation
        delta = np.random.normal(0, T, size=x.shape)
        x_new = x + delta
        # Clip to square
        x_new = np.clip(x_new, -0.5, 0.5)
        
        score_new = _score(x_new.reshape(N, 2))
        delta_score = score_new - current_score
        
        # Acceptance criterion
        if delta_score > 0 or np.random.random() < np.exp(delta_score / T):
            x = x_new
            current_score = score_new
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_x, best_score

def _nelder_mead_polish(x0, time_limit=5.0):
    """
    Polish the configuration using Nelder-Mead.
    """
    start_time = time.time()
    try:
        result = minimize(
            _objective, 
            x0, 
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return x0
        return result.x
    except Exception:
        return x0

def construct():
    start_time = time.time()
    
    # Get a good initial guess
    cents0, r0 = _hex_initial()
    x0 = cents0.reshape(-1)
    
    best_x = x0.copy()
    best_r = r0
    
    # Multi-start Simulated Annealing
    n_restarts = 20
    time_per_restart = (DEADLINE - (time.time() - start_time)) / n_restarts
    time_per_restart = min(time_per_restart, 15.0)  # Cap per restart
    
    for i in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Create a diverse initial guess
        if i == 0:
            x_start = x0.copy()
        else:
            # Random perturbation of the best known solution
            noise = np.random.normal(0, 0.05, size=x0.shape)
            x_start = np.clip(x0 + noise, -0.5, 0.5)
        
        # Simulated Annealing
        x_sa, r_sa = _simulated_annealing(x_start, time_limit=time_per_restart)
        
        if r_sa > best_r:
            best_r = r_sa
            best_x = x_sa.copy()
        
        # Nelder-Mead polish
        if time.time() - start_time < DEADLINE - 5:
            x_polished = _nelder_mead_polish(x_sa, time_limit=5.0)
            r_polished = _score(x_polished.reshape(N, 2))
            if r_polished > best_r:
                best_r = r_polished
                best_x = x_polished.copy()
    
    # Final check: ensure the best_x is valid and inside bounds
    best_cents = np.clip(best_x, -0.5, 0.5).reshape(N, 2)
    
    # Calculate final score to be sure
    final_score = _score(best_cents)
    
    return best_cents

# EVOLVE-BLOCK-END
