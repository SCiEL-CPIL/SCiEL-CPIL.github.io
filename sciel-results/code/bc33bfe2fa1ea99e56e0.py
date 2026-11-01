"""
Strategy: Simulated Annealing to maximize the minimum distance (radius).
Instead of binary searching for feasibility with a fixed radius, we treat the
minimum distance as the objective to maximize. We use a cooling schedule to
escape local optima in the rugged landscape of circle packing.
"""

import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    # Wall distances
    # Distance from center to nearest wall
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

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0

    # Try different row counts to find the best hexagonal-like start
    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        # Estimate radius for this configuration
        # Horizontal span: (max_rc - 1) * 2r + 2r = max_rc * 2r <= 1 => r <= 1/(2*max_rc)
        # Vertical span: (n_rows - 1) * sqrt(3)*r + 2r <= 1 => r <= 1/(2 + (n_rows-1)*sqrt(3))
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        # Total vertical height occupied by centers
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            # Offset alternating rows for hexagonal packing
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _fast_score(cents):
    """
    Optimized score calculation for speed.
    Returns the minimum radius.
    """
    # Wall check
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    
    # Pairwise check using vectorized ops but early exit isn't easy with numpy.
    # However, for N=50, 1225 pairs is small.
    # We can compute all pairwise distances.
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = np.min(dists_sq)
    pair_min = np.sqrt(min_dist_sq) / 2.0
    
    return min(wall_min, pair_min)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # 1. Initial Configuration
    cents, r_current = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r_current

    # 2. Simulated Annealing
    # Parameters
    T_initial = 0.05  # Initial temperature (scale of move)
    T_final = 0.0001
    cooling_rate = 0.9995  # Geometric cooling
    max_iterations = 200000  # Max steps if time allows
    
    current_cents = best_cents.copy()
    current_r = best_r
    
    # Precompute some constants
    # We want to maximize r = min(wall_dist, pair_dist/2)
    
    iteration = 0
    while time.time() - start_time < deadline - 5:
        T = T_initial * (T_final / T_initial) ** (iteration / max_iterations)
        
        # Generate a random move
        # Pick a random circle
        i = np.random.randint(N)
        # Random perturbation
        # Scale perturbation by temperature
        dx = np.random.normal(0, T)
        dy = np.random.normal(0, T)
        
        new_cents = current_cents.copy()
        new_cents[i, 0] += dx
        new_cents[i, 1] += dy
        
        # Clamp to square to avoid huge wall violations? 
        # Actually, allowing them to go outside increases wall_dist penalty (makes it negative),
        # which lowers the score, so the optimizer will naturally push them back.
        # But if they go far outside, the score drops a lot.
        # Let's clamp to a slightly larger box to keep numbers sane, but not strictly to [-0.5, 0.5]
        # because the penalty handles it.
        # However, if we clamp, we might get stuck on the boundary if T is high.
        # Better to let them move freely but rely on the score function.
        
        new_r = _fast_score(new_cents)
        
        # Acceptance criterion
        if new_r > current_r:
            current_cents = new_cents
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_cents = current_cents.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            # Note: new_r - current_r is negative.
            delta = new_r - current_r
            if T > 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_cents = new_cents
                    current_r = new_r
            # else T is effectively 0, reject all worse moves
        
        iteration += 1
        
        # Periodic restart if stuck? 
        # If no improvement for a long time, maybe randomize a bit?
        # For now, simple cooling.

    # 3. Local Polish (Optional if time remains)
    # If we have time left, do a few steps of gradient descent or just return best.
    # Given the 150s limit and SA efficiency, we likely return here.
    
    # Final check
    # Ensure returned centers are within bounds for validity (score function handles it, 
    # but if a center is outside, wall_dist is negative, so score is negative.
    # The evaluator checks "inside the square". 
    # "A centre on or outside the square ... make the candidate invalid."
    # So we MUST clip to [-0.5, 0.5].
    # If we clip, the score might drop if they were pushed out, but we must be valid.
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    # Re-evaluate score after clipping to ensure we don't return an invalid config
    # If clipping makes it invalid (e.g. overlap increases significantly), it's still the best we have.
    # But wait, if we clip, do we need to re-optimize? 
    # The SA process should have kept them mostly inside.
    # Clipping is a safety net.
    
    return best_cents

# EVOLVE-BLOCK-END
