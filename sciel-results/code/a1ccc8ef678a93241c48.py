"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _valid(arr):
    """Check if the array of circles is valid (disjoint, inside unit square)."""
    if arr.shape != (N, 3):
        return False
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    # Check inside unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    if np.any(r < -1e-9):
        return False
    # Check disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True


def _objective(arr):
    """Return the sum of radii if valid, else -inf."""
    if _valid(arr):
        return np.sum(arr[:, 2])
    return -np.inf


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    rng = np.random.default_rng(42)

    # Initial guess: grid of equal circles
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4])
    best = np.array(cells[:N], dtype=np.float64)
    best_val = _objective(best)

    # Simulated annealing with greedy initialization
    # We'll use a penalty-based approach: minimize violation - alpha * sum(radii)
    
    def _penalty_loss(arr, alpha=10.0):
        """Compute penalty loss: sum of violations - alpha * sum(radii).
        Lower is better."""
        x = arr[:, 0]
        y = arr[:, 1]
        r = arr[:, 2]
        
        # Boundary violations
        bound_viol = 0.0
        bound_viol += np.sum(np.maximum(0, -r - x))
        bound_viol += np.sum(np.maximum(0, x + r - 1))
        bound_viol += np.sum(np.maximum(0, -r - y))
        bound_viol += np.sum(np.maximum(0, y + r - 1))
        
        # Overlap violations
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap_viol += np.maximum(0, r[i] + r[j] - dist)
        
        # Negative radius violation
        neg_r_viol = np.sum(np.maximum(0, -r))
        
        return bound_viol + overlap_viol + neg_r_viol - alpha * np.sum(r)

    def _gradient_penalty(arr, alpha=10.0):
        """Compute gradient of penalty loss w.r.t. positions and radii."""
        x = arr[:, 0].copy()
        y = arr[:, 1].copy()
        r = arr[:, 2].copy()
        
        gx = np.zeros(N)
        gy = np.zeros(N)
        gr = np.zeros(N)
        
        # Boundary gradients
        for i in range(N):
            if x[i] < r[i]:
                gx[i] -= 1.0
                gr[i] += 1.0
            if x[i] + r[i] > 1:
                gx[i] += 1.0
                gr[i] += 1.0
            if y[i] < r[i]:
                gy[i] -= 1.0
                gr[i] += 1.0
            if y[i] + r[i] > 1:
                gy[i] += 1.0
                gr[i] += 1.0
        
        # Overlap gradients
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    continue
                if r[i] + r[j] > dist:
                    overlap = r[i] + r[j] - dist
                    ux = dx / dist
                    uy = dy / dist
                    gx[i] += ux
                    gy[i] += uy
                    gx[j] -= ux
                    gy[j] -= uy
                    gr[i] += 1.0
                    gr[j] += 1.0
        
        # Negative radius gradient
        for i in range(N):
            if r[i] < 0:
                gr[i] += 1.0
        
        # Objective gradient: -alpha for each radius
        gr -= alpha
        
        return gx, gy, gr

    # Simulated annealing loop
    T = 0.1
    T_min = 0.0001
    cooling = 0.995
    step_scale = 0.02
    
    current = best.copy()
    current_loss = _penalty_loss(current)
    
    # Greedy growth phase: try to increase radii while maintaining validity
    for _ in range(200):
        if time.time() > deadline - 30:
            break
        improved = False
        for _ in range(50):
            # Try increasing a random circle's radius
            i = rng.integers(N)
            old_r = current[i, 2]
            new_r = old_r + rng.uniform(0.001, 0.01)
            current[i, 2] = new_r
            if _valid(current):
                improved = True
                val = _objective(current)
                if val > best_val:
                    best = current.copy()
                    best_val = val
            else:
                current[i, 2] = old_r
        if not improved:
            break

    # Simulated annealing phase
    while T > T_min:
        if time.time() > deadline - 10:
            break
        
        # Generate perturbation
        pert = current.copy()
        # Perturb positions
        pert[:, 0] += rng.normal(0, step_scale * T, N)
        pert[:, 1] += rng.normal(0, step_scale * T, N)
        # Perturb radii
        pert[:, 2] += rng.normal(0, step_scale * T * 0.5, N)
        # Clamp to reasonable bounds
        pert[:, 2] = np.maximum(pert[:, 2], 0.001)
        pert[:, 0] = np.clip(pert[:, 0], 0, 1)
        pert[:, 1] = np.clip(pert[:, 1], 0, 1)
        
        pert_loss = _penalty_loss(pert)
        delta_loss = pert_loss - current_loss
        
        if delta_loss < 0 or rng.random() < np.exp(-delta_loss / T):
            current = pert
            current_loss = pert_loss
            # Check if valid and update best
            if _valid(current):
                val = _objective(current)
                if val > best_val:
                    best = current.copy()
                    best_val = val
        
        T *= cooling
        
        # Adaptive step size
        if T < 0.05:
            step_scale = 0.005
        elif T < 0.01:
            step_scale = 0.002

    # Final local optimization: gradient descent on penalty
    for epoch in range(500):
        if time.time() > deadline - 5:
            break
        gx, gy, gr = _gradient_penalty(current, alpha=10.0)
        lr = 0.01 * T / 0.1  # scale with temperature
        current[:, 0] -= lr * gx
        current[:, 1] -= lr * gy
        current[:, 2] -= lr * gr
        current[:, 2] = np.maximum(current[:, 2], 0.001)
        current[:, 0] = np.clip(current[:, 0], 0, 1)
        current[:, 1] = np.clip(current[:, 1], 0, 1)
        
        if _valid(current):
            val = _objective(current)
            if val > best_val:
                best = current.copy()
                best_val = val

    return best

# EVOLVE-BLOCK-END
