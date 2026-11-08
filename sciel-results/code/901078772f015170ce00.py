"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    # Objective: maximize sum of radii, subject to:
    # 1. Each circle inside [0,1]^2: x >= r, x <= 1-r, y >= r, y <= 1-r
    # 2. Disjoint interiors: dist(i,j) >= r_i + r_j for all i != j

    def evaluate(state):
        """Compute objective value (sum of radii) with penalty for violations."""
        n = len(state)
        x = state[0]
        y = state[1]
        r = state[2]
        
        # Sum of radii
        val = np.sum(r)
        
        # Penalty for circles going out of bounds
        # Each circle must satisfy: x >= r, x <= 1-r, y >= r, y <= 1-r
        # Equivalently: r <= x, r <= 1-x, r <= y, r <= 1-y
        # Penalty: sum of max(0, r - min(x, 1-x, y, 1-y)) for each circle
        min_dist_to_edge = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        bound_viol = np.maximum(0, r - min_dist_to_edge)
        val -= 100.0 * np.sum(bound_viol)
        
        # Penalty for overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    val -= 100.0 * overlap
        
        return val

    def repair(state):
        """Repair a state to be feasible if possible."""
        n = len(state)
        x = state[0].copy()
        y = state[1].copy()
        r = state[2].copy()
        
        # Ensure circles are within bounds
        for i in range(n):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            r[i] = max(0.001, min(r[i], max_r))
        
        # Try to resolve overlaps by shrinking
        for _ in range(50):
            max_overlap = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    overlap = r[i] + r[j] - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
            if max_overlap <= 1e-10:
                break
            # Shrink all circles slightly
            r *= (1 - max_overlap * 0.01)
            for i in range(n):
                max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
                r[i] = max(0.001, min(r[i], max_r))
        
        return np.array([x, y, r])

    def is_valid(state, tol=1e-6):
        """Check if state is valid."""
        n = len(state)
        x, y, r = state
        for i in range(n):
            if x[i] < r[i] - tol or x[i] > 1 - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > 1 - r[i] + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    # Initial state: grid-based
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side))
    while len(cells) < N:
        cells.append((0.5, 0.5, 0.01))
    
    best_state = np.array([c[0] for c in cells], dtype=float)
    best_state_y = np.array([c[1] for c in cells], dtype=float)
    best_state_r = np.array([c[2] for c in cells], dtype=float)
    best_score = np.sum(best_state_r)

    # Simulated annealing
    T = 0.05
    T_min = 0.001
    alpha = 0.999
    n_steps = 500000

    state_x = best_state.copy()
    state_y = best_state_y.copy()
    state_r = best_state_r.copy()
    current_score = best_score

    rng = np.random.default_rng(42)

    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Perturb: move a random circle slightly
        idx = rng.integers(0, N)
        dx = rng.normal(0, 0.02)
        dy = rng.normal(0, 0.02)
        dr = rng.normal(0, 0.01)
        
        new_x = state_x.copy()
        new_y = state_y.copy()
        new_r = state_r.copy()
        
        new_x[idx] += dx
        new_y[idx] += dy
        new_r[idx] = max(0.001, new_r[idx] + dr)
        
        # Clamp to bounds
        new_x[idx] = np.clip(new_x[idx], new_r[idx], 1 - new_r[idx])
        new_y[idx] = np.clip(new_y[idx], new_r[idx], 1 - new_r[idx])
        
        new_score = evaluate(np.array([new_x, new_y, new_r]))
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            state_x = new_x
            state_y = new_y
            state_r = new_r
            current_score = new_score
            
            if current_score > best_score:
                best_state = state_x.copy()
                best_state_y = state_y.copy()
                best_state_r = state_r.copy()
                best_score = current_score
        
        T = max(T_min, T * alpha)

    # Final repair and validation
    final_state = np.array([best_state, best_state_y, best_state_r])
    repaired = repair(final_state)
    
    # Check validity, if not valid, try to repair further
    if not is_valid(repaired):
        # Aggressive repair: shrink all circles
        for _ in range(100):
            r = repaired[2]
            x = repaired[0]
            y = repaired[1]
            max_overlap = 0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    overlap = r[i] + r[j] - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
            if max_overlap <= 1e-8:
                break
            r *= (1 - max_overlap * 0.005)
            for i in range(N):
                max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
                r[i] = max(0.0001, min(r[i], max_r))
            repaired = np.array([x, y, r])
    
    # Final validation
    if not is_valid(repaired, tol=1e-5):
        # Fallback to a simple valid grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
        while len(cells) < N:
            cells.append((0.5, 0.5, 0.001))
        return np.array(cells[:N])
    
    return repaired
# EVOLVE-BLOCK-END
