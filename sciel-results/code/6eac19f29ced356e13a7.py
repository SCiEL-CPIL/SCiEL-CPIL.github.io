"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_max_overlap(h, N):
    """Compute the maximum overlap of h with shifted 1-h."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _compute_overlap_vector(h, N):
    """Compute overlap for all shifts."""
    one_minus_h = 1.0 - h
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted)
    return overlaps


def _project_to_constraint(h, N):
    """Project h to satisfy 0<=h[i]<=1 and sum(h)=N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    current_sum = np.sum(h)
    if abs(current_sum - target) < 1e-12:
        return h
    # Adjust to meet sum constraint while keeping bounds
    diff = target - current_sum
    # Distribute adjustment, respecting bounds
    for _ in range(10):
        if diff > 0:
            # Need to increase sum; add to elements < 1
            mask = h < 1.0 - 1e-15
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        else:
            # Need to decrease sum; subtract from elements > 0
            mask = h > 1e-15
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
        if abs(diff) < 1e-10:
            break
    return h


def construct():
    deadline = time.time() + 140.0  # 140s to stay safely under 150s
    rng = np.random.default_rng(42)
    
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h, N)
    
    # Start with a few random initializations
    candidates = [best_h.copy()]
    
    # Generate diverse initial candidates
    for _ in range(5):
        h = rng.uniform(0.0, 1.0, size=N)
        h = _project_to_constraint(h, N)
        candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        score = _compute_max_overlap(h, N)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Simulated annealing
    T = 0.1
    T_min = 1e-8
    cooling = 0.9995
    
    h = best_h.copy()
    current_score = best_score
    
    while time.time() < deadline and T > T_min:
        # Propose a perturbation
        n_perturb = rng.integers(1, max(2, N // 10))
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = h.copy()
        h_new[idx] += rng.normal(0, 0.1, size=n_perturb)
        h_new = _project_to_constraint(h_new, N)
        
        new_score = _compute_max_overlap(h_new, N)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            h = h_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        
        T *= cooling
    
    return best_h
# EVOLVE-BLOCK-END
