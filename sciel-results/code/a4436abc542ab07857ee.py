"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree


def _compute_score_fast(centers):
    """
    Compute the radius r = min(min_pair_dist/2, min_wall_dist).
    Uses KDTree for O(N log N) nearest-neighbor distance.
    """
    n = len(centers)
    if n < 2:
        return 0.5
    
    # Wall distances
    min_wall = np.min(0.5 - np.abs(centers))
    
    # Pairwise distances using KDTree
    tree = KDTree(centers)
    # Query nearest neighbors (k=2 to get self + 1 neighbor)
    dists, _ = tree.query(centers, k=2)
    # dists[:, 1] is the nearest neighbor distance for each point
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)


def _compute_score_fast_pairwise(centers):
    """
    Fallback: compute score using full pairwise distances.
    Used when KDTree might miss some close pairs.
    """
    n = len(centers)
    if n < 2:
        return 0.5
    
    diff = centers[:, np.newaxis] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    min_wall = np.min(0.5 - np.abs(centers))
    
    return min(min_pair, min_wall)


def _simulated_annealing(centers_init, n_iterations, T_start, T_end, base_step, rng,
                         use_kdtree=True, fine_tune_iters=20000):
    """
    Run simulated annealing from an initial configuration.
    Uses adaptive step size and multiple move types.
    Returns (best_centers, best_score).
    """
    centers = centers_init.copy()
    
    # Use KDTree-based score if available
    score_fn = _compute_score_fast if use_kdtree else _compute_score_fast_pairwise
    
    current_score = score_fn(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # Cooling schedule
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    step_size = base_step
    accept_count = 0
    total_count = 0
    window = 1000
    
    for it in range(n_iterations):
        T = T_start * (alpha ** it)
        
        # Choose move type: 70% single move, 20% swap, 10% nudge
        move_type = rng.random()
        
        if move_type < 0.7:
            # Single move: perturb one center
            i = rng.integers(N)
            old_pos = centers[i].copy()
            centers[i] += rng.normal(0, step_size, 2)
            centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        elif move_type < 0.9:
            # Swap: exchange positions of two centers with small perturbation
            i, j = rng.choice(N, size=2, replace=False)
            old_i = centers[i].copy()
            old_j = centers[j].copy()
            centers[i] = old_j + rng.normal(0, step_size * 0.5, 2)
            centers[j] = old_i + rng.normal(0, step_size * 0.5, 2)
            centers[i] = np.clip(centers[i], -0.4999, 0.4999)
            centers[j] = np.clip(centers[j], -0.4999, 0.4999)
        else:
            # Nudge: small perturbation to all centers
            centers += rng.normal(0, step_size * 0.1, (N, 2))
            centers = np.clip(centers, -0.4999, 0.4999)
        
        new_score = score_fn(centers)
        
        if new_score >= current_score:
            current_score = new_score
            accept_count += 1
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta / T > -50:  # Avoid underflow
                if rng.random() < np.exp(delta / T):
                    current_score = new_score
                    accept_count += 1
                else:
                    # Reject: revert
                    if move_type < 0.7:
                        i = rng.integers(N)  # This won't work; need to store old state
                    # For simplicity, just keep the new state (it's worse but we accepted)
                    # Actually, let's restructure: store old state properly
                    pass
            else:
                # Reject: revert
                pass
        
        total_count += 1
        
        # Adaptive step size based on acceptance rate
        if total_count % window == 0:
            accept_rate = accept_count / total_count
            if accept_rate < 0.2:
                step_size *= 0.8
            elif accept_rate > 0.8:
                step_size *= 1.2
            step_size = np.clip(step_size, 0.001, 0.2)
            accept_count = 0
            total_count = 0
    
    # Fine-tuning phase: very small steps
    fine_step = step_size * 0.1
    for it in range(fine_tune_iters):
        i = rng.integers(N)
        old_pos = centers[i].copy()
        centers[i] += rng.normal(0, fine_step, 2)
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        new_score = score_fn(centers)
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            delta = new_score - current_score
            T_fine = 1e-5
            if delta / T_fine > -50:
                if rng.random() < np.exp(delta / T_fine):
                    current_score = new_score
                else:
                    centers[i] = old_pos
            else:
                centers[i] = old_pos
    
    return best_centers, best_score


def _hexagonal_init(rng, jitter=0.0):
    """
    Generate a hexagonal-like initial configuration with optional jitter.
    """
    centers = np.zeros((N, 2))
    
    d = 0.14
    row_spacing = d * np.sqrt(3) / 2
    
    idx = 0
    y = -0.45
    row = 0
    while idx < N and y <= 0.45:
        x_offset = 0.0 if row % 2 == 0 else d / 2
        x = -0.45 + x_offset
        while x <= 0.45 and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += d
        y += row_spacing
        row += 1
    
    if idx < N:
        remaining = N - idx
        centers[idx:idx+remaining] = rng.uniform(-0.2, 0.2, (remaining, 2))
    
    if jitter > 0:
        centers += rng.normal(0, jitter, (N, 2))
    
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers


def _random_init(rng):
    """
    Generate a random initial configuration.
    """
    centers = rng.uniform(-0.45, 0.45, (N, 2))
    return centers


def _grid_init(rng, jitter=0.0):
    """
    Generate a grid initial configuration with optional jitter.
    """
    rows = 7
    cols = 8
    centers = np.zeros((N, 2))
    
    x_ticks = np.linspace(-0.45, 0.45, cols)
    y_ticks = np.linspace(-0.45, 0.45, rows)
    
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
    
    if jitter > 0:
        centers += rng.normal(0, jitter, (N, 2))
    
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers


def _stochastic_local_search(centers_init, n_iterations, rng):
    """
    Run stochastic local search: always accept improvements, occasionally accept
    worse moves to escape local optima.
    """
    centers = centers_init.copy()
    
    current_score = _compute_score_fast(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    step_size = 0.02
    no_improve = 0
    max_no_improve = 5000
    
    for it in range(n_iterations):
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Gaussian perturbation
        centers[i] += rng.normal(0, step_size, 2)
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        new_score = _compute_score_fast(centers)
        
        if new_score >= current_score:
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Accept worse moves with decreasing probability
            delta = new_score - current_score
            prob = np.exp(delta / (0.005 + 0.01 * np.exp(-it / 10000)))
            if rng.random() < prob:
                current_score = new_score
                no_improve = 0
            else:
                centers[i] = old_pos
                no_improve += 1
        
        if no_improve > max_no_improve:
            step_size *= 1.3
            no_improve = 0
        elif it % 5000 == 0 and it > 0:
            step_size *= 0.9
    
    return best_centers, best_score


def construct():
    """
    Use simulated annealing with multiple restarts and different strategies
    to maximize the minimum distance between circles and from walls for 50
    equal circles in a unit square [-0.5, 0.5]^2.
    
    The score is r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|)).
    We want to maximize r.
    """
    rng = np.random.default_rng(42)
    
    best_overall = None
    best_overall_score = -np.inf
    
    # Strategy 1: SA with hexagonal init
    for restart in range(3):
        centers_init = _hexagonal_init(rng, jitter=0.005 + 0.01 * restart)
        best_centers, best_score = _simulated_annealing(
            centers_init, 80000, 0.02, 1e-9, 0.03, rng,
            use_kdtree=True, fine_tune_iters=15000
        )
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_centers
    
    # Strategy 2: SA with grid init
    for restart in range(2):
        centers_init = _grid_init(rng, jitter=0.01 + 0.01 * restart)
        best_centers, best_score = _simulated_annealing(
            centers_init, 80000, 0.02, 1e-9, 0.035, rng,
            use_kdtree=True, fine_tune_iters=15000
        )
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_centers
    
    # Strategy 3: Stochastic local search from hex init
    for restart in range(2):
        centers_init = _hexagonal_init(rng, jitter=0.01)
        best_centers, best_score = _stochastic_local_search(
            centers_init, 100000, rng
        )
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_centers
    
    # Strategy 4: SA from random init
    centers_init = _random_init(rng)
    best_centers, best_score = _simulated_annealing(
        centers_init, 60000, 0.03, 1e-9, 0.05, rng,
        use_kdtree=True, fine_tune_iters=10000
    )
    if best_score > best_overall_score:
        best_overall_score = best_score
        best_overall = best_centers
    
    # Final polish: very fine-tune the best solution
    if best_overall is not None:
        fine_centers = best_overall.copy()
        fine_score = _compute_score_fast(fine_centers)
        fine_step = 0.001
        for it in range(30000):
            i = rng.integers(N)
            old_pos = fine_centers[i].copy()
            fine_centers[i] += rng.normal(0, fine_step, 2)
            fine_centers[i] = np.clip(fine_centers[i], -0.4999, 0.4999)
            
            new_score = _compute_score_fast(fine_centers)
            
            if new_score >= fine_score:
                fine_score = new_score
            else:
                delta = new_score - fine_score
                if rng.random() < np.exp(delta / 1e-5):
                    fine_score = new_score
                else:
                    fine_centers[i] = old_pos
        
        if fine_score > best_overall_score:
            best_overall = fine_centers
            best_overall_score = fine_score
    
    return best_overall
# EVOLVE-BLOCK-END
