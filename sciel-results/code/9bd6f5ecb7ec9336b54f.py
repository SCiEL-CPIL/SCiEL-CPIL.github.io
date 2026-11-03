"""Optimized 50-sphere packing in unit cube via BCC init + simulated annealing + L-BFGS-B polish."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds to leave before deadline


def _compute_score(centres):
    """Compute the packing score: min of (min pairwise dist / 2, min distance to wall)."""
    n = len(centres)
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: cube is [-0.5, 0.5]^3
    # For each centre, distance to nearest wall
    wall_dists = np.min(0.5 - np.abs(centres), axis=1)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)


def _negative_score(centres):
    """Objective to minimize: negative of the score."""
    return -_compute_score(centres)


def _bcc_initialization():
    """
    Generate a BCC-like lattice initialization for 50 spheres.
    BCC unit cell has 2 atoms. We use a grid of BCC cells.
    For 50 spheres, we need 25 BCC cells.
    Use a 3x3x3 grid of BCC cells = 27 cells = 54 atoms, take first 50.
    Actually, let's use a more compact arrangement.
    
    Alternative: Use a simple approach - place spheres in a pattern that approximates
    the known optimal packing structure.
    """
    # Try a 5x5x2 arrangement with some perturbation
    # 5*5*2 = 50 exactly
    # Place them in a grid with alternating offsets
    
    # Create a 5x5x2 grid
    xs = np.linspace(-0.5, 0.5, 5)
    ys = np.linspace(-0.5, 0.5, 5)
    zs = np.array([-0.25, 0.25])
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    
    centres = np.array(centres, dtype=np.float64)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centres += rng.normal(0, 0.01, centres.shape)
    
    # Clip to stay inside cube
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _fcc_like_initialization():
    """
    Alternative initialization using an FCC-like structure.
    """
    # Use a body-centered arrangement
    # For 50 spheres, try a different grid
    # 5x5x2 with body-centered offsets
    
    rng = np.random.default_rng(123)
    
    # Start with a 5x5x2 grid
    xs = np.linspace(-0.45, 0.45, 5)
    ys = np.linspace(-0.45, 0.45, 5)
    zs = np.array([-0.22, 0.22])
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    
    centres = np.array(centres, dtype=np.float64)
    
    # Add random perturbation
    centres += rng.normal(0, 0.02, centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _random_initialization(seed):
    """Generate a random valid initialization."""
    rng = np.random.default_rng(seed)
    
    # Start with a grid and add noise
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].astype(np.float64)
    
    # Add random perturbation
    centres += rng.normal(0, 0.05, centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _simulated_annealing(centres, n_iter=5000, t_start=0.1, t_end=0.001, seed=0):
    """
    Perform simulated annealing to optimize the packing.
    """
    rng = np.random.default_rng(seed)
    current = centres.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    n = len(current)
    
    for i in range(n_iter):
        # Linear cooling schedule
        t = t_start * (t_end / t_start) ** (i / n_iter)
        
        # Random perturbation
        delta = rng.normal(0, t, current.shape)
        proposed = current + delta
        proposed = np.clip(proposed, -0.49, 0.49)
        
        proposed_score = _compute_score(proposed)
        delta_score = proposed_score - current_score
        
        # Accept if better, or with probability exp(delta_score / t)
        if delta_score > 0 or rng.random() < np.exp(delta_score / t):
            current = proposed
            current_score = proposed_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
    
    return best, best_score


def _lbfgs_polish(centres, max_iter=100):
    """
    Apply L-BFGS-B optimization to polish the packing.
    """
    try:
        result = minimize(
            _negative_score,
            centres,
            method='L-BFGS-B',
            bounds=[(-0.49, 0.49)] * 3 * N,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < _negative_score(centres):
            return result.x
        else:
            return centres
    except Exception:
        return centres


def construct():
    """
    Main entry point: construct 50 sphere centres in unit cube.
    Uses BCC/FCC initialization, simulated annealing, and L-BFGS-B polish.
    """
    start_time = time.time()
    
    # Try multiple initializations and keep the best
    best_centres = None
    best_score = -np.inf
    
    initializations = [
        ("bcc", _bcc_initialization()),
        ("fcc", _fcc_like_initialization()),
    ]
    
    # Add a few random initializations
    for seed in [7, 17, 27]:
        initializations.append((f"rand_{seed}", _random_initialization(seed)))
    
    for name, init_centres in initializations:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Simulated annealing
        sa_centres, sa_score = _simulated_annealing(
            init_centres,
            n_iter=3000,
            t_start=0.05,
            t_end=0.0005,
            seed=hash(name) % 2**32
        )
        
        # L-BFGS polish
        polished = _lbfgs_polish(sa_centres, max_iter=80)
        final_score = _compute_score(polished)
        
        if final_score > best_score:
            best_score = final_score
            best_centres = polished.copy()
    
    # Final polish with more iterations if time permits
    if best_centres is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        final = _lbfgs_polish(best_centres, max_iter=150)
        if _compute_score(final) > best_score:
            best_centres = final
            best_score = _compute_score(final)
    
    if best_centres is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N].astype(np.float64)
    
    # Ensure valid output
    best_centres = np.clip(best_centres, -0.49, 0.49)
    
    return best_centres
# EVOLVE-BLOCK-END
