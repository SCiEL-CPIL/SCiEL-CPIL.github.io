"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# Hard deadline with safety margin (evaluator kills at 150s; keep ~10s buffer)
_DEADLINE = 140.0


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|xi-xj|) using math.fsum for precision."""
    n = pts.shape[0]
    # Compute all pairwise distances
    # pts: (N, 3)
    # diff: (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Get upper triangle indices (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Filter out zeros (shouldn't happen if valid, but safe)
    mask = d > 1e-9
    if not np.all(mask):
        # If any points are too close, energy is effectively -inf (or very large negative)
        # But the task says "Points closer than 1e-9 ... make the candidate invalid".
        # So we should avoid returning such points.
        # For the purpose of the optimizer, if we encounter this, we treat it as bad energy.
        # However, log(1/d) goes to +inf as d->0? No, log(1/d) = -log(d). As d->0, -log(d) -> +inf.
        # Wait, the task says "minimize the logarithmic energy, the sum over all pairs of log(1/distance)".
        # If distance -> 0, 1/distance -> inf, log(inf) -> inf.
        # So minimizing energy means maximizing distances.
        # If points coincide, energy is +inf. We want to minimize, so we want to avoid coincidence.
        # The "lowest published energy" is negative. This implies distances are generally > 1?
        # No, on a unit sphere, max distance is 2. log(1/2) is negative.
        # If distance < 1, log(1/d) > 0. If distance > 1, log(1/d) < 0.
        # So we want points to be as far apart as possible, ideally > 1 apart.
        # If d < 1e-9, log(1/d) is huge positive. We want to minimize, so this is bad.
        # So returning such a config is invalid.
        return 1e100
    vals = np.log(1.0 / d[mask])
    return math.fsum(vals)


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return pts / norms


def _random_sphere(n):
    """Generate n random points on unit sphere."""
    vecs = np.random.normal(size=(n, 3))
    return _normalize(vecs)


def _fibonacci_sphere(n):
    """Generate n Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    
    # Initialize best point and energy
    # Start with Fibonacci spiral as a strong baseline
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy(best_pts)
    
    # Also try a few random starts to see if we can beat Fibonacci quickly
    for _ in range(5):
        if time.time() > _DEADLINE:
            break
        pts = _random_sphere(N)
        e = _log_energy(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            
    # Basin Hopping
    # We will perform random perturbations and accept if energy decreases.
    # This is essentially a simple stochastic local search.
    # To be more effective, we can use a "kick" size that decays or stays constant.
    # Given the time limit, a constant moderate kick size with many iterations is robust.
    
    current_pts = best_pts.copy()
    current_energy = best_energy
    
    # Parameters
    kick_sigma = 0.1  # Standard deviation for random perturbations
    # We can adaptively adjust kick_sigma, but let's keep it simple first.
    # Maybe start larger and decay?
    # Let's try a fixed sigma for simplicity and speed, relying on the large number of iterations.
    # Or maybe a schedule.
    
    # To improve efficiency, we can vectorize the energy calculation?
    # _log_energy uses math.fsum which is not vectorized.
    # However, N=150 is small. N^2 = 22500. 
    # math.fsum on 22500 elements is fast.
    # The bottleneck is likely the Python loop overhead if we do it in pure Python, 
    # but we are using NumPy for distance calculation and math.fsum for the sum.
    
    # Let's try to speed up _log_energy by using numpy sum first, then fsum?
    # Actually, math.fsum is for precision. The evaluator uses math.fsum.
    # If we use np.sum, we might get slightly different precision, but the score is based on the evaluator's calculation.
    # The evaluator calculates the score using math.fsum.
    # So my internal energy calculation just needs to be a good proxy.
    # np.sum is faster than math.fsum. Let's use np.sum for the internal optimization loop,
    # and only use math.fsum if we want to be super careful, but np.sum should be close enough for acceptance criteria.
    # Actually, let's stick to np.sum for speed in the loop. The difference in value is negligible for comparison.
    
    def _fast_log_energy(pts):
        n = pts.shape[0]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(n, k=1)
        d = dist[iu]
        # If any d < 1e-9, return large positive
        if np.any(d < 1e-9):
            return 1e100
        return np.sum(np.log(1.0 / d))
    
    # Re-evaluate best with fast energy to be consistent
    best_energy = _fast_log_energy(best_pts)
    current_energy = best_energy
    
    # Main loop
    # We want to maximize the number of iterations.
    # Each iteration:
    # 1. Generate random kick (N x 3 normal)
    # 2. Add to current points
    # 3. Normalize
    # 4. Calculate energy
    # 5. Compare
    
    # Let's benchmark one iteration to estimate steps per second.
    # But we can just loop until deadline.
    
    # To allow for better exploration, we can occasionally "reset" the kick size or do a larger jump.
    # Let's implement a simple adaptive kick:
    # If we accept a move, keep kick size.
    # If we reject, maybe slightly increase kick size? Or keep it?
    # Standard basin hopping often uses a fixed kick or a slowly decaying one.
    # Let's try a fixed kick of 0.05 to 0.1.
    
    kick_sigma = 0.08
    
    # We can also try to restart from a random point if we get stuck?
    # But since we accept only downhill moves, we never get "stuck" in the sense of cycling, 
    # we just converge to a local minimum.
    # To find a better global minimum, we need multiple starts or larger kicks.
    # Let's alternate between small kicks (refinement) and large kicks (exploration).
    
    # Strategy:
    # 80% of time: small kick (0.02) to refine current best.
    # 20% of time: large kick (0.5) to explore new basins.
    # If large kick leads to better energy, we adopt it.
    
    small_sigma = 0.02
    large_sigma = 0.5
    
    while time.time() < _DEADLINE:
        # Choose kick size
        if np.random.random() < 0.2:
            sigma = large_sigma
        else:
            sigma = small_sigma
            
        # Generate kick
        kick = np.random.normal(0.0, sigma, size=(N, 3))
        new_pts = current_pts + kick
        new_pts = _normalize(new_pts)
        
        new_energy = _fast_log_energy(new_pts)
        
        if new_energy < current_energy:
            current_pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        # Optional: if we are stuck for a long time, maybe do a full random restart?
        # But the "large kick" acts as a soft restart.
        
    # Final check: ensure the returned points are valid (no close pairs)
    # The evaluator checks for points closer than 1e-9.
    # If our best_pts has such pairs, it's invalid.
    # We should verify.
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    if np.any(d < 1e-9):
        # If invalid, try to fix by separating them? Or return a known valid config like Fibonacci.
        # Fibonacci is likely valid.
        return _fibonacci_sphere(N)
        
    return best_pts
# EVOLVE-BLOCK-END
