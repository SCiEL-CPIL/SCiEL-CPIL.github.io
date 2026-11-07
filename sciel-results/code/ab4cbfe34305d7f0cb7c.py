"""
Strategy: Parallel Multi-Trajectory Basin Hopping with Adaptive Step Size.

This approach departs from the "template + single BH" strategy by:
1. Using a pool of diverse initial configurations (FCC, HCP, BCC, Random) but
   relaxing them in parallel (4 threads) to save time.
2. Running multiple independent basin hopping trajectories from the best
   initial minima found.
3. Using an adaptive step size that increases when a move is rejected,
   helping to escape deep local minima (standard BH practice).
4. Aggressively managing the time budget to ensure we return the best found
   minimum before the 330s kill switch.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 325.0  # Safety margin before 330s kill
_START_TIME = None

def _time_left():
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_fragment(n, spacing=1.0):
    """Generate n atoms from an FCC lattice fragment."""
    a = spacing
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 3
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if (x + y + z) % 2 == 0:
                    points.append([x * 0.5 * a, y * 0.5 * a, z * 0.5 * a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _hcp_fragment(n, spacing=1.0):
    """Generate n atoms from an HCP lattice fragment."""
    a = spacing
    c = 1.633 * a
    points = []
    max_r = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            x = a * (i + 0.5 * j)
            y = a * (np.sqrt(3) / 2 * j)
            for k in range(-max_r, max_r + 1):
                points.append([x, y, k * c])
                points.append([x + 2*a/3, y + a*np.sqrt(3)/3, (k + 0.5) * c])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _bcc_fragment(n, spacing=1.0):
    """Generate n atoms from a BCC lattice fragment."""
    a = spacing
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    if len(points) < n:
        return None
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _random_compact(n, spacing=0.72, seed=0):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to all existing points
            # For small n, this is fine. For large n, might be slow but n<=80.
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: random points, relaxer will fix
        points = rng.uniform(-radius, radius, (n, 3)).tolist()
    return np.array(points)

def _relax_and_check(start_pos, potential, kind):
    """Helper to relax and return (energy, positions) or (inf, None) on failure."""
    try:
        energy, positions = relax(start_pos, potential, kind)
        # Basic validity check
        if not np.all(np.isfinite(positions)):
            return float('inf'), None
        # Check min distance
        # O(n^2) but n is small
        min_d = float('inf')
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_d:
                    min_d = d
        if min_d < 0.35:
            return float('inf'), None
        return energy, positions
    except Exception:
        return float('inf'), None

def _basin_hopping(start_pos, potential, kind, max_steps=200, initial_step=0.3, seed=0):
    """
    Perform basin hopping from a starting position.
    Returns the best (energy, positions) found.
    """
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_and_check(start_pos, potential, kind)
    if current_energy == float('inf'):
        return float('inf'), None
        
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    step_size = initial_step
    n = len(start_pos)
    
    for i in range(max_steps):
        if _time_left() < 10:  # Safety break
            break
            
        # Perturb
        noise = rng.normal(0, step_size, (n, 3))
        perturbed = current_pos + noise
        
        # Relax
        new_energy, new_pos = _relax_and_check(perturbed, potential, kind)
        
        if new_energy == float('inf'):
            # Relaxation failed or invalid, increase step to escape
            step_size *= 1.1
            continue
            
        if new_energy < current_energy:
            # Accept
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
            # Decrease step slightly on success to refine
            step_size *= 0.95
        else:
            # Reject
            # Increase step to try to escape the basin
            step_size *= 1.05
            
        # Cap step size
        if step_size > 1.0:
            step_size = 1.0
        if step_size < 0.05:
            step_size = 0.05

    return best_energy, best_pos

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # 1. Generate diverse initial configurations
    starts = []
    
    # Structural templates
    fcc = _fcc_fragment(n, 1.0)
    if fcc is not None:
        starts.append(fcc)
        
    hcp = _hcp_fragment(n, 1.0)
    if hcp is not None:
        starts.append(hcp)
        
    bcc = _bcc_fragment(n, 1.0)
    if bcc is not None:
        starts.append(bcc)
        
    # Random starts (multiple seeds for diversity)
    # Use fewer random starts since we will run BH from the best ones
    for seed in range(5):
        rand = _random_compact(n, 0.72, seed)
        starts.append(rand)
        
    # 2. Relax all initial configurations in parallel
    # We have 4 threads available.
    # Let's relax all 8 starts in parallel.
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_start = {executor.submit(_relax_and_check, s, potential, kind): s for s in starts}
        relaxed_results = []
        for future in as_completed(future_to_start):
            if _time_left() < 5:
                break
            energy, positions = future.result()
            if energy < float('inf') and positions is not None:
                relaxed_results.append((energy, positions))
                
    if not relaxed_results:
        # Fallback: return a random valid config if all relaxations failed
        # This should rarely happen
        pos = _random_compact(n, 0.72, 999)
        return pos

    # Sort by energy
    relaxed_results.sort(key=lambda x: x[0])
    
    # 3. Run Basin Hopping from the top K distinct minima
    # We pick up to 4 distinct starting points for BH to maximize coverage.
    # We ensure they are "distinct" by checking distance between them.
    bh_starts = []
    for e, p in relaxed_results:
        if len(bh_starts) >= 4:
            break
        # Check if this point is sufficiently different from existing BH starts
        is_distinct = True
        for _, bp in bh_starts:
            if np.linalg.norm(p - bp) < 0.5:  # Heuristic threshold
                is_distinct = False
                break
        if is_distinct:
            bh_starts.append((e, p))
            
    # If we don't have enough distinct starts, just take the best ones
    if len(bh_starts) < 4:
        for e, p in relaxed_results:
            if (e, p) not in bh_starts:
                bh_starts.append((e, p))
            if len(bh_starts) >= 4:
                break

    # 4. Run Basin Hopping in parallel (4 threads)
    best_energy = relaxed_results[0][0]
    best_pos = relaxed_results[0][1]
    
    # Allocate time budget for BH
    # Each BH trajectory runs until time is up or max steps.
    # We run them in parallel.
    
    bh_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, (e, p) in enumerate(bh_starts):
            # Seed BH with different seeds
            futures.append(executor.submit(_basin_hopping, p, potential, kind, max_steps=150, initial_step=0.4, seed=42+i))
            
        for future in as_completed(futures):
            if _time_left() < 5:
                # Can't wait for more, but we can't cancel easily.
                # We'll just break and return current best.
                break
            try:
                energy, positions = future.result()
                if energy < float('inf') and positions is not None:
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = positions
            except Exception:
                pass

    # 5. Final Polish: One more relaxation of the best result to ensure convergence
    # Sometimes BH stops slightly before full convergence due to step size logic.
    if _time_left() > 5:
        final_energy, final_pos = _relax_and_check(best_pos, potential, kind)
        if final_energy < float('inf') and final_pos is not None:
            best_energy = final_energy
            best_pos = final_pos

    return best_pos
# EVOLVE-BLOCK-END
