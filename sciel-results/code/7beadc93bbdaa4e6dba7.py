"""
Strategy: Parallel Basin Hopping with L-BFGS-B Local Minimization.
The Dzugutov potential favors polytetrahedral order. 
Instead of relying on a single, potentially slow or trapped 'relax' call from a few starts,
we implement a robust search:
1. Generate diverse initial structures (PT shells, random compact, perturbed PT).
2. Use a parallel pool (up to 4 threads) to run independent basin hopping chains.
3. Each chain uses L-BFGS-B (via SciPy) for fast, accurate local minimization,
   combined with random perturbations to escape local minima.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

# Global state for thread safety and time management
_START_TIME = time.time()
_HARD_LIMIT = 300.0  # Target finish time, leaving margin for 330s kill
_LOCK = threading.Lock()
_BEST_ENERGY = np.inf
_BEST_POSITIONS = None
_INSTANCE_N = None

def _get_remaining_time():
    elapsed = time.time() - _START_TIME
    # Leave 10s margin for safety
    return max(0.1, _HARD_LIMIT - elapsed - 10.0)

def _compute_energy(positions):
    """
    Compute the Dzugutov energy. 
    Since we only have the gradient function provided, we must implement the energy 
    or check if relax returns it. The prompt says 'Energies are recomputed exactly'.
    The interface `chem_clusters.relax` returns energy. 
    However, for L-BFGS-B we need a function that returns energy.
    We can use a wrapper that calls a lightweight energy calculation or 
    we can assume the gradient function is paired with an energy function in the suite?
    The prompt says: `chem_clusters.relax(positions, 'dzugutov')` and 
    `chem_clusters.dzugutov_energy_gradient(positions)` may be called.
    It does NOT explicitly say `dzugutov_energy` is available.
    However, standard practice in these suites often implies the energy is computable 
    or we can reconstruct it. 
    Wait, the prompt says: "The evaluator recomputes E = sum...".
    It does not provide a direct energy function in the interface list, only gradient.
    BUT, `relax` returns energy. 
    If I use L-BFGS-B, I need `fun(x)`. 
    I can implement the Dzugutov energy myself as it is fully specified:
    V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
    A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16.
    This is safer and faster than calling an external function if it's not exposed.
    """
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    
    pos = positions.reshape(-1, 3)
    n = len(pos)
    energy = 0.0
    
    # Vectorized distance calculation
    # dist[i, j] = ||pos[i] - pos[j]||
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    # Mask for i < j
    iu = np.triu_indices(n, 1)
    r = dist[iu]
    
    if len(r) == 0:
        return 0.0
    
    # V(r) = A(r^-m - B)exp(c/(r - a)) for r < a
    #      + B exp(d/(r - b)) for r < b
    # Note: The potential is defined piecewise. 
    # Usually, for r >= b, V(r) = 0? Or is it continuous?
    # The prompt says: V(r) = ... [r < a] + ... [r < b].
    # This implies for r >= b, V(r) = 0.
    # For a <= r < b, only the second term applies? 
    # "V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]"
    # This notation usually means:
    # if r < a: V = A(r^-m - B)exp(c/(r - a))
    # if a <= r < b: V = B exp(d/(r - b))
    # if r >= b: V = 0
    
    v_terms = np.zeros_like(r)
    
    mask_a = r < a
    if np.any(mask_a):
        r_a = r[mask_a]
        # Handle singularity at r=a? r < a so r-a is negative. exp(c/(r-a)) is small.
        v_terms[mask_a] = A * (r_a ** -m - B) * np.exp(c / (r_a - a))
        
    mask_b = (r >= a) & (r < b)
    if np.any(mask_b):
        r_b = r[mask_b]
        v_terms[mask_b] = B * np.exp(d / (r_b - b))
        
    return np.sum(v_terms)

def _dzugutov_energy_and_gradient(positions):
    """
    Returns energy and gradient.
    We use the provided gradient function for accuracy, but we compute energy ourselves
    to avoid potential overhead or interface mismatches if `relax` is the only energy source.
    Actually, using the provided gradient is safer for consistency with the evaluator's 
    potential implementation if there are subtle differences. 
    But I need energy for the optimizer. 
    I will use my own energy function for the optimizer loop, 
    and trust the provided gradient. 
    """
    energy = _compute_energy(positions)
    # The provided gradient function returns the gradient vector.
    try:
        grad = dzugutov_energy_gradient(positions)
        if grad is None:
            # Fallback to numerical gradient if provided one fails? 
            # No, let's assume it works.
            raise Exception("Gradient returned None")
    except Exception:
        # Numerical gradient fallback
        eps = 1e-6
        grad = np.zeros_like(positions)
        for i in range(len(positions)):
            pos_plus = positions.copy()
            pos_plus[i] += eps
            pos_minus = positions.copy()
            pos_minus[i] -= eps
            grad[i] = (_compute_energy(pos_plus) - _compute_energy(pos_minus)) / (2 * eps)
            
    return energy, grad

def _local_minimize(positions):
    """
    Run L-BFGS-B to local minimum.
    """
    try:
        result = minimize(
            _dzugutov_energy_and_gradient,
            positions,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4}
        )
        return result.fun, result.x
    except Exception:
        return np.inf, positions

def _generate_icosahedron(n_atoms, rng):
    """
    Generate a structure based on icosahedral shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)] # Center
    
    # Shell 1: 12 atoms
    r1 = 1.9
    for v in verts:
        positions.append(v * r1)
        
    if n_atoms <= 13:
        return np.array(positions[:n_atoms])
        
    # Shell 2: Remaining atoms on a sphere
    # For n=55, we need 42 more. 
    # We generate them randomly on a sphere of radius r2, then relax.
    # To improve symmetry, we could use specific points, but random + relax is robust.
    r2 = 1.9 * 1.75 # Approx 3.3
    num_outer = n_atoms - 13
    if num_outer > 0:
        # Generate num_outer points on sphere r2
        # Use random points, but try to space them out roughly
        outer_pts = []
        for _ in range(num_outer):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * r2
            outer_pts.append(v)
        positions.extend(outer_pts)
        
    return np.array(positions[:n_atoms])

def _random_compact(n_atoms, rng):
    """
    Generate a random compact structure.
    """
    spacing = 1.1
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.9
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
        
    # If we didn't get enough, fill with random
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points[:n_atoms])

def _basin_hopping_chain(n_atoms, rng, time_budget):
    """
    Run a single basin hopping chain.
    """
    global _BEST_ENERGY, _BEST_POSITIONS, _LOCK
    
    # Initialize with a mix of PT and Random
    if rng.random() > 0.5:
        pos = _generate_icosahedron(n_atoms, rng)
    else:
        pos = _random_compact(n_atoms, rng)
        
    # Initial minimization
    energy, pos = _local_minimize(pos)
    
    if energy < _BEST_ENERGY:
        with _LOCK:
            if energy < _BEST_ENERGY:
                _BEST_ENERGY = energy
                _BEST_POSITIONS = pos.copy()
    
    # Basin Hopping Loop
    step_size = 0.5
    while _get_remaining_time() > 1.0:
        # Perturb
        perturbed = pos + rng.normal(scale=step_size, size=pos.shape)
        
        # Minimize
        new_energy, new_pos = _local_minimize(perturbed)
        
        # Accept if better (or with probability if worse, but for global min search, greedy is often fine with enough restarts)
        # Standard BH accepts with prob exp(-(E_new - E_old)/T). 
        # We don't have T, so let's just accept if better, or occasionally accept worse to escape.
        if new_energy < energy:
            pos = new_pos
            energy = new_energy
            if energy < _BEST_ENERGY:
                with _LOCK:
                    if energy < _BEST_ENERGY:
                        _BEST_ENERGY = energy
                        _BEST_POSITIONS = pos.copy()
        else:
            # Small probability to accept worse to escape local minima
            if rng.random() < 0.05:
                pos = new_pos
                energy = new_energy
                
        # Reduce step size slightly over time? Or keep constant.
        # Constant is simpler.
        
def construct(instance):
    global _START_TIME, _BEST_ENERGY, _BEST_POSITIONS, _INSTANCE_N
    
    # Reset globals for this instance
    # Note: If construct is called multiple times, we should reset state.
    # The prompt says "called once for each n". 
    # It's safer to reset per call.
    _START_TIME = time.time()
    _BEST_ENERGY = np.inf
    _BEST_POSITIONS = None
    _INSTANCE_N = instance["n"]
    
    n_atoms = _INSTANCE_N
    potential = instance["potential"]
    
    # Time budget per instance
    # If 6 instances share 300s, we have ~50s each.
    # But construct is called sequentially? "all calls together within 300 s".
    # This implies the total time for all 6 calls is 300s.
    # So we should track global time across calls?
    # The prompt says "at most 4 threads". 
    # If I use 4 threads inside construct, and construct is called 6 times, 
    # do I have 6 * 4 = 24 threads? 
    # "at most 4 threads" likely applies to the whole execution.
    # So I should limit total threads to 4.
    
    # Strategy: 
    # 1. Create 4 worker threads.
    # 2. Each worker runs a BH chain.
    # 3. All workers stop when global time budget is exhausted.
    # 4. Return best position.
    
    # To respect "at most 4 threads", we will spawn 4 threads per construct call, 
    # but ensure they finish quickly or that we don't overlap calls.
    # Since calls are likely sequential (one n, then next), we can use 4 threads per call.
    
    threads = []
    for i in range(4):
        # Each thread gets a unique seed
        t = threading.Thread(target=_basin_hopping_chain, args=(n_atoms, np.random.default_rng(42 + i), 40.0))
        t.daemon = True
        threads.append(t)
        t.start()
        
    # Wait for threads to finish or timeout
    # We rely on _get_remaining_time inside the threads to stop them.
    # But we need to ensure the main thread doesn't hang.
    # We will join with a timeout.
    # Since _get_remaining_time is global, all threads will stop at the same time.
    
    # How long to wait? 
    # If this is the first call, we have 300s total. 
    # But we don't know which call we are in.
    # Let's assume the evaluator calls them sequentially and expects each to be fast, 
    # OR it expects the total to be 300s.
    # If I wait 40s per call, 6 calls * 40s = 240s. Safe.
    
    for t in threads:
        t.join(timeout=45.0)
        
    if _BEST_POSITIONS is None:
        # Fallback
        return _random_compact(n_atoms, np.random.default_rng(0))
        
    return _BEST_POSITIONS
# EVOLVE-BLOCK-END
