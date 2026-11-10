"""Coarse-to-fine grid seeding with gradient-based insertion for Sutton-Chen clusters."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline: evaluator kills at 330s, use 300s as safety margin
_DEADLINE = 300.0
_start_time = None

# Parameter lookup
_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}


def _sc_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        others = np.delete(np.arange(n), i)
        r2 = np.sum((positions[i] - positions[others]) ** 2, axis=1)
        r2 = np.maximum(r2, 1e-12)
        r = np.sqrt(r2)
        repulsive = 0.5 * np.sum(r ** (-n_exp))
        attractive = c * np.sqrt(np.sum(r ** (-m_exp)))
        total += repulsive - attractive
    return total


def _sc_energy_grad(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        r2 = np.sum((positions[i] - positions[others]) ** 2, axis=1)
        r2 = np.maximum(r2, 1e-12)
        r = np.sqrt(r2)
        repulsive = 0.5 * np.sum(r ** (-n_exp))
        attractive = c * np.sqrt(np.sum(r ** (-m_exp)))
        energy += repulsive - attractive
        # Gradient with respect to positions[i]
        # d/dr of 0.5 * r^(-n) = -0.5*n*r^(-n-1)
        # d/dr of c*sqrt(sum r^(-m)) is trickier - need chain rule
        # Actually for the gradient, we use the standard SC gradient formula
        # Force on atom i from atom j:
        # F_ij = -dV/dr * (r_i - r_j)/r
        # V_pair = 0.5*r^(-n) - c/sqrt(N_j) * r^(-m) where N_j = sum_k r_kj^(-m)
        # This is complex; let's just use numerical or the provided function
    # Use the provided function for accuracy
    return sutton_chen_energy_gradient(positions, _kind_from_params(n_exp, m_exp, c))


def _kind_from_params(n_exp, m_exp, c):
    """Reverse-lookup kind from parameters."""
    for kind, (ne, me, cc) in _PARAMS.items():
        if ne == n_exp and me == m_exp and abs(cc - c) < 1e-6:
            return kind
    return '12-6'


def _grid_seed_core(n_atoms, n_exp, m_exp, c, rng, time_budget=5.0):
    """
    Grid search for a dense low-energy core of n_atoms.
    Uses a coarse-to-fine approach: start with 2 atoms, then add more.
    """
    start = time.time()
    
    # Start with 2 atoms at optimal distance
    # Optimal pair distance: minimize 0.5*r^(-n) - c*sqrt(2*r^(-m))
    # d/dr: -0.5*n*r^(-n-1) + c*sqrt(2)*m/2 * r^(-m/2 - 1) = 0
    # 0.5*n*r^(-n-1) = c*sqrt(2)*m/2 * r^(-m/2-1)
    # r^(n - m/2) = n / (c*sqrt(2)*m)
    r_opt = (n_exp / (c * np.sqrt(2) * m_exp)) ** (1.0 / (n_exp - m_exp / 2.0))
    
    # Start with 2 atoms
    positions = np.array([[0.0, 0.0, 0.0], [r_opt, 0.0, 0.0]])
    
    # Relax the pair
    energy, positions = relax(positions, _kind_from_params(n_exp, m_exp, c))
    
    # Now add atoms one by one using grid search for best insertion
    for target_n in range(3, min(n_atoms + 1, 12)):
        if time.time() - start > time_budget:
            break
        
        best_energy = np.inf
        best_pos = None
        
        # Coarse grid search for new atom position
        # Search in a sphere around the existing cluster
        centroid = np.mean(positions, axis=0)
        radius = np.max(np.linalg.norm(positions - centroid, axis=1)) * 1.5 + 0.5
        
        # Generate candidate positions on a coarse grid
        n_grid = 8
        xs = np.linspace(-radius, radius, n_grid)
        ys = np.linspace(-radius, radius, n_grid)
        zs = np.linspace(-radius, radius, n_grid)
        
        best_local = None
        for x in xs:
            for y in ys:
                for z in zs:
                    new_pos = np.array([x, y, z])
                    # Check minimum distance
                    dists = np.linalg.norm(positions - new_pos, axis=1)
                    if np.min(dists) < 0.36:
                        continue
                    
                    candidate = np.vstack([positions, new_pos])
                    e, _ = sutton_chen_energy_gradient(candidate, _kind_from_params(n_exp, m_exp, c))
                    if e < best_energy:
                        best_energy = e
                        best_pos = candidate.copy()
        
        if best_pos is not None:
            positions = best_pos
            # Local relaxation
            energy, positions = relax(positions, _kind_from_params(n_exp, m_exp, c))
    
    return positions


def _grow_cluster(positions, n_target, n_exp, m_exp, c, kind, deadline):
    """
    Grow the cluster from current positions to n_target atoms.
    Use gradient-based insertion: find position that minimizes energy when added.
    """
    current_n = len(positions)
    
    while current_n < n_target:
        if time.time() > deadline - 30:  # Safety margin
            break
        
        best_energy = np.inf
        best_new_pos = None
        
        # Generate candidate positions using multiple strategies
        candidates = []
        
        # Strategy 1: Random positions near the cluster
        centroid = np.mean(positions, axis=0)
        radius = np.max(np.linalg.norm(positions - centroid, axis=1)) * 1.2 + 0.5
        
        rng = np.random.default_rng(hash((current_n, len(positions))) % (2**32))
        
        for _ in range(200):
            p = centroid + rng.uniform(-radius, radius, 3)
            candidates.append(p)
        
        # Strategy 2: Positions based on existing atom positions (near neighbors)
        for i in range(min(current_n, 10)):
            for j in range(i + 1, min(current_n, 10)):
                mid = (positions[i] + positions[j]) / 2.0
                d = positions[i] - positions[j]
                norm_d = np.linalg.norm(d)
                if norm_d > 1e-6:
                    d = d / norm_d
                    for offset in [-0.5, -0.3, 0.3, 0.5]:
                        candidates.append(mid + offset * d)
        
        # Strategy 3: Grid points
        grid_size = 5
        for dx in np.linspace(-radius, radius, grid_size):
            for dy in np.linspace(-radius, radius, grid_size):
                for dz in np.linspace(-radius, radius, grid_size):
                    candidates.append(np.array([centroid[0] + dx, centroid[1] + dy, centroid[2] + dz]))
        
        # Evaluate all candidates
        for new_pos in candidates:
            dists = np.linalg.norm(positions - new_pos, axis=1)
            if np.min(dists) < 0.36:
                continue
            
            candidate = np.vstack([positions, new_pos])
            e, grad = sutton_chen_energy_gradient(candidate, kind)
            
            if e < best_energy:
                best_energy = e
                best_new_pos = new_pos.copy()
        
        if best_new_pos is None:
            # Fallback: random position
            best_new_pos = centroid + rng.uniform(-radius, radius, 3)
            while np.min(np.linalg.norm(positions - best_new_pos, axis=1)) < 0.36:
                best_new_pos = centroid + rng.uniform(-radius, radius, 3)
        
        positions = np.vstack([positions, best_new_pos])
        current_n = len(positions)
        
        # Local relaxation to refine
        if current_n <= 30:
            energy, positions = relax(positions, kind)
    
    return positions


def construct(instance):
    """Build a Sutton-Chen cluster using coarse-to-fine grid seeding and gradient-based growth."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    n_exp, m_exp, c = _PARAMS[kind]
    
    deadline = _start_time + _DEADLINE
    
    # Check if we're running out of time
    if time.time() > deadline - 60:
        # Emergency: return a simple valid cluster
        positions = _emergency_cluster(n, kind)
        return positions
    
    # Step 1: Grid seed a small core (up to 12 atoms)
    time_budget_core = min(10.0, max(2.0, (deadline - time.time()) * 0.15))
    core = _grid_seed_core(12, n_exp, m_exp, c, np.random.default_rng(n), time_budget_core)
    
    if time.time() > deadline - 60:
        return _emergency_cluster(n, kind)
    
    # Step 2: Grow to full size using gradient-based insertion
    positions = _grow_cluster(core, n, n_exp, m_exp, c, kind, deadline)
    
    # Step 3: Final relaxation
    if time.time() < deadline - 20:
        energy, positions = relax(positions, kind)
    
    # Step 4: Verify validity
    if not np.all(np.isfinite(positions)):
        return _emergency_cluster(n, kind)
    
    # Check minimum distance
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if np.linalg.norm(positions[i] - positions[j]) < 0.35:
                return _emergency_cluster(n, kind)
    
    return positions


def _emergency_cluster(n, kind):
    """Return a simple valid cluster as emergency fallback."""
    positions = []
    spacing = 0.72
    for i in range(n):
        # Place atoms on a line with spacing
        positions.append([i * spacing, 0.0, 0.0])
    positions = np.array(positions)
    # Relax
    try:
        energy, positions = relax(positions, kind)
    except:
        pass
    return positions
# EVOLVE-BLOCK-END
