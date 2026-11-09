"""Dzugutov cluster global minima search with Langevin dynamics annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
_TOTAL_BUDGET = 310.0  # 330s limit with 20s safety margin
_HARD_LIMIT = 325.0
_START_TIME = None
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_RESULTS = {}

A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return _TOTAL_BUDGET - _elapsed()


def _hard_remaining():
    return _HARD_LIMIT - _elapsed()


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        diffs = positions[i+1:] - positions[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        mask_a = dists < a_DZ
        mask_b = (dists >= a_DZ) & (dists < b_DZ)
        
        if np.any(mask_a):
            da = dists[mask_a]
            da = np.maximum(da, 1e-10)
            v_a = A_DZ * (da ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (da - a_DZ))
            total += np.sum(v_a)
        
        if np.any(mask_b):
            db = dists[mask_b]
            db = np.maximum(db, 1e-10)
            v_b = B_DZ * np.exp(d_DZ / (db - b_DZ))
            total += np.sum(v_b)
    
    return total


def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy."""
    try:
        return dzugutov_energy_gradient(positions)
    except Exception:
        # Finite difference fallback
        n = len(positions)
        grad = np.zeros_like(positions)
        eps = 1e-5
        E0 = _dzugutov_energy(positions)
        for i in range(n):
            for j in range(3):
                pos_p = positions.copy()
                pos_p[i, j] += eps
                E_p = _dzugutov_energy(pos_p)
                grad[i, j] = (E_p - E0) / eps
        return grad


def _generate_initial_positions(n_atoms, rng, method='random_sphere'):
    """Generate initial positions for Langevin dynamics."""
    if method == 'random_sphere':
        radius = 1.5 * n_atoms ** (1.0/3.0)
        points = rng.uniform(-radius, radius, (n_atoms, 3))
        # Ensure minimum distance by rejection
        min_dist = 0.52
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    points[j] += rng.normal(0, 0.1, 3)
        return points
    elif method == 'bcc_lattice':
        a = 1.1
        max_coord = int(np.ceil(n_atoms ** (1.0/3.0) / 2)) + 2
        points = []
        for ix in range(-max_coord, max_coord + 1):
            for iy in range(-max_coord, max_coord + 1):
                for iz in range(-max_coord, max_coord + 1):
                    p = np.array([ix, iy, iz]) * a
                    if len(points) < n_atoms:
                        points.append(p)
                    p2 = (np.array([ix, iy, iz]) + 0.5) * a
                    if len(points) < n_atoms:
                        points.append(p2)
        points = np.array(points[:n_atoms*2])
        dists = np.linalg.norm(points, axis=1)
        order = np.argsort(dists)
        return points[order[:n_atoms]]
    else:  # icosahedral_seed
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        edge = 2 * phi
        verts = verts / edge * 1.1
        points = [np.array([0.0, 0.0, 0.0])]
        points.extend([v.copy() for v in verts])
        if n_atoms <= 13:
            return np.array(points[:n_atoms])
        remaining = n_atoms - 13
        outer_radius = 1.1 * 1.8
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(remaining):
            y = 1 - 2 * i / max(1, remaining - 1)
            r = np.sqrt(1 - y**2)
            theta = golden_angle * i + rng.uniform(0, 2*np.pi)
            points.append(np.array([np.cos(theta)*r, y, np.sin(theta)*r]) * outer_radius)
        return np.array(points[:n_atoms])


def _langevin_anneal(positions, rng, time_limit=10.0, dt=0.001, T_start=2.0, T_end=0.05,
                      n_steps=None, friction=0.5, mass=1.0):
    """
    Langevin dynamics with exponential cooling schedule.
    Uses the Dzugutov gradient to drive atoms toward low-energy configurations
    while thermal noise helps escape local minima.
    """
    start_time = time.time()
    n = len(positions)
    
    if n_steps is None:
        # Estimate steps from time limit
        n_steps = int(time_limit / dt / 10)  # Conservative estimate
        n_steps = max(100, min(n_steps, 50000))
    
    pos = positions.copy().astype(float)
    vel = np.zeros_like(pos)
    T = T_start
    
    best_pos = pos.copy()
    best_energy = _dzugutov_energy(pos)
    
    # Precompute cooling factor
    if T_end > 0:
        cool_factor = (T_end / T_start) ** (1.0 / n_steps)
    else:
        cool_factor = 0.999
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient
        try:
            grad = dzugutov_energy_gradient(pos)
        except Exception:
            break
        
        # Langevin update: m*dv/dt = -grad - gamma*v + random_force
        random_force = np.sqrt(2 * friction * T * mass / dt) * rng.standard_normal(pos.shape)
        vel = vel * (1 - friction * dt / mass) - grad * dt / mass + random_force * dt / mass
        pos = pos + vel * dt
        
        # Keep atoms within reasonable bounds
        max_radius = 2.0 * n ** (1.0/3.0) + 5.0
        norms = np.linalg.norm(pos, axis=1)
        for i in range(n):
            if norms[i] > max_radius:
                pos[i] *= max_radius / norms[i]
        
        # Compute energy periodically (every 10 steps for speed)
        if step % 10 == 0 or step == n_steps - 1:
            energy = _dzugutov_energy(pos)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        
        # Cool down
        T *= cool_factor
        if T < 0.01:
            T = 0.01
    
    # Final relaxation
    try:
        final_energy, final_pos = relax(best_pos, 'dzugutov', None)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = final_pos
    except Exception:
        pass
    
    return best_pos, best_energy


def _parallel_langevin(n_atoms, rng, time_budget, n_runs=4):
    """
    Run multiple Langevin annealing simulations in parallel with different
    initial conditions and cooling schedules.
    """
    start_time = time.time()
    time_per_run = time_budget / max(1, n_runs) - 2.0  # Leave 2s margin
    time_per_run = max(1.0, time_per_run)
    
    results = []
    
    def _run_langevin(run_id):
        run_rng = np.random.default_rng(run_id * 12345 + 678)
        
        # Different initial conditions and parameters for each run
        init_methods = ['random_sphere', 'bcc_lattice', 'icosahedral_seed', 'random_sphere']
        init_method = init_methods[run_id % len(init_methods)]
        
        # Vary temperature schedule
        T_starts = [2.0, 3.0, 1.5, 2.5]
        T_ends = [0.05, 0.03, 0.08, 0.04]
        T_start = T_starts[run_id % len(T_starts)]
        T_end = T_ends[run_id % len(T_ends)]
        
        # Vary friction
        frictions = [0.5, 0.3, 0.7, 0.4]
        friction = frictions[run_id % len(frictions)]
        
        positions = _generate_initial_positions(n_atoms, run_rng, init_method)
        
        pos, energy = _langevin_anneal(
            positions, run_rng,
            time_limit=time_per_run,
            dt=0.001,
            T_start=T_start,
            T_end=T_end,
            friction=friction
        )
        
        return pos, energy
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=min(4, n_runs)) as executor:
        futures = [executor.submit(_run_langevin, i) for i in range(n_runs)]
        for future in as_completed(futures):
            if time.time() - start_time > time_budget - 5.0:
                break
            try:
                pos, energy = future.result(timeout=1.0)
                results.append((energy, pos))
            except Exception:
                pass
    
    if not results:
        # Fallback: single sequential run
        try:
            positions = _generate_initial_positions(n_atoms, rng, 'random_sphere')
            pos, energy = _langevin_anneal(positions, rng, time_limit=time_budget - 5.0)
            results.append((energy, pos))
        except Exception:
            pass
    
    if not results:
        # Last resort: random positions
        radius = 1.5 * n_atoms ** (1.0/3.0)
        positions = rng.uniform(-radius, radius, (n_atoms, 3))
        try:
            energy, positions = relax(positions, 'dzugutov', None)
        except Exception:
            energy = _dzugutov_energy(positions)
        results.append((energy, positions))
    
    # Return best result
    results.sort(key=lambda x: x[0])
    return results[0][1]


def _self_assemble_cluster(n_atoms, rng, time_limit=5.0):
    """
    Physics-driven self-assembly: grow cluster by adding atoms one at a time,
    allowing the system to relax after each addition.
    """
    start_time = time.time()
    
    # Start with a small seed (5 atoms in a tetrahedron + center)
    seed = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5]
    ])
    
    try:
        energy, seed = relax(seed, 'dzugutov', None)
    except Exception:
        pass
    
    current = seed.copy()
    current_energy = _dzugutov_energy(current)
    
    for target_n in range(6, n_atoms + 1):
        if time.time() - start_time > time_limit:
            break
        
        best_pos = None
        best_e = np.inf
        
        n_candidates = 6
        for c in range(n_candidates):
            if c == 0:
                new_p = np.array([0.0, 0.0, 0.0])
            else:
                idx = rng.integers(0, len(current))
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                new_p = current[idx] + direction * 1.0
            
            valid = True
            for q in current:
                if np.linalg.norm(new_p - q) < 0.51:
                    valid = False
                    break
            if not valid:
                continue
            
            test = np.vstack([current, new_p])
            try:
                e_test, relaxed_test = relax(test, 'dzugutov', None)
                if e_test < best_e:
                    best_e = e_test
                    best_pos = relaxed_test
            except Exception:
                pass
        
        if best_pos is not None:
            current = best_pos
        else:
            new_p = rng.uniform(-2.0, 2.0, 3)
            current = np.vstack([current, new_p])
            try:
                current_energy, current = relax(current, 'dzugutov', None)
            except Exception:
                pass
    
    return current[:n_atoms]


def _basin_hopping(positions, rng, max_iter=50, step_size=0.3, time_limit=10.0):
    """
    Basin hopping: perturb, relax, accept if lower energy.
    """
    start_time = time.time()
    current = positions.copy()
    current_energy = _dzugutov_energy(current)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        perturbed = current + rng.normal(0, step_size, current.shape)
        
        try:
            e_new, relaxed = relax(perturbed, 'dzugutov', None)
        except Exception:
            continue
        
        if e_new < current_energy or rng.random() < 0.1:
            current = relaxed
            current_energy = e_new
        
        step_size *= 0.98
    
    return current, current_energy


def _process_instance(size, potential, parameter, time_budget):
    """
    Process a single instance using Langevin dynamics annealing as the primary
    search strategy, with basin hopping refinement.
    """
    rng = np.random.default_rng(size * 1000 + 7)
    
    # Allocate time: 70% Langevin, 20% basin hopping, 10% buffer
    langevin_time = time_budget * 0.70
    bh_time = time_budget * 0.20
    
    # Primary strategy: parallel Langevin annealing
    best_positions = None
    best_energy = np.inf
    
    try:
        best_positions = _parallel_langevin(size, rng, langevin_time, n_runs=4)
        best_energy = _dzugutov_energy(best_positions)
    except Exception:
        pass
    
    # If Langevin failed, fall back to self-assembly
    if best_positions is None or best_energy > 1e10:
        try:
            assembled = _self_assemble_cluster(size, np.random.default_rng(size * 7000 + 3), 
                                                time_limit=langevin_time * 0.5)
            best_positions = assembled
            best_energy = _dzugutov_energy(best_positions)
        except Exception:
            # Last resort
            radius = 1.5 * size ** (1.0/3.0)
            best_positions = rng.uniform(-radius, radius, (size, 3))
            try:
                best_energy, best_positions = relax(best_positions, 'dzugutov', None)
            except Exception:
                best_energy = _dzugutov_energy(best_positions)
    
    # Refine with basin hopping if time remains
    if _hard_remaining() > 10.0 and best_positions is not None:
        try:
            bh_pos, bh_energy = _basin_hopping(
                best_positions,
                np.random.default_rng(size * 9000 + 5),
                max_iter=40,
                step_size=0.25,
                time_limit=bh_time
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except Exception:
            pass
    
    # Final relaxation
    try:
        final_energy, final_pos = relax(best_positions, 'dzugutov', None)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    except Exception:
        pass
    
    # Ensure valid output
    if best_positions is None or len(best_positions) != size:
        radius = 1.5 * size ** (1.0/3.0)
        best_positions = rng.uniform(-radius, radius, (size, 3))
    
    return best_positions


def _relax_and_score(positions, potential, parameter):
    """Relax positions and return energy, positions."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return np.inf, positions


def construct(instance):
    """Find low-energy Dzugutov cluster using Langevin dynamics annealing."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    remaining_instances = [s for s in _INSTANCE_ORDER if s >= size]
    time_per_instance = _remaining() / max(1, len(remaining_instances))
    time_per_instance = min(time_per_instance, 45.0)
    
    # Check if we already have a result
    if size in _INSTANCE_RESULTS:
        return _INSTANCE_RESULTS[size]
    
    # Process this instance
    result = _process_instance(size, potential, parameter, time_per_instance)
    
    # Cache the result
    _INSTANCE_RESULTS[size] = result
    
    return result
# EVOLVE-BLOCK-END
