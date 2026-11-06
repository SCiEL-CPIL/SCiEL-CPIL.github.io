"""Morse cluster global minima: parallel basin hopping with gradient-guided refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 285.0
    return _T0


def _time_left():
    global _HARD_DEADLINE
    return max(0.0, _HARD_DEADLINE - time.time())


def _enforce_min_dist(P, min_d=0.5):
    """Vectorized: ensure all pair distances >= min_d by local pushes."""
    P = np.array(P, dtype=float)
    n = P.shape[0]
    for _ in range(30):
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        mask = dists < min_d
        if not np.any(mask):
            break
        iu, ju = np.where(np.triu(mask, 1))
        for i, j in zip(iu, ju):
            d = P[i] - P[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.array([1.0, 0.0, 0.0])
            else:
                d = d / dist
            push = (min_d - dist) / 2.0 + 0.003
            P[i] += push * d
            P[j] -= push * d
    return P


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice, centered."""
    pts = []
    a = spacing
    basis = [np.array([0, 0, 0]), np.array([0, a/2, a/2]),
             np.array([a/2, 0, a/2]), np.array([a/2, a/2, 0])]
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                base = np.array([i, j, k]) * a
                for b in basis:
                    pts.append(base + b)
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _hcp_points(n, spacing=1.0):
    """Generate n points from an HCP lattice, centered."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0)/2.0 * j
                z = c * (k + (1.0/3.0) * ((j % 2) + 1))
                pts.append(np.array([x, y, z]))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _bcc_points(n, spacing=1.0):
    """Generate n points from a BCC lattice, centered."""
    pts = []
    a = spacing
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                base = np.array([i, j, k]) * a
                pts.append(base)
                pts.append(base + np.array([a/2, a/2, a/2]))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice, centered."""
    pts = []
    a = spacing
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                pts.append(np.array([i, j, k]) * a)
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _icosahedral_shells(n):
    """Generate points on nested icosahedral shells."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]
    shells = [1.0, 1.4, 1.8, 2.15, 2.5, 2.85, 3.2, 3.55]
    for s in shells:
        if len(pts) >= n:
            break
        shell_pts = ico * s
        rotations = [np.eye(3),
                     np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                     np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])]
        for rot in rotations:
            rotated = shell_pts @ rot.T
            for p in rotated:
                if len(pts) >= n:
                    break
                pts.append(p)
    pts = np.array(pts[:n])
    return pts


def _random_compact(n, spacing=1.0, seed=42):
    """Generate a random compact cluster with minimum distance enforcement."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    max_tries = 50000
    tries = 0
    while len(points) < n and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or np.min(np.linalg.norm(np.array(points) - p, axis=1)) >= 0.52 * spacing:
                points.append(p)
        tries += 1
    if len(points) < n:
        for i in range(n - len(points)):
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            r = radius * 0.9
            p = np.array([r * np.sin(phi) * np.cos(theta),
                          r * np.sin(phi) * np.sin(theta),
                          r * np.cos(phi)])
            points.append(p)
    return np.array(points)


def _perturb(P, scale, rng):
    """Add Gaussian noise to positions."""
    return P + rng.normal(0, scale, P.shape)


def _relax_one(P, potential, parameter):
    """Relax a single configuration, returning (energy, positions)."""
    try:
        energy, positions = relax(P, potential, parameter)
        return energy, positions
    except Exception:
        return np.inf, P


def _gradient_refine(P, potential, parameter, n_steps=5, lr=0.02, rng=None):
    """Gradient-based local refinement using morse_energy_gradient."""
    if rng is None:
        rng = np.random.default_rng()
    best_E, best_P = np.inf, P.copy()
    try:
        E, _ = relax(P, potential, parameter)
        best_E = E
        best_P = P.copy()
    except Exception:
        return best_E, best_P
    
    for step in range(n_steps):
        if _time_left() < 2.0:
            break
        try:
            E, grad = morse_energy_gradient(P, parameter)
            if E < best_E:
                best_E = E
                best_P = P.copy()
            # Gradient step with adaptive learning rate
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-10:
                break
            step_size = lr / (1.0 + gnorm * 0.1)
            P = P - step_size * grad
            P = _enforce_min_dist(P, 0.5)
        except Exception:
            break
    return best_E, best_P


def _parallel_basin_hopping(chains, potential, parameter, n_chains, time_budget):
    """Run multiple independent basin-hopping chains with shared global best."""
    n_chains = len(chains)
    global_best_E = np.inf
    global_best_P = None
    chain_states = []
    
    # Initialize chains
    for i in range(n_chains):
        P = _enforce_min_dist(chains[i], 0.5)
        E, P_relaxed = _relax_one(P, potential, parameter)
        if E < global_best_E:
            global_best_E = E
            global_best_P = P_relaxed.copy()
        chain_states.append({'P': P_relaxed, 'E': E, 'scale': 0.25 + 0.1 * i})
    
    # Adaptive basin hopping
    max_iterations = 200
    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break
        
        for ci in range(n_chains):
            if _time_left() < 3.0:
                    break
            
            state = chain_states[ci]
            rng = np.random.default_rng(ci * 10000 + iteration * 7)
            
            # Perturb
            scale = state['scale'] * (0.8 + 0.4 * rng.random())
            P_pert = _perturb(state['P'], scale, rng)
            P_pert = _enforce_min_dist(P_pert, 0.5)
            
            # Relax
            E_new, P_new = _relax_one(P_pert, potential, parameter)
            
            # Accept if better than chain's current, or with Metropolis acceptance
            if E_new < state['E']:
                state['P'] = P_new
                state['E'] = E_new
                if E_new < global_best_E:
                    global_best_E = E_new
                    global_best_P = P_new.copy()
            else:
                # Metropolis acceptance for exploration
                delta_E = E_new - state['E']
                if delta_E < 0 or rng.random() < np.exp(-delta_E / max(0.01, state['E'] * 0.001)):
                    state['P'] = P_new
                    state['E'] = E_new
            
            # Re-seed from global best if chain stagnates
            if global_best_P is not None and iteration % 20 == 0 and ci > 0:
                if state['E'] > global_best_E * 1.05:
                    state['P'] = _perturb(global_best_P, 0.3, rng)
                    state['P'] = _enforce_min_dist(state['P'], 0.5)
                    E_res, P_res = _relax_one(state['P'], potential, parameter)
                    state['P'] = P_res
                    state['E'] = E_res
            
            # Adaptively adjust kick scale
            if E_new < state['E'] or (E_new - state['E']) < 0:
                state['scale'] = max(0.03, state['scale'] * 0.9)
            else:
                state['scale'] = min(0.5, state['scale'] * 1.1)
    
    return global_best_E, global_best_P


def construct(instance):
    """Multi-chain parallel basin hopping with gradient refinement."""
    _init_timer()
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", 6))

    rng = np.random.default_rng(size * 1000 + int(parameter))

    # Phase 1: Generate diverse initial candidates
    candidates = []
    candidates.append(_fcc_points(size, 1.0))
    candidates.append(_hcp_points(size, 1.0))
    candidates.append(_bcc_points(size, 1.0))
    candidates.append(_sc_points(size, 1.0))

    if size <= 60:
        candidates.append(_icosahedral_shells(size))

    for seed in [size, size + 7, size + 13, size + 23]:
        candidates.append(_random_compact(size, 1.0, seed))

    # Perturbed lattice variants
    for scale in [0.1, 0.2]:
        candidates.append(_fcc_points(size, 1.0) + rng.normal(0, scale, (size, 3)))
    candidates.append(_hcp_points(size, 1.0) + rng.normal(0, 0.15, (size, 3)))

    # Relax all candidates
    best_energy = np.inf
    best_positions = None
    relaxed_candidates = []

    for i, cand in enumerate(candidates):
        if _time_left() < 10.0:
            break
        cand = _enforce_min_dist(cand, 0.5)
        energy, positions = _relax_one(cand, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
        relaxed_candidates.append((energy, positions))

    # Phase 2: Parallel basin hopping with multiple chains
    if best_positions is not None:
        # Create chains from best candidates plus perturbations of global best
        chains = []
        # Use top relaxed candidates as chain starts
        relaxed_candidates.sort(key=lambda x: x[0])
        for E_c, P_c in relaxed_candidates[:5]:
            chains.append(P_c)
        # Add perturbed versions of global best
        for scale in [0.2, 0.35, 0.5]:
            chains.append(_perturb(best_positions, scale, rng))
        
        # Run parallel basin hopping
        time_for_bh = min(_time_left() - 15.0, 180.0)
        if time_for_bh > 20.0:
            bh_E, bh_P = _parallel_basin_hopping(chains, potential, parameter, len(chains), time_for_bh)
            if bh_E < best_energy:
                best_energy = bh_E
                best_positions = bh_P

    # Phase 3: Gradient-based refinement of the best result
    if best_positions is not None and _time_left() > 5.0:
        grad_E, grad_P = _gradient_refine(best_positions, potential, parameter, 
                                           n_steps=8, lr=0.015, rng=rng)
        if grad_E < best_energy:
            best_energy = grad_E
            best_positions = grad_P

    # Phase 4: Final small-kick polish
    if best_positions is not None and _time_left() > 3.0:
        for _ in range(4):
            if _time_left() < 2.0:
                break
            perturbed = _perturb(best_positions, 0.02, rng)
            perturbed = _enforce_min_dist(perturbed, 0.5)
            energy, positions = _relax_one(perturbed, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()

    if best_positions is None:
        best_positions = _enforce_min_dist(candidates[0], 0.5)

    return best_positions
# EVOLVE-BLOCK-END
