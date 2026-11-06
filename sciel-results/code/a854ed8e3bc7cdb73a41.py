"""Icosahedral symmetry-based variational optimization for Dzugutov clusters."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Icosahedral vertex directions (normalized)
_ICO_VERTICES = np.array([
    [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
    [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    [1, 0, -phi], [-1, 0, -phi], [1, 0, phi], [-1, 0, phi],
    [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0],
    [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
    [0, phi, 1], [0, -phi, 1], [0, phi, -1], [0, -phi, -1],
    [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
], dtype=float)
_phi = (1 + np.sqrt(5)) / 2
_ICO_VERTICES = np.array([
    [0, 1, _phi], [0, -1, _phi], [0, 1, -_phi], [0, -1, -_phi],
    [1, _phi, 0], [-1, _phi, 0], [1, -_phi, 0], [-1, -_phi, 0],
    [_phi, 0, 1], [-_phi, 0, 1], [_phi, 0, -1], [-_phi, 0, -1],
    [1, 0, -_phi], [-1, 0, -_phi], [1, 0, _phi], [-1, 0, _phi],
    [_phi, 1, 0], [-_phi, 1, 0], [_phi, -1, 0], [-_phi, -1, 0],
    [1, _phi, 0], [-1, _phi, 0], [1, -_phi, 0], [-1, -_phi, 0],
    [0, _phi, 1], [0, -_phi, 1], [0, _phi, -1], [0, -_phi, -1],
    [_phi, 0, 1], [-_phi, 0, 1], [_phi, 0, -1], [-_phi, 0, -1],
], dtype=float)
# Remove duplicates and normalize
_unique = []
for v in _ICO_VERTICES:
    v = v / np.linalg.norm(v)
    if not any(np.allclose(v, u, atol=1e-10) for u in _unique):
        _unique.append(v)
_ICO_VERTICES = np.array(_unique)

# Dzugutov potential parameters
_A, _B, _a, _b, _c, _d, _m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16


def _dzugutov_pair_energy(r):
    """Compute Dzugutov pair potential at distance r."""
    if r < _a:
        return _A * (r**(-_m) - _B) * np.exp(_c / (r - _a))
    elif r < _b:
        return _B * np.exp(_d / (r - _b))
    else:
        return 0.0


def _dzugutov_pair_energy_vec(r):
    """Vectorized Dzugutov pair potential."""
    r = np.asarray(r, dtype=float)
    e = np.zeros_like(r)
    mask_a = r < _a
    mask_b = (r >= _a) & (r < _b)
    e[mask_a] = _A * (r[mask_a]**(-_m) - _B) * np.exp(_c / (r[mask_a] - _a))
    e[mask_b] = _B * np.exp(_d / (r[mask_b] - _b))
    return e


def _total_energy(positions):
    """Compute total Dzugutov energy for a set of positions."""
    n = len(positions)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            total += _dzugutov_pair_energy(r)
    return total


def _build_icosahedral_cluster(n_atoms, shell_radii, twist_angles=None):
    """
    Build a cluster with icosahedral symmetry.
    
    n_atoms: total number of atoms
    shell_radii: list of radii for each shell (center + up to 3 shells)
    twist_angles: optional list of twist angles for each shell (radians)
    
    Returns positions array of shape (n_atoms, 3)
    """
    positions = []
    remaining = n_atoms
    
    # Central atom
    if remaining > 0:
        positions.append(np.array([0.0, 0.0, 0.0]))
        remaining -= 1
    
    # Shell 1: 12 atoms at icosahedral vertices
    if remaining >= 12:
        r1 = shell_radii[0] if len(shell_radii) > 0 else 1.0
        tw1 = twist_angles[0] if twist_angles and len(twist_angles) > 0 else 0.0
        # Apply twist around z-axis
        ct, st = np.cos(tw1), np.sin(tw1)
        for v in _ICO_VERTICES[:12]:
            v_rot = np.array([v[0]*ct - v[1]*st, v[0]*st + v[1]*ct, v[2]])
            positions.append(r1 * v_rot)
        remaining -= 12
    
    # Shell 2: 30 atoms (icosahedral face centers / edge midpoints)
    if remaining >= 30:
        r2 = shell_radii[1] if len(shell_radii) > 1 else 1.5
        tw2 = twist_angles[1] if twist_angles and len(twist_angles) > 1 else 0.0
        # Generate 30 positions: 60 half-edges of icosahedron, take 30
        # These are the midpoints of edges, normalized
        ico = _ICO_VERTICES[:12]
        # Find all pairs that are edges (distance ~ 2*sin(pi/5) * |v|)
        edge_midpoints = []
        seen = set()
        for i in range(12):
            for j in range(i+1, 12):
                mid = (ico[i] + ico[j]) / 2.0
                norm = np.linalg.norm(mid)
                if norm > 0.1:
                    mid_norm = mid / norm
                    key = tuple(np.round(mid_norm, 4))
                    if key not in seen:
                        seen.add(key)
                        edge_midpoints.append(mid_norm)
        # We should have 30 unique edge midpoints
        if len(edge_midpoints) < 30:
            # Fallback: use face centers
            # 20 faces of icosahedron
            face_centers = []
            # Approximate: use all 30 edge midpoints we found
            pass
        ct, st = np.cos(tw2), np.sin(tw2)
        for v in edge_midpoints[:30]:
            v_rot = np.array([v[0]*ct - v[1]*st, v[0]*st + v[1]*ct, v[2]])
            positions.append(r2 * v_rot)
        remaining -= 30
    
    # Shell 3: up to 30 more atoms
    if remaining > 0 and remaining <= 30:
        r3 = shell_radii[2] if len(shell_radii) > 2 else 2.0
        tw3 = twist_angles[2] if twist_angles and len(twist_angles) > 2 else 0.0
        # Use the remaining edge midpoints or face centers
        # For simplicity, place them at a slightly larger radius using icosahedral directions
        ct, st = np.cos(tw3), np.sin(tw3)
        # Use a subset of icosahedral vertices scaled to r3
        for i in range(remaining):
            v = _ICO_VERTICES[i % len(_ICO_VERTICES)]
            v_rot = np.array([v[0]*ct - v[1]*st, v[0]*st + v[1]*ct, v[2]])
            positions.append(r3 * v_rot)
        remaining = 0
    
    # If we still have atoms, place them at the next shell
    if remaining > 0:
        r_next = shell_radii[-1] * 1.3 if shell_radii else 2.5
        rng = np.random.default_rng(42)
        for _ in range(remaining):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v)
            positions.append(r_next * v)
    
    return np.array(positions)


def _objective(params, n_atoms):
    """
    Objective function for optimization.
    params: [r1, r2, r3, tw1, tw2, tw3]
    """
    shell_radii = params[:3]
    twist_angles = params[3:6]
    # Ensure radii are positive
    shell_radii = np.maximum(shell_radii, 0.3)
    positions = _build_icosahedral_cluster(n_atoms, shell_radii, twist_angles)
    return _total_energy(positions)


def _build_cluster_for_n(n):
    """Build initial icosahedral cluster for given n."""
    if n <= 1:
        return np.zeros((n, 3))
    
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    
    # Shell 1: 12 atoms
    if remaining >= 12:
        r1 = 1.0
        for v in _ICO_VERTICES[:12]:
            positions.append(r1 * v)
        remaining -= 12
    
    # Shell 2: 30 atoms
    if remaining >= 30:
        r2 = 1.5
        ico = _ICO_VERTICES[:12]
        edge_midpoints = []
        seen = set()
        for i in range(12):
            for j in range(i+1, 12):
                mid = (ico[i] + ico[j]) / 2.0
                norm = np.linalg.norm(mid)
                if norm > 0.1:
                    mid_norm = mid / norm
                    key = tuple(np.round(mid_norm, 4))
                    if key not in seen:
                        seen.add(key)
                        edge_midpoints.append(mid_norm)
        for v in edge_midpoints[:30]:
            positions.append(r2 * v)
        remaining -= 30
    
    # Remaining atoms
    if remaining > 0:
        r3 = 2.0
        for i in range(remaining):
            v = _ICO_VERTICES[i % len(_ICO_VERTICES)]
            positions.append(r3 * v)
    
    return np.array(positions)


def construct(instance):
    """Construct low-energy Dzugutov cluster using icosahedral symmetry optimization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Build initial icosahedral cluster
    positions = _build_cluster_for_n(size)
    
    # First, do a local relaxation to get a good starting point
    energy, positions_relaxed = relax(positions, potential, parameter)
    
    # Now optimize shell parameters
    # Extract approximate shell radii from relaxed positions
    dists = np.linalg.norm(positions_relaxed, axis=1)
    
    # Cluster distances into shells
    shell_radii = [1.0, 1.5, 2.0]
    twist_angles = [0.0, 0.0, 0.0]
    
    # Optimize using L-BFGS-B
    x0 = np.array([1.0, 1.5, 2.0, 0.0, 0.0, 0.0])
    
    def obj_wrapper(params):
        r = np.maximum(params[:3], 0.3)
        tw = params[3:6]
        pos = _build_icosahedral_cluster(size, list(r), list(tw))
        return _total_energy(pos)
    
    try:
        result = minimize(obj_wrapper, x0, method='L-BFGS-B', 
                          bounds=[(0.3, 5.0), (0.3, 5.0), (0.3, 5.0), 
                                  (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)],
                          options={'maxiter': 200, 'ftol': 1e-10})
        if result.success:
            optimal_params = result.x
            pos_opt = _build_icosahedral_cluster(size, list(np.maximum(optimal_params[:3], 0.3)), 
                                                  list(optimal_params[3:6]))
            # Relax the optimized structure
            e_opt, pos_opt_relaxed = relax(pos_opt, potential, parameter)
            if e_opt < energy:
                positions_relaxed = pos_opt_relaxed
                energy = e_opt
    except Exception:
        pass
    
    # Also try a few random perturbations of the icosahedral structure
    best_energy = energy
    best_positions = positions_relaxed
    
    rng = np.random.default_rng(size)
    for _ in range(3):
        # Perturb shell radii
        r_pert = [1.0 + rng.normal(0, 0.1), 1.5 + rng.normal(0, 0.1), 2.0 + rng.normal(0, 0.1)]
        tw_pert = [rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)]
        pos_pert = _build_icosahedral_cluster(size, r_pert, tw_pert)
        e_pert, pos_pert_relaxed = relax(pos_pert, potential, parameter)
        if e_pert < best_energy:
            best_energy = e_pert
            best_positions = pos_pert_relaxed
    
    return best_positions
# EVOLVE-BLOCK-END
