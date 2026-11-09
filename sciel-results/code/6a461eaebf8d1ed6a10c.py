"""Sutton-Chen transition-metal cluster global minima: coarse-to-fine annealing with dynamic potential softening."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _get_params(kind):
    """Return (n_exp, m_exp, c) for the given kind."""
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41


def _sc_energy_gradient(pos, n_exp, m_exp, c):
    """Compute Sutton-Chen energy and gradient for given parameters.
    E = sum_i [1/2 sum_{j!=i} r_ij^-n_exp - c * sqrt(sum_{j!=i} r_ij^-m_exp)]
    """
    n = pos.shape[0]
    E = 0.0
    grad = np.zeros_like(pos)
    
    for i in range(n):
        pair_sum_rep = 0.0
        pair_sum_att = 0.0
        grad_i = np.zeros(3)
        
        for j in range(n):
            if i == j:
                continue
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                continue
            r = np.sqrt(r2)
            r_inv = 1.0 / r
            r_inv_n = r_inv ** n_exp
            r_inv_m = r_inv ** m_exp
            pair_sum_rep += r_inv_n
            pair_sum_att += r_inv_m
        
        E += 0.5 * pair_sum_rep - c * np.sqrt(pair_sum_att)
        
        # Gradient computation
        sqrt_att = np.sqrt(max(pair_sum_att, 1e-15))
        for j in range(n):
            if i == j:
                continue
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                continue
            r = np.sqrt(r2)
            r_inv = 1.0 / r
            
            # dE/dr for pair (i,j) contribution to atom i
            # E_i = 0.5 * sum_j r_ij^-n - c * sqrt(sum_j r_ij^-m)
            # dE_i/dr_ij = -0.5 * n * r_ij^-(n+1) - c * (1/(2*sqrt(S_m))) * (-m * r_ij^-(m+1))
            #            = -0.5 * n * r^-(n+1) + c * m * r^-(m+1) / (2 * sqrt(S_m))
            dE_dr = -0.5 * n_exp * r ** (-(n_exp + 1)) + c * m_exp * r ** (-(m_exp + 1)) / (2.0 * sqrt_att)
            
            # grad_i += dE/dr * (pos_i - pos_j)/r
            grad_i += dE_dr * rij / r
        
        grad[i] = grad_i
    
    return E, grad


def _soft_relax(pts, kind, n_scale, m_scale, c_scale, maxiter=300):
    """Relax using a softened version of the potential."""
    n_exp_target, m_exp_target, c_target = _get_params(kind)
    n_exp = max(2, int(round(n_exp_target * n_scale)))
    m_exp = max(2, int(round(m_exp_target * m_scale)))
    c = c_target * c_scale
    
    def objective(x):
        pos = x.reshape(-1, 3)
        E, grad = _sc_energy_gradient(pos, n_exp, m_exp, c)
        return E, grad.flatten()
    
    x0 = pts.flatten()
    try:
        res = minimize(objective, x0, jac=True, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'gtol': 1e-6, 'maxls': 20})
        if res.success or np.isfinite(res.fun):
            return res.fun, res.x.reshape(-1, 3)
    except Exception:
        pass
    return np.inf, pts.copy()


def _ensure_min_dist(pts, dmin=0.355):
    """Push apart any pair closer than dmin so the result is always valid."""
    pts = np.asarray(pts, dtype=float).copy()
    n = pts.shape[0]
    if n < 2:
        return pts
    for _ in range(300):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                r = np.linalg.norm(d)
                if r < dmin:
                    if r < 1e-9:
                        pts[j] += np.array([dmin, 0.0, 0.0])
                    else:
                        push = (dmin - r) * 0.5
                        pts[i] += push * d / r
                        pts[j] -= push * d / r
                    moved = True
        if not moved:
            break
    return pts


def _validate_positions(pos):
    """Check if positions are finite and no atoms are closer than 0.35."""
    if not np.all(np.isfinite(pos)):
        return False
    if pos.shape[0] < 2:
        return True
    # Vectorized check
    n = pos.shape[0]
    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                cubic = np.array([i, j, k], dtype=float)
                for b in basis:
                    p = (cubic + b) * spacing
                    sites.append(p)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster by filling hcp lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    sites = []
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                sites.append([x, y, z])
                sites.append([x + a * 0.5, y + a * np.sqrt(3.0) / 6.0, z])
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral shell cluster."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, -phi, 1/phi], [0, phi, -1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [-phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    
    for v in ico_verts[:12]:
        points.append(v * spacing * 1.0)
    
    if n <= 13:
        selected = np.array(points[:n])
        return selected - selected.mean(axis=0)
    
    num_second = min(42, n - 13)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for k in range(num_second):
        z = 1.0 - 2.0 * (k + 0.5) / num_second
        r = np.sqrt(1.0 - z * z)
        theta = golden_angle * k
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * spacing * 1.8)
    
    remaining = n - len(points)
    if remaining > 0:
        num_third = remaining
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(num_third):
            z = 1.0 - 2.0 * (k + 0.5) / num_third
            r = np.sqrt(1.0 - z * z)
            theta = golden_angle * (k + 0.3)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * spacing * 2.6)
    
    selected = np.array(points[:n])
    return selected - selected.mean(axis=0)


def _random_compact(n, spacing, seed):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.7 * spacing
    pts = []
    pts_arr = np.empty((0, 3))
    attempts = 0
    max_attempts = min(15000, n * 300)
    while len(pts) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(pts_arr) == 0:
                pts.append(p)
                pts_arr = np.array([p])
            else:
                dists = np.linalg.norm(pts_arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    pts.append(p)
                    pts_arr = np.vstack([pts_arr, p])
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _anneal_phase(pts, kind, t_start, deadline, phase_duration, n_scale_start, m_scale_start, c_scale_start,
                  n_scale_end, m_scale_end, c_scale_end, maxiter_per_step=100, n_steps=10):
    """Perform annealing from soft to hard potential.
    
    Parameters:
    - pts: initial positions
    - kind: potential kind
    - t_start, deadline: time bounds
    - phase_duration: total time for this phase
    - n_scale_start, m_scale_start, c_scale_start: initial (soft) scales
    - n_scale_end, m_scale_end, c_scale_end: final (hard) scales
    - maxiter_per_step: max L-BFGS-B iterations per annealing step
    - n_steps: number of annealing steps
    
    Returns: (energy, positions)
    """
    n_exp_target, m_exp_target, c_target = _get_params(kind)
    pos = pts.copy()
    best_energy = np.inf
    best_pos = pos.copy()
    
    time_per_step = phase_duration / n_steps
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Linear interpolation of scales
        frac = step / max(n_steps - 1, 1)
        n_scale = n_scale_start + (n_scale_end - n_scale_start) * frac
        m_scale = m_scale_start + (m_scale_end - m_scale_start) * frac
        c_scale = c_scale_start + (c_scale_end - c_scale_start) * frac
        
        # Relax with current softened potential
        e, p_new = _soft_relax(pos, kind, n_scale, m_scale, c_scale, maxiter=maxiter_per_step)
        
        if np.isfinite(e) and _validate_positions(p_new):
            # Evaluate with the true potential to track progress
            e_true, _ = sutton_chen_energy_gradient(p_new, kind)
            if np.isfinite(e_true) and e_true < best_energy:
                best_energy = e_true
                best_pos = p_new.copy()
            pos = p_new
        else:
            # If relaxation failed, keep current position
            pass
    
    return best_energy, best_pos


def _basin_hopping_refine(best_pos, best_energy, potential, kind, rng, deadline, n_iter=30, max_step=0.05):
    """Fine basin hopping with small steps to polish the solution."""
    pos = best_pos.copy()
    energy = best_energy
    for it in range(n_iter):
        if time.time() > deadline - 8.0:
            break
        step = max_step * (1.0 - 0.3 * it / max(n_iter, 1))
        noise = rng.normal(0, step, size=pos.shape)
        new_pos = pos + noise
        
        if not _validate_positions(new_pos):
            continue
        
        try:
            e_new, p_new = relax(new_pos, potential, kind)
        except Exception:
            continue
        
        if np.isfinite(e_new) and e_new < energy and _validate_positions(p_new):
            energy = e_new
            pos = p_new
    return energy, pos


def construct(instance):
    """Coarse-to-fine annealing with dynamic potential softening for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    t_start = time.time()
    deadline = t_start + 280.0  # 280s out of 330s, leave 50s safety margin
    
    # Generate diverse initial geometries
    spacing = 0.72
    geometries = []
    
    # 1. FCC-like
    try:
        fcc = _fcc_cluster(size, spacing)
        if fcc.shape[0] == size:
            geometries.append(fcc)
    except Exception:
        pass
    
    # 2. HCP-like
    try:
        hcp = _hcp_cluster(size, spacing)
        if hcp.shape[0] == size:
            geometries.append(hcp)
    except Exception:
        pass
    
    # 3. Icosahedral
    try:
        ico = _icosahedral_cluster(size, spacing)
        if ico.shape[0] == size:
            geometries.append(ico)
    except Exception:
        pass
    
    # 4. Random compact with a few seeds
    for s in [size, size + 101, size + 202]:
        try:
            rc = _random_compact(size, spacing, s)
            if rc.shape[0] == size:
                geometries.append(rc)
        except Exception:
            pass
    
    # Ensure we have at least one geometry
    if not geometries:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * 0.5
        geometries.append(positions)
    
    # Phase 1: Annealing from soft to hard potential for each geometry
    best_energy = np.inf
    best_positions = None
    
    # Allocate time: 60% for annealing, 30% for basin hopping, 10% for final polish
    anneal_time = 160.0
    bh_time = 80.0
    polish_time = 20.0
    
    anneal_deadline = t_start + anneal_time
    bh_deadline = t_start + anneal_time + bh_time
    
    for geom in geometries:
        if time.time() > anneal_deadline - 10.0:
            break
        
        # Run annealing phase: start very soft, end at full hardness
        e, p = _anneal_phase(
            geom, kind, t_start, anneal_deadline,
            phase_duration=anneal_deadline - t_start,
            n_scale_start=0.3, m_scale_start=0.3, c_scale_start=0.2,
            n_scale_end=1.0, m_scale_end=1.0, c_scale_end=1.0,
            maxiter_per_step=80, n_steps=8
        )
        
        if np.isfinite(e) and _validate_positions(p) and e < best_energy:
            best_energy = e
            best_positions = p
    
    if best_positions is None:
        best_positions = geometries[0]
        best_energy, _ = sutton_chen_energy_gradient(best_positions, kind)
    
    # Phase 2: Basin hopping refinement
    if time.time() < bh_deadline - 5.0:
        rng = np.random.default_rng(42)
        best_energy, best_positions = _basin_hopping_refine(
            best_positions, best_energy, potential, kind, rng, bh_deadline,
            n_iter=25, max_step=0.06
        )
    
    # Final validity guarantee
    best_positions = _ensure_min_dist(best_positions)
    
    if not np.all(np.isfinite(best_positions)):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.5
    
    return best_positions
# EVOLVE-BLOCK-END
