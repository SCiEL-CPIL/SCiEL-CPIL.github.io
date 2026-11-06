"""PSO + basin-hopping search for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_DEADLINE = time.time() + 280.0  # safety margin before 300s kill
_PER_INSTANCE_BUDGET = 45.0  # ~6 instances * 45s = 270s


def _fcc_points(n):
    """Generate n FCC lattice points (NN=1) sorted by radius from origin."""
    a = np.sqrt(2.0)
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=float) * a
    pts, pts_set = [], set()
    R = 0
    while len(pts) < n and R < 100:
        for ix in range(-R, R + 1):
            for iy in range(-R, R + 1):
                for iz in range(-R, R + 1):
                    for b in basis:
                        p = np.array([ix, iy, iz]) * a + b
                        key = tuple(np.round(p, 6))
                        if key not in pts_set:
                            pts_set.add(key)
                            pts.append(p)
        R += 1
    pts = np.array(pts)
    norms = np.linalg.norm(pts, axis=1)
    order = np.argsort(norms)
    return pts[order[:n]]


def _simple_cubic_points(n):
    """Generate n simple cubic lattice points (NN=1) sorted by radius."""
    pts, pts_set = [], set()
    R = 0
    while len(pts) < n and R < 100:
        for ix in range(-R, R + 1):
            for iy in range(-R, R + 1):
                for iz in range(-R, R + 1):
                    key = (ix, iy, iz)
                    if key not in pts_set:
                        pts_set.add(key)
                        pts.append(np.array([ix, iy, iz], dtype=float))
        R += 1
    pts = np.array(pts)
    norms = np.linalg.norm(pts, axis=1)
    order = np.argsort(norms)
    return pts[order[:n]]


def _bcc_points(n):
    """Generate n BCC lattice points (NN=1)."""
    a = 2.0 / np.sqrt(3.0)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=float) * a
    pts, pts_set = [], set()
    R = 0
    while len(pts) < n and R < 100:
        for ix in range(-R, R + 1):
            for iy in range(-R, R + 1):
                for iz in range(-R, R + 1):
                    for b in basis:
                        p = np.array([ix, iy, iz]) * a + b
                        key = tuple(np.round(p, 6))
                        if key not in pts_set:
                            pts_set.add(key)
                            pts.append(p)
        R += 1
    pts = np.array(pts)
    norms = np.linalg.norm(pts, axis=1)
    order = np.argsort(norms)
    return pts[order[:n]]


def _hcp_points(n):
    """Generate n HCP lattice points (NN=1) sorted by radius."""
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    basis = np.array([
        [0, 0, 0],
        [a, 0, 0],
        [a/2, a*np.sqrt(3)/2, 0],
        [a/2, a*np.sqrt(3)/6, c/2],
        [a, a*np.sqrt(3)/3, c/2],
        [a/2 + a, a*np.sqrt(3)/6, c/2],
    ], dtype=float)
    pts, pts_set = [], set()
    R = 0
    while len(pts) < n and R < 100:
        for ix in range(-R, R + 1):
            for iy in range(-R, R + 1):
                for iz in range(-R, R + 1):
                    for b in basis:
                        p = np.array([ix * a, iy * a * np.sqrt(3), iz * c]) + b
                        key = tuple(np.round(p, 6))
                        if key not in pts_set:
                            pts_set.add(key)
                            pts.append(p)
        R += 1
    pts = np.array(pts)
    norms = np.linalg.norm(pts, axis=1)
    order = np.argsort(norms)
    return pts[order[:n]]


def _random_compact(n, seed):
    """Random compact cluster with minimum distance ~0.55."""
    rng = np.random.default_rng(seed)
    pts = []
    radius = n ** (1.0 / 3.0) * 0.9 + 2.0
    attempts = 0
    while len(pts) < n and attempts < 100000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        ok = True
        for q in pts:
            if np.linalg.norm(p - q) < 0.55:
                ok = False
                break
        if ok:
            pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _enforce_min_dist(pts, min_d=0.501):
    """Vectorized enforcement of minimum pairwise distance."""
    n = len(pts)
    for _ in range(50):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = (dists < min_d) & (np.arange(n)[:, None] < np.arange(n)[None, :])
        if not np.any(mask):
            break
        for i, j in zip(*np.where(mask)):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(3)
                dist = np.linalg.norm(d)
            d = d / dist * (min_d - dist) * 0.5
            pts[i] += d
            pts[j] -= d
    return pts


def _morse_energy(pos, rho):
    """Compute Morse energy."""
    n = len(pos)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            x = rho * (1.0 - r)
            total += np.exp(x) * (np.exp(x) - 2.0)
    return total


def _basin_hopping(pts, potential, rho, rng, max_iter=50, kick_scale=0.3):
    """Basin-hopping: perturb + relax iteratively."""
    best_e, best_p = np.inf, pts.copy()
    cur_e, cur_p = np.inf, pts.copy()
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        try:
            e, p = relax(cur_p, potential, rho)
            cur_e, cur_p = e, p
        except Exception:
            break
        
        if cur_e < best_e:
            best_e, best_p = cur_e, cur_p.copy()
        
        kick = rng.normal(0, kick_scale, (len(cur_p), 3))
        new_p = cur_p + kick
        new_p = _enforce_min_dist(new_p)
        kick_scale *= 0.95
        cur_p = new_p
    
    return best_e, best_p


def _pso_search(seed_pts, potential, rho, rng, n_particles=8, max_iter=30,
                c1=1.5, c2=1.5, w_init=0.9, w_min=0.4, step_max=0.5):
    """
    Gradient-guided PSO: particles update position using velocity driven by
    analytical Morse gradient, with cognitive/social terms. After each PSO
    iteration, the best particle is relaxed to a local minimum.
    """
    n = len(seed_pts)
    dim = n * 3
    # Initialize particles around seed
    particles = np.zeros((n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    for i in range(n_particles):
        particles[i] = seed_pts.flatten() + rng.normal(0, 0.2, dim)
    
    # Evaluate initial energies
    p_energies = np.full(n_particles, np.inf)
    for i in range(n_particles):
        try:
            p_energies[i], _ = relax(particles[i].reshape(n, 3), potential, rho)
        except Exception:
            p_energies[i] = _morse_energy(particles[i].reshape(n, 3), rho)
    
    p_best = particles.copy()
    p_best_e = p_energies.copy()
    g_best_idx = np.argmin(p_best_e)
    g_best = p_best[g_best_idx].copy()
    g_best_e = p_best_e[g_best_idx]
    
    w = w_init
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        w = w_init + (w_min - w_init) * (it / max(max_iter - 1, 1))
        
        for i in range(n_particles):
            if time.time() > _DEADLINE:
                break
            
            # Get gradient for this particle
            try:
                pos_i = p_best[i].reshape(n, 3)
                grad = morse_energy_gradient(pos_i, rho)
                grad_flat = grad.flatten()
            except Exception:
                # Fallback: finite difference
                eps = 1e-4
                grad_flat = np.zeros(dim)
                for d in range(dim):
                    plus = p_best[i].copy()
                    minus = p_best[i].copy()
                    plus[d] += eps
                    minus[d] -= eps
                    try:
                        ep = _morse_energy(plus.reshape(n, 3), rho)
                        em = _morse_energy(minus.reshape(n, 3), rho)
                    except Exception:
                        ep = em = 0.0
                    grad_flat[d] = (ep - em) / (2 * eps)
            
            # Gradient-based velocity update (negative gradient direction)
            grad_step = -grad_flat * 0.01  # small gradient step
            
            # PSO velocity update
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            cognitive = c1 * r1 * (p_best[i] - particles[i])
            social = c2 * r2 * (g_best - particles[i])
            velocities[i] = w * velocities[i] + cognitive + social + grad_step
            
            # Clamp velocity
            v_norm = np.linalg.norm(velocities[i])
            if v_norm > step_max:
                velocities[i] = velocities[i] * (step_max / v_norm)
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Enforce min distance
            pos_i = particles[i].reshape(n, 3)
            pos_i = _enforce_min_dist(pos_i)
            particles[i] = pos_i.flatten()
            
            # Evaluate new position
            try:
                e_new, _ = relax(particles[i].reshape(n, 3), potential, rho)
            except Exception:
                e_new = _morse_energy(particles[i].reshape(n, 3), rho)
            
            # Update personal best
            if e_new < p_best_e[i]:
                p_best_e[i] = e_new
                p_best[i] = particles[i].copy()
            
            # Update global best
            if e_new < g_best_e:
                g_best_e = e_new
                g_best = particles[i].copy()
                g_best_idx = i
        
        # Every 5 iterations, do a full relax of global best
        if (it + 1) % 5 == 0:
            try:
                e_relaxed, p_relaxed = relax(g_best.reshape(n, 3), potential, rho)
                if e_relaxed < g_best_e:
                    g_best_e = e_relaxed
                    g_best = p_relaxed.flatten()
            except Exception:
                pass
    
    return g_best_e, g_best.reshape(n, 3)


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    t_start = time.time()
    instance_deadline = t_start + _PER_INSTANCE_BUDGET
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial seeds
    seeds = []
    try:
        seeds.append(_simple_cubic_points(size))
    except:
        pass
    try:
        seeds.append(_fcc_points(size))
    except:
        pass
    try:
        seeds.append(_bcc_points(size))
    except:
        pass
    try:
        seeds.append(_hcp_points(size))
    except:
        pass
    for s in [size, size + 42, size + 123, size + 777]:
        try:
            seeds.append(_random_compact(size, s))
        except:
            pass
    
    if not seeds:
        # Fallback: random
        rng = np.random.default_rng(42)
        seeds.append(rng.uniform(-5, 5, (size, 3)))
    
    # Phase 1: PSO from each seed (gradient-guided, collective search)
    for idx, seed_pts in enumerate(seeds):
        if time.time() > min(_DEADLINE, instance_deadline):
            break
        
        rng = np.random.default_rng(idx * 1000 + size)
        
        # PSO with adaptive settings based on cluster size
        if size <= 50:
            np_particles = 10
            max_iter = 25
        else:
            np_particles = 6
            max_iter = 20
        
        e, p = _pso_search(seed_pts, potential, parameter, rng,
                           n_particles=np_particles, max_iter=max_iter)
        
        if e < best_energy:
            best_energy = e
            best_positions = p
    
    # Phase 2: Basin-hopping refinement from best PSO result and top seeds
    if best_positions is not None:
        if time.time() < min(_DEADLINE, instance_deadline - 5.0):
            rng = np.random.default_rng(99999 + size)
            e, p = _basin_hopping(best_positions, potential, parameter, rng,
                                   max_iter=30, kick_scale=0.2)
            if e < best_energy:
                best_energy = e
                best_positions = p
    
    # Phase 3: Additional basin-hopping from diverse seeds if time permits
    for idx, seed_pts in enumerate(seeds[:4]):
        if time.time() > min(_DEADLINE, instance_deadline - 3.0):
            break
        
        rng = np.random.default_rng(50000 + idx * 1000 + size)
        for kick_init in [0.4, 0.2]:
            if time.time() > min(_DEADLINE, instance_deadline - 3.0):
                break
            e, p = _basin_hopping(seed_pts, potential, parameter, rng,
                                   max_iter=20, kick_scale=kick_init)
            if e < best_energy:
                best_energy = e
                best_positions = p
    
    # If no valid result, use first seed
    if best_positions is None:
        best_positions = seeds[0].copy()
    
    # Final min-distance enforcement
    best_positions = _enforce_min_dist(best_positions)
    
    return best_positions
# EVOLVE-BLOCK-END
