# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _energy_morse(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r)) (exp(rho(1-r))-2)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    r = dists[iu]
    e = np.exp(rho * (1.0 - r))
    return np.sum(e * (e - 2.0))


def _min_distance(pos):
    """Compute minimum interatomic distance."""
    if pos.shape[0] < 2:
        return np.inf
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)


def _gradient_morse(pos, rho):
    """Compute gradient of Morse energy."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    e = np.exp(rho * (1.0 - dists))
    dE_dr = rho * e * (2.0 * e - 2.0)
    mask = ~np.eye(n, dtype=bool)
    inv_r = np.where(mask, 1.0 / dists, 0.0)
    factor = -dE_dr * inv_r
    factor = factor * mask
    grad = np.einsum('ij,ijk->ik', factor, diff)
    return grad


def _bfgs_local(pos, rho, max_iter=200, tol=1e-10):
    """Local optimization using L-BFGS-B."""
    from scipy.optimize import minimize
    
    def fun(x):
        p = x.reshape(-1, 3)
        return _energy_morse(p, rho)
    
    def jac(x):
        p = x.reshape(-1, 3)
        g = _gradient_morse(p, rho)
        return g.ravel()
    
    x0 = pos.ravel()
    res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    return res.fun, res.x.reshape(-1, 3)


def _make_fcc(n):
    """Generate FCC lattice points, select n closest to center."""
    a = 2 ** 0.5
    max_coord = a * (n ** (1/3) + 3)
    L = int(2 * max_coord / a) + 2
    pts = []
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = a * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= max_coord:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) >= n:
        centers = np.linalg.norm(pts, axis=1)
        idx = np.argsort(centers)[:n]
        return pts[idx].copy()
    return None


def _make_bcc(n):
    """Generate BCC lattice points, select n closest to center."""
    b = 2.0 / np.sqrt(3)
    max_coord = b * (n ** (1/3) + 3)
    L = int(2 * max_coord / b) + 2
    pts = []
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = b * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= max_coord:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) >= n:
        centers = np.linalg.norm(pts, axis=1)
        idx = np.argsort(centers)[:n]
        return pts[idx].copy()
    return None


def _make_hcp(n):
    """Generate HCP lattice points, select n closest to center."""
    spacing = 1.0
    layer_spacing = spacing * np.sqrt(2/3)
    max_layers = int(n ** (1/3) * 2) + 4
    max_coord = layer_spacing * max_layers + 2
    pts = []
    for layer in range(-max_layers, max_layers + 1):
        z = layer * layer_spacing
        offset = 0.5 if layer % 2 != 0 else 0.0
        for i in range(-max_layers, max_layers + 1):
            for j in range(-max_layers, max_layers + 1):
                x = spacing * (i + 0.5 * (j % 2)) + offset * spacing
                y = spacing * np.sqrt(3) / 2 * j
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= max_coord:
                    pts.append(p)
    pts = np.array(pts)
    if len(pts) >= n:
        centers = np.linalg.norm(pts, axis=1)
        idx = np.argsort(centers)[:n]
        return pts[idx].copy()
    return None


def _make_sc(n):
    """Generate simple cubic lattice points, select n closest to center."""
    spacing = 1.0
    max_coord = spacing * (n ** (1/3) + 3)
    L = int(2 * max_coord / spacing) + 2
    pts = []
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                p = spacing * np.array([i, j, k])
                if np.linalg.norm(p) <= max_coord:
                    pts.append(p)
    pts = np.array(pts)
    if len(pts) >= n:
        centers = np.linalg.norm(pts, axis=1)
        idx = np.argsort(centers)[:n]
        return pts[idx].copy()
    return None


def _make_random_compact(n, rng, radius_factor=0.8):
    """Generate random compact configuration with minimum distance constraint."""
    radius = n ** (1/3) * 0.8 * radius_factor
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 300000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(pts) == 0 or all(np.linalg.norm(p - q) >= 0.51 for q in pts):
                pts.append(p)
        attempts += 1
    if len(pts) >= n:
        return np.array(pts[:n])
    return None


def _make_shells(n, rng):
    """Generate shell-like configuration using Fibonacci spheres."""
    shell_pts = [np.zeros(3)]
    shell_r = 1.0
    while len(shell_pts) < n:
        num_on_shell = min(n - len(shell_pts), max(2, int(12 * shell_r**2)))
        golden = (1 + np.sqrt(5)) / 2
        for m in range(num_on_shell):
            z = 1 - 2 * (m + 0.5) / num_on_shell
            r_z = np.sqrt(max(0, 1 - z*z))
            theta = golden * m * 2 * np.pi
            p = shell_r * np.array([r_z * np.cos(theta), r_z * np.sin(theta), z])
            shell_pts.append(p)
        shell_r += 1.8
    if len(shell_pts) >= n:
        return np.array(shell_pts[:n])
    return None


def _make_initial_guesses(n, rng):
    """Generate multiple initial configurations."""
    guesses = []
    
    # Lattice-based guesses
    for gen_func in [_make_fcc, _make_bcc, _make_hcp, _make_sc]:
        g = gen_func(n)
        if g is not None:
            guesses.append(g)
    
    # Shell-based
    g = _make_shells(n, rng)
    if g is not None:
        guesses.append(g)
    
    # Random compact with varying densities
    for rf in [0.6, 0.8, 1.0, 1.2, 1.4]:
        g = _make_random_compact(n, rng, rf)
        if g is not None:
            guesses.append(g)
    
    # Multiple random starts
    for _ in range(5):
        g = _make_random_compact(n, rng, 1.0)
        if g is not None:
            guesses.append(g)
    
    return guesses


def _relax_and_check(pos, rho, potential='morse'):
    """Relax a configuration and check minimum distance constraint."""
    try:
        energy, positions = relax(pos, potential, rho)
        min_d = _min_distance(positions)
        if min_d >= 0.5 - 1e-6:
            return energy, positions
    except Exception:
        pass
    return None, None


def _sa_search(best_pos, best_energy, n, rho, rng, time_budget, potential='morse'):
    """Simulated annealing search to escape local minima."""
    current_pos = best_pos.copy()
    current_energy = best_energy
    best_energy_local = best_energy
    best_pos_local = best_pos.copy()
    
    T = 0.3
    T_min = 0.005
    cooling = 0.97
    
    start = time.time()
    while time.time() - start < time_budget and T > T_min:
        # Perturb a random subset
        n_perturb = max(1, n // 5)
        idx = rng.choice(n, n_perturb, replace=False)
        noise_scale = 0.15 * (1 + T)
        perturbed = current_pos.copy()
        perturbed[idx] += rng.normal(0, noise_scale, (n_perturb, 3))
        
        energy, positions = _relax_and_check(perturbed, rho, potential)
        if energy is None:
            T *= cooling
            continue
        
        delta_e = energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pos = positions
            current_energy = energy
            if current_energy < best_energy_local:
                best_energy_local = current_energy
                best_pos_local = positions.copy()
        
        T *= cooling
    
    return best_energy_local, best_pos_local


def _kick_search(best_pos, best_energy, n, rho, rng, time_budget, potential='morse'):
    """Kick-out search: move one atom far away and relax."""
    best_e = best_energy
    best_p = best_pos.copy()
    start = time.time()
    
    while time.time() - start < time_budget:
        idx = rng.integers(0, n)
        perturbed = best_p.copy()
        perturbed[idx] = best_p[idx] + rng.normal(0, 1.5, 3)
        
        energy, positions = _relax_and_check(perturbed, rho, potential)
        if energy is not None and energy < best_e:
            best_e = energy
            best_p = positions.copy()
    
    return best_e, best_p


def _perturb_search(best_pos, best_energy, n, rho, rng, time_budget, potential='morse'):
    """Perturbation-based search with varying noise scales."""
    best_e = best_energy
    best_p = best_pos.copy()
    start = time.time()
    trial = 0
    
    while time.time() - start < time_budget:
        # Cycle through different noise scales
        noise_scale = 0.1 + 0.05 * (trial % 20)
        perturbed = best_p + rng.normal(0, noise_scale, best_p.shape)
        
        energy, positions = _relax_and_check(perturbed, rho, potential)
        if energy is not None and energy < best_e:
            best_e = energy
            best_p = positions.copy()
        
        trial += 1
    
    return best_e, best_p


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    start_time = time.time()
    deadline = 325.0
    safety_margin = 5.0
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size * 7919 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate initial guesses
    guesses = _make_initial_guesses(size, rng)
    
    # Relax each guess and keep the best
    for i, guess in enumerate(guesses):
        if time.time() - start_time > deadline - safety_margin:
            break
        
        # Use the provided relax function
        energy, positions = _relax_and_check(guess, rho, potential)
        if energy is not None:
            if energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        
        # Also try our own BFGS for potentially better local min
        if time.time() - start_time > deadline - safety_margin:
            break
        try:
            e2, p2 = _bfgs_local(guess, rho, max_iter=300)
            min_d = _min_distance(p2)
            if min_d >= 0.5 - 1e-6:
                if e2 < best_energy:
                    best_energy = e2
                    best_pos = p2.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: return a valid configuration
        best_pos = np.zeros((size, 3))
        for i in range(size):
            best_pos[i, 0] = i * 1.0
        best_pos -= best_pos.mean(axis=0)
        return best_pos
    
    # Now do global search improvements if we have time
    remaining = deadline - safety_margin - (time.time() - start_time)
    
    if remaining > 15:
        # Split remaining time among different search strategies
        sa_time = min(remaining * 0.4, 60)
        kick_time = min(remaining * 0.2, 30)
        perturb_time = min(remaining * 0.3, 45)
        
        # Simulated annealing
        if sa_time > 5:
            e_sa, p_sa = _sa_search(best_pos, best_energy, size, rho, rng, sa_time, potential)
            if e_sa < best_energy:
                best_energy = e_sa
                best_pos = p_sa
        
        # Kick search
        if kick_time > 5:
            e_kick, p_kick = _kick_search(best_pos, best_energy, size, rho, rng, kick_time, potential)
            if e_kick < best_energy:
                best_energy = e_kick
                best_pos = p_kick
        
        # Perturbation search
        if perturb_time > 5:
            e_pert, p_pert = _perturb_search(best_pos, best_energy, size, rho, rng, perturb_time, potential)
            if e_pert < best_energy:
                best_energy = e_pert
                best_pos = p_pert
        
        # If still have time, do another round of SA from the new best
        remaining2 = deadline - safety_margin - (time.time() - start_time)
        if remaining2 > 10:
            e_sa2, p_sa2 = _sa_search(best_pos, best_energy, size, rho, rng, remaining2 * 0.8, potential)
            if e_sa2 < best_energy:
                best_energy = e_sa2
                best_pos = p_sa2
    
    return best_pos
# EVOLVE-BLOCK-END
