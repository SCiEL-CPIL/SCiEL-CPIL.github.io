# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _energy_morse(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r)) (exp(rho(1-r))-2)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    e = np.exp(rho * (1.0 - dists))
    pair_energy = e * (e - 2.0)
    mask = ~np.eye(n, dtype=bool)
    total_energy = 0.5 * np.sum(pair_energy * mask)
    return total_energy


def _gradient_morse(pos, rho):
    """Compute gradient of Morse energy using full matrix approach."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e-15)
    dists = np.sqrt(dists_sq)
    e = np.exp(rho * (1.0 - dists))
    dE_dr = rho * e * (2.0 * e - 2.0)
    mask = ~np.eye(n, dtype=bool)
    inv_r = np.where(mask, 1.0 / dists, 0.0)
    factor = -dE_dr * inv_r
    factor = factor * mask
    grad = np.einsum('ij,ijk->ik', factor, diff)
    return grad


def _bfgs_local(pos, rho, time_limit=3.0):
    """Local optimization using L-BFGS-B with time guard."""
    n = pos.shape[0]
    start_time = time.time()
    stop_flag = [False]

    def fun(x):
        p = x.reshape(n, 3)
        return _energy_morse(p, rho)

    def jac(x):
        p = x.reshape(n, 3)
        return _gradient_morse(p, rho).ravel()

    def callback(xk):
        if time.time() - start_time > time_limit:
            stop_flag[0] = True
        return stop_flag[0]

    x0 = pos.ravel()
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30},
                       callback=callback)
        return res.fun, res.x.reshape(n, 3)
    except Exception:
        return np.inf, pos


def _min_distance(pos):
    """Compute minimum interatomic distance."""
    if pos.shape[0] < 2:
        return np.inf
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    return np.sqrt(np.min(dists_sq))


def _check_valid(pos, min_dist=0.5):
    """Check if configuration is valid (min distance constraint)."""
    return _min_distance(pos) >= min_dist - 1e-6


def _generate_fcc(n):
    """Generate FCC lattice cluster."""
    a = np.sqrt(2.0)
    basis = np.array([[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]) * a
    L = int((n / 4) ** (1/3)) + 4
    points = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                for b in basis:
                    p = a * np.array([i, j, k]) + b
                    points.append(p)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_bcc(n):
    """Generate BCC lattice cluster."""
    a = 2.0 / np.sqrt(3.0)
    basis = np.array([[0,0,0],[0.5,0.5,0.5]]) * a
    L = int((n / 2) ** (1/3)) + 4
    points = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                for b in basis:
                    p = a * np.array([i, j, k]) + b
                    points.append(p)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_hcp(n):
    """Generate HCP-like lattice cluster."""
    layer_spacing = np.sqrt(2.0 / 3.0)
    max_layers = int(n ** (1/3) * 2) + 4
    points = []
    for layer in range(-max_layers, max_layers + 1):
        z = layer * layer_spacing
        for i in range(-max_layers, max_layers + 1):
            for j in range(-max_layers, max_layers + 1):
                x = i + 0.5 * (j % 2)
                y = np.sqrt(3.0) / 2.0 * j
                p = np.array([x, y, z])
                points.append(p)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_shells(n):
    """Generate Fibonacci-sphere shell configuration."""
    shell_pts = [np.zeros(3)]
    shell_r = 1.0
    golden = (1 + np.sqrt(5)) / 2
    while len(shell_pts) < n:
        num_on_shell = min(n - len(shell_pts), max(2, int(12 * shell_r**2)))
        for m in range(num_on_shell):
            z = 1 - 2 * (m + 0.5) / num_on_shell
            r_z = np.sqrt(max(0, 1 - z*z))
            theta = golden * m * 2 * np.pi
            p = shell_r * np.array([r_z * np.cos(theta), r_z * np.sin(theta), z])
            shell_pts.append(p)
        shell_r += 1.8
        if len(shell_pts) >= n:
            break
    return np.array(shell_pts[:n])


def _generate_random_valid(n, rng, min_dist=0.51):
    """Generate random valid configuration."""
    r_max = 1.8 * n ** (1/3)
    pts = []
    attempts = 0
    max_attempts = 50000
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-r_max, r_max, 3)
        if np.linalg.norm(p) <= r_max:
            if len(pts) == 0:
                pts.append(p)
            else:
                arr = np.array(pts)
                if np.min(np.linalg.norm(arr - p, axis=1)) >= min_dist:
                    pts.append(p)
        attempts += 1
    if len(pts) == n:
        return np.array(pts)
    # Fallback: grid
    side = int(n ** (1/3)) + 2
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(pts) < n:
                    pts.append([(i - side//2) * 1.0, (j - side//2) * 1.0, (k - side//2) * 1.0])
    return np.array(pts[:n])


def _simulate_annealing(pos, rho, steps=4000, t_start=1.5, t_end=0.001, time_limit=8.0, rng=None):
    """Simulated Annealing with constraint handling."""
    if rng is None:
        rng = np.random.default_rng()
    n = pos.shape[0]
    current_pos = pos.copy()
    current_energy = _energy_morse(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy

    start_time = time.time()

    for i in range(steps):
        if time.time() - start_time > time_limit:
            break
        t = t_start * (t_end / t_start) ** (i / max(steps, 1))
        step_size = 0.04 + 0.25 * (t / t_start)
        displacement = rng.normal(0, step_size, (n, 3))
        new_pos = current_pos + displacement

        # Quick min-distance check
        if _min_distance(new_pos) < 0.5:
            continue

        new_energy = _energy_morse(new_pos, rho)
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(t, 1e-10)):
            current_pos = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()

    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    start_time = time.time()
    deadline = 325.0  # safety margin from 330s kill

    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))

    rng = np.random.default_rng(42 + n * 7)

    best_energy = np.inf
    best_pos = None

    def update_best(e, p):
        nonlocal best_energy, best_pos
        if _check_valid(p) and e < best_energy:
            best_energy = e
            best_pos = p.copy()

    # Generate diverse initial guesses
    guesses = []
    guesses.append(_generate_fcc(n))
    guesses.append(_generate_bcc(n))
    guesses.append(_generate_hcp(n))
    guesses.append(_generate_shells(n))
    for _ in range(4):
        guesses.append(_generate_random_valid(n, rng))

    # Phase 1: Local optimization from all guesses
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    phase1_budget = 0.30 * remaining
    per_guess_time = min(2.5, phase1_budget / max(len(guesses), 1))

    for guess in guesses:
        if time.time() - start_time > deadline:
            break
        time_left = deadline - (time.time() - start_time)
        if time_left < 1.0:
            break
        tl = min(per_guess_time, time_left * 0.9)
        e, p = _bfgs_local(guess, rho, time_limit=tl)
        update_best(e, p)

    if best_pos is None:
        best_pos = guesses[0]
        best_energy = _energy_morse(best_pos, rho)

    # Phase 2: Simulated Annealing from multiple starts
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    phase2_budget = 0.50 * remaining

    sa_starts = [best_pos.copy()]
    for scale in [0.2, 0.4, 0.7, 1.0]:
        perturbed = best_pos + rng.normal(0, scale, (n, 3))
        sa_starts.append(perturbed)
    for g in guesses[:3]:
        sa_starts.append(g.copy())

    per_sa_time = phase2_budget / max(len(sa_starts), 1)

    for start_pos in sa_starts:
        if time.time() - start_time > deadline:
            break
        time_left = deadline - (time.time() - start_time)
        if time_left < 3.0:
            break

        tl_sa = min(per_sa_time * 0.65, time_left * 0.55)
        tl_loc = min(per_sa_time * 0.35, time_left * 0.30)

        e_sa, p_sa = _simulate_annealing(start_pos, rho, steps=3000, time_limit=tl_sa, rng=rng)
        update_best(e_sa, p_sa)

        if time.time() - start_time < deadline - 1.0:
            e_loc, p_loc = _bfgs_local(p_sa, rho, time_limit=tl_loc)
            update_best(e_loc, p_loc)

    # Phase 3: Additional SA restarts from best if time remains
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5.0 and best_pos is not None:
        phase3_budget = remaining * 0.6
        num_extra = max(1, int(phase3_budget / 5.0))
        for _ in range(num_extra):
            if time.time() - start_time > deadline - 3.0:
                break
            time_left = deadline - (time.time() - start_time)
            if time_left < 4.0:
                break
            scale = rng.uniform(0.3, 1.2)
            perturbed = best_pos + rng.normal(0, scale, (n, 3))
            tl_sa = min(3.0, time_left * 0.6)
            e_sa, p_sa = _simulate_annealing(perturbed, rho, steps=2000, time_limit=tl_sa, rng=rng)
            update_best(e_sa, p_sa)
            if time.time() - start_time < deadline - 2.0:
                e_loc, p_loc = _bfgs_local(p_sa, rho, time_limit=min(2.0, time_left * 0.3))
                update_best(e_loc, p_loc)

    # Phase 4: Final polish
    if best_pos is not None and time.time() - start_time < deadline - 2.0:
        try:
            e_relax, p_relax = relax(best_pos, 'morse', rho)
            update_best(e_relax, p_relax)
        except Exception:
            pass

        if time.time() - start_time < deadline - 1.0:
            e_final, p_final = _bfgs_local(best_pos, rho, time_limit=min(2.0, deadline - time.time() - 0.5))
            update_best(e_final, p_final)

    # Fallback
    if best_pos is None or not _check_valid(best_pos):
        best_pos = _generate_random_valid(n, rng)
        if not _check_valid(best_pos):
            best_pos = np.zeros((n, 3))
            for i in range(n):
                best_pos[i, 0] = i * 1.0
            best_pos -= best_pos.mean(axis=0)

    return best_pos

# EVOLVE-BLOCK-END
