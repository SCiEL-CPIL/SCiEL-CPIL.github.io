"""Sutton-Chen transition-metal cluster global minima: coarse-to-fine annealed
basin-hopping with multi-start from diverse structured seeds."""
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize


# ---- hard wall-clock deadline (evaluator kills at 330 s; keep a safety margin) ----
_DEADLINE_SECONDS = 312.0
_START_TIME = time.time()


def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


# ---------------------------------------------------------------------------
# Seed generators
# ---------------------------------------------------------------------------

def _fcc_seed(n, rng, scale=1.0):
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0]])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 5
    cand = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                cell = np.array([i, j, k])
                for b in basis:
                    p = cell + b
                    cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _hcp_seed(n, rng, scale=1.0):
    a = 1.0
    h = np.sqrt(2.0 / 3.0) * a
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 5
    cand = []
    for layer in range(-max_shell, max_shell + 1):
        z = layer * h
        is_b = (layer % 2 != 0)
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * j)
                y = a * (np.sqrt(3.0) / 2.0) * j
                if is_b:
                    x += a * 0.5
                    y += a * np.sqrt(3.0) / 6.0
                p = np.array([x, y, z])
                cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _icosahedral_seed(n, rng, scale=1.0):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    pts_list = [p * 1.0 for p in ico]
    fcc = _fcc_seed(n, rng, scale=1.0)
    all_pts = list(pts_list)
    for p in fcc:
        if len(all_pts) >= n:
            break
        ok = True
        for q in all_pts:
            if np.linalg.norm(p - q) < 0.45:
                ok = False
                break
        if ok:
            all_pts.append(p)
    if len(all_pts) < n:
        for p in fcc:
            if len(all_pts) >= n:
                break
            all_pts.append(p)
    pts = np.array(all_pts[:n], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _random_compact_seed(n, rng, scale=1.0, min_dist=0.55):
    radius = scale * n ** (1.0 / 3.0) * 0.85
    pts = []
    tries = 0
    max_tries = 20000
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    if len(pts) < n:
        md = min_dist
        while len(pts) < n:
            md *= 0.95
            if md < 0.36:
                while len(pts) < n:
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) <= radius:
                        pts.append(p)
                break
            tries = 0
            while len(pts) < n and tries < max_tries:
                tries += 1
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                if all(np.linalg.norm(p - q) >= md for q in pts):
                    pts.append(p)
    pts = np.array(pts, dtype=float)
    pts = pts - pts.mean(axis=0)
    return pts


def _jitter(seed, rng, amount):
    return seed + rng.normal(0.0, amount, seed.shape)


def _is_valid(pts, min_dist=0.35):
    if not np.all(np.isfinite(pts)):
        return False
    n = len(pts)
    if n < 2:
        return True
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    return dist.min() >= min_dist


# ---------------------------------------------------------------------------
# Softened energy for coarse-to-fine annealing
# ---------------------------------------------------------------------------

def _softened_energy_and_grad(positions, n_exp, m_exp, c, n_exp_soft, m_exp_soft, c_soft):
    """Compute energy and gradient with softened exponents and coupling."""
    n = len(positions)
    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, 1.0)  # avoid division by zero; diagonal won't matter

    # Pairwise terms
    r_inv_n = np.power(r, -n_exp_soft)
    r_inv_m = np.power(r, -m_exp_soft)
    np.fill_diagonal(r_inv_n, 0.0)
    np.fill_diagonal(r_inv_m, 0.0)

    # Energy: sum_i [1/2 sum_j r_ij^-n - c * sqrt(sum_j r_ij^-m)]
    pair_n = r_inv_n.sum(axis=1)  # sum over j for each i
    pair_m = r_inv_m.sum(axis=1)

    energy = 0.5 * np.sum(pair_n) - c_soft * np.sum(np.sqrt(np.maximum(pair_m, 0.0)))

    # Gradient
    # dE/dr_ij = -n_exp_soft * r_ij^(-n_exp_soft-1) - c_soft * (1/(2*sqrt(pair_m_i))) * (-m_exp_soft * r_ij^(-m_exp_soft-1))
    #           for the i-component; symmetric for j.
    # Actually let's compute via chain rule more carefully.
    # E = 0.5 * sum_i sum_{j!=i} r_ij^-n  -  c * sum_i sqrt(S_i)  where S_i = sum_{j!=i} r_ij^-m
    # dE/dx_i = sum_{j!=i} [ -n * r_ij^(-n-1) * (x_i - x_j)/r_ij  -  c * (1/(2*sqrt(S_i))) * (-m * r_ij^(-m-1)) * (x_i - x_j)/r_ij ]
    #         = sum_{j!=i} [ -(n+1) * r_ij^(-n-2) * (x_i - x_j)  +  c*m/(2*sqrt(S_i)) * r_ij^(-m-2) * (x_i - x_j) ]

    grad = np.zeros_like(positions)
    for i in range(n):
        si = pair_m[i]
        if si <= 0:
            continue
        inv_2sqrt_si = 1.0 / (2.0 * np.sqrt(si))
        for j in range(n):
            if i == j:
                continue
            rij = r[i, j]
            if rij < 1e-10:
                continue
            diff_ij = positions[i] - positions[j]
            coeff_rep = -(n_exp_soft + 1.0) * np.power(rij, -(n_exp_soft + 2.0))
            coeff_att = c_soft * m_exp_soft * inv_2sqrt_si * np.power(rij, -(m_exp_soft + 2.0))
            grad[i] += (coeff_rep + coeff_att) * diff_ij

    return energy, grad


def _softened_relax(positions, n_exp, m_exp, c, n_exp_soft, m_exp_soft, c_soft, maxiter=1000):
    """Relax using softened potential."""
    def fun_and_grad(x):
        x = x.reshape(-1, 3)
        e, g = _softened_energy_and_grad(x, n_exp, m_exp, c, n_exp_soft, m_exp_soft, c_soft)
        return e, g.flatten()

    try:
        res = minimize(fun_and_grad, positions.flatten(), method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6})
        if res.success or res.status in (0, 1, 2, 3, 4, 5, 6):
            pos = res.x.reshape(-1, 3)
            if np.all(np.isfinite(pos)) and _is_valid(pos, 0.35):
                e, _ = _softened_energy_and_grad(pos, n_exp, m_exp, c, n_exp_soft, m_exp_soft, c_soft)
                return float(e), pos
    except Exception:
        pass
    return np.inf, None


# ---------------------------------------------------------------------------
# Basin hopping with coarse-to-fine annealing
# ---------------------------------------------------------------------------

def _basin_hopping_coarse_to_fine(start_pos, kind, n_exp, m_exp, c, rng, max_stages=6, max_iter_per_stage=200):
    """
    Coarse-to-fine basin hopping:
    - Start with softened potential (lower exponents, reduced c)
    - Perform basin hopping steps with Metropolis acceptance
    - Gradually ramp exponents and c back to target values
    - In later stages, use the true potential for relaxation
    """
    n = len(start_pos)
    current_pos = start_pos.copy()

    # Define annealing schedule: 6 stages
    # Stage 0: very soft  ->  Stage 5: full potential
    n_exp_schedule = [n_exp * 0.35, n_exp * 0.5, n_exp * 0.65, n_exp * 0.8, n_exp * 0.95, n_exp]
    m_exp_schedule = [m_exp * 0.35, m_exp * 0.5, m_exp * 0.65, m_exp * 0.8, m_exp * 0.95, m_exp]
    c_schedule = [c * 0.15, c * 0.3, c * 0.5, c * 0.7, c * 0.9, c]

    best_energy = np.inf
    best_pos = None

    # Initial evaluation with full potential
    try:
        e0, _ = sutton_chen_energy_gradient(start_pos, kind)
        best_energy = float(e0)
        best_pos = start_pos.copy()
    except Exception:
        pass

    for stage in range(max_stages):
        if _remaining() < 10.0:
            break

        ne_s = n_exp_schedule[stage]
        me_s = m_exp_schedule[stage]
        c_s = c_schedule[stage]

        # Number of BH steps per stage: more in early (softer) stages
        n_steps = max(3, int(max_iter_per_stage * (1.0 - 0.3 * stage / max_stages)))
        if stage >= 3:
            n_steps = max(2, int(max_iter_per_stage * 0.5))

        temp = 0.5 * (1.0 - stage / max_stages) + 0.05  # cooling schedule

        for step in range(n_steps):
            if _remaining() < 8.0:
                break

            # Perturbation: larger in early stages
            perturb_scale = 0.3 * (1.0 - stage / max_stages) + 0.08
            perturbed = current_pos + rng.normal(0.0, perturb_scale, current_pos.shape)

            # Relax with softened potential
            e_soft, pos_soft = _softened_relax(perturbed, n_exp, m_exp, c, ne_s, me_s, c_s,
                                               maxiter=max(100, int(300 * (1.0 - stage / max_stages))))
            if pos_soft is None:
                continue

            # Evaluate with true potential
            try:
                e_true, _ = sutton_chen_energy_gradient(pos_soft, kind)
                e_true = float(e_true)
            except Exception:
                continue

            if not np.all(np.isfinite(pos_soft)) or not _is_valid(pos_soft, 0.35):
                continue

            # Metropolis acceptance based on true energy
            if e_true < best_energy:
                best_energy = e_true
                best_pos = pos_soft.copy()
                current_pos = pos_soft.copy()
            else:
                delta_e = e_true - best_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 0.01)):
                    current_pos = pos_soft.copy()

    return best_energy, best_pos


def _try_relax(seed, potential, kind, maxiter=2000):
    try:
        energy, positions = relax(seed, potential, kind)
    except Exception:
        return np.inf, None
    if not np.all(np.isfinite(positions)):
        return np.inf, None
    if not _is_valid(positions):
        return np.inf, None
    return float(energy), positions


# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()

    n = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")

    # Extract potential parameters
    if kind == "12-6":
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == "9-6":
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == "10-8":
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        n_exp, m_exp, c = 12, 6, 144.41

    base_scale = 1.0
    rng = np.random.default_rng(12345 + n)

    # Build diverse seed list
    seeds = []
    for gen in (_fcc_seed, _hcp_seed, _icosahedral_seed):
        s = gen(n, rng, scale=base_scale)
        if _is_valid(s, 0.4):
            seeds.append(s)
        for amt in (0.05, 0.12, 0.25):
            sj = _jitter(s, rng, amt)
            if _is_valid(sj, 0.36):
                seeds.append(sj)

    for _ in range(4):
        s = _random_compact_seed(n, rng, scale=base_scale, min_dist=0.6)
        if _is_valid(s, 0.36):
            seeds.append(s)

    best_energy = np.inf
    best_positions = None

    # Phase 1: Quick relax of top structured seeds to get an initial best
    for seed in seeds[:6]:
        if _remaining() < 15.0:
            break
        e, pos = _try_relax(seed, potential, kind)
        if e < best_energy:
            best_energy = e
            best_positions = pos

    # Phase 2: Coarse-to-fine basin hopping from best and a few other seeds
    bh_seeds = []
    if best_positions is not None:
        bh_seeds.append(best_positions.copy())
    for seed in seeds[6:12]:
        if _remaining() < 20.0:
            break
        e, pos = _try_relax(seed, potential, kind)
        if e < best_energy:
            best_energy = e
            best_positions = pos
        if pos is not None:
            bh_seeds.append(pos.copy())

    # Run basin hopping from each BH seed
    for bh_seed in bh_seeds:
        if _remaining() < 20.0:
            break
        e_bh, pos_bh = _basin_hopping_coarse_to_fine(
            bh_seed, kind, n_exp, m_exp, c, rng,
            max_stages=6, max_iter_per_stage=150
        )
        if pos_bh is not None and e_bh < best_energy:
            best_energy = e_bh
            best_positions = pos_bh

    # Phase 3: Final polish - high-precision local relax of best
    if best_positions is not None and _remaining() > 10.0:
        e_final, pos_final = _try_relax(best_positions, potential, kind)
        if pos_final is not None and e_final < best_energy:
            best_energy = e_final
            best_positions = pos_final

    # Phase 4: If time remains, a few more random restarts + BH
    if best_positions is not None and _remaining() > 30.0:
        for _ in range(4):
            if _remaining() < 20.0:
                break
            perturbed = _jitter(best_positions, rng, 0.2)
            e_p, pos_p = _try_relax(perturbed, potential, kind)
            if pos_p is not None:
                e_bh, pos_bh = _basin_hopping_coarse_to_fine(
                    pos_p, kind, n_exp, m_exp, c, rng,
                    max_stages=4, max_iter_per_stage=100
                )
                if pos_bh is not None and e_bh < best_energy:
                    best_energy = e_bh
                    best_positions = pos_bh

    # Fallback
    if best_positions is None:
        s = _fcc_seed(n, rng, scale=base_scale)
        if not _is_valid(s, 0.35):
            s = _random_compact_seed(n, rng, scale=base_scale, min_dist=0.36)
        e, pos = _try_relax(s, potential, kind)
        if pos is not None:
            best_positions = pos
        else:
            side = int(np.ceil(n ** (1.0 / 3.0)))
            grid = []
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        grid.append([i - side // 2, j - side // 2, k - side // 2])
            best_positions = np.array(grid[:n], dtype=float) * 0.75

    return best_positions
