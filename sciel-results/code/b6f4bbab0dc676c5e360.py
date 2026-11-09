"""Sutton-Chen transition-metal cluster global minima: multi-start from diverse
structured seeds (fcc, hcp, icosahedral, Mackay), followed by basin hopping
refinement with Metropolis acceptance on the best candidates.
Uses a strict wall-clock deadline."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


# ---- hard wall-clock deadline (evaluator kills at 330 s; keep a safety margin) ----
_DEADLINE_SECONDS = 318.0
_START_TIME = time.time()


def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


# ---------------------------------------------------------------------------
# Seed generators: produce compact, valid (min-pair-distance) (n,3) arrays.
# ---------------------------------------------------------------------------

def _fcc_seed(n, rng, scale=1.0):
    """Fill the fcc lattice with the n sites closest to the origin."""
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
                    p = (cell + b)
                    cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _hcp_seed(n, rng, scale=1.0):
    """Fill an hcp-like lattice (two interpenetrating triangular layers)."""
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
    """Start from an icosahedron and fill with fcc points."""
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


def _bcp_seed(n, rng, scale=1.0):
    """Bicapped icosahedron / Mackay-like seed for transition metals."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]  # center
    
    # First shell: 12 icosahedron vertices
    for v in ico_verts:
        pts.append(v * 1.0)
    
    # Second shell: 30 points (edges of icosahedron midpoints)
    if n > 13:
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 1.0:  # edge length ~1
                    edges.append((i, j))
        for i, j in edges:
            if len(pts) >= n:
                break
            mid = (ico_verts[i] + ico_verts[j]) / 2.0
            pts.append(mid * 1.62)  # slightly larger radius
    
    # Third shell: 20 points (face centers)
    if n > 43:
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                    d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                    d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                    if d1 < 1.0 and d2 < 1.0 and d3 < 1.0:
                        faces.append((i, j, k))
        for i, j, k in faces:
            if len(pts) >= n:
                break
            centroid = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
            centroid = centroid / np.linalg.norm(centroid)
            pts.append(centroid * 2.1)
    
    # Fill remaining with fcc
    if len(pts) < n:
        fcc = _fcc_seed(n, rng, scale=1.0)
        for p in fcc:
            if len(pts) >= n:
                break
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.4:
                    ok = False
                    break
            if ok:
                pts.append(p)
    
    pts = np.array(pts[:n], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _random_compact_seed(n, rng, scale=1.0, min_dist=0.55):
    """Random points in a ball with a minimum-pair-distance constraint."""
    radius = scale * n ** (1.0 / 3.0) * 0.85
    pts = []
    tries = 0
    max_tries = 30000
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


def _softened_energy_gradient(positions, n_exp, m_exp, c):
    """Compute energy and gradient for a softened Sutton-Chen potential."""
    n = len(positions)
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)

    inv_r_n = dist ** (-n_exp)
    inv_r_m = dist ** (-m_exp)
    np.fill_diagonal(inv_r_n, 0.0)
    np.fill_diagonal(inv_r_m, 0.0)

    pair_rep = 0.5 * np.sum(inv_r_n)
    S_i = np.sum(inv_r_m, axis=1)
    pair_att = -c * np.sum(np.sqrt(np.maximum(S_i, 1e-12)))
    energy = pair_rep + pair_att

    inv_r_np2 = dist ** (-(n_exp + 2))
    np.fill_diagonal(inv_r_np2, 0.0)
    row_sum_n = np.sum(inv_r_np2, axis=1)
    grad_rep = 0.5 * n_exp * (inv_r_np2 @ positions - row_sum_n[:, None] * positions)

    inv_r_mp2 = dist ** (-(m_exp + 2))
    np.fill_diagonal(inv_r_mp2, 0.0)
    row_sum_m = np.sum(inv_r_mp2, axis=1)
    sqrt_S = np.sqrt(np.maximum(S_i, 1e-12))
    sum_term = inv_r_mp2 @ positions - row_sum_m[:, None] * positions
    grad_att = -c * m_exp / (2.0 * sqrt_S) * sum_term

    grad = grad_rep + grad_att
    return energy, grad


def _relax_soft(positions, n_exp, m_exp, c, maxiter=300):
    """Relax using a softened potential via L-BFGS-B."""
    from scipy.optimize import minimize
    try:
        def fun_and_grad(x):
            pos = x.reshape(-1, 3)
            e, g = _softened_energy_gradient(pos, n_exp, m_exp, c)
            return e, g.reshape(-1)
        res = minimize(fun_and_grad, positions.reshape(-1), jac=True,
                       method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-7})
        if np.all(np.isfinite(res.x)):
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    return positions


def _polish(positions, kind, maxiter=5000):
    """High-precision polish with the true potential."""
    from scipy.optimize import minimize
    try:
        def fun_and_grad(x):
            pos = x.reshape(-1, 3)
            e, g = sutton_chen_energy_gradient(pos, kind)
            return e, g.reshape(-1)
        res = minimize(fun_and_grad, positions.reshape(-1), jac=True,
                       method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        if np.all(np.isfinite(res.x)):
            pos = res.x.reshape(-1, 3)
            if _is_valid(pos):
                e, _ = sutton_chen_energy_gradient(pos, kind)
                return float(e), pos
    except Exception:
        pass
    # Fallback to library relax
    try:
        e, pos = relax(positions, 'sutton_chen', kind)
        if np.all(np.isfinite(pos)) and _is_valid(pos):
            return float(e), pos
    except Exception:
        pass
    return np.inf, None


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


def _try_relax(seed, potential, kind, maxiter=3000):
    """Relax a seed; return (energy, positions) or (inf, None) on failure."""
    try:
        energy, positions = relax(seed, potential, kind)
    except Exception:
        return np.inf, None
    if not np.all(np.isfinite(positions)):
        return np.inf, None
    if not _is_valid(positions):
        return np.inf, None
    return float(energy), positions


def construct(instance):
    global _START_TIME
    _START_TIME = time.time()

    n = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    parameter = kind

    base_scale = 1.0
    rng = np.random.default_rng(12345 + n)

    # Build a diverse list of seeds
    seeds = []

    # 1. Structured seeds (Mackay icosahedral, fcc, hcp, icosahedral)
    for gen in (_bcp_seed, _fcc_seed, _hcp_seed, _icosahedral_seed):
        s = gen(n, rng, scale=base_scale)
        if _is_valid(s, 0.4):
            seeds.append(s)
        # Jittered variants to break symmetry
        for amt in (0.08, 0.15, 0.25):
            sj = _jitter(s, rng, amt)
            if _is_valid(sj, 0.36):
                seeds.append(sj)

    # 2. Random compact seeds
    for _ in range(6):
        s = _random_compact_seed(n, rng, scale=base_scale, min_dist=0.55)
        if _is_valid(s, 0.36):
            seeds.append(s)

    best_energy = np.inf
    best_positions = None
    local_minima = []  # Store (energy, positions) for basin hopping

    # Phase 1: Relax seeds one at a time under the deadline
    # Spend ~50% of budget on initial relaxations
    phase1_deadline = 0.5 * _DEADLINE_SECONDS
    for seed in seeds:
        if time.time() - _START_TIME > phase1_deadline:
            break
        if _remaining() < 15.0:
            break
        e, pos = _try_relax(seed, potential, parameter)
        if e < best_energy:
            best_energy = e
            best_positions = pos
        if pos is not None:
            local_minima.append((e, pos))

    # Fallback if nothing valid
    if best_positions is None:
        s = _fcc_seed(n, rng, scale=base_scale)
        if not _is_valid(s, 0.35):
            s = _random_compact_seed(n, rng, scale=base_scale, min_dist=0.36)
        e, pos = _try_relax(s, potential, parameter)
        if pos is not None:
            best_energy = e
            best_positions = pos
            local_minima.append((e, pos))
        else:
            side = int(np.ceil(n ** (1.0 / 3.0)))
            grid = []
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        grid.append([i - side // 2, j - side // 2, k - side // 2])
            pts = np.array(grid[:n], dtype=float) * 0.75
            best_positions = pts

    # Phase 2: Coarse-to-Fine Annealing Basin Hopping
    # Use softened potential for relaxation during coarse phase to allow
    # large structural rearrangements, then ramp to true potential.
    
    # Sort local minima by energy
    local_minima.sort(key=lambda x: x[0])
    
    # Take the best 4 unique minima (or fewer if not available)
    candidates = []
    for e, pos in local_minima:
        if e < 1e-9:
            continue
        candidates.append((e, pos))
        if len(candidates) >= 4:
            break
            
    if not candidates and best_positions is not None:
        candidates.append((best_energy, best_positions))

    # Determine target parameters from kind
    kind_params = {
        '12-6': (12, 6, 144.41),
        '9-6': (9, 6, 39.432),
        '10-8': (10, 8, 34.408),
    }
    target_n_exp, target_m_exp, target_c = kind_params.get(parameter, (12, 6, 144.41))

    # Soft potential parameters for coarse phase
    soft_n_exp = 3.0
    soft_m_exp = 3.0
    soft_c = target_c * 0.3

    for start_e, start_pos in candidates:
        if _remaining() < 20.0:
            break
        
        current_e = start_e
        current_pos = start_pos.copy()
        
        # Coarse-to-fine annealing schedule
        # Each stage: (n_exp_frac, m_exp_frac, c_frac, amplitude, T, num_cycles)
        # n_exp_frac interpolates from soft to target
        stages = [
            # (n_interp, m_interp, c_interp, amp, T, cycles)
            (0.0, 0.0, 0.0, 0.40, 0.5, 8),   # Very soft, large moves
            (0.15, 0.15, 0.15, 0.35, 0.4, 8),
            (0.3, 0.3, 0.3, 0.30, 0.35, 6),
            (0.5, 0.5, 0.5, 0.25, 0.3, 6),
            (0.7, 0.7, 0.7, 0.20, 0.25, 5),
            (0.85, 0.85, 0.85, 0.15, 0.2, 4),
            (1.0, 1.0, 1.0, 0.12, 0.15, 4),  # True potential
            (1.0, 1.0, 1.0, 0.08, 0.1, 3),
            (1.0, 1.0, 1.0, 0.04, 0.05, 3),
            (1.0, 1.0, 1.0, 0.02, 0.02, 2),
        ]
        
        for n_frac, m_frac, c_frac, amp, T, num_cycles in stages:
            if _remaining() < 5.0:
                break
            
            # Interpolate potential parameters
            cur_n = soft_n_exp + (target_n_exp - soft_n_exp) * n_frac
            cur_m = soft_m_exp + (target_m_exp - soft_m_exp) * m_frac
            cur_c = soft_c + (target_c - soft_c) * c_frac
            
            for _ in range(num_cycles):
                if _remaining() < 5.0:
                    break
                
                # Perturb
                perturbed = _jitter(current_pos, rng, amp)
                
                # Relax: use softened potential for coarse stages, true for fine
                if n_frac < 1.0:
                    pos_new = _relax_soft(perturbed, cur_n, cur_m, cur_c, maxiter=200)
                    # Evaluate true energy for acceptance
                    try:
                        e_new, _ = sutton_chen_energy_gradient(pos_new, parameter)
                    except Exception:
                        e_new = np.inf
                    if not np.all(np.isfinite(pos_new)) or not _is_valid(pos_new):
                        continue
                else:
                    e_new, pos_new = _try_relax(perturbed, potential, parameter)
                    if pos_new is None:
                        continue
                
                # Metropolis acceptance based on true energy
                if e_new < current_e:
                    current_e = e_new
                    current_pos = pos_new
                else:
                    delta_e = e_new - current_e
                    if delta_e < 10.0:
                        accept_prob = np.exp(-delta_e / T)
                        if rng.random() < accept_prob:
                            current_e = e_new
                            current_pos = pos_new
                
                # Update global best
                if current_e < best_energy:
                    best_energy = current_e
                    best_positions = current_pos.copy()

    # Phase 3: High-precision polish on the best candidate
    if _remaining() > 15.0 and best_positions is not None:
        e_polished, pos_polished = _polish(best_positions, parameter, maxiter=5000)
        if pos_polished is not None and e_polished < best_energy:
            best_energy = e_polished
            best_positions = pos_polished

    return best_positions
# EVOLVE-BLOCK-END
