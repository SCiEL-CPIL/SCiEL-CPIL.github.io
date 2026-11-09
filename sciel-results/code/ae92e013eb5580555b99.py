"""Morse cluster global minima finder using structured initializers and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Global time management
# ---------------------------------------------------------------------------
_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 320.0  # 10 s safety margin


def _time_left():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


# ---------------------------------------------------------------------------
# Fast vectorized Morse energy and gradient
# E = sum_{i<j} exp(rho(1-r_ij)) * (exp(rho(1-r_ij)) - 2)
# ---------------------------------------------------------------------------
def _morse_energy(pos, rho):
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_pairs = r[mask]
    if r_pairs.size == 0:
        return 0.0
    u = np.exp(rho * (1.0 - r_pairs))
    return float(np.sum(u * (u - 2.0)))


def _morse_grad_fast(pos, rho):
    """Fully vectorized gradient."""
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    u = np.exp(rho * (1.0 - r))
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = (2.0 * u - 2.0) * rho * u / r
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    factor = np.where(mask, factor, 0.0)
    A = factor - factor.T  # antisymmetric
    grad = np.sum(A[:, :, None] * diff, axis=1)
    return grad


# ---------------------------------------------------------------------------
# Local minimization
# ---------------------------------------------------------------------------
def _local_min(pos, rho, timeout=5.0):
    n = pos.shape[0]
    flat = pos.reshape(-1)
    t0 = time.time()
    try:
        res = minimize(
            lambda x: _morse_energy(x.reshape(n, 3), rho),
            flat,
            jac=lambda x: _morse_grad_fast(x.reshape(n, 3), rho).reshape(-1),
            method="L-BFGS-B",
            options={"maxiter": 600, "maxfun": 3000,
                     "gtol": 1e-8, "ftol": 1e-12, "disp": False},
        )
        if time.time() - t0 > timeout:
            return pos, _morse_energy(pos, rho)
        e = _morse_energy(res.x.reshape(n, 3), rho)
        return res.x.reshape(n, 3), e
    except Exception:
        pass
    # Fallback with fewer iterations
    try:
        res = minimize(
            lambda x: _morse_energy(x.reshape(n, 3), rho),
            flat,
            jac=lambda x: _morse_grad_fast(x.reshape(n, 3), rho).reshape(-1),
            method="L-BFGS-B",
            options={"maxiter": 200, "maxfun": 1000,
                     "gtol": 1e-6, "ftol": 1e-10, "disp": False},
        )
        e = _morse_energy(res.x.reshape(n, 3), rho)
        return res.x.reshape(n, 3), e
    except Exception:
        return pos, _morse_energy(pos, rho)


# ---------------------------------------------------------------------------
# Seed generators
# ---------------------------------------------------------------------------
def _fcc_seed(n, spacing=1.0, rng=None):
    """Generate n atoms in FCC shell arrangement."""
    if rng is None:
        rng = np.random.default_rng(0)
    pts = []
    maxc = int(np.ceil((n * 2.0) ** (1.0 / 3.0))) + 3
    for i in range(-maxc, maxc + 1):
        for j in range(-maxc, maxc + 1):
            for k in range(-maxc, maxc + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([i * 0.577, j * 0.577, k * 0.577])
    pts = np.array(pts, dtype=float)
    pts -= pts.mean(axis=0)
    radii = np.linalg.norm(pts, axis=1)
    order = np.argsort(radii)
    pts = pts[order[:n]] * spacing
    return pts


def _hcp_seed(n, spacing=1.0, rng=None):
    """Generate n atoms in HCP-like layered arrangement."""
    if rng is None:
        rng = np.random.default_rng(1)
    pts = []
    layer = 0
    while len(pts) < n:
        R = int(np.ceil(np.sqrt(max(1, n - len(pts)) / 3.0)))
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                if i * i + j * j <= R * R:
                    x = (i + 0.5 * (j % 2)) * 0.5
                    y = j * 0.433
                    z = layer * 0.408
                    pts.append([x, y, z])
        layer += 1
    pts = np.array(pts, dtype=float)
    pts -= pts.mean(axis=0)
    radii = np.linalg.norm(pts, axis=1)
    order = np.argsort(radii)
    pts = pts[order[:n]] * spacing
    return pts


def _icosahedral_seed(n, spacing=1.0, rng=None):
    """Generate n atoms in icosahedral shell arrangement."""
    if rng is None:
        rng = np.random.default_rng(2)
    pts = [[0.0, 0.0, 0.0]]
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    pts.extend(ico.tolist())
    if n > 13:
        verts = ico
        edges = []
        for a in range(12):
            for b in range(a + 1, 12):
                d = np.linalg.norm(verts[a] - verts[b])
                if 0.9 < d < 1.1:
                    mid = (verts[a] + verts[b]) / 2.0
                    mid /= np.linalg.norm(mid)
                    edges.append(mid)
        pts.extend([p.tolist() for p in edges])
    if n > 43:
        # Third shell: 60 points (face centers of icosahedron, normalized)
        faces = []
        for a in range(12):
            for b in range(a + 1, 12):
                for c in range(b + 1, 12):
                    da = np.linalg.norm(verts[a] - verts[b])
                    db = np.linalg.norm(verts[b] - verts[c])
                    dc = np.linalg.norm(verts[c] - verts[a])
                    if 0.9 < da < 1.1 and 0.9 < db < 1.1 and 0.9 < dc < 1.1:
                        centroid = (verts[a] + verts[b] + verts[c]) / 3.0
                        centroid /= np.linalg.norm(centroid)
                        faces.append(centroid)
        pts.extend([p.tolist() for p in faces])
    pts = np.array(pts, dtype=float)
    pts *= spacing * 0.5
    if len(pts) < n:
        extra = n - len(pts)
        random_pts = rng.normal(size=(extra, 3))
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts *= spacing * 0.7
        pts = np.vstack([pts, random_pts])
    else:
        pts = pts[:n]
    return pts


def _spherical_shell_seed(n, spacing=1.0, rng=None):
    """Generate n atoms on concentric spherical shells."""
    if rng is None:
        rng = np.random.default_rng(4)
    pts = []
    # Place atoms on shells with radii proportional to shell index
    shell_idx = 0
    remaining = n
    while remaining > 0:
        # Number of atoms on this shell
        if shell_idx == 0:
            shell_count = 1
        else:
            shell_count = min(remaining, int(2 * shell_idx * (shell_idx + 1) + 1))
        shell_radius = 0.5 * spacing * shell_idx
        for _ in range(shell_count):
            # Fibonacci sphere for even distribution
            theta = np.arccos(1 - 2 * (len(pts) + 1) / (n + 1))
            phi_angle = np.pi * (1 + 5 ** 0.5) * (len(pts) + 1)
            x = shell_radius * np.sin(theta) * np.cos(phi_angle)
            y = shell_radius * np.sin(theta) * np.sin(phi_angle)
            z = shell_radius * np.cos(theta)
            pts.append([x, y, z])
        remaining -= shell_count
        shell_idx += 1
    pts = np.array(pts[:n], dtype=float)
    pts -= pts.mean(axis=0)
    return pts


def _random_compact(n, spacing=1.0, rng=None):
    """Random compact cluster with minimum separation."""
    if rng is None:
        rng = np.random.default_rng(3)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 50000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n], dtype=float)


# ---------------------------------------------------------------------------
# Basin hopping
# ---------------------------------------------------------------------------
def _basin_hop(pos, rho, n, budget, rng, kick_scale=0.3):
    """Basin hopping: perturb, local minimize, Metropolis accept."""
    best_pos = pos.copy()
    best_e = _morse_energy(pos, rho)
    t0 = time.time()
    cur_pos = pos.copy()
    cur_e = best_e
    temp = 2.0
    step = 0
    while (time.time() - t0) < budget and _time_left() > 5.0:
        kick = kick_scale * rng.normal(size=cur_pos.shape)
        new_pos = cur_pos + kick
        min_pos, min_e = _local_min(new_pos, rho, timeout=min(3.0, _time_left() / 4))
        if min_e < cur_e:
            cur_pos, cur_e = min_pos, min_e
            if min_e < best_e:
                best_pos, best_e = min_pos, min_e
        else:
            delta = min_e - cur_e
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-8)):
                cur_pos, cur_e = min_pos, min_e
        step += 1
        temp = max(0.05, 2.0 * (0.95 ** step))
        kick_scale = max(0.05, 0.3 * (0.98 ** step))
    return best_pos, best_e


# ---------------------------------------------------------------------------
# Enforce minimum separation
# ---------------------------------------------------------------------------
def _enforce_min_sep(pos, min_d, rng):
    """Nudge atoms that are too close apart."""
    n = pos.shape[0]
    for _ in range(30):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                r = np.linalg.norm(d)
                if r < min_d:
                    if r < 1e-8:
                        d = rng.normal(size=3)
                        r = np.linalg.norm(d)
                    push = (min_d - r) * 0.5 * (d / r)
                    pos[i] += push
                    pos[j] -= push
                    moved = True
        if not moved:
            break
    return pos


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    spacing = 1.0

    remaining = _time_left()
    inst_budget = min(remaining * 0.9, 50.0)
    inst_budget = max(inst_budget, 3.0)

    rng = np.random.default_rng(n * 1000 + int(rho))

    # Build candidate seeds
    seeds = []
    try:
        seeds.append(_fcc_seed(n, spacing, rng))
    except Exception:
        pass
    try:
        seeds.append(_hcp_seed(n, spacing, rng))
    except Exception:
        pass
    try:
        seeds.append(_icosahedral_seed(n, spacing, rng))
    except Exception:
        pass
    try:
        seeds.append(_spherical_shell_seed(n, spacing, rng))
    except Exception:
        pass
    for k in range(3):
        try:
            seeds.append(_random_compact(n, spacing, rng))
        except Exception:
            pass
    # Jittered crystal seeds
    for base in seeds[:4]:
        try:
            jittered = base + 0.1 * rng.normal(size=base.shape)
            seeds.append(jittered)
        except Exception:
            pass

    if not seeds:
        seeds.append(np.random.default_rng(n).normal(size=(n, 3)))

    best_pos = None
    best_e = np.inf

    # Phase 1: Local minimize all seeds
    for idx, seed in enumerate(seeds):
        if _time_left() < 5.0:
            break
        try:
            min_pos, min_e = _local_min(seed, rho, timeout=min(4.0, _time_left() / 4))
        except Exception:
            min_pos, min_e = seed, _morse_energy(seed, rho)
        if min_e < best_e:
            best_e = min_e
            best_pos = min_pos.copy()

    # Phase 2: Basin hopping from top seeds
    if _time_left() > 10.0:
        # Sort seeds by their local min energy, take top 4 for basin hopping
        seed_results = []
        for seed in seeds[:6]:
            if _time_left() < 5.0:
                break
            try:
                min_pos, min_e = _local_min(seed, rho, timeout=min(3.0, _time_left() / 5))
            except Exception:
                min_pos, min_e = seed, _morse_energy(seed, rho)
            seed_results.append((min_e, min_pos))
        seed_results.sort(key=lambda x: x[0])
        
        for min_e, min_pos in seed_results[:4]:
            if _time_left() < 10.0:
                break
            bh_budget = min(10.0, _time_left() * 0.3)
            try:
                bh_pos, bh_e = _basin_hop(min_pos, rho, n, bh_budget, rng)
                if bh_e < best_e:
                    best_e = bh_e
                    best_pos = bh_pos.copy()
            except Exception:
                pass

    # Final safety
    if best_pos is None:
        best_pos = seeds[0]
    best_pos = _enforce_min_sep(best_pos, 0.5, rng)
    return best_pos
# EVOLVE-BLOCK-END
