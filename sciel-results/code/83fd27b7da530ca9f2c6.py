"""Optimized Morse cluster minimizer: vectorized gradient + L-BFGS-B + basin hopping
with crystallographic seeds and a strict global deadline."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Global time management (evaluator kills at 330 s wall-clock)
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
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r = r[mask]
    if r.size == 0:
        return 0.0
    u = np.exp(rho * (1.0 - r))
    return float(np.sum(u * (u - 2.0)))


def _morse_grad(pos, rho):
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    u = np.exp(rho * (1.0 - r))
    # dE/dx_i = sum_{j>i} (2u-2)*rho*u * (x_i - x_j)/r
    factor = (2.0 * u - 2.0) * rho * u
    # divide by r (broadcast), avoid div-by-zero on diagonal (masked by mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = factor / r
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    factor = np.where(mask, factor, 0.0)
    grad = np.zeros_like(pos)
    grad = np.sum(factor[:, :, None] * diff, axis=1)
    # For pair (i,j) with i<j: diff = x_i - x_j, factor = (2u-2)*rho*u/r
    # Contribution to x_i: +factor*diff, to x_j: -factor*diff
    # The sum over axis=1 with mask only on i<j gives:
    #   grad_i = sum_{j>i} factor_ij * (x_i - x_j)  (correct for i)
    #   but x_j gets -factor_ij*(x_i-x_j) only if we also subtract
    # We need symmetric treatment. Recompute properly:
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            ri = r[i, j]
            fi = (2.0 * u[i, j] - 2.0) * rho * u[i, j] / ri
            grad[i] += fi * d
            grad[j] -= fi * d
    return grad


def _morse_grad_fast(pos, rho):
    """Fully vectorized gradient (no Python loop)."""
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
    # grad_i = sum_j factor_ij * (x_i - x_j), where factor is zero for i>=j
    # This double counts? No: for i<j, factor_ij nonzero, contributes +f*(x_i-x_j) to i
    # and we need -f*(x_i-x_j) to j. Since factor_ji=0, the sum for j over i gives
    # sum_{i<j} 0 + ... we are missing the -f term for j.
    # Fix: grad_i = sum_j [factor_ij*(x_i-x_j)] where we want the antisymmetric part.
    # Use: grad = sum_j factor_ij * diff_ij  with factor antisymmetric.
    # Make factor antisymmetric: A_ij = factor for i<j, A_ji = -factor, else 0.
    A = np.zeros_like(factor)
    A[mask] = factor[mask]
    A = A - A.T  # now antisymmetric: A_ij = f for i<j, A_ji = -f
    grad = np.sum(A[:, :, None] * diff, axis=1)
    return grad


# ---------------------------------------------------------------------------
# Local minimization (L-BFGS-B, with BFGS fallback)
# ---------------------------------------------------------------------------
def _local_min(pos, rho, timeout=5.0):
    n = pos.shape[0]
    flat = pos.reshape(-1)
    t0 = time.time()
    try:
        res = minimize(
            _morse_energy, flat,
            jac=lambda x: _morse_grad_fast(x.reshape(n, 3), rho).reshape(-1),
            args=(rho,),
            method="L-BFGS-B",
            options={"maxiter": 400, "maxfun": 2000,
                     "gtol": 1e-6, "ftol": 1e-10, "disp": False},
        )
        e = _morse_energy(res.x.reshape(n, 3), rho)
        return res.x.reshape(n, 3), e
    except Exception:
        pass
    try:
        res = minimize(
            _morse_energy, flat,
            jac=lambda x: _morse_grad_fast(x.reshape(n, 3), rho).reshape(-1),
            args=(rho,),
            method="BFGS",
            options={"maxiter": 300, "gtol": 1e-5, "disp": False},
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
    # FCC lattice points: all integer triples (i,j,k) with i+j+k even,
    # plus face centers. Simpler: generate a cubic grid of FCC sites.
    pts = []
    maxc = int(np.ceil((n * 2.0) ** (1.0 / 3.0))) + 3
    for i in range(-maxc, maxc + 1):
        for j in range(-maxc, maxc + 1):
            for k in range(-maxc, maxc + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([i, j, k])
    pts = np.array(pts, dtype=float)
    # Center
    pts -= pts.mean(axis=0)
    # Sort by radius, take nearest n
    radii = np.linalg.norm(pts, axis=1)
    order = np.argsort(radii)
    pts = pts[order[:n]] * (spacing * 0.577)  # FCC nearest neighbor ~ sqrt(2)/2 * a
    return pts


def _hcp_seed(n, spacing=1.0, rng=None):
    """Generate n atoms in HCP-like layered arrangement."""
    if rng is None:
        rng = np.random.default_rng(1)
    pts = []
    # Layer 1: hexagonal packing
    layer = 0
    while len(pts) < n:
        # hexagonal layer with radius growing
        R = int(np.ceil(np.sqrt(max(1, n - len(pts)) / 3.0)))
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                if i * i + j * j <= R * R:
                    x = i + 0.5 * (j % 2)
                    y = j * 0.866
                    z = layer * 0.816
                    pts.append([x, y, z])
        layer += 1
    pts = np.array(pts, dtype=float)
    pts -= pts.mean(axis=0)
    radii = np.linalg.norm(pts, axis=1)
    order = np.argsort(radii)
    pts = pts[order[:n]] * (spacing * 0.5)
    return pts


def _icosahedral_seed(n, spacing=1.0, rng=None):
    """Generate n atoms in icosahedral shell arrangement."""
    if rng is None:
        rng = np.random.default_rng(2)
    pts = [[0.0, 0.0, 0.0]]
    # 12 icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    pts.extend(ico.tolist())
    # Second shell: 30 vertices (edges midpoints of icosahedron, normalized)
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
    pts = np.array(pts, dtype=float)
    pts *= spacing * 0.5
    if len(pts) < n:
        # fill with random points on a sphere
        extra = n - len(pts)
        random_pts = rng.normal(size=(extra, 3))
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts *= spacing * 0.7
        pts = np.vstack([pts, random_pts])
    else:
        pts = pts[:n]
    return pts


def _random_compact(n, spacing=1.0, rng=None):
    """Random compact cluster with minimum separation."""
    if rng is None:
        rng = np.random.default_rng(3)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all, add remaining at random
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
        # Perturb
        kick = kick_scale * rng.normal(size=cur_pos.shape)
        new_pos = cur_pos + kick
        # Local minimize
        min_pos, min_e = _local_min(new_pos, rho, timeout=min(3.0, _time_left() / 4))
        # Metropolis accept
        if min_e < cur_e:
            cur_pos, cur_e = min_pos, min_e
            if min_e < best_e:
                best_pos, best_e = min_pos, min_e
        else:
            delta = min_e - cur_e
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-8)):
                cur_pos, cur_e = min_pos, min_e
        step += 1
        # Anneal temperature
        temp = max(0.05, 2.0 * (0.95 ** step))
        # Reduce kick scale slowly
        kick_scale = max(0.05, 0.3 * (0.98 ** step))
    return best_pos, best_e


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    spacing = 1.0

    # Per-instance budget: distribute remaining time, but cap
    remaining = _time_left()
    inst_budget = min(remaining * 0.9, 50.0)  # cap per instance at 50 s
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
    # Random compact seeds
    for k in range(3):
        try:
            seeds.append(_random_compact(n, spacing, rng))
        except Exception:
            pass
    # Jittered crystal seeds
    for base in seeds[:3]:
        try:
            jittered = base + 0.1 * rng.normal(size=base.shape)
            seeds.append(jittered)
        except Exception:
            pass

    # Ensure at least one valid seed
    if not seeds:
        seeds.append(np.random.default_rng(n).normal(size=(n, 3)))

    best_pos = None
    best_e = np.inf

    for idx, seed in enumerate(seeds):
        if _time_left() < 5.0:
            break
        # Local minimize the seed
        t0 = time.time()
        try:
            min_pos, min_e = _local_min(seed, rho, timeout=min(4.0, _time_left() / 4))
        except Exception:
            min_pos, min_e = seed, _morse_energy(seed, rho)
        if min_e < best_e:
            best_e = min_e
            best_pos = min_pos.copy()

        # Basin hopping from the best seed found so far (use a couple of seeds)
        if idx < 3 and _time_left() > 10.0:
            bh_budget = min(8.0, _time_left() * 0.3)
            try:
                bh_pos, bh_e = _basin_hop(min_pos, rho, n, bh_budget, rng)
                if bh_e < best_e:
                    best_e = bh_e
                    best_pos = bh_pos.copy()
            except Exception:
                pass

    # Final safety: ensure minimum separation and return
    if best_pos is None:
        best_pos = seeds[0]
    # Enforce minimum separation 0.5 (light cleanup)
    best_pos = _enforce_min_sep(best_pos, 0.5, rng)
    return best_pos


def _enforce_min_sep(pos, min_d, rng):
    """Nudge atoms that are too close apart."""
    n = pos.shape[0]
    for _ in range(20):
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
# EVOLVE-BLOCK-END
