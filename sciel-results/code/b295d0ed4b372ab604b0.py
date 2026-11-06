import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s kill


def _remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _morse_energy(pos, rho):
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    r = dists[np.triu_indices(n, 1)]
    val = rho * (1.0 - r)
    ev = np.exp(val)
    return float(np.sum(ev * (ev - 2.0)))


def _morse_grad(pos, rho):
    n = len(pos)
    g = np.zeros_like(pos)
    if n < 2:
        return g
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    ev = np.exp(rho * (1.0 - dists))
    pf = -2.0 * rho * ev * (ev - 1.0)
    np.fill_diagonal(pf, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        W = pf / dists
    W = np.where(dists > 1e-10, W, 0.0)
    Ws = W.sum(axis=1)
    g = Ws[:, None] * pos - W @ pos
    return g


def _min_dist_ok(pos, min_d=0.5):
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    return np.min(d) >= min_d - 1e-9


def _fix_min_dist(pos, min_d=0.51, max_iter=300):
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        if _min_dist_ok(pos, min_d):
            break
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        ii, jj = np.unravel_index(np.argmin(d), d.shape)
        rij = pos[ii] - pos[jj]
        r = np.linalg.norm(rij)
        if r < 1e-10:
            rij = np.array([1.0, 0.0, 0.0])
            r = 1.0
        push = (min_d - r + 0.005) / 2.0
        pos[ii] += push * rij / r
        pos[jj] -= push * rij / r
    return pos


def _fibonacci_sphere(n, radius=1.0):
    """Evenly distributed points on a sphere using Fibonacci lattice."""
    if n == 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.zeros((1, 3))
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    pts = np.column_stack([
        radius * np.sin(phi) * np.cos(theta),
        radius * np.sin(phi) * np.sin(theta),
        radius * np.cos(phi)
    ])
    return pts


def _mckay_shells(n):
    """
    Generate Mackay icosahedral shell points:
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (vertices of icosahedron)
    Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
    Total: 1, 13, 55, 147, 309...
    We generate enough points and take the closest n to origin.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    pts = [np.zeros(3)]

    # Shell 1: 12 vertices
    shell1 = verts.copy()
    for v in shell1:
        pts.append(v.copy())

    # Shell 2: 30 edge midpoints + 12 face centers = 42
    # Edge midpoints
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(shell1[i] - shell1[j])
            if 0.9 < d < 1.1:
                mid = (shell1[i] + shell1[j]) / 2.0
                r = np.linalg.norm(mid)
                if r > 1e-10:
                    mid = mid / r
                    edge_mids.append(mid)

    # Remove duplicates
    unique_mids = []
    for m in edge_mids:
        if not any(np.linalg.norm(m - u) < 0.05 for u in unique_mids):
            unique_mids.append(m)

    for m in unique_mids:
        pts.append(m.copy())

    # Face centers
    faces = [
        [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
        [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
        [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
        [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
        [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
    ]
    for f in faces:
        center = (shell1[f[0]] + shell1[f[1]] + shell1[f[2]]) / 3.0
        r = np.linalg.norm(center)
        if r > 1e-10:
            center = center / r
            pts.append(center)

    # Shell 3: 92 more atoms (to reach 147 total)
    # Add points on edges and faces of the icosahedron at finer subdivisions
    if n > 55:
        # Subdivide edges into 3 parts
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(shell1[i] - shell1[j])
                if 0.9 < d < 1.1:
                    for t in [0.25, 0.75]:
                        p = (1 - t) * shell1[i] + t * shell1[j]
                        r = np.linalg.norm(p)
                        if r > 1e-10:
                            p = p / r
                            pts.append(p)

        # Subdivide faces into 4 parts (midpoints of edges of each face)
        for f in faces:
            v0, v1, v2 = shell1[f[0]], shell1[f[1]], shell1[f[2]]
            # Midpoints of triangle edges
            m01 = (v0 + v1) / 2
            m12 = (v1 + v2) / 2
            m20 = (v2 + v0) / 2
            # Center of face
            fc = (v0 + v1 + v2) / 3
            # Points at 1/3 and 2/3 along edges from center to vertices
            for v in [v0, v1, v2]:
                for t in [1/3, 2/3]:
                    p = (1 - t) * fc + t * v
                    r = np.linalg.norm(p)
                    if r > 1e-10:
                        p = p / r
                        pts.append(p)

    # Shell 4: if we need more (for n=80)
    if n > 147:
        # Add a second outer shell at radius ~1.5
        outer_verts = shell1 * 1.5
        for v in outer_verts:
            pts.append(v.copy())
        for m in unique_mids:
            pts.append(m * 1.5)
        for f in faces:
            center = (shell1[f[0]] + shell1[f[1]] + shell1[f[2]]) / 3.0
            r = np.linalg.norm(center)
            if r > 1e-10:
                pts.append((center / r) * 1.5)

    pts = np.array(pts)

    # Sort by distance from origin and take first n
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    pts = pts[idx].copy()

    # Scale to unit nearest-neighbor distance
    if len(pts) > 1:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d > 1e-10:
            pts = pts / min_d

    return pts


def _fcc_truncated(n):
    """Generate truncated FCC cluster."""
    pts = []
    a = 1.0
    max_r = int(np.ceil(n ** (1/3))) + 6
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    pts = pts[idx].copy()
    # Scale to unit NN distance
    if len(pts) > 1:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d > 1e-10:
            pts = pts / min_d
    return pts


def _hcp_truncated(n):
    """Generate truncated HCP cluster."""
    pts = []
    a = 1.0
    c_over_a = np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 6
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1.0/3.0, j + 2.0/3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    pts = pts[idx].copy()
    if len(pts) > 1:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d > 1e-10:
            pts = pts / min_d
    return pts


def _sphere_shells(n, rng):
    """Generate points on concentric spherical shells."""
    pts = [np.zeros(3)]
    remaining = n - 1
    shell_sizes = []
    # Distribute remaining atoms over shells
    shell_radii = [1.0, 1.6, 2.2, 2.8, 3.4]
    for i, r in enumerate(shell_radii):
        if remaining <= 0:
            break
        if i == len(shell_radii) - 1:
            shell_sizes.append(remaining)
        else:
            # Roughly proportional to surface area
            frac = r ** 2 / sum(rr ** 2 for rr in shell_radii[i:])
            sz = max(1, int(remaining * frac))
            sz = min(sz, remaining)
            shell_sizes.append(sz)
            remaining -= sz

    for r, sz in zip(shell_radii, shell_sizes):
        if sz > 0:
            pts.extend(_fibonacci_sphere(sz, r).tolist())

    pts = np.array(pts[:n])
    return pts


def _random_compact(n, rng):
    """Random compact configuration."""
    radius = 0.8 * (n ** (1/3)) + 1.0
    pts = [np.zeros(3)]
    min_d = 0.52
    attempts = 0
    max_attempts = 200000
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in pts:
                if np.linalg.norm(p - q) < min_d:
                    valid = False
                    break
            if valid:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _relax_fast(pos, rho, maxiter=500):
    """Fast BFGS relaxation for basin finding."""
    n = len(pos)
    x0 = pos.flatten()

    def obj(x):
        return _morse_energy(x.reshape(n, 3), rho)

    def grad(x):
        return _morse_grad(x.reshape(n, 3), rho).flatten()

    try:
        res = minimize(obj, x0, jac=grad, method='BFGS',
                       options={'maxiter': maxiter, 'gtol': 1e-6})
        if res.fun < _morse_energy(pos, rho) + 1e-8:
            return res.x.reshape(n, 3), float(res.fun)
    except Exception:
        pass
    return pos, _morse_energy(pos, rho)


def _relax_precise(pos, rho, maxiter=2000):
    """Precise L-BFGS-B relaxation for polishing."""
    n = len(pos)
    x0 = pos.flatten()

    def obj(x):
        return _morse_energy(x.reshape(n, 3), rho)

    def grad(x):
        return _morse_grad(x.reshape(n, 3), rho).flatten()

    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-14})
        if res.fun < _morse_energy(pos, rho) + 1e-8:
            return res.x.reshape(n, 3), float(res.fun)
    except Exception:
        pass
    return pos, _morse_energy(pos, rho)


def _relax_two_stage(pos, rho):
    """Two-stage relaxation: fast then precise."""
    pos1, e1 = _relax_fast(pos, rho, maxiter=300)
    if _min_dist_ok(pos1):
        pos2, e2 = _relax_precise(pos1, rho, maxiter=1500)
        if _min_dist_ok(pos2) and e2 < e1:
            return pos2, e2
        return pos1, e1
    return pos, _morse_energy(pos, rho)


def construct(instance):
    size = instance["n"]
    rho = instance.get("rho", 10)

    deadline = _START_TIME + _TIME_LIMIT - 5.0  # 5s safety margin
    rng = np.random.default_rng(size * 31 + int(rho) * 17 + 99)

    best_energy = np.inf
    best_pos = None

    def try_relax(pos, stage=2):
        nonlocal best_energy, best_pos
        if time.time() > deadline:
            return
        pos = _fix_min_dist(pos)
        if not _min_dist_ok(pos):
            return
        if stage == 2:
            rpos, e = _relax_two_stage(pos, rho)
        else:
            rpos, e = _relax_fast(pos, rho, maxiter=400)
        if _min_dist_ok(rpos) and e < best_energy:
            best_energy = e
            best_pos = rpos.copy()

    # Phase 1: Try high-symmetry seeds
    seeds = []

    # Mackay icosahedral
    seeds.append(_mckay_shells(size))

    # FCC
    seeds.append(_fcc_truncated(size))

    # HCP
    seeds.append(_hcp_truncated(size))

    # Fibonacci sphere shells
    seeds.append(_sphere_shells(size, rng))

    # For short-range (low rho), FCC/HCP are more relevant
    if rho <= 6:
        seeds.append(_fcc_truncated(size))
        seeds.append(_hcp_truncated(size))
    # For long-range (high rho), icosahedral is more relevant
    if rho >= 10:
        seeds.append(_mckay_shells(size))

    # Random compact
    seeds.append(_random_compact(size, rng))

    # Perturbed versions of top seeds
    for base in seeds[:3]:
        for sigma in [0.05, 0.15, 0.3]:
            seeds.append(base + rng.normal(0, sigma, base.shape))

    # Relax all seeds
    for pos in seeds:
        if time.time() > deadline - 10.0:
            break
        try_relax(pos, stage=2)

    if best_pos is None:
        pos = _random_compact(size, rng)
        try_relax(pos, stage=2)
        if best_pos is None:
            # Absolute fallback
            best_pos = _random_compact(size, rng)
            best_energy = _morse_energy(best_pos, rho)

    # Phase 2: Basin hopping with adaptive kicks
    # Start with small kicks, gradually increase
    kick_schedule = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    # Estimate how many kicks we can do
    # Each two-stage relax takes roughly 50-200ms depending on n
    elapsed = time.time() - _START_TIME
    remaining = max(0.0, _TIME_LIMIT - 5.0 - elapsed)

    # Conservative estimate: ~10 kicks per second
    total_kicks = int(remaining * 8)
    kicks_per_size = max(1, total_kicks // len(kick_schedule))

    for kick_size in kick_schedule:
        for k in range(kicks_per_size):
            if time.time() > deadline - 5.0:
                break

            # Adaptive kick: mix of random and directional
            if rng.random() < 0.3:
                # Directional kick: push along gradient
                grad = _morse_grad(best_pos, rho)
                norm_g = np.linalg.norm(grad)
                if norm_g > 1e-10:
                    kick_dir = -grad / norm_g  # downhill direction
                    kicked = best_pos + kick_size * kick_dir + rng.normal(0, kick_size * 0.3, best_pos.shape)
                else:
                    kicked = best_pos + rng.normal(0, kick_size, best_pos.shape)
            else:
                # Random kick
                kicked = best_pos + rng.normal(0, kick_size, best_pos.shape)

            try_relax(kicked, stage=1 if kick_size > 0.3 else 2)

    # Phase 3: Final polish
    if best_pos is not None and time.time() < deadline - 3.0:
        rpos, e = _relax_precise(best_pos, rho, maxiter=3000)
        if _min_dist_ok(rpos) and e < best_energy:
            best_pos = rpos
            best_energy = e

    # Ensure minimum distance constraint
    best_pos = _fix_min_dist(best_pos, min_d=0.51)

    return best_pos
