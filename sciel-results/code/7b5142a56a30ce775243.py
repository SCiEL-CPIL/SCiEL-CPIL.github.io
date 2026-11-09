"""Sutton-Chen cluster global minima via coarse-to-fine basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    # efficient pairwise min distance via broadcasting in chunks to save memory
    for i in range(n - 1):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        if d.size and d.min() < 0.35:
            return False
    return True


def _fcc_cluster(n, lattice=0.72):
    """Compact FCC-like cluster: take n nearest FCC lattice points to origin."""
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    pts.append(base + np.array(off))
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _hcp_cluster(n, lattice=0.72):
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                pts.append([x, y, z])
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _icosahedral_cluster(n):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v.copy())
    # second shell: edge midpoints pushed out
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.35
                pts.append(mid)
    # third shell: face centers pushed out
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * 1.7
                    pts.append(fc)
    pts = np.array(pts)
    if len(pts) >= n:
        return pts[:n]
    rng = np.random.default_rng(44)
    while len(pts) < n:
        p = rng.normal(size=3)
        p = p / np.linalg.norm(p) * 2.0
        pts = np.vstack([pts, p])
    return pts[:n]


def _random_compact(n, seed, min_dist=0.8):
    rng = np.random.default_rng(seed)
    max_radius = 0.9 * n ** (1.0 / 3.0)
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 400:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            if pts:
                arr = np.array(pts)
                if np.min(np.linalg.norm(arr - p, axis=1)) < min_dist:
                    ok = False
            if ok:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        pts.append(p)
    return np.array(pts)


def _short_relax(pos, potential, kind, max_iter=300):
    """Quick L-BFGS-B polish."""
    try:
        e, p = relax(pos, potential, kind)
        if np.isfinite(e) and _validate_positions(p):
            return e, p
    except Exception:
        pass
    return np.inf, pos


def construct(instance):
    start_time = time.time()
    deadline = start_time + 320.0  # safety margin vs 330s kill

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))

    # --- Phase 1: get a good initial configuration from structured seeds ---
    seeds = [
        _fcc_cluster(n, lattice=0.72),
        _hcp_cluster(n, lattice=0.72),
        _icosahedral_cluster(n),
        _random_compact(n, seed=7, min_dist=0.8),
        _random_compact(n, seed=13, min_dist=0.9),
    ]

    best_e = np.inf
    best_pos = None
    for s in seeds:
        if time.time() > deadline:
            break
        if not _validate_positions(s):
            continue
        e, p = _short_relax(s, potential, kind)
        if e < best_e:
            best_e = e
            best_pos = p.copy()

    if best_pos is None:
        # fallback
        pos = _random_compact(n, seed=99, min_dist=0.8)
        best_e, best_pos = _short_relax(pos, potential, kind)
        if best_e == np.inf:
            return pos

    # --- Phase 2: Coarse-to-fine basin hopping ---
    # Perturbation amplitudes decrease over time (coarse -> fine)
    # Each accepted perturbation is polished with a short L-BFGS-B
    rng = np.random.default_rng(12345)

    # Amplitude schedule: start broad, narrow down
    n_atoms = n
    # Typical interatomic distance ~0.8-1.0; perturbation should be a fraction
    amp_schedule = [
        (1.2, 40),   # (amplitude, n_trials) - coarse
        (0.7, 60),
        (0.4, 80),
        (0.2, 100),
        (0.1, 120),
        (0.05, 150),  # fine
        (0.02, 200),
    ]

    for amp, n_trials in amp_schedule:
        if time.time() > deadline:
            break
        for t in range(n_trials):
            if time.time() > deadline:
                break

            # Generate perturbation: random displacement per atom
            # Use a mix of small Gaussian and occasional larger jumps
            if t % 5 == 0:
                # occasional larger perturbation to escape basin
                disp = rng.normal(0, amp * 1.5, size=(n_atoms, 3))
            else:
                disp = rng.normal(0, amp, size=(n_atoms, 3))

            perturbed = best_pos + disp

            # Quick validity check (cheap for small n; for large n do sampling)
            if n_atoms <= 80:
                valid = _validate_positions(perturbed)
            else:
                # sample-based check for speed
                idx = rng.choice(n_atoms, size=min(20, n_atoms), replace=False)
                valid = True
                for ii in idx:
                    d = np.linalg.norm(perturbed[ii + 1:] - perturbed[ii], axis=1) if ii + 1 < n_atoms else np.array([1.0])
                    if d.size and d.min() < 0.35:
                        valid = False
                        break

            if not valid:
                continue

            # Polish the perturbed configuration
            e_new, p_new = _short_relax(perturbed, potential, kind)

            if e_new < best_e:
                best_e = e_new
                best_pos = p_new.copy()

    # --- Phase 3: Final full polish on the best configuration ---
    if time.time() < deadline - 10.0:
        e_final, p_final = relax(best_pos, potential, kind)
        if np.isfinite(e_final) and _validate_positions(p_final):
            best_e = e_final
            best_pos = p_final

    return best_pos
# EVOLVE-BLOCK-END
