"""
LJ38 global minimum search.
Strategy:
1. Construct a high-quality FCC truncated-octahedron starting geometry
   (the known global minimum topology for N=38).
2. Perform a careful local relaxation with L-BFGS-B using analytic gradient.
3. Also try a few perturbed FCC starts and a couple of random starts to
   guard against local minima, keeping the best energy found.
4. Enforce the minimum-distance constraint (>= 0.5) by rejecting/repairing.
5. Deadline guard: return best result before ~140 s wall-clock.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE = time.time() + 140.0


def _lj_energy(pos):
    """Total Lennard-Jones energy 4*sum_{i<j}(r^-12 - r^-6)."""
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1e12)
    dist = np.sqrt(dist_sq)
    inv_r6 = dist ** (-6)
    inv_r12 = inv_r6 ** 2
    # Sum over upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    return 4.0 * np.sum(inv_r12[iu] - inv_r6[iu])


def _lj_energy_grad(pos):
    """Energy and gradient of LJ potential."""
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1e12)
    dist = np.sqrt(dist_sq)
    inv_r6 = dist ** (-6)
    inv_r12 = inv_r6 ** 2

    iu = np.triu_indices(n, k=1)
    energy = 4.0 * np.sum(inv_r12[iu] - inv_r6[iu])

    # Gradient: for pair (i,j), force on i is -dE/dr_ij * (r_i - r_j)/r_ij
    # dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # grad_i = sum_{j!=i} 4*(12*inv_r12 - 6*inv_r6)/dist_sq * (pos[i]-pos[j])
    factor = 4.0 * (12.0 * inv_r12 - 6.0 * inv_r6) / dist_sq
    np.fill_diagonal(factor, 0.0)
    grad = np.einsum('ij,ijk->ik', factor, diff)
    return energy, grad


def _min_pair_dist(pos):
    """Minimum pairwise distance."""
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    iu = np.triu_indices(n, k=1)
    return np.sqrt(np.min(dist_sq[iu]))


def _relax(pos, max_iter=5000):
    """Local relaxation via L-BFGS-B with analytic gradient."""
    try:
        res = minimize(_lj_energy_grad, pos, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-14,
                                'gtol': 1e-10, 'maxls': 50})
        return res.x, res.fun
    except Exception:
        return pos, _lj_energy(pos)


def _fcc_truncated_octahedron(spacing=1.12):
    """
    Generate FCC lattice points and select the 38 closest to the origin.
    FCC primitive lattice: points (i,j,k)*a/2 where i+j+k is even.
    """
    a = spacing
    sites = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    sites.append([i * a / 2.0, j * a / 2.0, k * a / 2.0])
    sites = np.array(sites)
    dists = np.sum(sites ** 2, axis=1)
    order = np.argsort(dists)
    sites = sites[order]
    # Take the 38 closest
    sites = sites[:N].copy()
    # Center
    sites -= np.mean(sites, axis=0)
    return sites


def _icosahedral_start():
    """
    Icosahedral arrangement: 1 center + 12 vertices + 25 from second shell.
    This is a known local minimum topology for LJ38.
    """
    phi = (1 + np.sqrt(5)) / 2.0
    r1 = 1.12
    r2 = 2.2

    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    verts *= r1

    # Second shell: 30 edge midpoints of icosahedron
    # Find edges: pairs of vertices at minimum distance
    vdist = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    min_d = np.min(vdist[np.triu_indices(12, k=1)])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(vdist[i, j] - min_d) < 1e-6:
                edges.append((i, j))

    mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid /= np.linalg.norm(mid)
        mids.append(mid)
    mids = np.array(mids)
    mids *= r2

    # We have 1 + 12 + 30 = 43. Need 38. Remove 5 farthest from center.
    positions = np.vstack([[np.zeros(3)], verts, mids])
    dists = np.linalg.norm(positions, axis=1)
    # Keep the 38 closest
    order = np.argsort(dists)
    positions = positions[order[:N]].copy()
    positions -= np.mean(positions, axis=0)
    return positions


def _cubic_start():
    """Simple cubic lattice fragment, 38 points."""
    side = int(np.ceil(N ** (1.0 / 3.0)))
    points = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                points.append([i, j, k])
    points.sort(key=lambda p: (p[0] - (side - 1) / 2.0) ** 2 +
                              (p[1] - (side - 1) / 2.0) ** 2 +
                              (p[2] - (side - 1) / 2.0) ** 2)
    cubic = np.array(points[:N], dtype=float) * 1.12
    cubic -= np.mean(cubic, axis=0)
    return cubic


def construct():
    global DEADLINE
    DEADLINE = time.time() + 140.0

    best_pos = None
    best_energy = np.inf

    # Generate candidate starting geometries
    candidates = []

    # FCC truncated octahedron with several spacings
    for sp in [1.08, 1.10, 1.12, 1.14, 1.16]:
        try:
            fcc = _fcc_truncated_octahedron(sp)
            candidates.append(fcc)
        except Exception:
            pass

    # Icosahedral
    try:
        ico = _icosahedral_start()
        candidates.append(ico)
    except Exception:
        pass

    # Cubic
    try:
        cubic = _cubic_start()
        candidates.append(cubic)
    except Exception:
        pass

    # A few random starts
    np.random.seed(123)
    for _ in range(4):
        rand = np.random.uniform(-2.5, 2.5, size=(N, 3))
        candidates.append(rand)

    # Process candidates
    for idx, pos in enumerate(candidates):
        if time.time() > DEADLINE - 8.0:
            break

        # Ensure minimum distance >= 0.5 by iterative perturbation
        for _ in range(50):
            md = _min_pair_dist(pos)
            if md >= 0.5:
                break
            # Push apart close pairs
            n = pos.shape[0]
            diff = pos[:, None, :] - pos[None, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(dist_sq, 1e12)
            dist = np.sqrt(dist_sq)
            # Find pairs too close
            iu = np.triu_indices(n, k=1)
            close = dist[iu] < 0.5
            if not np.any(close):
                break
            # Perturb
            pos = pos + np.random.uniform(-0.05, 0.05, size=pos.shape)

        # Relax
        pos_relaxed, energy = _relax(pos, max_iter=4000)

        # Check minimum distance after relaxation
        md = _min_pair_dist(pos_relaxed)
        if md < 0.5:
            # Try to repair
            for _ in range(20):
                diff = pos_relaxed[:, None, :] - pos_relaxed[None, :, :]
                dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
                np.fill_diagonal(dist_sq, 1e12)
                dist = np.sqrt(dist_sq)
                iu = np.triu_indices(N, k=1)
                close_idx = np.argwhere(dist[iu] < 0.5)
                if len(close_idx) == 0:
                    break
                for ci in close_idx[:5]:
                    i, j = ci
                    d = pos_relaxed[i] - pos_relaxed[j]
                    norm = np.linalg.norm(d)
                    if norm > 1e-10:
                        push = (d / norm) * (0.5 - norm + 0.01)
                        pos_relaxed[i] += push / 2.0
                        pos_relaxed[j] -= push / 2.0
                # Quick re-relax
                pos_relaxed, energy = _relax(pos_relaxed, max_iter=2000)
                md = _min_pair_dist(pos_relaxed)

        if md >= 0.5 and energy < best_energy:
            best_energy = energy
            best_pos = pos_relaxed.copy()

    # Fallback
    if best_pos is None:
        best_pos = _cubic_start()

    # Final safety check
    md = _min_pair_dist(best_pos)
    if md < 0.5:
        # Last resort: spread out
        best_pos = best_pos * (0.5 / md + 0.1)
        md = _min_pair_dist(best_pos)
        if md < 0.5:
            # Use cubic
            best_pos = _cubic_start()

    return best_pos
