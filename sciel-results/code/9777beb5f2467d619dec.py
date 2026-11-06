"""
Morse cluster global minima: differentiable particle swarm optimization (PSO)
driven by the analytical Morse energy gradient, with diverse structured seeds.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # safety margin before the 300s kill

def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(positions, rho):
    """Vectorized Morse energy: E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, None, :] - positions[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(n, k=1)
    r = dists[iu]
    x = np.exp(rho * (1.0 - r))
    return float(np.sum(x * (x - 2.0)))


def _enforce_min_dist(positions, min_d=0.5):
    """Push any pair closer than min_d apart along their separation vector."""
    n = len(positions)
    positions = positions.copy()
    for _ in range(20):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-9:
                        d = np.array([1.0, 0.0, 0.0])
                    else:
                        d = d / dist
                    push = (min_d - dist) / 2.0
                    positions[i] += push * d
                    positions[j] -= push * d
                    moved = True
        if not moved:
            break
    return positions


def _fcc_points(n, rng):
    """FCC lattice sites, centered, randomly subsampled to n."""
    m = int(np.ceil(n ** (1.0 / 3.0)) * 2) + 1
    pts = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                pts.append([i, j, k])
                pts.append([i, j + 0.5, k + 0.5])
                pts.append([i + 0.5, j, k + 0.5])
                pts.append([i + 0.5, j + 0.5, k])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    idx = rng.choice(len(pts), n, replace=False)
    return _enforce_min_dist(pts[idx])


def _bcc_points(n, rng):
    """BCC lattice sites, centered, randomly subsampled to n."""
    m = int(np.ceil(n ** (1.0 / 3.0)) * 2) + 1
    pts = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    idx = rng.choice(len(pts), n, replace=False)
    return _enforce_min_dist(pts[idx])


def _sc_points(n, rng):
    """Simple cubic lattice sites, centered, randomly subsampled to n."""
    m = int(np.ceil(n ** (1.0 / 3.0)) * 2) + 1
    pts = np.array([[i, j, k] for i in range(m) for j in range(m) for k in range(m)])
    pts -= pts.mean(axis=0)
    idx = rng.choice(len(pts), n, replace=False)
    return _enforce_min_dist(pts[idx])


def _random_compact(n, rng):
    """Random points inside a sphere, compact, min-distance enforced."""
    radius = 1.0 * n ** (1.0 / 3.0) * 0.9
    pts = rng.uniform(-radius, radius, (n, 3))
    return _enforce_min_dist(pts)


def _icosahedral(n, rng):
    """Icosahedral shells; returns None if n too large."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts /= np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    pts.extend(verts)
    if n > len(pts):
        # second shell: icosidodecahedron-ish (midpoints of edges)
        for a in range(len(verts)):
            for b in range(a + 1, len(verts)):
                v = verts[a] + verts[b]
                nn = np.linalg.norm(v)
                if nn > 1e-9:
                    pts.append(v / nn)
    if n > len(pts):
        return None
    pts = np.array(pts[:n])
    pts += rng.normal(0, 0.05, pts.shape)
    return _enforce_min_dist(pts)


def _gradient_refine(pos, potential, rho, rng):
    """Run the provided relax (gradient-based) and return (energy, positions)."""
    try:
        e, p = relax(pos, potential, rho)
        p = _enforce_min_dist(p)
        return float(e), p
    except Exception:
        e = _morse_energy(pos, rho)
        return e, pos


def _pso_chain(pos0, potential, rho, rng, time_budget):
    """
    Differentiable PSO: particles carry positions AND velocities; the velocity
    update is driven by the analytical Morse gradient (smooth, collective),
    with inertia, cognitive and social terms. Final gradient polish per particle.
    """
    n = len(pos0)
    npart = 4
    X = np.empty((npart, n, 3))
    V = np.empty((npart, n, 3))
    for p in range(npart):
        X[p] = pos0 + rng.normal(0, 0.3, (n, 3))
        V[p] = rng.normal(0, 0.1, (n, 3))
    X = np.array([_enforce_min_dist(X[p]) for p in range(npart)])

    # initial energies
    E = np.array([_morse_energy(X[p], rho) for p in range(npart)])
    pbest = X.copy()
    pbest_E = E.copy()
    gidx = int(np.argmin(E))
    gbest = X[gidx].copy()
    gbest_E = float(E[gidx])

    w = 0.7      # inertia
    c1 = 1.3     # cognitive
    c2 = 1.3     # social
    vmax = 0.6

    it = 0
    while _time_left() > time_budget:
        it += 1
        # decay coefficients for convergence
        scale = max(0.3, 1.0 - 0.02 * it)
        for p in range(npart):
            g, _ = morse_energy_gradient(X[p], rho)
            g = np.asarray(g, dtype=float)
            # gradient descent term (negative gradient, scaled)
            grad_step = -0.05 * g * scale
            r1 = rng.random((n, 3))
            r2 = rng.random((n, 3))
            V[p] = (w * V[p]
                    + c1 * r1 * (pbest[p] - X[p])
                    + c2 * r2 * (gbest - X[p])
                    + grad_step)
            # clip velocity
            V[p] = np.clip(V[p], -vmax, vmax)
            X[p] = _enforce_min_dist(X[p] + V[p])
            E[p] = _morse_energy(X[p], rho)
            if E[p] < pbest_E[p]:
                pbest_E[p] = E[p]
                pbest[p] = X[p].copy()
            if E[p] < gbest_E:
                gbest_E = E[p]
                gbest = X[p].copy()
                gidx = p
    # final gradient polish on the global best
    e, pos = _gradient_refine(gbest, potential, rho, rng)
    if e < gbest_E:
        gbest_E = e
        gbest = pos
    return gbest_E, gbest


def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]

    if _time_left() < 6.0:
        rng = np.random.default_rng()
        radius = 1.0 * n ** (1.0 / 3.0) * 0.9
        pts = rng.uniform(-radius, radius, (n, 3))
        _, pos = _gradient_refine(_enforce_min_dist(pts), potential, rho, rng)
        return pos

    rng = np.random.default_rng()

    # Diverse structured seeds
    seeds = []
    seeds.append(_random_compact(n, rng))
    seeds.append(_fcc_points(n, rng))
    seeds.append(_bcc_points(n, rng))
    seeds.append(_sc_points(n, rng))
    ico = _icosahedral(n, rng)
    if ico is not None:
        seeds.append(ico)
    seeds.append(_random_compact(n, rng))

    # Relax each seed to a local minimum (cheap gradient polish)
    best_E = float('inf')
    best_pos = None
    for s in seeds:
        if _time_left() < 8.0:
            break
        e, pos = _gradient_refine(s, potential, rho, rng)
        if e < best_E:
            best_E = e
            best_pos = pos.copy()

    if best_pos is None:
        e, best_pos = _gradient_refine(seeds[0], potential, rho, rng)
        best_E = e

    # Run PSO chains from the best relaxed seed and a couple of diverse seeds.
    # Each chain gets a time budget so the total stays under the deadline.
    chain_starts = [best_pos]
    # add the fcc and random seeds (already relaxed) as additional swarm origins
    extra = []
    for s in seeds:
        e, pos = _gradient_refine(s, potential, rho, rng)
        extra.append(pos)
        if e < best_E:
            best_E = e
            best_pos = pos.copy()
    chain_starts.extend(extra[:2])

    nchains = len(chain_starts)
    per_budget = 4.0  # seconds of slack per chain (total ~12s, well under deadline)
    for start in chain_starts:
        if _time_left() < per_budget + 6.0:
            break
        e, pos = _pso_chain(start, potential, rho, rng, time_budget=per_budget)
        if e < best_E:
            best_E = e
            best_pos = pos.copy()

    # Final safety: ensure min-distance and return
    best_pos = _enforce_min_dist(best_pos)
    return best_pos
# EVOLVE-BLOCK-END
