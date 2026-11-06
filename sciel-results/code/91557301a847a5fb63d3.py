"""Parallel basin-hopping with MD-enhanced exploration and adaptive step sizes.

Strategy:
  1. Generate diverse structural seeds (FCC, BCC, icosahedral, random compact).
  2. Run multiple independent basin-hopping chains in parallel.
  3. Each chain uses adaptive perturbation scales, MD-based exploration,
     and periodic re-seeding from the global best to escape local minima.
  4. Final high-precision polish on the best configuration.
  5. Return the best configuration found within the time budget.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 285.0
    return _T0


def _morse_E(P, rho):
    """Exact energy E = sum_{i<j} exp(rho(1-r))(exp(rho(1-r))-2)."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    r = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    r = r[iu]
    t = np.exp(rho * (1.0 - r))
    return float((t * (t - 2.0)).sum())


def _morse_grad(P, rho):
    """Gradient of exact Morse energy w.r.t. coordinates (n,3)."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    D = P[:, None, :] - P[None, :, :]
    r = np.linalg.norm(D, axis=2)
    np.fill_diagonal(r, np.inf)
    t = np.exp(rho * (1.0 - r))
    c = rho * t * (2.0 * t - 2.0)
    inv_r = np.where(r > 1e-12, 1.0 / r, 0.0)
    W = c * inv_r
    rowsum = W.sum(axis=1)
    grad = -(rowsum[:, None] * P - W @ P)
    return grad


def _enforce_min_dist(P, min_d=0.5):
    """Robustly nudge so every pair is >= min_d apart."""
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    for _ in range(500):
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        bad = (d < min_d)
        np.fill_diagonal(bad, False)
        if not bad.any():
            break
        idx = np.argwhere(bad)
        if len(idx) == 0:
            break
        i, j = idx[0]
        gap = min_d - d[i, j] + 0.01
        v = P[i] - P[j]
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            v = np.array([1.0, 0.0, 0.0]); nv = 1.0
        v = v / nv
        P[i] += 0.5 * gap * v
        P[j] -= 0.5 * gap * v
    return P


def _local_opt(P0, rho, max_iter=300, ftol=1e-10):
    """L-BFGS-B on exact Morse energy; returns (E, P)."""
    P0 = np.asarray(P0, dtype=float)
    n = P0.shape[0]

    def fun(x):
        P = x.reshape(n, 3)
        return _morse_E(P, rho)

    def gfun(x):
        P = x.reshape(n, 3)
        return _morse_grad(P, rho).ravel()

    try:
        res = minimize(fun, P0.ravel(), jac=gfun, method="L-BFGS-B",
                       options={"maxiter": max_iter, "maxfun": 10000,
                                "ftol": ftol, "gtol": 1e-7, "iprint": 0})
        P = res.x.reshape(n, 3)
        E = _morse_E(P, rho)
        return E, P
    except Exception:
        return None, None


def _fcc_seeds(n, rho):
    """Generate FCC lattice seeds with varying lattice constants."""
    seeds = []
    R = n ** (1/3.0) * 1.2
    for a in [0.9, 1.0, 1.1, 1.2]:
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ])
        pts = []
        extent = int(np.ceil(R / a)) + 2
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                for k in range(-extent, extent + 1):
                    for b in basis:
                        p = np.array([i, j, k]) * a + b * a
                        if np.linalg.norm(p) <= R:
                            pts.append(p)
        if len(pts) >= n:
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            order = np.argsort(dists)
            pts = pts[order][:n]
            seeds.append(pts)
    return seeds


def _bcc_seeds(n, rho):
    """Generate BCC lattice seeds."""
    seeds = []
    R = n ** (1/3.0) * 1.2
    for a in [0.9, 1.0, 1.1]:
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ])
        pts = []
        extent = int(np.ceil(R / a)) + 2
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                for k in range(-extent, extent + 1):
                    for b in basis:
                        p = np.array([i, j, k]) * a + b * a
                        if np.linalg.norm(p) <= R:
                            pts.append(p)
        if len(pts) >= n:
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            order = np.argsort(dists)
            pts = pts[order][:n]
            seeds.append(pts)
    return seeds


def _icosahedral_seeds(n, rho):
    """Generate icosahedral/Mackay-like seeds for small clusters."""
    seeds = []
    if n > 60:
        return seeds

    ico12 = np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
    ]) * (1.0 / np.sqrt(2.0))

    if n >= 13:
        pts = [np.zeros(3)]
        pts.extend([p.copy() for p in ico12])
        rng = np.random.RandomState(42)
        for _ in range(42):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z*z)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * 1.55
            pts.append(p)

        pts = np.array(pts[:n])
        seeds.append(pts)

    if n >= 13 and n <= 20:
        pts = np.vstack([np.zeros(3), ico12])
        if n > 13:
            rng = np.random.RandomState(99)
            extra = []
            for _ in range(n - 13):
                theta = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(-1, 1)
                r = np.sqrt(1 - z*z)
                p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * 1.55
                extra.append(p)
            pts = np.vstack([pts, np.array(extra)])
        seeds.append(pts)

    return seeds


def _random_compact(n, seed=None):
    """Random compact cluster with minimum distance enforcement."""
    rng = np.random.RandomState(seed)
    R = n ** (1/3.0) * 1.5
    P = rng.uniform(-R, R, size=(n, 3))
    return P


def _random_shell(n, seed=None):
    """Random points on a spherical shell (good for small clusters)."""
    rng = np.random.RandomState(seed)
    R = 1.0 + 0.5 * (n / 50.0)
    P = np.zeros((n, 3))
    for i in range(n):
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z*z)
        P[i] = R * np.array([r * np.cos(theta), r * np.sin(theta), z])
    return P


def _md_phase(P, rho, n_steps=40, dt=0.008, T=0.2, deadline=None):
    """Simple MD phase with velocity Verlet and thermostatting."""
    P = np.array(P, dtype=float, copy=True)
    V = np.zeros_like(P)
    m = 1.0
    n = P.shape[0]

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        F = -_morse_grad(P, rho)
        V += F / m * dt
        # Thermostat: scale velocities to maintain target temperature
        KE = 0.5 * m * np.sum(V**2)
        target_KE = 1.5 * n * T
        if KE > 1e-12:
            V *= np.sqrt(target_KE / KE)
        P += V * dt
        P = _enforce_min_dist(P)

    return P


def _basin_hopping_chain(P0, rho, n, deadline, rng_seed, global_best_tracker):
    """
    Run a basin-hopping chain with adaptive step size, MD exploration,
    and periodic re-seeding.
    """
    rng = np.random.RandomState(rng_seed)

    # Initial local optimization
    P0 = _enforce_min_dist(P0)
    E, P = _local_opt(P0, rho, max_iter=300)
    if E is None:
        E, P = 1e10, P0

    best_E = E
    best_P = P.copy()

    # Adaptive perturbation scale
    step = 0.5
    stagnant_iters = 0
    max_stagnant = 20

    it = 0
    max_iters = 400

    while it < max_iters:
        if time.time() > deadline - 3.0:
            break

        # Periodic re-seeding: if stagnant, jump from global best with MD
        if stagnant_iters > max_stagnant and global_best_tracker.get('best_P') is not None:
            gp = global_best_tracker['best_P']
            found = False
            for noise_scale in [0.2, 0.4, 0.6]:
                noise = rng.normal(0, noise_scale, size=gp.shape)
                P_try = _enforce_min_dist(gp + noise)
                # Use MD to explore, then local opt
                P_try = _md_phase(P_try, rho, n_steps=20, T=0.15, deadline=deadline - 3.0)
                E_try, P_try = _local_opt(P_try, rho, max_iter=200)
                if E_try is not None and E_try < best_E:
                    best_E = E_try
                    best_P = P_try.copy()
                    E, P = E_try, P_try
                    stagnant_iters = 0
                    found = True
                    break
            if not found:
                noise = rng.normal(0, 0.15, size=gp.shape)
                P = _enforce_min_dist(gp + noise)
                P = _md_phase(P, rho, n_steps=15, T=0.1, deadline=deadline - 3.0)
                E, P = _local_opt(P, rho, max_iter=200)
                if E is None:
                    E, P = 1e10, gp
                stagnant_iters = 0

        # Perturb
        noise = rng.normal(0, step, size=P.shape)
        P_new = P + noise
        P_new = _enforce_min_dist(P_new)

        # Use MD for enhanced exploration on every other step
        if it % 2 == 0:
            T_md = 0.1 + 0.1 * (stagnant_iters / max_stagnant)
            P_new = _md_phase(P_new, rho, n_steps=15, T=T_md, deadline=deadline - 3.0)

        # Local optimization
        E_new, P_new = _local_opt(P_new, rho, max_iter=200)
        if E_new is None:
            E_new, P_new = 1e10, P

        # Track best
        if E_new < best_E:
            best_E = E_new
            best_P = P_new.copy()
            stagnant_iters = 0
        else:
            stagnant_iters += 1

        # Update global best tracker
        if E_new < global_best_tracker.get('best_E', float('inf')):
            global_best_tracker['best_E'] = E_new
            global_best_tracker['best_P'] = P_new.copy()

        # Metropolis acceptance with adaptive temperature
        T = 0.5 * (0.01 / 0.5) ** (it / max(1, max_iters - 1))
        if E_new <= E or rng.random() < np.exp(-(E_new - E) / max(T, 1e-12)):
            E, P = E_new, P_new
            step = min(step * 1.03, 1.0)
        else:
            step = max(step * 0.93, 0.15)

        it += 1

    return best_E, best_P


def _run_chain_wrapper(args):
    """Wrapper for parallel execution."""
    P0, rho, n, deadline, rng_seed, global_best_tracker = args
    try:
        return _basin_hopping_chain(P0, rho, n, deadline, rng_seed, global_best_tracker)
    except Exception:
        return None, None


def construct(instance):
    _init_timer()
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))

    call_start = time.time()
    deadline = min(_HARD_DEADLINE, call_start + 45.0)

    best_E = float('inf')
    best_P = None

    # Generate diverse seeds
    seeds = []
    seeds.extend(_fcc_seeds(n, rho))
    seeds.extend(_bcc_seeds(n, rho))
    seeds.extend(_icosahedral_seeds(n, rho))
    for s in range(4):
        seeds.append(_random_compact(n, seed=100 + s))
    for s in range(2):
        seeds.append(_random_shell(n, seed=200 + s))

    if not seeds:
        seeds = [_random_compact(n, seed=0)]

    # Quick local optimization on all seeds to get baseline
    seed_results = []
    for i, s in enumerate(seeds):
        if time.time() > deadline - 15.0:
            break
        E, P = _local_opt(_enforce_min_dist(s), rho, max_iter=150)
        if E is not None:
            seed_results.append((E, P))
            if E < best_E:
                best_E = E
                best_P = P

    # Sort seeds by energy, use best ones for basin hopping
    seed_results.sort(key=lambda x: x[0])
    top_seeds = seed_results[:8] if len(seed_results) >= 8 else seed_results

    # Shared tracker for global best across chains
    global_best_tracker = {'best_E': best_E, 'best_P': best_P.copy() if best_P is not None else None}

    # Run basin hopping chains in parallel (up to 4 threads)
    n_chains = min(4, len(top_seeds))
    if n_chains > 0 and time.time() < deadline - 10.0:
        chain_args = []
        for i in range(n_chains):
            P0 = top_seeds[i][1]
            chain_args.append((P0, rho, n, deadline, 1000 + i, global_best_tracker))

        with _cf.ThreadPoolExecutor(max_workers=n_chains) as pool:
            futures = [pool.submit(_run_chain_wrapper, args) for args in chain_args]

            for f in futures:
                try:
                    E, P = f.result()
                    if E is not None and E < best_E:
                        best_E = E
                        best_P = P
                except Exception:
                    pass

        if global_best_tracker['best_E'] < best_E:
            best_E = global_best_tracker['best_E']
            best_P = global_best_tracker['best_P']

    # If we still have time, do a final polish on the best configuration
    if best_P is not None and time.time() < deadline - 5.0:
        E, P = _local_opt(_enforce_min_dist(best_P), rho, max_iter=800, ftol=1e-13)
        if E is not None and E < best_E:
            best_E = E
            best_P = P

    # Fallback if nothing worked
    if best_P is None:
        best_P = _random_compact(n, seed=42)
        best_E = _morse_E(best_P, rho)

    # Final validity check
    best_P = _enforce_min_dist(best_P)

    # One last quick relaxation to ensure validity
    if time.time() < deadline - 1.0:
        E, P = _local_opt(best_P, rho, max_iter=100)
        if E is not None:
            best_P = P

    best_P = _enforce_min_dist(best_P)

    return best_P

# EVOLVE-BLOCK-END
