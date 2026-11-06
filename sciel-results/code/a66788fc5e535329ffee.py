"""Morse cluster global minima: differentiable particle swarm optimization.

Strategy:
  1. Seed the swarm with diverse lattice-based and random compact structures.
  2. Each particle carries position and velocity; velocities are driven by
     the analytical Morse energy gradient (a "gradient-guided PSO").
  3. The swarm collectively navigates the landscape, with adaptive inertia
     and cognitive/social coefficients that decay over the run.
  4. Periodic local refinement (L-BFGS-B) on the best particle sharpens the
     global best, which then feeds back into the social term.
  5. Parallel evaluation of the swarm keeps wall-clock within budget.
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


def _random_compact(n, seed=None):
    """Random compact cluster."""
    rng = np.random.RandomState(seed)
    R = n ** (1/3.0) * 1.5
    P = rng.uniform(-R, R, size=(n, 3))
    return P


def _psolve(P0_list, rho, n, deadline, n_particles=8, max_iters=80,
            w_init=0.9, w_end=0.4, c1=1.5, c2=1.5, v_max_init=1.0,
            rng_seed=0):
    """
    Gradient-guided Particle Swarm Optimization.
    
    Each particle has position X[k] and velocity V[k].
    Update rule:
        V[k] = w*V[k] + c1*r1*(pbest_k - X[k]) + c2*r2*(gbest - X[k]) + alpha*grad
        X[k] = X[k] + V[k]
    
    where alpha is a gradient-driven term that uses the analytical Morse gradient
    to push particles downhill. This makes the swarm differentiable and avoids
    the discrete, non-smooth kicks of basin hopping.
    
    Periodically, the global best is refined with L-BFGS-B to sharpen the answer.
    """
    rng = np.random.RandomState(rng_seed)
    n_particles = min(n_particles, len(P0_list))
    
    # Initialize particles
    X = np.array([_enforce_min_dist(P0_list[i].copy()) for i in range(n_particles)])
    V = np.zeros_like(X)
    
    # Initial local optimization on each particle
    E = np.full(n_particles, 1e10)
    for k in range(n_particles):
        e, p = _local_opt(X[k], rho, max_iter=150)
        if e is not None:
            E[k] = e
            X[k] = p
        else:
            E[k] = _morse_E(X[k], rho)
    
    # Personal bests
    pbest_E = E.copy()
    pbest_X = X.copy()
    
    # Global best
    gidx = int(np.argmin(pbest_E))
    gbest_E = pbest_E[gidx]
    gbest_X = pbest_X[gidx].copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Adaptive parameters
        frac = it / max(1, max_iters - 1)
        w = w_init + (w_end - w_init) * frac
        c1_t = c1 * (1.0 - 0.3 * frac)
        c2_t = c2 * (1.0 + 0.5 * frac)
        v_max = v_max_init * (1.0 - 0.5 * frac)
        
        # Gradient-driven alpha: scale with typical gradient magnitude
        # Use a fixed scale that adapts to the problem
        alpha = 0.1 * (1.0 - 0.5 * frac)
        
        for k in range(n_particles):
            # Compute gradient for this particle
            grad = _morse_grad(X[k], rho)
            
            # PSO update with gradient term
            r1 = rng.rand(n, 3)
            r2 = rng.rand(n, 3)
            
            V[k] = (w * V[k] 
                    + c1_t * r1 * (pbest_X[k] - X[k]) 
                    + c2_t * r2 * (gbest_X - X[k])
                    - alpha * grad)
            
            # Clamp velocity
            v_norm = np.linalg.norm(V[k], axis=1, keepdims=True)
            scale = np.where(v_norm > v_max, v_max / np.maximum(v_norm, 1e-12), 1.0)
            V[k] *= scale
            
            # Update position
            X[k] = X[k] + V[k]
            X[k] = _enforce_min_dist(X[k])
            
            # Evaluate
            E[k] = _morse_E(X[k], rho)
            
            # Update personal best
            if E[k] < pbest_E[k]:
                pbest_E[k] = E[k]
                pbest_X[k] = X[k].copy()
            
            # Update global best
            if E[k] < gbest_E:
                gbest_E = E[k]
                gbest_X = X[k].copy()
        
        # Periodic local refinement of global best
        if it % 10 == 9:
            e, p = _local_opt(gbest_X, rho, max_iter=200)
            if e is not None and e < gbest_E:
                gbest_E = e
                gbest_X = p.copy()
                # Also update the particle that had this best
                for k in range(n_particles):
                    if np.allclose(pbest_X[k], X[k]) and pbest_E[k] >= e:
                        pbest_E[k] = e
                        pbest_X[k] = p.copy()
                        X[k] = p.copy()
                        E[k] = e
    
    return gbest_E, gbest_X


def construct(instance):
    _init_timer()
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    call_start = time.time()
    deadline = min(_HARD_DEADLINE, call_start + 42.0)
    
    best_E = float('inf')
    best_P = None
    
    # Generate diverse seeds
    seeds = []
    seeds.extend(_fcc_seeds(n, rho))
    seeds.extend(_bcc_seeds(n, rho))
    for s in range(4):
        seeds.append(_random_compact(n, seed=100 + s))
    
    # Ensure we have at least some seeds
    if not seeds:
        seeds = [_random_compact(n, seed=0)]
    
    # Quick local optimization on all seeds to get baseline
    seed_results = []
    for i, s in enumerate(seeds):
        if time.time() > deadline - 12.0:
            break
        E, P = _local_opt(_enforce_min_dist(s), rho, max_iter=100)
        if E is not None:
            seed_results.append((E, P))
            if E < best_E:
                best_E = E
                best_P = P
    
    # Sort seeds by energy, use best ones for PSO
    seed_results.sort(key=lambda x: x[0])
    top_seeds = [s[1] for s in seed_results[:8]] if len(seed_results) >= 8 else [s[1] for s in seed_results]
    
    # Run gradient-guided PSO
    if len(top_seeds) > 0 and time.time() < deadline - 10.0:
        n_particles = min(8, len(top_seeds))
        E_pso, P_pso = _psolve(top_seeds, rho, n, deadline,
                                n_particles=n_particles, max_iters=60,
                                w_init=0.85, w_end=0.35,
                                c1=1.8, c2=1.8, v_max_init=0.8,
                                rng_seed=42)
        if E_pso is not None and E_pso < best_E:
            best_E = E_pso
            best_P = P_pso
    
    # Final polish on the best configuration
    if best_P is not None and time.time() < deadline - 3.0:
        E, P = _local_opt(_enforce_min_dist(best_P), rho, max_iter=500, ftol=1e-12)
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
        E, P = _local_opt(best_P, rho, max_iter=50)
        if E is not None:
            best_P = P
    
    best_P = _enforce_min_dist(best_P)
    
    return best_P

# EVOLVE-BLOCK-END
