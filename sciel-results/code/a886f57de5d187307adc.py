"""Morse cluster global minima: hybrid PSO-basin hopping with parallel chains.

Strategy:
  1. Generate diverse structured seeds (FCC, BCC, icosahedral, random compact).
  2. Run multiple parallel optimization chains, each combining:
     - Gradient-guided PSO for smooth navigation
     - Basin hopping perturbations to escape local minima
  3. Periodic L-BFGS-B refinement sharpens the best configuration.
  4. Parallel evaluation keeps wall-clock within budget.
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
    for _ in range(200):
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
    """Generate icosahedral shell seeds."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    seeds = []
    for scale in [1.0, 1.2, 1.5]:
        pts = [np.zeros(3)]
        # First shell
        for v in verts:
            if len(pts) < n:
                pts.append(v * scale)
        # Second shell
        if len(pts) < n:
            for v in verts:
                if len(pts) < n:
                    pts.append(v * 2.0 * scale)
        # Fill remaining
        rng = np.random.RandomState(42)
        while len(pts) < n:
            p = rng.uniform(-2.5*scale, 2.5*scale, 3)
            pts.append(p)
        pts = np.array(pts[:n])
        seeds.append(pts)
    return seeds


def _random_compact(n, seed=None):
    """Random compact cluster."""
    rng = np.random.RandomState(seed)
    R = n ** (1/3.0) * 1.5
    P = rng.uniform(-R, R, size=(n, 3))
    return P


def _hybrid_chain(P0, rho, n, deadline, chain_id=0, max_iters=100):
    """
    Hybrid PSO-basin hopping chain.
    
    Alternates between:
    - Gradient-guided PSO steps (smooth navigation)
    - Basin hopping perturbations (escape local minima)
    
    Returns (best_E, best_P) for this chain.
    """
    rng = np.random.RandomState(1000 + chain_id)
    
    # Initialize with local optimization
    E, P = _local_opt(_enforce_min_dist(P0.copy()), rho, max_iter=150)
    if E is None:
        P = _enforce_min_dist(P0.copy())
        E = _morse_E(P, rho)
    
    best_E = E
    best_P = P.copy()
    current_E = E
    current_P = P.copy()
    
    # PSO state
    V = np.zeros_like(P)
    pbest_E = E
    pbest_X = P.copy()
    
    # Adaptive parameters
    w_init, w_end = 0.85, 0.35
    c1, c2 = 1.8, 1.8
    v_max_init = 0.8
    
    # Basin hopping state
    bh_scale = 0.5
    bh_count = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        frac = it / max(1, max_iters - 1)
        w = w_init + (w_end - w_init) * frac
        c1_t = c1 * (1.0 - 0.3 * frac)
        c2_t = c2 * (1.0 + 0.5 * frac)
        v_max = v_max_init * (1.0 - 0.5 * frac)
        
        # Decide: PSO step or basin hop?
        # More basin hops as we progress (exploration)
        do_basin_hop = (it % 5 == 4) or (frac > 0.5 and rng.random() < 0.3)
        
        if do_basin_hop:
            # Basin hopping: perturb and relax
            perturbation = rng.normal(0, bh_scale, size=current_P.shape)
            perturbed = current_P + perturbation
            perturbed = _enforce_min_dist(perturbed)
            
            e, p = _local_opt(perturbed, rho, max_iter=100)
            if e is not None:
                if e < current_E:
                    current_E = e
                    current_P = p.copy()
                    bh_scale *= 1.1
                else:
                    # Metropolis acceptance
                    delta_e = e - current_E
                    if rng.random() < np.exp(-delta_e / 0.1):
                        current_P = p.copy()
                        current_E = e
                    bh_scale *= 0.95
                bh_count += 1
                
                if current_E < best_E:
                    best_E = current_E
                    best_P = current_P.copy()
        else:
            # PSO step with gradient guidance
            grad = _morse_grad(current_P, rho)
            alpha = 0.1 * (1.0 - 0.5 * frac)
            
            r1 = rng.rand(n, 3)
            r2 = rng.rand(n, 3)
            
            V = (w * V 
                 + c1_t * r1 * (pbest_X - current_P) 
                 + c2_t * r2 * (best_P - current_P)
                 - alpha * grad)
            
            # Clamp velocity
            v_norm = np.linalg.norm(V, axis=1, keepdims=True)
            scale = np.where(v_norm > v_max, v_max / np.maximum(v_norm, 1e-12), 1.0)
            V *= scale
            
            # Update position
            new_P = current_P + V
            new_P = _enforce_min_dist(new_P)
            
            # Evaluate
            new_E = _morse_E(new_P, rho)
            
            # Update personal best
            if new_E < pbest_E:
                pbest_E = new_E
                pbest_X = new_P.copy()
            
            # Update global best for this chain
            if new_E < best_E:
                best_E = new_E
                best_P = new_P.copy()
            
            # Accept step
            current_P = new_P
            current_E = new_E
        
        # Periodic refinement of best
        if it % 20 == 19:
            e, p = _local_opt(best_P, rho, max_iter=200)
            if e is not None and e < best_E:
                best_E = e
                best_P = p.copy()
    
    return best_E, best_P


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
    seeds.extend(_icosahedral_seeds(n, rho))
    for s in range(4):
        seeds.append(_random_compact(n, seed=100 + s))
    
    # Ensure we have at least some seeds
    if not seeds:
        seeds = [_random_compact(n, seed=0)]
    
    # Quick local optimization on all seeds to get baseline
    seed_results = []
    for i, s in enumerate(seeds):
        if time.time() > deadline - 15.0:
            break
        E, P = _local_opt(_enforce_min_dist(s), rho, max_iter=80)
        if E is not None:
            seed_results.append((E, P))
            if E < best_E:
                best_E = E
                best_P = P
    
    # Sort seeds by energy, use best ones for hybrid chains
    seed_results.sort(key=lambda x: x[0])
    top_seeds = [s[1] for s in seed_results[:6]] if len(seed_results) >= 6 else [s[1] for s in seed_results]
    
    # Run parallel hybrid chains
    if len(top_seeds) > 0 and time.time() < deadline - 10.0:
        n_chains = min(4, len(top_seeds))
        
        def run_chain(args):
            idx, seed_P = args
            return _hybrid_chain(seed_P, rho, n, deadline, chain_id=idx, max_iters=80)
        
        with _cf.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_chain, (i, top_seeds[i])) for i in range(n_chains)]
            for f in _cf.as_completed(futures):
                try:
                    E_chain, P_chain = f.result()
                    if E_chain < best_E:
                        best_E = E_chain
                        best_P = P_chain
                except Exception:
                    pass
    
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
