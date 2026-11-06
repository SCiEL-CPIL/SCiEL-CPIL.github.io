"""Langevin Simulated Annealing solver for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_POOL = None
_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 288.0
    return _T0


def _pair_dists(P):
    n = P.shape[0]
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    return d[iu], iu


def _morse_E(P, rho):
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    r, _ = _pair_dists(P)
    t = np.exp(rho * (1.0 - r))
    return float((t * (t - 2.0)).sum())


def _morse_grad(P, rho):
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
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    for _ in range(300):
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        bad = (d < min_d)
        np.fill_diagonal(bad, False)
        if not bad.any():
            break
        i, j = np.argwhere(bad)[0]
        gap = min_d - d[i, j] + 0.01
        v = P[i] - P[j]
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            v = np.array([1.0, 0.0, 0.0]); nv = 1.0
        v = v / nv
        P[i] += 0.5 * gap * v
        P[j] -= 0.5 * gap * v
    return P


def _local_opt(P0, rho, max_iter=400):
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
                       options={"maxiter": max_iter, "maxfun": 20000,
                                "ftol": 1e-12, "gtol": 1e-8, "iprint": 0})
        P = res.x.reshape(n, 3)
        E = _morse_E(P, rho)
        return E, P
    except Exception:
        return None, None


def _get_pool():
    global _POOL
    if _POOL is None:
        _POOL = _cf.ThreadPoolExecutor(max_workers=4)
    return _POOL


def _langevin_sa(P_init, rho, t_start, t_end, n_steps, deadline, best):
    """
    Run Langevin dynamics with Simulated Annealing.
    Updates best if a lower energy is found.
    """
    P = P_init.copy()
    n = P.shape[0]
    T = t_start
    cooling_factor = (t_end / t_start) ** (1.0 / n_steps) if t_start > 0 else 1.0
    
    # Initial step size estimate
    step_size = 0.05
    
    for i in range(n_steps):
        if time.time() > deadline - 0.1:
            break
            
        # Calculate gradient
        grad = _morse_grad(P, rho)
        E = _morse_E(P, rho)
        
        # Update best
        if best[0] is None or E < best[0]:
            best[0] = E
            best[1] = P.copy()
            
        # Langevin step
        # F = -grad
        # v_{t+1} = v_t * gamma + F * dt + sqrt(2 * k_B * T * dt / m) * noise
        # Here we use a simplified position-based Langevin (overdamped limit mostly, but with some inertia)
        # Let's use a standard Langevin form:
        # P_{t+1} = P_t - grad * dt + sqrt(2 * T * dt) * noise
        
        noise = np.random.normal(0, 1, P.shape)
        P = P - grad * step_size + np.sqrt(2 * T * step_size) * noise
        
        # Adaptive step size: if energy increases too much, reduce step
        E_new = _morse_E(P, rho)
        if E_new > E + 5.0: # Too big a jump
            step_size *= 0.8
            # Revert or dampen? For simplicity, just reduce step size and continue
            # Or revert to P_old and retry with smaller step?
            # Let's just reduce step size.
        else:
            # If energy decreased or stayed similar, maybe increase step slightly
            if E_new < E:
                step_size = min(step_size * 1.05, 0.5)
            else:
                step_size = max(step_size * 0.95, 0.01)
                
        # Cool down
        T *= cooling_factor
        
    return best


def _run_sa_batch(cands, rho, deadline, best):
    """Run Langevin SA for multiple candidates in parallel."""
    pool = _get_pool()
    futs = []
    for c in cands:
        # Each SA run gets a portion of the time? No, they run concurrently.
        # We need to pass the deadline.
        # How many steps? Depends on time.
        # Let's estimate steps based on time remaining.
        # Actually, it's better to run a fixed number of steps and check time inside.
        # Let's say 5000 steps per trajectory.
        futs.append(pool.submit(_langevin_sa, c, rho, 10.0, 0.01, 5000, deadline, best))
        
    for f in futs:
        try:
            f.result()
        except Exception:
            pass
    return best


def _make_seeds_langevin(n, rng):
    """
    Generate diverse initial conditions for Langevin SA.
    1. Ideal gas (uniform random in a sphere)
    2. Structured but noisy (FCC/BCC with high noise)
    """
    base = n ** (1.0 / 3.0)
    R = base * 1.2
    
    seeds = []
    # 1. Uniform random in sphere (Ideal gas)
    for _ in range(3):
        P = rng.uniform(-R, R, size=(n, 3))
        # Ensure inside sphere
        dists = np.linalg.norm(P, axis=1)
        mask = dists > R
        while mask.any():
            P[mask] = rng.uniform(-R, R, size=(mask.sum(), 3))
            dists = np.linalg.norm(P, axis=1)
            mask = dists > R
        seeds.append(P)
        
    # 2. Noisy FCC
    a = 1.0
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                pts.append([i * a, j * a, k * a])
                pts.append([(i + 0.5) * a, (j + 0.5) * a, k * a])
                pts.append([(i + 0.5) * a, j * a, (k + 0.5) * a])
                pts.append([i * a, (j + 0.5) * a, (k + 0.5) * a])
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order][:n]
    seeds.append(pts + rng.normal(0, 0.5, pts.shape))
    
    # 3. Noisy BCC
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                pts.append([i * a, j * a, k * a])
                pts.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order][:n]
    seeds.append(pts + rng.normal(0, 0.5, pts.shape))

    return seeds


def construct(instance):
    _init_timer()
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size * 1000003 + 17)

    call_start = time.time()
    deadline = min(_HARD_DEADLINE, call_start + 40.0)
    best = [None, None]  # [energy, positions]

    # Phase 1: Langevin SA from diverse seeds
    seeds = _make_seeds_langevin(size, rng)
    _run_sa_batch(seeds, rho, deadline, best)

    if best[1] is None:
        # Fallback
        P = rng.uniform(-size**(1/3), size**(1/3), size=(size, 3))
        best = [_morse_E(P, rho), P]

    # Phase 2: Polishing and further SA from the best found
    # Run more SA iterations starting from the current best
    # This helps escape shallow local minima if the first pass got stuck
    # We can run a second batch of SA starting from the best found so far
    # with slightly lower initial temperature to focus on refinement
    if time.time() < deadline - 2.0:
        cands = [best[1].copy() + rng.normal(0, 0.2, best[1].shape)]
        # Add a few more perturbations
        for _ in range(3):
            cands.append(best[1].copy() + rng.normal(0, 0.5, best[1].shape))
        
        # Run SA with lower T_start
        pool = _get_pool()
        futs = [pool.submit(_langevin_sa, c, rho, 5.0, 0.01, 3000, deadline, best) for c in cands]
        for f in futs:
            try:
                f.result()
            except Exception:
                pass

    # Final polish: BFGS on the best
    if time.time() < _HARD_DEADLINE - 1.0:
        E, P = _local_opt(best[1], rho, max_iter=600)
        if E is not None and E < best[0]:
            best[0], best[1] = E, P

    P = _enforce_min_dist(best[1], 0.5)
    return P
# EVOLVE-BLOCK-END
