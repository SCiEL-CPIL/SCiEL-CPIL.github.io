"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np

_DEADLINE = 150.0
_MARGIN = 10.0  # safety margin in seconds


def _radius(pts):
    """Return the feasible common radius for a set of points (side-1 square, centred)."""
    pts = np.asarray(pts)
    # edge distance: min over i,k of (1/2 - |c_ik|)
    edge = 0.5 - np.max(np.abs(pts), axis=1)
    r_edge = float(np.min(edge))
    # pair distance: min over i<j of |c_i - c_j| / 2
    # compute pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    # exclude self
    np.fill_diagonal(d, np.inf)
    r_pair = float(np.min(d)) / 2.0
    return min(r_edge, r_pair)


def _soft_radius(pts, beta):
    """Smooth surrogate: soft-min of edge and pair distances, scaled."""
    pts = np.asarray(pts)
    edge = 0.5 - np.max(np.abs(pts), axis=1)
    # soft-min of edge terms
    e_min = -np.log(np.sum(np.exp(-beta * edge))) / beta
    # pair distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    d_min = -np.log(np.sum(np.exp(-beta * d))) / beta
    r_pair = d_min / 2.0
    return min(e_min, r_pair)


def _initial_guess():
    """Square grid initialisation (valid baseline)."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(float)


def _clamp(pts):
    """Project points back into the square (hard constraint)."""
    pts = np.clip(pts, -0.5, 0.5)
    return pts


def _repulsive_forces(pts, strength, exponent):
    """Hard-core-like repulsive force: for pairs closer than a threshold, push apart."""
    # threshold ~ current radius estimate * 1.1
    r = _radius(pts)
    thr = max(1e-6, r * 1.15)
    forces = np.zeros_like(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1e-9)
    # only pairs within threshold contribute
    mask = d < thr
    # force magnitude ~ (1 - d/thr)^exponent
    mags = np.zeros_like(d)
    mags[mask] = (1.0 - d[mask] / thr) ** exponent
    # direction: from j to i is diff[i,j]
    # force on i = sum over j of mags[i,j] * diff[i,j]/d[i,j]
    unit = diff / d[:, :, None]
    f = np.sum(mags[:, :, None] * unit, axis=1)
    return strength * f


def _anneal_schedule(total_time, n_iters):
    """Return arrays of temperature and repulsion exponent over iterations."""
    temps = np.linspace(0.05, 0.0005, n_iters)
    temps = temps ** 1.5  # faster early cooling
    exponents = np.linspace(2.0, 6.0, n_iters)
    return temps, exponents


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN

    best_pts = _initial_guess()
    best_r = _radius(best_pts)

    rng = np.random.default_rng(12345)

    # Total iterations budget: we'll run as many as fit in time.
    # Estimate: each iteration ~ O(N^2) with N=50 -> ~2500 ops, very fast.
    # Use a fixed number of iterations, but check deadline.
    n_iters = 20000

    temps, exponents = _anneal_schedule(_DEADLINE, n_iters)

    pts = best_pts.copy()
    # Start with a slightly perturbed grid to break symmetry
    pts = pts + rng.normal(0, 0.005, size=pts.shape)
    pts = _clamp(pts)

    for it in range(n_iters):
        if time.time() > deadline:
            break

        T = temps[it]
        exp = exponents[it]

        # Repulsive force to keep circles apart
        F = _repulsive_forces(pts, strength=1.0, exponent=exp)

        # Gradient-free stochastic move: Langevin-style
        # Move = force + random noise scaled by temperature
        move = F + rng.normal(0, T, size=pts.shape)
        # Limit step size to avoid huge jumps
        step = np.linalg.norm(move, axis=1, keepdims=True)
        max_step = 0.05
        move = move * (max_step / (step + 1e-9))

        new_pts = _clamp(pts + move)

        # Accept if better, or with Metropolis probability
        r_new = _radius(new_pts)
        r_old = _radius(pts)
        if r_new >= r_old:
            pts = new_pts
            if r_new > best_r:
                best_r = r_new
                best_pts = new_pts.copy()
        else:
            # Metropolis acceptance
            delta = r_new - r_old  # negative
            if rng.random() < np.exp(delta / (T + 1e-9)):
                pts = new_pts

    # Polish: local refinement using a smooth objective (soft-min) with a
    # simple gradient-free coordinate-descent / Nelder-Mead-like step.
    if time.time() < deadline:
        # Use scipy if available for a final polish
        try:
            from scipy.optimize import minimize
            from scipy.spatial.distance import cdist

            def obj(x):
                p = x.reshape(N, 2)
                p = np.clip(p, -0.5, 0.5)
                # maximize min radius -> minimize negative
                edge = 0.5 - np.max(np.abs(p), axis=1)
                d = cdist(p, p)
                np.fill_diagonal(d, np.inf)
                r = min(float(np.min(edge)), float(np.min(d)) / 2.0)
                return -r

            x0 = best_pts.reshape(-1)
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                    'adaptive': True})
            p = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
            r_pol = _radius(p)
            if r_pol > best_r:
                best_r = r_pol
                best_pts = p
        except Exception:
            pass

    # Final safety: ensure finite and inside
    if not np.all(np.isfinite(best_pts)):
        best_pts = _initial_guess()
    best_pts = np.clip(best_pts, -0.5, 0.5)

    return best_pts
# EVOLVE-BLOCK-END
