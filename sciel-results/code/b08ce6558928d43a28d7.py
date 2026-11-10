"""Combined program: structural seeding (FCC/BCC) + robust relaxation with time guards."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_DEADLINE = 310.0  # safety margin before 330 s kill


def _remaining():
    return _DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Compute exact Morse energy: sum exp(rho(1-r))(exp(rho(1-r))-2) for all pairs."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    x = np.exp(rho * (1.0 - r))
    e = x * (x - 2.0)
    iu = np.triu_indices(n, k=1)
    return float(np.sum(e[iu]))


def _morse_grad(pos, rho):
    """Compute gradient of Morse energy."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    r2 = (diff * diff).sum(-1)
    r = np.sqrt(r2)
    np.fill_diagonal(r, np.inf)
    r2[np.eye(n, dtype=bool)] = 1.0  # avoid division by zero
    x = np.exp(rho * (1.0 - r))
    # dE/dr = 2 * rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1)
    # dE/dpos_i = sum_j dE/dr_ij * (pos_i - pos_j) / r_ij
    coeff = 2.0 * rho * x * (x - 1.0) / r
    coeff[np.eye(n, dtype=bool)] = 0.0
    grad = (coeff[:, :, np.newaxis] * diff).sum(1)
    return grad


def _fcc_cluster(n, a=1.0):
    """Build a compact FCC cluster of exactly n atoms."""
    pts = []
    maxcoord = int(3.0 * n ** (1.0 / 3.0) + 6.0)
    for i in range(-maxcoord, maxcoord + 1):
        for j in range(-maxcoord, maxcoord + 1):
            for k in range(-maxcoord, maxcoord + 1):
                r2 = i * i + j * j + k * k
                if r2 > maxcoord * maxcoord:
                    continue
                # FCC: all lattice points where i+j+k is even
                if (i + j + k) % 2 == 0:
                    pts.append([i * a / 2.0, j * a / 2.0, k * a / 2.0])
    pts = np.array(pts, dtype=float)
    d2 = (pts ** 2).sum(1)
    order = np.argsort(d2)
    return pts[order[:n]].copy()


def _bcc_cluster(n, a=1.0):
    """Build a compact BCC cluster of exactly n atoms."""
    pts = []
    maxcoord = int(3.0 * n ** (1.0 / 3.0) + 6.0)
    for i in range(-maxcoord, maxcoord + 1):
        for j in range(-maxcoord, maxcoord + 1):
            for k in range(-maxcoord, maxcoord + 1):
                r2 = i * i + j * j + k * k
                if r2 > maxcoord * maxcoord:
                    continue
                # BCC: corners and body centers
                if (i + j + k) % 2 == 0:
                    pts.append([i * a / 2.0, j * a / 2.0, k * a / 2.0])
                else:
                    # body center of cube [i-1,i]x[j-1,j]x[k-1,k] if i,j,k all odd
                    if i % 2 != 0 and j % 2 != 0 and k % 2 != 0:
                        pts.append([(i - 1) * a / 2.0, (j - 1) * a / 2.0, (k - 1) * a / 2.0])
    pts = np.array(pts, dtype=float)
    d2 = (pts ** 2).sum(1)
    order = np.argsort(d2)
    return pts[order[:n]].copy()


def _random_cluster(n, rng, spacing=1.0):
    """Random compact cluster with minimum separation."""
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    pts = []
    minsep = 0.55
    maxtries = 100000
    t = 0
    while len(pts) < n and t < maxtries:
        t += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= minsep for q in pts):
            pts.append(p)
    if len(pts) < n:
        # pad with a jittered FCC
        base = _fcc_cluster(n)
        base = base + rng.normal(0, 0.05, base.shape)
        return base
    return np.array(pts, dtype=float)


def _lbfgs_refine(pos, rho, max_iter=3000):
    """Direct L-BFGS-B refinement with time guard."""
    pos = np.asarray(pos, dtype=float).copy()
    t0 = time.time()
    timeout = 25.0  # max seconds per refinement

    def fe(x):
        if time.time() - t0 > timeout:
            return float("inf")
        e = _morse_energy(x, rho)
        if not np.isfinite(e):
            return 1e9
        return e

    def ge(x):
        if time.time() - t0 > timeout:
            return np.zeros_like(x)
        return _morse_grad(x, rho)

    try:
        res = minimize(fe, pos, jac=ge,
                       method="L-BFGS-B",
                       options={"maxiter": max_iter, "maxls": 40, "ftol": 1e-10})
        if res.x is not None and np.isfinite(_morse_energy(res.x, rho)):
            return np.asarray(res.x, dtype=float)
    except Exception:
        pass
    return pos


def _check_valid(pos, minsep=0.5):
    """Check that all pairs are at least minsep apart and positions are finite."""
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d2 = (diff * diff).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min()) >= minsep - 1e-6 and np.all(np.isfinite(pos))


def construct(instance):
    """Build compact clusters (FCC, BCC, random) and relax them to local minima."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10.0))

    candidates = []

    # 1) FCC (primary, structure-derived)
    fcc = _fcc_cluster(n, a=1.0)
    candidates.append(("fcc", fcc))

    # 2) BCC (fallback structure)
    bcc = _bcc_cluster(n, a=1.0)
    candidates.append(("bcc", bcc))

    # 3) Random compact seeds
    rng = np.random.default_rng(n)
    for k in range(2):
        rc = _random_cluster(n, rng)
        candidates.append(("rand%d" % k, rc))

    best = None
    best_e = float("inf")

    # First pass: use chem_clusters.relax on each candidate
    for name, seed in candidates:
        if _remaining() < 15.0:
            break
        try:
            e, pos = relax(np.asarray(seed, dtype=float), potential, rho)
            e = float(e)
            if np.isfinite(e) and _check_valid(pos):
                if e < best_e:
                    best_e = e
                    best = np.asarray(pos, dtype=float)
        except Exception:
            pass

    # Second pass: L-BFGS-B refinement on the best candidates for further improvement
    if _remaining() > 40.0:
        # Refine the top 2 candidates
        refined = []
        for name, seed in candidates[:2]:
            if _remaining() < 30.0:
                break
            try:
                ref = _lbfgs_refine(np.asarray(seed, dtype=float), rho)
                e = _morse_energy(ref, rho)
                if np.isfinite(e) and _check_valid(ref):
                    refined.append((e, ref))
            except Exception:
                pass

        for e, ref in refined:
            if e < best_e:
                best_e = e
                best = ref

    # Final safety: if we have a valid best, return it
    if best is not None and _check_valid(best):
        return best

    # Last-resort fallback: return the FCC seed (should be valid)
    fcc = _fcc_cluster(n, a=1.0)
    return fcc
# EVOLVE-BLOCK-END
