import numpy as np
import time

N = 300


def _score(pts):
    # Largest common radius: min over pairs (dist/2) and boundary (1 - |p|)
    if pts is None or pts.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    bmin = float(boundary.min())
    if bmin < 0.0:
        return bmin
    # Pairwise distances via broadcasting
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    pmin = float(d[iu].min()) / 2.0
    return min(pmin, bmin)


def _hex_init(n):
    # Build a hexagonal lattice, keep points inside the unit disk with a small margin,
    # take the n closest to the centre.
    best = None
    for trial in range(60):
        spacing = 2.0 / (np.sqrt(n) + 1.0) * (0.985 ** trial)
        k = int(1.0 / spacing) + 3
        xs = (np.arange(-k, k + 1)) * spacing
        js = np.arange(-k, k + 1)
        X, J = np.meshgrid(xs, js, indexing="ij")
        Y = (J + 0.5 * ((J % 2) + 2) % 2) * spacing * np.sqrt(3.0) / 2.0
        # proper offset: rows shifted by 0.5*spacing when j odd
        Y = J * spacing * np.sqrt(3.0) / 2.0
        X = X + (spacing / 2.0) * ((J % 2) + 2) % 2
        P = np.column_stack([X.ravel(), Y.ravel()])
        rad = np.hypot(P[:, 0], P[:, 1])
        keep = rad <= 1.0 - spacing / 2.0 - 1e-9
        Pk = P[keep]
        if len(Pk) >= n:
            Pk = Pk[np.argsort(np.hypot(Pk[:, 0], Pk[:, 1]))[:n]]
            if best is None or len(Pk) >= n:
                best = Pk
        if best is not None and len(best) >= n and spacing < 2.0 / (np.sqrt(n) + 1.0) * 0.6:
            break
    if best is None or len(best) < n:
        # Fallback: uniform random inside disk
        theta = np.random.rand(n) * 2 * np.pi
        rho = np.sqrt(np.random.rand(n)) * 0.98
        best = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
    return best[:n].astype(float)


def _soft_min(vals, beta):
    # Smooth approximation of min via -log-sum-exp / beta
    m = vals.min()
    return m - np.log(np.sum(np.exp(-beta * (vals - m)))) / beta


def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin

    pts = _hex_init(N)
    best_pts = pts.copy()
    best_r = _score(pts)

    # Precompute static structure
    iu = np.triu_indices(N, k=1)
    pair_rows = iu[0]
    pair_cols = iu[1]

    # ---- Stage 1: gradient-free (Nelder-Mead style) with soft-min objective ----
    # We optimise a smooth surrogate: f = -min_pair_dist/2 blended with boundary term.
    # Use a simple projected-gradient approach on a soft penalty.

    def objective(P, beta=50.0):
        # Returns a scalar to minimise; lower = better packing
        boundary = 1.0 - np.linalg.norm(P, axis=1)
        bmin = boundary.min()
        # Pairwise
        d = np.sqrt(np.sum((P[pair_rows] - P[pair_cols]) ** 2, axis=1))
        pmin = d.min() / 2.0
        # We want to maximise min(pmin, bmin). Use a soft-min of the two "radii"
        # but with a strong penalty if any constraint violated.
        # Surrogate: soft_min of all pairwise half-dists and boundary terms
        # Build vector of all "radius candidates"
        # To keep memory low, compute soft-min of d/2 values and boundary values separately
        dm = d / 2.0
        sm_d = _soft_min(dm, beta)
        sm_b = _soft_min(boundary, beta)
        # Combine with another soft-min (outer)
        sm = _soft_min(np.array([sm_d, sm_b]), beta)
        # Penalty for any hard violation (push to keep feasible)
        viol = max(0.0, -sm_d, -sm_b)
        return -sm + 10.0 * viol

    # Random restarts + local optimisation via scipy
    try:
        from scipy.optimize import minimize
        have_scipy = True
    except Exception:
        have_scipy = False

    beta_schedule = [20, 50, 200, 500, 2000, 20000, 100000]
    rng = np.random.default_rng(12345)

    current = pts.copy()
    cur_r = best_r

    # Anneal beta and do local minimisation
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        if have_scipy:
            x0 = current.ravel()

            def f_vec(x):
                P = x.reshape(N, 2)
                return objective(P, beta=beta)

            # Use a few restarts
            for restart in range(3):
                if time.time() > deadline - 20.0:
                    break
                if restart == 0:
                    x_init = current.ravel()
                else:
                    # Perturb current best
                    x_init = current.ravel() + rng.normal(0, 0.02, size=N * 2)
                    x_init = x_init.reshape(N, 2)
                    # Pull back inside
                    norms = np.linalg.norm(x_init, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-9)
                    x_init = x_init * np.minimum(1.0, 0.99 / norms)
                    x_init = x_init.ravel()
                try:
                    res = minimize(f_vec, x_init, method="BFGS",
                                   options={"maxiter": 400, "gtol": 1e-8, "factr": 1e-6})
                    Pnew = res.x.reshape(N, 2)
                    # Project to feasibility lightly
                    norms = np.linalg.norm(Pnew, axis=1, keepdims=True)
                    mask = norms > 0.999
                    Pnew[mask] = Pnew[mask] * (0.999 / norms[mask])
                    rnew = _score(Pnew)
                    if rnew > cur_r:
                        cur_r = rnew
                        current = Pnew
                        if rnew > best_r:
                            best_r = rnew
                            best_pts = Pnew.copy()
                except Exception:
                    pass
        # After a beta stage, if improved, keep; otherwise keep previous

    # ---- Stage 2: High-precision polish using exact min (nonsmooth) with L-BFGS on soft ----
    if have_scipy and time.time() < deadline - 15.0:
        for beta in [5000, 20000, 100000, 300000]:
            if time.time() > deadline - 15.0:
                break
            x0 = best_pts.ravel()

            def f_vec(x):
                P = x.reshape(N, 2)
                return objective(P, beta=beta)

            try:
                res = minimize(f_vec, x0, method="L-BFGS-B",
                               options={"maxiter": 2000, "gtol": 1e-10, "maxls": 50})
                Pnew = res.x.reshape(N, 2)
                norms = np.linalg.norm(Pnew, axis=1, keepdims=True)
                mask = norms > 0.9999
                Pnew[mask] = Pnew[mask] * (0.9999 / norms[mask])
                rnew = _score(Pnew)
                if rnew > best_r:
                    best_r = rnew
                    best_pts = Pnew.copy()
            except Exception:
                pass

    # ---- Stage 3: Pairwise repulsion local search (exact objective) ----
    # Simple: iteratively move each point away from its nearest neighbour / boundary.
    if time.time() < deadline - 10.0:
        P = best_pts.copy()
        for it in range(20000):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 10.0:
                    break
                xi = P[i].copy()
                # find nearest other point
                d = np.sqrt(np.sum((P - xi) ** 2, axis=1))
                d[i] = np.inf
                j = int(np.argmin(d))
                dist_j = d[j]
                bound = 1.0 - np.linalg.norm(xi)
                # Determine the limiting constraint
                half = dist_j / 2.0
                # Move away from j by a small step if close, or toward centre if boundary limiting
                if half < bound:
                    # move away from j
                    direction = xi - P[j]
                    nrm = np.linalg.norm(direction)
                    if nrm > 1e-12:
                        direction = direction / nrm
                    step = 0.5 * max(0.0, 0.02 - half)
                    cand = xi + direction * step
                else:
                    # move toward centre
                    nrm = np.linalg.norm(xi)
                    if nrm > 1e-12:
                        direction = -xi / nrm
                    else:
                        direction = np.array([0.01, 0.0])
                    step = 0.5 * max(0.0, bound - 0.02)
                    cand = xi + direction * step
                # keep inside
                cnorm = np.linalg.norm(cand)
                if cnorm > 0.9999:
                    cand = cand * (0.9999 / cnorm)
                # Accept if local min radius improves
                r_old_local = min(half, bound)
                # compute new local
                d_new = np.sqrt(np.sum((P - cand) ** 2, axis=1))
                d_new[i] = np.inf
                half_new = d_new.min() / 2.0
                bound_new = 1.0 - np.linalg.norm(cand)
                r_new_local = min(half_new, bound_new)
                if r_new_local > r_old_local + 1e-12:
                    P[i] = cand
                    improved = True
            if not improved:
                break
        rnew = _score(P)
        if rnew > best_r:
            best_r = rnew
            best_pts = P.copy()

    # Final sanity: ensure exactly N points, finite, shape correct
    if best_pts is None or best_pts.shape != (N, 2) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    # Hard project inside unit disk
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    mask = norms > 0.99999
    if np.any(mask):
        best_pts[mask] = best_pts[mask] * (0.99999 / norms[mask])

    return best_pts
