"""
Strategy: Complex-polynomial discriminant optimization + dual annealing + coordinate polish.
1. Represent the 8 points as the roots of a monic complex polynomial P(z) = prod (z - z_i).
2. The product of all 28 pairwise distances squared equals |Disc(P)|, so maximizing
   log|Disc(P)| - 56*log(soft-max modulus of roots) is a smooth (mostly) 16-D problem.
3. Use scipy.optimize.dual_annealing (global stochastic search, structurally different
   from the current L-BFGS-B + MDS + DE + Nelder-Mead pipeline) to find a good coefficient
   vector, then extract roots and polish in coordinate space.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, dual_annealing

_START_TIME = time.time()
_DEADLINE = 135.0


def _log_score(points):
    """Compute log of Delta/n^n where Delta is product of distances over ordered pairs,
    after scaling to diameter 2."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    if np.any(valid_dists <= 0):
        return -np.inf
    diameter = np.max(dists)
    if diameter <= 0:
        return -np.inf
    n_pairs = n * (n - 1)
    log_delta = n_pairs * np.log(2.0) + np.sum(np.log(valid_dists)) - n_pairs * np.log(diameter)
    log_score = log_delta - n * np.log(n)
    return log_score


def _neg_log_score(params):
    """Negative log score for coordinate-space minimization."""
    if time.time() > _DEADLINE:
        return 1e10
    pts = params.reshape(8, 2)
    for i in range(8):
        for j in range(i + 1, 8):
            d = math.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
            if d < 1e-10:
                return 1e10
    ls = _log_score(pts)
    if not np.isfinite(ls):
        return 1e10
    return -ls


def _regular_octagon():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _poly_objective(coef):
    """
    Negative objective in coefficient space.
    coef: 16-element real array representing 8 complex coefficients c_7..c_0
    of P(z) = c_7 z^7 + c_6 z^6 + ... + c_0 (monic, so c_7 is fixed to 1; we
    actually use 15 free coefficients for c_6..c_0, but to keep it simple we
    use 16 and ignore the first / fix it).
    
    We maximize log|Disc(P)| - 56 * log(softmax_modulus).
    """
    if time.time() > _DEADLINE:
        return 1e10

    # Reconstruct complex coefficients: c_k for k=7..0
    # We store 16 reals: [Re(c7), Im(c7), Re(c6), Im(c6), ..., Re(c0), Im(c0)]
    # But P is monic, so c7 = 1. We'll fix c7 and only use 15 free params.
    # Actually, let's use 15 free coefficients for c6..c0 (30 reals).
    # To match the 16-element input, let's reinterpret: use 16 reals as
    # [Re(c6), Im(c6), Re(c5), Im(c5), ..., Re(c0), Im(c0)] -> 14 reals for c6..c0.
    # That's 7*2 = 14. Let's use 14 free params.

    # Let me restructure: use 14 real params for c6..c0.
    if len(coef) < 14:
        return 1e10

    # Build complex coefficients c7=1, c6..c0 from 14 reals
    c = [1.0 + 0.0j]  # c7
    for k in range(6, -1, -1):
        idx = (6 - k) * 2
        re = coef[idx]
        im = coef[idx + 1]
        c.append(complex(re, im))
    # c is now [c7, c6, c5, c4, c3, c2, c1, c0]
    c = np.array(c, dtype=np.complex128)

    # Find roots
    try:
        roots = np.roots(c)
    except Exception:
        return 1e10

    if np.any(np.abs(roots) > 1e8):
        return 1e10
    if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return 1e10

    # Check distinctness
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(roots[i] - roots[j]) < 1e-12:
                return 1e10

    # Compute discriminant via resultant of P and P'
    # P'(z) coefficients: [7*c7, 6*c6, ..., 1*c1]
    cp = np.array([k * c[k] for k in range(7, -1, -1)], dtype=np.complex128)
    # Remove leading zeros from cp
    while len(cp) > 0 and abs(cp[0]) < 1e-15:
        cp = cp[1:]
    if len(cp) == 0:
        return 1e10

    # Resultant via Sylvester matrix
    deg_p = 8
    deg_cp = len(cp) - 1
    size = deg_p + deg_cp
    syl = np.zeros((size, size), dtype=np.complex128)
    # Fill rows for P shifted
    for i in range(deg_cp):
        syl[i, i:i + deg_p + 1] = c
    # Fill rows for P' shifted
    for i in range(deg_p):
        syl[deg_cp + i, i:i + deg_cp + 1] = cp

    try:
        det_val = np.linalg.det(syl)
    except Exception:
        return 1e10

    disc = det_val  # For monic polynomial, Disc(P) = (-1)^(n(n-1)/2) * Res(P, P') / a_n
    # For monic (a_n=1), Disc(P) = (-1)^(n(n-1)/2) * Res(P, P')
    # n=8, n(n-1)/2 = 28, so (-1)^28 = 1
    # So Disc(P) = Res(P, P')
    abs_disc = abs(disc)
    if abs_disc < 1e-300:
        return 1e10

    log_disc = np.log(abs_disc)

    # Soft-max modulus of roots
    moduli = np.abs(roots)
    beta = 50.0
    shifted = beta * moduli
    shifted = np.clip(shifted, -500, 500)
    max_s = np.max(shifted)
    log_sum_exp = max_s + np.log(np.sum(np.exp(shifted - max_s)))
    smooth_max_mod = (max_s + log_sum_exp) / beta  # This is wrong; let me fix

    # Correct soft-max: logsumexp(beta * x) / beta
    smooth_max_mod = (max_s + np.log(np.sum(np.exp(shifted - max_s)))) / beta

    # We want max modulus <= 1 (after scaling, diameter = 2 means max distance between
    # any two points = 2, but the max modulus from origin is different. Let's use
    # the actual diameter constraint instead.)
    # Actually, the score scales to diameter 2, so the constraint is on the diameter,
    # not the max modulus. Let's compute the diameter from roots.
    diff = roots[:, np.newaxis] - roots[np.newaxis, :]
    dists = np.abs(diff)
    np.fill_diagonal(dists, 0)
    diameter = np.max(dists)
    if diameter < 1e-10:
        return 1e10

    # Smooth diameter penalty: logsumexp(beta * (dists - 2)) but we want dists <= 2
    # Actually, the score normalizes by diameter, so we just need to maximize
    # log_disc - 56 * log(diameter)
    # But that's not smooth. Let's use the soft-max on squared distances.
    sq_dists = dists ** 2
    mask = ~np.eye(8, dtype=bool)
    valid_sq = sq_dists[mask]
    beta_d = 30.0
    shifted_d = beta_d * valid_sq
    shifted_d = np.clip(shifted_d, -500, 500)
    max_sd = np.max(shifted_d)
    log_sum_exp_d = max_sd + np.log(np.sum(np.exp(shifted_d - max_sd)))
    smooth_max_sq_diam = (max_sd + log_sum_exp_d) / beta_d
    smooth_diam = np.sqrt(max(smooth_max_sq_diam, 1e-10))

    objective = -log_disc + 56.0 * np.log(smooth_diam)
    if not np.isfinite(objective):
        return 1e10
    return objective


def _coeffs_to_points(coef):
    """Convert 14 real coefficients (c6..c0) to 8 points."""
    c = [1.0 + 0.0j]
    for k in range(6, -1, -1):
        idx = (6 - k) * 2
        re = coef[idx]
        im = coef[idx + 1]
        c.append(complex(re, im))
    c = np.array(c, dtype=np.complex128)
    try:
        roots = np.roots(c)
    except Exception:
        return _regular_octagon()
    if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return _regular_octagon()
    # Check distinctness
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(roots[i] - roots[j]) < 1e-10:
                return _regular_octagon()
    pts = np.column_stack([np.real(roots), np.imag(roots)])
    return pts


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 135.0

    best_points = _regular_octagon()
    best_log_score = _log_score(best_points)

    rng = np.random.default_rng(42)

    # === Stage 1: Generate initial coefficient vectors ===
    init_coeffs_list = []

    # From regular octagon
    oct_pts = _regular_octagon()
    oct_roots = oct_pts[:, 0] + 1j * oct_pts[:, 1]
    # Compute coefficients from roots
    c_oct = np.poly(oct_roots)
    # c_oct has 9 elements: [1, c7, c6, ..., c0]
    # We need c6..c0 (7 complex = 14 reals)
    init_coeffs_list.append(np.concatenate([np.real(c_oct[1:]), np.imag(c_oct[1:])]))

    # From two concentric squares
    for r1 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        for r2 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
            if abs(r1 - r2) < 0.2:
                continue
            for rot in [0.0, np.pi / 8, np.pi / 4, np.pi / 6]:
                pts = []
                for k in range(4):
                    a = rot + 2 * np.pi * k / 4
                    pts.append([r1 * np.cos(a), r1 * np.sin(a)])
                for k in range(4):
                    a = rot + np.pi / 4 + 2 * np.pi * k / 4
                    pts.append([r2 * np.cos(a), r2 * np.sin(a)])
                pts = np.array(pts)
                roots = pts[:, 0] + 1j * pts[:, 1]
                try:
                    c = np.poly(roots)
                    init_coeffs_list.append(np.concatenate([np.real(c[1:]), np.imag(c[1:])]))
                except Exception:
                    pass

    # Random coefficient vectors
    for _ in range(30):
        init_coeffs_list.append(rng.uniform(-2.0, 2.0, 14))

    # === Stage 2: Dual annealing on top initial candidates ===
    bounds = [(-3.0, 3.0)] * 14

    # Score initial candidates to pick the best few
    scored_inits = []
    for init_c in init_coeffs_list:
        try:
            obj = _poly_objective(init_c)
            if np.isfinite(obj) and obj < 1e10:
                scored_inits.append((obj, init_c))
        except Exception:
            pass

    scored_inits.sort(key=lambda x: x[0])

    best_poly_obj = np.inf
    best_poly_coefs = None

    for obj, init_c in scored_inits[:10]:
        if time.time() > _DEADLINE - 40:
            break
        try:
            result = dual_annealing(
                _poly_objective,
                x0=init_c,
                bounds=bounds,
                maxiter=100,
                seed=42,
                initial_temp=5.0,
                restart_temp_ratio=2e-5,
                visit=[1.0, 1.0, 1.0, 1.0],
                accept=[1.0, 1.0, 1.0, 1.0],
                maxiter_no_improvement=20,
                polish=True,
                vectorized=False
            )
            if result.fun < 1e10 and result.fun < best_poly_obj:
                best_poly_obj = result.fun
                best_poly_coefs = result.x.copy()
        except Exception:
            pass

    # === Stage 3: Convert best polynomial roots to points and polish ===
    candidate_points_list = []

    if best_poly_coefs is not None:
        pts_from_poly = _coeffs_to_points(best_poly_coefs)
        candidate_points_list.append(pts_from_poly)

    # Also convert a few other good polynomial solutions
    for obj, init_c in scored_inits[:5]:
        if time.time() > _DEADLINE - 35:
            break
        try:
            # Do a quick local polish in coefficient space
            res = minimize(
                _poly_objective,
                init_c,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < 1e10:
                pts_from_poly = _coeffs_to_points(res.x)
                candidate_points_list.append(pts_from_poly)
        except Exception:
            pass

    # Add regular octagon and random points
    candidate_points_list.append(_regular_octagon())
    for _ in range(10):
        radii = rng.uniform(0.3, 1.8, 8)
        angles = rng.uniform(0, 2 * np.pi, 8)
        candidate_points_list.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))

    # Antipodal pairs
    for _ in range(8):
        base_angles = rng.uniform(0, np.pi / 2, 4)
        radii = rng.uniform(0.5, 1.5, 4)
        pts = []
        for i in range(4):
            a = base_angles[i]
            r = radii[i]
            pts.append([r * np.cos(a), r * np.sin(a)])
            pts.append([-r * np.cos(a), -r * np.sin(a)])
        candidate_points_list.append(np.array(pts))

    # Score and sort
    scored_candidates = []
    for cand in candidate_points_list:
        if time.time() > _DEADLINE - 30:
            break
        try:
            ls = _log_score(cand)
            if np.isfinite(ls):
                scored_candidates.append((ls, cand))
        except Exception:
            pass

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Update best
    for ls, cand in scored_candidates:
        if ls > best_log_score:
            best_log_score = ls
            best_points = cand

    # === Stage 4: Coordinate-space polish ===
    # 4a: Nelder-Mead on top candidates
    if time.time() < _DEADLINE - 15:
        for ls, cand in scored_candidates[:10]:
            if time.time() > _DEADLINE - 10:
                break
            x0 = cand.flatten()
            try:
                res = minimize(_neg_log_score, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < 1e10:
                    opt_pts = res.x.reshape(8, 2)
                    ls_val = _log_score(opt_pts)
                    if np.isfinite(ls_val) and ls_val > best_log_score:
                        best_log_score = ls_val
                        best_points = opt_pts
            except Exception:
                pass

    # 4b: Smooth coordinate-space optimization with increasing beta
    if time.time() < _DEADLINE - 8:
        def _neg_smooth_coord(params, beta):
            if time.time() > _DEADLINE:
                return 1e10
            pts = params.reshape(8, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            mask = ~np.eye(8, dtype=bool)
            valid_dists = dists[mask]
            if np.any(valid_dists <= 0):
                return 1e10
            max_d = np.max(valid_dists)
            if max_d <= 0:
                return 1e10
            shifted = beta * (valid_dists - max_d)
            shifted = np.clip(shifted, -500, 500)
            max_s = np.max(shifted)
            log_sum_exp = max_s + np.log(np.sum(np.exp(shifted - max_s)))
            smooth_log_diam = max_d + log_sum_exp / beta
            n_pairs = 56
            log_delta = n_pairs * np.log(2.0) + np.sum(np.log(valid_dists)) - n_pairs * smooth_log_diam
            log_score = log_delta - 8 * np.log(8)
            if not np.isfinite(log_score):
                return 1e10
            return -log_score

        current = best_points.copy()
        for beta in [50.0, 200.0, 500.0, 1000.0]:
            if time.time() > _DEADLINE - 5:
                break
            try:
                res = minimize(lambda x, b=beta: _neg_smooth_coord(x, b), current.flatten(),
                               method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < 1e10:
                    refined = res.x.reshape(8, 2)
                    ls_val = _log_score(refined)
                    if np.isfinite(ls_val) and ls_val > best_log_score:
                        best_log_score = ls_val
                        best_points = refined
                        current = refined.copy()
            except Exception:
                pass

    # 4c: Final Nelder-Mead polish
    if time.time() < _DEADLINE - 3:
        try:
            res = minimize(_neg_log_score, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < 1e10:
                refined = res.x.reshape(8, 2)
                ls_val = _log_score(refined)
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
                    best_points = refined
        except Exception:
            pass

    # Ensure coordinates are within bounds
    final_result = []
    for p in best_points:
        x = max(-1e6, min(1e6, float(p[0])))
        y = max(-1e6, min(1e6, float(p[1])))
        final_result.append([x, y])

    return final_result
