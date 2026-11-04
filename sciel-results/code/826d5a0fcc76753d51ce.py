import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)
])

def _compute_areas(points):
    """Compute all triangle areas efficiently using vectorized cross products."""
    p = points
    # C[i,j] = p[i][0]*p[j][1] - p[i][1]*p[j][0]
    C = p[:, 0:1] * p[:, 1:2].T - p[:, 1:2] * p[:, 0:1].T
    i_idx = TRIPLE_INDICES[:, 0]
    j_idx = TRIPLE_INDICES[:, 1]
    k_idx = TRIPLE_INDICES[:, 2]
    vals = C[j_idx, k_idx] + C[i_idx, j_idx] - C[i_idx, k_idx]
    return 0.5 * np.abs(vals)

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, temperature=0.015):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    s = np.sum(np.exp(shifted / temperature))
    soft_min = m - temperature * np.log(s)
    return soft_min / hull_area

def _two_temp_soft_min(points, temp1=0.008, temp2=0.04):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    s1 = np.sum(np.exp(shifted / temp1))
    soft1 = m - temp1 * np.log(s1)
    s2 = np.sum(np.exp(shifted / temp2))
    soft2 = m - temp2 * np.log(s2)
    return (soft1 * soft2) ** 0.5 / hull_area

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + phi_rel
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    return pts_out

def _generate_pentagon_interior(n_out, n_in, r_in, phi_rel):
    """Pentagon on unit circle + interior points on a ring."""
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + phi_rel
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    return pts_out

def _differential_evolution(points, time_limit=20.0, pop_size=20):
    """Differential evolution global search over all 2N coordinates."""
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)
    rng = np.random.default_rng(42)

    # Initialize population
    pop = np.zeros((pop_size, N, 2))
    pop[0] = p
    for i in range(1, pop_size):
        # Mix: some near best, some random
        if i < pop_size // 2:
            pop[i] = p + rng.normal(0, 0.15, size=(N, 2))
        else:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.3, 0.3, size=N)
            radii = np.abs(rng.normal(0.7, 0.2, size=N))
            radii = np.clip(radii, 0.05, 1.3)
            pop[i] = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    scores = np.array([_calculate_score(pop[i]) for i in range(pop_size)])
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_points = pop[best_idx].copy()

    F = 0.8  # scale factor
    CR = 0.9  # crossover rate

    gen = 0
    while time.time() - start_time < time_limit:
        gen += 1
        # Adaptive parameters: Faster convergence
        F = max(0.2, 0.8 - 0.08 * gen)
        CR = min(0.95, 0.7 + 0.02 * gen)

        for i in range(pop_size):
            if time.time() - start_time > time_limit:
                break
            # Select three distinct random individuals
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(candidates, size=3, replace=False)

            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])

            # Crossover
            trial = pop[i].copy()
            j_rand = rng.integers(N * 2)
            for j in range(N * 2):
                if rng.random() < CR or j == j_rand:
                    trial.flat[j] = mutant.flat[j]

            # Evaluate
            s_trial = _calculate_score(trial)
            if s_trial > scores[i]:
                pop[i] = trial
                scores[i] = s_trial
                if s_trial > best_score:
                    best_score = s_trial
                    best_points = trial.copy()

        # Restart stagnating members
        for i in range(pop_size):
            if i == best_idx:
                continue
            if scores[i] < best_score * 0.85:
                if time.time() - start_time > time_limit:
                    break
                if rng.random() < 0.5:
                    pop[i] = best_points + rng.normal(0, 0.1, size=(N, 2))
                else:
                    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.2, 0.2, size=N)
                    radii = np.clip(rng.normal(0.7, 0.2, size=N), 0.05, 1.3)
                    pop[i] = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
                scores[i] = _calculate_score(pop[i])
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_points = pop[i].copy()

    return best_points, best_score

def _bottleneck_polish(points, time_limit=8.0):
    """Coordinate descent focusing on the worst-performing triangles."""
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    for iteration in range(6):
        if time.time() - start_time > time_limit:
            break
        areas = _compute_areas(p)
        min_area = np.min(areas)
        # Find indices of the worst triangles (below 1.2x min)
        threshold = min_area * 1.3
        bad_triples = TRIPLE_INDICES[areas < threshold]

        # Collect all point indices in bad triangles
        bad_pts = set()
        for t in bad_triples:
            bad_pts.add(t[0])
            bad_pts.add(t[1])
            bad_pts.add(t[2])

        # Optimize those points first, then others
        pt_list = sorted(bad_pts)
        remaining = [i for i in range(N) if i not in bad_pts]
        pt_list += remaining

        for i in pt_list:
            if time.time() - start_time > time_limit:
                break
            old_pt = p[i].copy()

            def obj_2d(x):
                p[i] = x
                return -_calculate_score(p)

            try:
                res = minimize(obj_2d, old_pt, method='Nelder-Mead',
                               options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 400, 'disp': False})
                p[i] = res.x
                s_new = _calculate_score(p)
                if s_new <= s - 1e-12:
                    p[i] = old_pt
            except Exception:
                p[i] = old_pt
        s = _calculate_score(p)

    return p, s

def _generate_three_ring(n1, n2, n3, r2, r3, phi2, phi3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    p1 = np.column_stack((np.cos(a1), np.sin(a1)))
    if n2 > 0:
        a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phi2
        p2 = np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2)))
    else:
        p2 = np.empty((0, 2))
    if n3 > 0:
        a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + phi3
        p3 = np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
    else:
        p3 = np.empty((0, 2))
    return np.vstack((p1, p2, p3))

def _optimize_two_ring(n_out, n_in, time_limit=8.0):
    start_time = time.time()
    best_score = 0.0
    best_params = None

    def objective(params):
        r_in, phi_rel = params
        if r_in < 0.0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        return -_calculate_score(pts)

    r_starts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    phi_starts = [0.0]
    if n_in > 0:
        phi_starts += [np.pi / n_in, np.pi / (2.0 * n_in), 2.0 * np.pi / n_in,
                       np.pi / (3.0 * n_in), np.pi / (4.0 * n_in)]
    if n_out > 0:
        phi_starts += [np.pi / n_out, np.pi / (2.0 * n_out)]

    for r0 in r_starts:
        for phi0 in phi_starts:
            if time.time() - start_time > time_limit:
                return best_score, best_params
            try:
                res = minimize(objective, [r0, phi0], method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500, 'disp': False})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    return best_score, best_params

def _optimize_three_ring(n1, n2, n3, time_limit=8.0):
    start_time = time.time()
    best_score = 0.0
    best_params = None

    def objective(params):
        r2, r3, phi2, phi3 = params
        if r2 < 0.0 or r2 > 1.5 or r3 < 0.0 or r3 > 1.5:
            return 1e6
        pts = _generate_three_ring(n1, n2, n3, r2, r3, phi2, phi3)
        return -_calculate_score(pts)

    r2_starts = [0.3, 0.5, 0.7, 0.9]
    r3_starts = [0.2, 0.4, 0.6, 0.8]
    phi2_starts = [0.0, np.pi / n2 if n2 > 0 else 0.0, np.pi / (2.0 * n2) if n2 > 0 else 0.0]
    phi3_starts = [0.0, np.pi / n3 if n3 > 0 else 0.0, np.pi / (2.0 * n3) if n3 > 0 else 0.0]

    count = 0
    for r2 in r2_starts:
        for r3 in r3_starts:
            for phi2 in phi2_starts:
                for phi3 in phi3_starts:
                    if time.time() - start_time > time_limit or count >= 18:
                        return best_score, best_params
                    count += 1
                    try:
                        res = minimize(objective, [r2, r3, phi2, phi3], method='Nelder-Mead',
                                       options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 400, 'disp': False})
                        if -res.fun > best_score:
                            best_score = -res.fun
                            best_params = res.x.copy()
                    except Exception:
                        pass
    return best_score, best_params

def _generate_single_ring(angles):
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _optimize_single_ring(time_limit=6.0):
    start_time = time.time()
    best_score = 0.0
    best_angles = None

    def score_from_angles(a):
        pts = _generate_single_ring(a)
        return _calculate_score(pts)

    def neg_score(a):
        return -score_from_angles(a)

    base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    s = score_from_angles(base)
    if s > best_score:
        best_score, best_angles = s, base.copy()

    rng = np.random.default_rng(7)
    for trial in range(6):
        if time.time() - start_time > time_limit:
            break
        a0 = base + rng.uniform(-0.15, 0.15, size=N)
        try:
            res = minimize(neg_score, a0, method='Nelder-Mead',
                           options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 600, 'disp': False})
            s = score_from_angles(res.x)
            if s > best_score:
                best_score, best_angles = s, res.x.copy()
        except Exception:
            pass
    return best_score, best_angles

def _coordinate_descent_polish(points, time_limit=5.0):
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    for iteration in range(4):
        if time.time() - start_time > time_limit:
            break
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            old_pt = p[i].copy()

            def obj_2d(x):
                p[i] = x
                return -_calculate_score(p)

            try:
                res = minimize(obj_2d, old_pt, method='Nelder-Mead',
                               options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 300, 'disp': False})
                p[i] = res.x
                s_new = _calculate_score(p)
                if s_new <= s - 1e-12:
                    p[i] = old_pt
            except Exception:
                p[i] = old_pt
        s = _calculate_score(p)
    return p, s

def _polish_points(points, time_limit=5.0):
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    def neg_two_temp(x, temp_scale=1.0):
        return -_two_temp_soft_min(x.reshape(N, 2), temp1=0.008 * temp_scale, temp2=0.04 * temp_scale)

    def neg_soft(x, temp_scale=1.0):
        return -_soft_min_score(x.reshape(N, 2), temperature=0.015 * temp_scale)

    def neg_hard(x):
        return -_calculate_score(x.reshape(N, 2))

    # Phase 1: High temperature (exploration)
    if time.time() < start_time + time_limit * 0.3:
        try:
            res = minimize(lambda x: neg_two_temp(x, 5.0), p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    # Phase 2: Medium temperature
    if time.time() < start_time + time_limit * 0.6:
        try:
            res = minimize(lambda x: neg_two_temp(x, 1.0), p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    # Phase 3: Low temperature (exploitation, close to hard min)
    if time.time() < start_time + time_limit * 0.8:
        try:
            res = minimize(lambda x: neg_soft(x, 0.5), p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    # Phase 4: Hard objective polish (Nelder-Mead on hard score)
    if time.time() < start_time + time_limit * 0.95:
        try:
            res = minimize(neg_hard, p.flatten(), method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-10, 'maxiter': 500, 'disp': False})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    return p, s

def _heptagon_radial_params_to_points(params):
    """Convert 10 params (7 radial offsets + 3 affine) to 14 points.
    
    The 14 points are placed on 7 radial axes (heptagon directions),
    with 2 points per axis at radii 1+offset_k and 1-offset_k.
    Affine: rotation, scale, shear.
    """
    offsets = params[:7]
    theta = params[7]
    scale = params[8]
    shear = params[9]
    
    base_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    points = np.zeros((14, 2))
    for k in range(7):
        r1 = 1.0 + offsets[k]
        r2 = 1.0 - offsets[k]
        ca = np.cos(base_angles[k])
        sa = np.sin(base_angles[k])
        # Original position before affine
        x1 = r1 * ca
        y1 = r1 * sa
        x2 = r2 * ca
        y2 = r2 * sa
        # Apply affine: rotate, scale, shear
        # Rotation: [x*cos_t - y*sin_t, x*sin_t + y*cos_t]
        # Scale: multiply by scale
        # Shear: x' = x + shear*y
        rx1 = x1 * cos_t - y1 * sin_t
        ry1 = x1 * sin_t + y1 * cos_t
        rx2 = x2 * cos_t - y2 * sin_t
        ry2 = x2 * sin_t + y2 * cos_t
        # Scale
        sx1 = rx1 * scale
        sy1 = ry1 * scale
        sx2 = rx2 * scale
        sy2 = ry2 * scale
        # Shear (x' = x + shear*y)
        points[2*k, 0] = sx1 + shear * sy1
        points[2*k, 1] = sy1
        points[2*k+1, 0] = sx2 + shear * sy2
        points[2*k+1, 1] = sy2
    return points

def _optimize_heptagon_radial(time_limit=12.0):
    """Optimize the 10-parameter heptagon-radial + affine model."""
    start_time = time.time()
    best_score = 0.0
    best_params = None
    
    def objective(params):
        pts = _heptagon_radial_params_to_points(params)
        return -_calculate_score(pts)
    
    # Multiple starting points
    rng = np.random.default_rng(99)
    n_starts = 8
    
    for trial in range(n_starts):
        if time.time() - start_time > time_limit:
            break
        
        if trial == 0:
            # Symmetric start: all offsets zero, identity affine
            p0 = np.zeros(10)
            p0[8] = 1.0  # scale = 1
        else:
            p0 = np.zeros(10)
            p0[:7] = rng.uniform(-0.2, 0.2, size=7)
            p0[7] = rng.uniform(-0.3, 0.3)
            p0[8] = rng.uniform(0.8, 1.2)
            p0[9] = rng.uniform(-0.1, 0.1)
        
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 800, 'disp': False})
            s = -res.fun
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
        except Exception:
            pass
    
    # If we found good params, do a second pass with tighter tolerance
    if best_params is not None and time.time() - start_time < time_limit * 0.8:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000, 'disp': False})
            s = -res.fun
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None:
        pts = _heptagon_radial_params_to_points(best_params)
        return best_score, pts
    return best_score, None

def _high_precision_polish(points, time_limit=8.0):
    """Final high-precision polish with very tight tolerances."""
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)
    
    # Phase 1: Very tight soft-min with L-BFGS-B
    def neg_soft_tight(x):
        return -_soft_min_score(x.reshape(N, 2), temperature=0.003)
    
    if time.time() - start_time < time_limit * 0.4:
        try:
            res = minimize(neg_soft_tight, p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass
    
    # Phase 2: Two-temp soft-min with very low temps
    def neg_two_temp_tight(x):
        return -_two_temp_soft_min(x.reshape(N, 2), temp1=0.002, temp2=0.01)
    
    if time.time() - start_time < time_limit * 0.7:
        try:
            res = minimize(neg_two_temp_tight, p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass
    
    # Phase 3: Hard Nelder-Mead with extreme precision
    def neg_hard(x):
        return -_calculate_score(x.reshape(N, 2))
    
    if time.time() - start_time < time_limit * 0.95:
        try:
            res = minimize(neg_hard, p.flatten(), method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000, 'disp': False})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass
    
    return p, s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    best_score = 0.0
    best_points = None

    def try_update(score, points):
        nonlocal best_score, best_points
        if points is not None and score > best_score:
            best_score = score
            best_points = points.copy()

    # 1. Single-ring angular-packing family
    if time.time() < deadline - 40:
        s, ang = _optimize_single_ring(time_limit=4.0)
        if ang is not None:
            pts = _generate_single_ring(ang)
            polished, ps = _polish_points(pts, time_limit=3.0)
            try_update(ps, polished)

    # 2. Two-ring sweep (reduced budget)
    splits = [(14, 0), (12, 2), (11, 3), (10, 4), (9, 5), (8, 6), (7, 7)]
    for n_out, n_in in splits:
        if time.time() > deadline - 40:
            break
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0.0, 0.0)
            polished, ps = _polish_points(pts, time_limit=2.0)
            try_update(ps, polished)
            continue
        tl = 4.0
        score, params = _optimize_two_ring(n_out, n_in, time_limit=tl)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            polished, ps = _polish_points(pts, time_limit=3.0)
            try_update(ps, polished)

    # 3. Pentagon + interior structural family (NEW)
    pent_splits = [(5, 9), (6, 8), (7, 7), (8, 6)]
    for n_out, n_in in pent_splits:
        if time.time() > deadline - 35:
            break
        # Optimize r_in and phi_rel for pentagon+interior
        def pent_obj(params):
            r_in, phi_rel = params
            if r_in < 0.0 or r_in > 1.5:
                return 1e6
            pts = _generate_pentagon_interior(n_out, n_in, r_in, phi_rel)
            return -_calculate_score(pts)

        r_starts = [0.2, 0.4, 0.6, 0.8, 1.0]
        phi_starts = [0.0, np.pi / n_in, np.pi / (2.0 * n_in)]
        local_best = 0.0
        local_params = None
        for r0 in r_starts:
            for phi0 in phi_starts:
                if time.time() > deadline - 35:
                    break
                try:
                    res = minimize(pent_obj, [r0, phi0], method='Nelder-Mead',
                                   options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 400, 'disp': False})
                    if -res.fun > local_best:
                        local_best = -res.fun
                        local_params = res.x.copy()
                except Exception:
                    pass
        if local_params is not None:
            pts = _generate_pentagon_interior(n_out, n_in, local_params[0], local_params[1])
            polished, ps = _polish_points(pts, time_limit=3.0)
            try_update(ps, polished)

    # 4. Three-ring exploration
    three_splits = [(10, 2, 2), (9, 3, 2), (8, 4, 2), (7, 4, 3)]
    for n1, n2, n3 in three_splits:
        if time.time() > deadline - 30:
            break
        score, params = _optimize_three_ring(n1, n2, n3, time_limit=4.0)
        if params is not None:
            pts = _generate_three_ring(n1, n2, n3, params[0], params[1], params[2], params[3])
            polished, ps = _polish_points(pts, time_limit=3.0)
            try_update(ps, polished)

    # 5. Heptagon-radial + affine parameterized family
    if time.time() < deadline - 35:
        hr_limit = min(12.0, deadline - time.time() - 25)
        if hr_limit > 5:
            hr_score, hr_pts = _optimize_heptagon_radial(time_limit=hr_limit)
            if hr_pts is not None:
                polished_hr, phr = _polish_points(hr_pts, time_limit=3.0)
                try_update(phr, polished_hr)
    
    # 6. DIFFERENTIAL EVOLUTION global search (NEW structural method)
    if best_points is not None and time.time() < deadline - 20:
        de_limit = min(25.0, deadline - time.time() - 10)
        if de_limit > 5:
            de_pts, de_score = _differential_evolution(best_points, time_limit=de_limit, pop_size=15)
            try_update(de_score, de_pts)

    # 7. Extra polish passes on the best result
    if best_points is not None and time.time() < deadline - 12:
        remaining = deadline - time.time() - 10
        if remaining > 3:
            polished, ps = _polish_points(best_points, time_limit=remaining)
            try_update(ps, polished)

    # 8. High-precision final polish
    if best_points is not None and time.time() < deadline - 6:
        remaining = deadline - time.time() - 3
        if remaining > 4:
            try:
                p_hp, s_hp = _high_precision_polish(best_points, time_limit=remaining)
                try_update(s_hp, p_hp)
            except Exception:
                pass

    # 9. Final bottleneck polish
    if best_points is not None and time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 2:
            try:
                p_b, s_b = _bottleneck_polish(best_points, time_limit=remaining * 0.5)
                try_update(s_b, p_b)
            except Exception:
                pass
            if time.time() < deadline - 2:
                remaining2 = deadline - time.time() - 1
                if remaining2 > 2:
                    try:
                        p2, s2 = _polish_points(best_points, time_limit=remaining2)
                        try_update(s2, p2)
                    except Exception:
                        pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
