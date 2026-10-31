import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _polar_to_cartesian(polar):
    """Convert (angles, radii) to Cartesian points.
    polar shape: (2N,) -> angles[0:N], radii[N:2N]
    """
    angles = polar[:N]
    radii = polar[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _polar_objective(polar):
    """Objective for polar-parameterized DE: minimize negative score."""
    points = _polar_to_cartesian(polar)
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _smooth_objective(x_flat, p=50):
    """Smooth surrogate using log-sum-exp based soft-min for better gradients."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _coordinate_descent_polish(points, deadline, max_iter=5000):
    """Coordinate descent on the hard objective: move each point's x and y independently."""
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                    trial = best_pts.copy()
                    trial[i, coord] += delta
                    trial_s = _score(trial)
                    if trial_s > best_s + 1e-14:
                        best_pts = trial
                        best_s = trial_s
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_pts, best_s

def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=80),
                best_x,
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 2:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
            if cd_s > best_s:
                best_s = cd_s
                best_x = cd_pts.flatten().copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _simulated_annealing(points, deadline):
    """Simulated annealing in Cartesian space with adaptive step size."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_s = current_score
    
    time_budget = deadline - 3.0
    total_time = max(0.5, time_budget - time.time())
    est_iters = int(total_time * 800)
    est_iters = max(500, min(est_iters, 60000))
    
    T0 = 0.03
    T_end = 0.00005
    alpha = (T_end / T0) ** (1.0 / max(1, est_iters - 1))
    
    rng = np.random.default_rng(42)
    
    accepted = 0
    total = 0
    step_scale = 0.5
    
    for it in range(est_iters):
        if time.time() > time_budget:
            break
        
        T = T0 * (alpha ** it)
        noise = rng.normal(0, T * step_scale, size=(N, 2))
        candidate = current + noise
        candidate = np.clip(candidate, -5, 5)
        
        cand_score = _score(candidate)
        total += 1
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            accepted += 1
            if current_score > best_s:
                best_s = current_score
                best = current.copy()
        else:
            delta = current_score - cand_score
            if delta < 1e-12:
                current = candidate
                current_score = cand_score
                accepted += 1
            else:
                prob = np.exp(-delta / max(T, 1e-10))
                if rng.random() < prob:
                    current = candidate
                    current_score = cand_score
                    accepted += 1
        
        if total % 500 == 0 and total > 0:
            acc_rate = accepted / total
            if acc_rate < 0.15:
                step_scale *= 0.7
            elif acc_rate > 0.4:
                step_scale *= 1.2
            step_scale = np.clip(step_scale, 0.05, 3.0)
            accepted = 0
            total = 0
    
    return best, best_s

def _fourier_convex_objective(params):
    """
    Objective for Fourier-series parameterized convex polygon.
    params: [r0, a1, b1, a2, b2] where the radius at angle theta is:
      r(theta) = r0 + a1*cos(theta) + b1*sin(theta) + a2*cos(2*theta) + b2*sin(2*theta)
    The 13 points are placed at evenly spaced angles.
    """
    r0, a1, b1, a2, b2 = params
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = r0 + a1 * np.cos(angles) + b1 * np.sin(angles) + a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
    # Ensure all radii are positive (convexity requires r > 0)
    if np.min(r) < 0.05:
        return 1e10
    points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _ellipse_objective(params):
    """
    Objective for ellipse-parameterized points.
    params: [a, b, angle_offsets...] where a,b are semi-axes and angle offsets adjust positions.
    Actually: [a, b, phase] for ellipse x = a*cos(t+phase), y = b*sin(t+phase)
    with 13 evenly spaced t values.
    """
    a, b, phase = params
    if a < 0.1 or b < 0.1:
        return 1e10
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + phase
    points = np.column_stack([a * np.cos(angles), b * np.sin(angles)])
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _elliptic_ring_objective(params):
    """
    Two-ring on ellipse: inner ring on smaller ellipse, outer on larger.
    params: [a_out, b_out, a_in, b_in, phase_out, phase_in, n_out]
    n_out is integer number of outer points (rest inner).
    """
    a_out, b_out, a_in, b_in, phase_out, phase_in = params[:6]
    n_out = int(round(params[6]))
    n_out = max(3, min(N - 3, n_out))
    n_in = N - n_out
    
    if a_out < 0.1 or b_out < 0.1 or a_in < 0.01 or b_in < 0.01:
        return 1e10
    
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False) + phase_out
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + phase_in
    
    pts_out = np.column_stack([a_out * np.cos(angles_out), b_out * np.sin(angles_out)])
    pts_in = np.column_stack([a_in * np.cos(angles_in), b_in * np.sin(angles_in)])
    points = np.vstack([pts_out, pts_in])
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # === Phase 1a: Fourier convex polygon optimization (low-dim global search) ===
    if time.time() < deadline - 50:
        try:
            bounds_fourier = [(0.3, 2.0), (-0.8, 0.8), (-0.8, 0.8), (-0.5, 0.5), (-0.5, 0.5)]
            de_fourier = differential_evolution(
                _fourier_convex_objective,
                bounds=bounds_fourier,
                seed=42,
                maxiter=200,
                popsize=40,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                updating='deferred'
            )
            if de_fourier.fun < 1e9:
                r0, a1, b1, a2, b2 = de_fourier.x
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                r = r0 + a1 * np.cos(angles) + b1 * np.sin(angles) + a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
                pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
                pts = pts - np.mean(pts, axis=0)
                _update_best(pts)
        except Exception:
            pass
    
    # === Phase 1b: Ellipse optimization ===
    if time.time() < deadline - 45:
        try:
            bounds_ellipse = [(0.5, 2.5), (0.5, 2.5), (0, 2 * np.pi)]
            de_ellipse = differential_evolution(
                _ellipse_objective,
                bounds=bounds_ellipse,
                seed=123,
                maxiter=150,
                popsize=30,
                tol=1e-8,
                polish=False,
                updating='deferred'
            )
            if de_ellipse.fun < 1e9:
                a, b, phase = de_ellipse.x
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + phase
                pts = np.column_stack([a * np.cos(angles), b * np.sin(angles)])
                pts = pts - np.mean(pts, axis=0)
                _update_best(pts)
        except Exception:
            pass
    
    # === Phase 1c: Elliptic two-ring optimization ===
    if time.time() < deadline - 40:
        try:
            bounds_ring = [(0.8, 2.5), (0.8, 2.5), (0.2, 1.2), (0.2, 1.2), (0, 2*np.pi), (0, 2*np.pi), (4, 9)]
            de_ring = differential_evolution(
                _elliptic_ring_objective,
                bounds=bounds_ring,
                seed=456,
                maxiter=100,
                popsize=30,
                tol=1e-7,
                polish=False,
                updating='deferred'
            )
            if de_ring.fun < 1e9:
                a_out, b_out, a_in, b_in, phase_out, phase_in = de_ring.x[:6]
                n_out = int(round(de_ring.x[6]))
                n_out = max(3, min(N - 3, n_out))
                n_in = N - n_out
                angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False) + phase_out
                angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + phase_in
                pts_out = np.column_stack([a_out * np.cos(angles_out), b_out * np.sin(angles_out)])
                pts_in = np.column_stack([a_in * np.cos(angles_in), b_in * np.sin(angles_in)])
                pts = np.vstack([pts_out, pts_in])
                pts = pts - np.mean(pts, axis=0)
                _update_best(pts)
        except Exception:
            pass
    
    # === Phase 1d: Polar DE (original) ===
    if time.time() < deadline - 30:
        bounds = [(0, 2 * np.pi)] * N + [(0.05, 3.0)] * N
        try:
            de_result = differential_evolution(
                _polar_objective,
                bounds=bounds,
                seed=42,
                maxiter=80,
                popsize=25,
                tol=1e-7,
                mutation=(0.4, 1.2),
                recombination=0.8,
                polish=False,
                updating='deferred'
            )
            if de_result.success or de_result.fun < 1e9:
                de_points = _polar_to_cartesian(de_result.x)
                de_points = de_points - np.mean(de_points, axis=0)
                _update_best(de_points)
        except Exception:
            pass
    
    # === Phase 2: Generate diverse initial configurations ===
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Elliptical variants
    for a_ratio in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        for phase in [0, np.pi/13, np.pi/6]:
            ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + phase
            pts = np.column_stack([a_ratio * np.cos(ang), np.sin(ang)])
            inits.append(pts)
    
    # 3. Two-ring configurations
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        if n_in < 3:
            continue
        for r_in in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts)
    
    # 4. Three-ring configurations
    for n1, n2, n3 in [(5, 4, 4), (6, 4, 3), (7, 3, 3), (6, 5, 2)]:
        if n1 + n2 + n3 != N:
            continue
        for r2 in [0.4, 0.5, 0.6]:
            for r3 in [0.15, 0.25, 0.35]:
                a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n1
                a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n2
                pts = np.vstack([
                    np.column_stack([np.cos(a1), np.sin(a1)]),
                    np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
                    np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
                ])
                inits.append(pts)
    
    # 5. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(15):
        r = np.sqrt(rng.random(N)) * 1.5
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)
    
    # 6. Hexagonal lattice-based starts
    center = np.array([[0.0, 0.0]])
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring2 = np.column_stack([2.0 * np.cos(angles2), 2.0 * np.sin(angles2)])
    inits.append(np.vstack([center, ring1, ring2]))
    
    rot = np.pi / 6
    ring1r = np.column_stack([np.cos(angles1 + rot), np.sin(angles1 + rot)])
    ring2r = np.column_stack([2.0 * np.cos(angles2 + rot), 2.0 * np.sin(angles2 + rot)])
    inits.append(np.vstack([center, ring1r, ring2r]))
    
    # 7. If we found good points, add their neighborhood
    if best_points is not None:
        rng_de = np.random.default_rng(777)
        for _ in range(5):
            perturbed = best_points + rng_de.normal(0, 0.1, size=(N, 2))
            inits.append(perturbed)
    
    # Shuffle inits
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # === Phase 3: Local optimization from each start ===
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        x_opt, s_opt = _local_polish(x0, deadline)
        _update_best(x_opt)
        
        if i % 15 == 0:
            elapsed = time.time() - start_time
            print(f"Init {i}/{len(inits)}: best_score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # === Phase 4: Simulated Annealing ===
    if best_points is not None and time.time() < deadline - 8:
        try:
            sa_best, sa_score = _simulated_annealing(best_points, deadline)
            _update_best(sa_best)
        except Exception:
            pass
    
    # === Phase 5: Final high-precision polish ===
    if best_points is not None and time.time() < deadline - 3:
        try:
            rng_final = np.random.default_rng(888)
            for attempt in range(5):
                if time.time() > deadline - 1:
                    break
                if attempt == 0:
                    start_x = best_points.flatten()
                else:
                    start_x = best_points.flatten() + rng_final.normal(0, 0.005, size=N*2)
                
                res_final = minimize(
                    _objective_full,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                _update_best(res_final.x.reshape(N, 2))
                
                if time.time() < deadline - 1:
                    cd_pts, cd_s = _coordinate_descent_polish(best_points, deadline)
                    _update_best(cd_pts)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
