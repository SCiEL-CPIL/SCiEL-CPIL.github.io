import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
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
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    # Handle numerical stability
    if np.any(shifted > 100 * temperature):
        # If some values are very far, exp(-shifted/temp) will be 0, which is fine
        pass
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    """Score using soft-min instead of hard min."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s

# ---------- Angular Representation Functions ----------

def _angles_to_points(angles):
    """Convert sorted angles to points on the unit circle."""
    # Ensure angles are in [0, 2*pi) and sorted
    angles = np.sort(angles)
    # Map to [0, 2*pi)
    angles = angles % (2 * np.pi)
    # Sort again after modulo just in case
    angles = np.sort(angles)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _score_from_angles(angles):
    """Calculate score directly from angles."""
    pts = _angles_to_points(angles)
    return _score(pts)

def _soft_score_from_angles(angles, temperature):
    """Calculate soft score directly from angles."""
    pts = _angles_to_points(angles)
    return _soft_score(pts, temperature)

def _neg_soft_score_from_angles(angles, temperature):
    s = _soft_score_from_angles(angles, temperature)
    if s <= 0:
        return 1e10
    return -s

# ---------- Seed Generators ----------

def _seed_regular():
    """Regular 13-gon."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])

def _seed_regular_angles():
    """Regular 13-gon angles."""
    return np.linspace(0, 2 * np.pi, N, endpoint=False)

def _seed_two_ring(n_out, r_inner, angle_offset):
    """Two-ring configuration."""
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])

# ---------- Optimization Routines ----------

def _optimize_angles(angles0, deadline, max_temp=0.05, min_temp=1e-5, n_steps=10):
    """
    Optimize the angular coordinates using annealed soft-min.
    """
    best_angles = angles0.copy()
    best_score = _score_from_angles(best_angles)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 5:
            break
        
        # L-BFGS-B on angles
        if time.time() < deadline - 3:
            try:
                def obj(x):
                    return _neg_soft_score_from_angles(x, temp)
                
                res = minimize(
                    obj, best_angles,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand_angles = res.x
                cand_pts = _angles_to_points(cand_angles)
                s = _score(cand_pts)
                if s > best_score:
                    best_score = s
                    best_angles = cand_angles.copy()
            except Exception:
                pass
        
        # Powell on angles
        if time.time() < deadline - 2:
            try:
                def obj2(x):
                    return _neg_soft_score_from_angles(x, temp)
                
                res = minimize(
                    obj2, best_angles,
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand_angles = res.x
                cand_pts = _angles_to_points(cand_angles)
                s = _score(cand_pts)
                if s > best_score:
                    best_score = s
                    best_angles = cand_angles.copy()
            except Exception:
                pass

    return _angles_to_points(best_angles), best_score

def _annealed_soft_minimize_cartesian(points, deadline, max_temp=0.1, min_temp=1e-5, n_steps=10):
    """
    Anneal the soft-min temperature and optimize Cartesian coordinates.
    """
    best = points.copy()
    best_s = _score(best)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 8:
            break
        
        # L-BFGS-B on Cartesian
        if time.time() < deadline - 6:
            try:
                def obj(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj, best.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        
        # Powell on Cartesian
        if time.time() < deadline - 4:
            try:
                def obj2(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj2, best.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
    
    return best, best_s

def _polish_true_objective(points, deadline):
    """Polish using the true (non-smooth) objective with Powell then Nelder-Mead."""
    best = points.copy()
    best_s = _score(best)
    
    # Powell
    if time.time() < deadline - 6:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Nelder-Mead fine polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1200, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    np.random.seed(42)
    
    best_points = None
    best_score = 0.0
    
    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # ---------- Phase 1: Angular Optimization ----------
    # Start with regular polygon angles
    angles0 = _seed_regular_angles()
    
    # Try a few perturbed angle seeds
    angle_seeds = [angles0]
    for _ in range(3):
        perturbed_angles = angles0 + 0.05 * np.random.randn(N)
        angle_seeds.append(perturbed_angles)
        
    for a_seed in angle_seeds:
        if time.time() > deadline - 30:
            break
        try:
            pts, s = _optimize_angles(a_seed, deadline - 20)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 2: Cartesian Optimization from Best Angular Result ----------
    if best_points is not None:
        if time.time() < deadline - 20:
            try:
                pts, s = _annealed_soft_minimize_cartesian(best_points, deadline - 10, max_temp=0.1, min_temp=1e-5, n_steps=8)
                _update(pts)
            except Exception:
                pass

    # ---------- Phase 3: Diverse Cartesian Seeds ----------
    # In case angular optimization got stuck, try some Cartesian seeds
    seeds = []
    seeds.append(_seed_regular())
    for n_out in [6, 7]:
        for r_in in [0.5, 0.7]:
            seeds.append(_seed_two_ring(n_out, r_in, 0.0))
            
    for seed in seeds:
        _update(seed)
        
    if best_points is not None:
        if time.time() < deadline - 10:
            try:
                pts, s = _annealed_soft_minimize_cartesian(best_points, deadline - 5, max_temp=0.05, min_temp=1e-6, n_steps=5)
                _update(pts)
            except Exception:
                pass

    # ---------- Phase 4: Final Polish ----------
    if best_points is not None:
        if time.time() < deadline - 5:
            try:
                pts, s = _polish_true_objective(best_points, deadline)
                _update(pts)
            except Exception:
                pass

    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
