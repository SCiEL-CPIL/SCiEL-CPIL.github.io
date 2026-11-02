import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 140.0  # Safety margin before the 150s kill

# Pre-compute triple indices
_ti = []
_tj = []
_tk = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _ti.append(_i)
            _tj.append(_j)
            _tk.append(_k)
_ti = np.array(_ti, dtype=np.int32)
_tj = np.array(_tj, dtype=np.int32)
_tk = np.array(_tk, dtype=np.int32)
_num_triples = len(_ti)

def _hull_area(pts):
    """Calculate convex hull area. Returns 0.0 on failure."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _triangle_areas(pts):
    """Calculate areas of all 286 triangles. Vectorized."""
    x = pts[:, 0]
    y = pts[:, 1]
    # Cross product magnitude / 2
    cross = np.abs(
        (x[_tj] - x[_ti]) * (y[_tk] - y[_ti])
        - (y[_tj] - y[_ti]) * (x[_tk] - x[_ti])
    )
    return 0.5 * cross

def _hard_score(pts):
    """Exact score: min triangle area / hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    
    h_area = _hull_area(pts)
    if h_area < 1e-12:
        return 0.0
        
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
        
    return float(min_area / h_area)

def _soft_score(pts, beta):
    """
    Soft score using Log-Sum-Exp to approximate the minimum.
    We want to maximize min(A_i) / H.
    This is equivalent to minimizing -min(A_i)/H.
    
    A common trick is to optimize log(min(A_i)) - log(H).
    log(min(A_i)) = - (1/beta) * log(sum(exp(-beta * A_i)))
    
    So objective to minimize:
    f = (1/beta) * log(sum(exp(-beta * A_i))) + log(H)
    """
    pts = np.asarray(pts, dtype=np.float64)
    h_area = _hull_area(pts)
    if h_area < 1e-12:
        return 1e10 # Large penalty
    
    areas = _triangle_areas(pts)
    
    # Shift for numerical stability in log-sum-exp
    # We are computing log(sum(exp(-beta * A_i)))
    # Let m = min(A_i). Then exp(-beta * A_i) = exp(-beta * (A_i - m)) * exp(-beta * m)
    # sum(exp(-beta * A_i)) = exp(-beta * m) * sum(exp(-beta * (A_i - m)))
    # log(sum(exp(-beta * A_i))) = -beta * m + log(sum(exp(-beta * (A_i - m))))
    # (1/beta) * log(sum(exp(-beta * A_i))) = -m + (1/beta) * log(sum(exp(-beta * (A_i - m))))
    
    m = np.min(areas)
    # If beta is large, only the min term matters.
    # exp(-beta * (A_i - m)) is <= 1.
    shifted_areas = -beta * (areas - m)
    # Clip to avoid overflow/underflow issues, though exp of negative is safe.
    # sum(exp(shifted_areas))
    log_sum = np.log(np.sum(np.exp(shifted_areas)))
    
    soft_min_term = -m + (1.0 / beta) * log_sum
    
    # We want to maximize soft_min / H
    # So minimize -soft_min / H
    # Or minimize log(H) - log(soft_min)
    # Let's just use the direct ratio for simplicity in SA, 
    # but the log-sum-exp is for the min part.
    
    return -soft_min_term / h_area

def _neg_hard_score(params):
    """For scipy minimization."""
    pts = params.reshape(N, 2)
    return -_hard_score(pts)

def _sa_run(rng, t_start, t_end, T_start, T_end, steps, beta_start, beta_end):
    """
    Run Simulated Annealing.
    Returns best points and score.
    """
    # Initial point: Random in unit square, or regular polygon
    # Regular polygon is a good start
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Randomize slightly to break symmetry
    pts += rng.normal(0, 0.05, size=(N, 2))
    
    current_score = _hard_score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    # Pre-allocate
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Beta schedule: increase beta to make the soft min sharper over time
    # T schedule: decrease temperature
    
    for step in range(steps):
        # Check time
        if time.time() > t_end:
            break
            
        # Update parameters
        progress = step / steps
        T = T_start * (T_end / T_start) ** progress
        beta = beta_start * (beta_end / beta_start) ** progress
        
        # Perturb
        # Step size should scale with T or be fixed?
        # Usually step size decreases as T decreases.
        step_size = 0.1 * (1.0 - 0.5 * progress) 
        delta = rng.normal(0, step_size, size=(N, 2))
        
        new_pts = pts + delta
        
        # Calculate new soft score
        new_soft_score = _soft_score(new_pts, beta)
        
        # We are minimizing -score. 
        # Let's work with "energy" = -soft_score.
        # Current energy
        current_soft_score = _soft_score(pts, beta)
        
        delta_e = new_soft_score - current_soft_score # This is change in -Score? 
        # Wait. _soft_score returns -soft_min/H. 
        # So higher _soft_score is better.
        # Let's define Energy = -_soft_score.
        # E_new = -new_soft_score
        # E_curr = -current_soft_score
        # dE = E_new - E_curr = -(new_soft_score - current_soft_score)
        
        dE = -(new_soft_score - current_soft_score)
        
        # Metropolis acceptance
        if dE < 0 or (T > 1e-10 and rng.random() < np.exp(-dE / T)):
            pts = new_pts
            # Update current score for hard metric tracking? 
            # We use soft for acceptance, but track hard for "best".
            # Calculating hard score every step is expensive.
            # We can just check hard score every K steps or at the end.
            
            # For "best" tracking, we need the hard score.
            # Let's calculate hard score periodically or if soft score is very good.
            # To save time, let's just update best_pts based on soft score?
            # No, soft score is an approximation. 
            # Let's calculate hard score every 10 steps or if dE < 0 (improvement in soft).
            if dE < 0:
                h_score = _hard_score(pts)
                if h_score > best_score:
                    best_score = h_score
                    best_pts = pts.copy()
        else:
            # Reject move, pts remains unchanged
            pass
            
    # Final hard score check for the last accepted state
    final_score = _hard_score(pts)
    if final_score > best_score:
        best_score = final_score
        best_pts = pts.copy()
        
    return best_pts, best_score

def _local_polish(pts, rng, max_iter=500):
    """
    Refine the points using Nelder-Mead on the hard objective.
    """
    try:
        res = minimize(
            _neg_hard_score, pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10,
                     'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)
    
    best_pts = None
    best_score = -1e18
    
    def _update(pts):
        nonlocal best_pts, best_score
        if pts is None or pts.shape != (N, 2):
            return
        s = _hard_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Initial guess: Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    _update(x0)
    
    # SA Parameters
    # We will run multiple SA runs with different random seeds/starts
    # and different beta schedules.
    
    # Time budget allocation
    # Reserve 10s for final polishing.
    sa_deadline = t0 + (DEADLINE - 10.0)
    
    num_sa_runs = 10
    steps_per_run = 5000
    
    for run in range(num_sa_runs):
        if time.time() > sa_deadline:
            break
            
        # Diversify start points
        if run == 0:
            start_pts = x0.copy()
        elif run == 1:
            # Two rings
            r1 = 0.5
            r2 = 1.0
            ang1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
            ang2 = np.linspace(0, 2*np.pi, 8, endpoint=False)
            p1 = np.column_stack([r1*np.cos(ang1), r1*np.sin(ang1)])
            p2 = np.column_stack([r2*np.cos(ang2), r2*np.sin(ang2)])
            start_pts = np.vstack([p1, p2])
        else:
            # Random start in unit disk
            r = np.sqrt(rng.uniform(0, 1, N))
            a = rng.uniform(0, 2*np.pi, N)
            start_pts = np.column_stack([r*np.cos(a), r*np.sin(a)])
        
        start_pts += rng.normal(0, 0.1, size=(N, 2))
        
        # SA Parameters
        # T_start high enough to explore, T_end low enough to converge
        T_start = 0.5
        T_end = 1e-5
        beta_start = 10.0
        beta_end = 1000.0
        
        p, s = _sa_run(rng, time.time(), sa_deadline, T_start, T_end, steps_per_run, beta_start, beta_end)
        _update(p)
        
        # Local polish on the best found so far
        if time.time() < sa_deadline - 2.0:
            polished = _local_polish(best_pts, rng, max_iter=200)
            _update(polished)

    # Final intensive polish if time permits
    if best_pts is not None and time.time() < DEADLINE - 2.0:
        # Try L-BFGS-B as well, it might be faster for smooth parts
        try:
            res = minimize(
                _neg_hard_score, best_pts.flatten(),
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (2 * N),
                options={'maxiter': 100, 'maxfun': 5000}
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass
        
        # Final NM
        polished = _local_polish(best_pts, rng, max_iter=500)
        _update(polished)

    if best_pts is None:
        return x0
        
    return best_pts
