"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 13
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

# Precompute index arrays for vectorized area computation
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])

def _tri_areas(pts):
    """Vectorized triangle areas for all 286 triples."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _score(pts):
    """Exact score: min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return float(np.min(_tri_areas(pts)) / ha)

def _neg_soft_min_ratio(pts, temp):
    """
    Smooth surrogate to maximize: -soft_min(areas) / hull_area.
    soft_min(a) = -T * log(sum(exp(-a/T)))  -> approximates min(a) as T->0.
    """
    pts = np.asarray(pts, dtype=np.float64)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e8
    a = _tri_areas(pts)
    a = np.maximum(a, 1e-15)
    s = -a / temp
    s = s - np.max(s)
    logsum = np.log(np.sum(np.exp(s)))
    soft_min = -logsum * temp
    return soft_min / ha

def _angular_to_cartesian(radii, angles):
    """Convert polar coordinates to Cartesian."""
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _cartesian_to_angular(pts):
    """Convert Cartesian to polar coordinates."""
    radii = np.linalg.norm(pts, axis=1)
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    angles = angles % (2 * np.pi)
    return radii, angles

def _evaluate_angular(radii, angles):
    """Evaluate score given polar parameters."""
    pts = _angular_to_cartesian(radii, angles)
    return _score(pts)

def _refine_cartesian(pts, time_budget):
    """
    Refine a Cartesian point set using L-BFGS on a smooth surrogate.
    This acts as the local polish step.
    """
    t0 = time.time()
    best = pts.copy()
    best_s = _score(pts)
    
    # Use a moderately small temperature for the soft-min to guide L-BFGS
    # towards the exact min without being too noisy.
    temp = 0.005
    
    def obj(x):
        p = x.reshape(N, 2)
        return _neg_soft_min_ratio(p, temp)
    
    try:
        res = minimize(obj, best.flatten(), method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s = s
            best = cand.copy()
    except Exception:
        pass
        
    # Second pass with tighter temperature if time permits
    if time.time() - t0 < time_budget - 1:
        temp2 = 0.001
        def obj2(x):
            p = x.reshape(N, 2)
            return _neg_soft_min_ratio(p, temp2)
        try:
            res = minimize(obj2, best.flatten(), method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
            
    return best, best_s

def _differential_evolution_angular(r0, a0, time_budget, rng):
    """
    Global search in angular (radius, angle) space using Differential Evolution.
    Parameters:
    r0: initial radii (N,)
    a0: initial angles (N,)
    time_budget: seconds to spend
    rng: random state
    """
    t0 = time.time()
    deadline = t0 + time_budget
    
    # Bounds for radii and angles
    r_min, r_max = 0.0, 1.5  # Allow points slightly outside unit circle if needed, 
                             # but hull area normalizes it. Actually, keeping them 
                             # within a reasonable bound helps. 
                             # For unit area hull, points should be roughly in [0, 1.5].
    a_min, a_max = 0.0, 2 * np.pi
    
    # Population size
    pop_size = 20
    
    # Initialize population
    # Mix of initial point and random perturbations
    pop_r = np.zeros((pop_size, N))
    pop_a = np.zeros((pop_size, N))
    
    pop_r[0] = r0
    pop_a[0] = a0
    
    for i in range(1, pop_size):
        # Random radii in [0.1, 1.2]
        pop_r[i] = rng.uniform(0.1, 1.2, N)
        # Random angles
        pop_a[i] = rng.uniform(0, 2 * np.pi, N)
        
    # Evaluate initial population
    pop_scores = np.zeros(pop_size)
    for i in range(pop_size):
        pop_scores[i] = _evaluate_angular(pop_r[i], pop_a[i])
        
    best_idx = np.argmax(pop_scores)
    best_r = pop_r[best_idx].copy()
    best_a = pop_a[best_idx].copy()
    best_score = pop_scores[best_idx]
    
    # DE Parameters
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover probability
    
    # Main DE Loop
    while time.time() < deadline - 5:  # Leave 5s for final polish
        new_pop_r = np.zeros_like(pop_r)
        new_pop_a = np.zeros_like(pop_a)
        new_scores = np.zeros(pop_size)
        
        for i in range(pop_size):
            # Select distinct random individuals a, b, c
            idxs = rng.choice(pop_size, 3, replace=False)
            a_idx, b_idx, c_idx = idxs
            
            # Mutation: v = a + F * (b - c)
            v_r = pop_r[a_idx] + F * (pop_r[b_idx] - pop_r[c_idx])
            v_a = pop_a[a_idx] + F * (pop_a[b_idx] - pop_a[c_idx])
            
            # Handle angle wrapping for mutation
            # Simple approach: just let it be, arctan2 in eval handles it, 
            # but for DE bounds, we should wrap.
            v_a = v_a % (2 * np.pi)
            
            # Crossover: binomial crossover
            j_rand = rng.randint(0, N)
            trial_r = np.zeros(N)
            trial_a = np.zeros(N)
            
            for j in range(N):
                if rng.random() < CR or j == j_rand:
                    trial_r[j] = v_r[j]
                    trial_a[j] = v_a[j]
                else:
                    trial_r[j] = pop_r[i][j]
                    trial_a[j] = pop_a[i][j]
                    
            # Bounds check (Radii must be positive)
            trial_r = np.clip(trial_r, r_min, r_max)
            
            # Evaluate trial
            trial_score = _evaluate_angular(trial_r, trial_a)
            
            # Selection
            if trial_score >= pop_scores[i]:
                new_pop_r[i] = trial_r
                new_pop_a[i] = trial_a
                new_scores[i] = trial_score
            else:
                new_pop_r[i] = pop_r[i]
                new_pop_a[i] = pop_a[i]
                new_scores[i] = pop_scores[i]
                
        # Update population
        pop_r = new_pop_r
        pop_a = new_pop_a
        pop_scores = new_scores
        
        # Update best
        current_best_idx = np.argmax(pop_scores)
        if pop_scores[current_best_idx] > best_score:
            best_score = pop_scores[current_best_idx]
            best_r = pop_r[current_best_idx].copy()
            best_a = pop_a[current_best_idx].copy()
            
    # Final Polish: Convert best angular to Cartesian and refine
    best_pts = _angular_to_cartesian(best_r, best_a)
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        best_pts, best_score = _refine_cartesian(best_pts, remaining_time)
        
    return best_pts, best_score

def construct():
    """Main entry: Multi-start Differential Evolution in Angular Space."""
    t0 = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    rng = np.random.RandomState(20240517)
    
    # Initial points: Regular polygon
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r_init = np.ones(N)
    init_pts = np.column_stack([r_init * np.cos(th), r_init * np.sin(th)])
    
    best_pts = init_pts.copy()
    best_score = _score(init_pts)
    
    # Run DE with a significant portion of the time
    # We run one main DE campaign. If time allows, we could run a second one
    # from a different start, but DE is global so one good campaign is usually enough.
    
    time_for_de = deadline - (time.time() - t0) - 10  # Leave 10s for safety/final checks
    
    if time_for_de > 20:
        try:
            pts, s = _differential_evolution_angular(r_init, th, time_for_de, rng)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Final polish on the absolute best found
    remaining = deadline - (time.time() - t0) - 2
    if remaining > 5:
        try:
            pts, s = _refine_cartesian(best_pts, remaining)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
            
    return best_pts
