import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution

N = 14
D = 3
DEADLINE = 145.0  # Leave 5s safety margin for 150s limit
_t0 = None

def _true_score(pts):
    """Calculate the squared ratio of max to min pairwise distance."""
    if pts.shape != (N, D):
        return np.inf
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _obj(x):
    """Objective function for differential evolution.
    We minimize the ratio (mx/mn)^2.
    To handle scale invariance, we normalize the points to have unit norm 
    or fix the scale by constraining the mean squared distance.
    However, DE is scale-invariant if the objective is. 
    To help convergence, we can penalize extreme scales.
    """
    pts = x.reshape(N, D)
    # Normalize to prevent overflow/underflow and help DE explore
    norm = np.linalg.norm(pts)
    if norm < 1e-6:
        return 1e6
    pts = pts / norm
    
    score = _true_score(pts)
    return score

def _init_double_shell():
    """
    Initialize points in a double-shell configuration.
    12 points on a sphere of radius R1 (approx icosahedron vertices).
    2 points near the center or on a smaller sphere.
    This breaks the symmetry of a single shell.
    """
    t = (1 + 5**0.5) / 2
    # 12 vertices of an icosahedron
    v_ico = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)
    ]
    v_ico = np.array(v_ico, dtype=float)
    v_ico /= np.linalg.norm(v_ico[0]) # Normalize to unit sphere
    
    # Scale icosahedron to radius 1
    pts_ico = v_ico * 1.0
    
    # Add 2 points near the center, slightly offset to avoid collision
    # Placing them on a small sphere of radius 0.4, opposite to each other
    r_inner = 0.4
    p13 = np.array([r_inner, 0, 0])
    p14 = np.array([-r_inner, 0, 0])
    
    pts = np.vstack([pts_ico, p13, p14])
    return pts

def _init_fib_sphere():
    """Fallback initialization: Fibonacci sphere."""
    i = np.arange(N)
    phi = np.arccos(1 - 2*(i+0.5)/N)
    theta = np.pi*(1+5**0.5)*i
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x,y,z])

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
        
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = np.inf
    
    def consider(pts):
        nonlocal best_pts, best_score
        if pts is None:
            return
        # Ensure shape
        if pts.shape != (N, D):
            return
        # Center the points to remove translational invariance
        pts = pts - pts.mean(axis=0)
        s = _true_score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts.copy()

    # 1. Evaluate initial topologies
    init_double = _init_double_shell()
    init_fib = _init_fib_sphere()
    
    consider(init_double)
    consider(init_fib)
    
    # 2. Run Differential Evolution
    # DE is a global optimization method. 
    # Bounds: We can restrict the search space to a box. 
    # Since the problem is scale-invariant, we can fix the scale by 
    # normalizing the objective, but DE works better with bounded search.
    # We'll search in a box [-2, 2]^3.
    bounds = [(-2, 2)] * (N * D)
    
    # We will run DE multiple times with different seeds and populations
    # to maximize the chance of finding the global minimum within the time limit.
    
    # Time budget for DE
    start_time = time.time()
    remaining = DEADLINE - (start_time - _t0)
    
    if remaining > 10:
        # Run DE with moderate population
        # maxiter is capped by time via callback
        def callback(xk, convergence):
            return time.time() - _t0 > DEADLINE - 15
        
        try:
            # Use 'de' strategy, popsize=20, maxiter=100
            # tol=1e-6
            result = differential_evolution(
                _obj, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                tol=1e-6, 
                seed=1,
                callback=callback,
                polish=True  # Use L-BFGS-B to polish the result
            )
            if result.success or result.fun < best_score:
                pts_de = result.x.reshape(N, D)
                consider(pts_de)
        except Exception:
            pass

    # 3. If time remains, run a second DE with different seed and larger population
    current_time = time.time()
    remaining = DEADLINE - (current_time - _t0)
    
    if remaining > 20 and best_score > 1.2: # If we haven't found a very good solution
        def callback2(xk, convergence):
            return time.time() - _t0 > DEADLINE - 10
        
        try:
            result2 = differential_evolution(
                _obj, 
                bounds, 
                maxiter=150, 
                popsize=30, 
                tol=1e-7, 
                seed=2,
                callback=callback2,
                polish=True
            )
            if result2.fun < best_score:
                pts_de2 = result2.x.reshape(N, D)
                consider(pts_de2)
        except Exception:
            pass

    # 4. Final local polish with Nelder-Mead on the best found so far
    if best_pts is not None:
        current_time = time.time()
        remaining = DEADLINE - (current_time - _t0)
        
        if remaining > 5:
            from scipy.optimize import minimize
            
            def _nm_obj(x):
                pts = x.reshape(N, D)
                pts = pts - pts.mean(axis=0)
                s = _true_score(pts)
                # Add a small regularization to avoid scale drift? 
                # The ratio is scale invariant, so NM might drift.
                # We fix scale by normalizing in the objective? 
                # _true_score doesn't normalize. 
                # Let's normalize inside the NM objective to help convergence.
                norm = np.linalg.norm(pts)
                if norm < 1e-6:
                    return 1e6
                pts_norm = pts / norm
                return _true_score(pts_norm)

            try:
                res_nm = minimize(
                    _nm_obj, 
                    best_pts.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res_nm.fun < best_score:
                    pts_nm = res_nm.x.reshape(N, D)
                    consider(pts_nm)
            except Exception:
                pass

    if best_pts is None:
        # Fallback
        best_pts = _init_fib_sphere()
        
    return best_pts
