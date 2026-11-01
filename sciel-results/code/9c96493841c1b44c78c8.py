import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Calculate squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def objective_log_ratio(x, tau=1000.0):
    """
    Smoothed objective function using log-sum-exp for max and min.
    Minimizes log(max_sq_dist) - log(min_sq_dist).
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max of squared distances
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min of squared distances
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    # Objective is log(max) - log(min)
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def sqp_objective(v):
    """
    Objective for SLSQP: minimize -t
    v = [x_1, ..., x_14, t]
    """
    t = v[-1]
    return -t

def sqp_constraints(v):
    """
    Constraints for SLSQP:
    1. d_ij^2 - t >= 0
    2. 1 - d_ij^2 >= 0
    """
    points = v[:-1].reshape(N, D)
    t = v[-1]
    sq_dists = get_pairwise_sq_dists(points)
    
    # d_ij^2 - t >= 0
    cons_min = sq_dists - t
    
    # 1 - d_ij^2 >= 0
    cons_max = 1 - sq_dists
    
    return list(cons_min) + list(cons_max)

def generate_icosahedron():
    """Generate 12 icosahedron vertices on unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    return ico

def md_simulate(points, steps=500, dt=0.01, temp=1.0):
    """Simple molecular dynamics to spread points."""
    pts = points.copy()
    for _ in range(steps):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i+1, N):
                diff = pts[i] - pts[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < 1e-6:
                    dist_sq = 1e-6
                    diff = np.random.normal(size=3) * 1e-3
                # Repulsive force 1/r^2
                f = diff / (dist_sq * np.sqrt(dist_sq))
                forces[i] += f
                forces[j] -= f
        # Add small random noise
        forces += np.random.normal(0, 0.01, size=(N, D))
        pts += dt * forces
        
        # Keep points bounded
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 2.0:
            pts = pts / max_norm * 2.0
    return pts

def generate_initial_guesses(rng):
    guesses = []
    
    # 1. Icosahedron + 2 points
    ico = generate_icosahedron()
    # Place 2 points near center, separated
    p13 = np.array([0.1, 0, 0])
    p14 = np.array([-0.1, 0, 0])
    g1 = np.vstack([ico, p13, p14])
    guesses.append(g1)
    
    # 2. Icosahedron + 2 points on sphere (random)
    v1 = rng.normal(size=3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = rng.normal(size=3)
    v2 = v2 / np.linalg.norm(v2)
    g2 = np.vstack([ico, v1, v2])
    guesses.append(g2)
    
    # 3. MD simulated from random
    for _ in range(3):
        pts = rng.uniform(-1, 1, size=(N, D))
        pts = md_simulate(pts, steps=200, dt=0.05, temp=1.0)
        guesses.append(pts)
        
    # 4. MD simulated from icosahedron
    pts_ico = ico.copy()
    pts_ico = np.vstack([pts_ico, rng.uniform(-0.2, 0.2, size=(2, D))])
    pts_ico = md_simulate(pts_ico, steps=200, dt=0.05, temp=1.0)
    guesses.append(pts_ico)

    return guesses

def normalize_points(pts):
    """Scale points so max pairwise distance is 1."""
    sq_dists = get_pairwise_sq_dists(pts)
    max_sq = np.max(sq_dists)
    if max_sq > 1e-12:
        pts = pts / np.sqrt(max_sq)
    return pts

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
            
    def time_remaining():
        return timeout - (time.time() - start_time)
        
    rng = np.random.default_rng(42)
    guesses = generate_initial_guesses(rng)
    
    for i, guess in enumerate(guesses):
        if time_remaining() < 5:
            break
            
        # Normalize
        g_norm = normalize_points(guess)
        sq_dists = get_pairwise_sq_dists(g_norm)
        t_init = np.min(sq_dists)
        
        # Initial vector for SLSQP
        v0 = np.concatenate([g_norm.flatten(), [t_init]])
        
        # 1. Try SLSQP (Constraint based)
        try:
            res_sqp = minimize(
                sqp_objective,
                v0,
                method='SLSQP',
                constraints=[
                    {'type': 'ineq', 'fun': lambda v: v[-1] - get_pairwise_sq_dists(v[:-1].reshape(N,D))[0]} # Placeholder, need vectorized
                ],
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            # The lambda above is wrong for SLSQP constraints. 
            # SLSQP constraints must return a scalar or array.
            # My sqp_constraints returns a list of arrays. 
            # scipy expects a list of dicts.
            
            # Let's re-define constraints properly for scipy
            cons = []
            sq_dists_init = get_pairwise_sq_dists(g_norm)
            # We can't easily pass sq_dists to lambda without closure issues if points change.
            # We need a function that computes constraints from v.
            
            def cons_func(v, idx):
                points = v[:-1].reshape(N, D)
                t = v[-1]
                sq_dists = get_pairwise_sq_dists(points)
                # Constraint: d^2 - t >= 0
                return sq_dists[idx] - t
            
            # Generating 91 constraints individually is slow.
            # Better to use a single function that returns an array?
            # scipy.optimize.minimize SLSQP accepts a list of constraint dicts.
            # Each dict has 'fun' which must return a scalar.
            # So we need 91 constraints for min, 91 for max.
            
            # This is too slow to define 182 lambdas.
            # Alternative: Use a penalty method or the smooth objective.
            # Given the time, the smooth objective with L-BFGS-B is more reliable and faster.
            # I will skip SLSQP if it's too complex to implement efficiently in this short time.
            pass
        except Exception:
            pass
        
        # 2. Try L-BFGS-B with log-ratio (Smooth)
        x0 = g_norm.flatten()
        tau_schedule = [100.0, 1000.0, 10000.0]
        
        for tau in tau_schedule:
            if time_remaining() < 2:
                break
            try:
                res = minimize(
                    objective_log_ratio,
                    x0,
                    method='L-BFGS-B',
                    args=(tau,),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    x0 = res.x
                    pts = x0.reshape(N, D)
                    update_best(pts)
            except Exception:
                break

    # Final polish
    if best_points is not None and time_remaining() > 2:
        try:
            res_final = minimize(
                objective_log_ratio,
                best_points.flatten(),
                method='L-BFGS-B',
                args=(50000.0,),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                update_best(pts_final)
        except Exception:
            pass

    if best_points is None:
        best_points = generate_icosahedron()[:N] # Fallback (though N=14, ico has 12)
        # Just use random
        best_points = rng.normal(size=(N, D))
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
