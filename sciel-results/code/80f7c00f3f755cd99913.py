import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _pairwise_distances(X):
    """Compute pairwise distances between points in X (N, 2)."""
    x2 = np.sum(X ** 2, axis=1)
    dot = X @ X.T
    dist_sq = x2[:, np.newaxis] + x2[np.newaxis, :] - 2 * dot
    dist_sq = np.maximum(dist_sq, 0)
    return np.sqrt(dist_sq)

def _score(X, r):
    """
    Score: how well X fits circles of radius r inside the unit disk.
    Returns the minimum of:
      - min pairwise distance / 2 - r  (overlap violation, positive means OK)
      - 1 - max |c_i| - r              (boundary violation, positive means OK)
    We want to maximize this. Equivalent to minimizing the negative.
    """
    dists = _pairwise_distances(X)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    dists_to_origin = np.linalg.norm(X, axis=1)
    max_origin = np.max(dists_to_origin)
    
    r_pair = min_pair / 2.0
    r_bound = 1.0 - max_origin
    
    r_actual = min(r_pair, r_bound)
    return r_actual

def _softmin_logsumexp(values, beta):
    """
    Smooth approximation of min(values) using log-sum-exp.
    -logsumexp(-beta * values) / beta approximates min(values) as beta -> inf.
    """
    values = np.asarray(values)
    # For numerical stability
    m = np.min(values)
    exp_vals = np.exp(-beta * (values - m))
    logsumexp = np.log(np.sum(exp_vals))
    return m - logsumexp / beta

def _smooth_objective(X_flat, beta, r_target):
    """
    Smooth objective for optimization. We want to find positions such that
    all pairwise distances >= 2*r_target and all points are within 1 - r_target.
    
    We minimize a smooth penalty that is 0 when all constraints are satisfied.
    Penalty = softmin(pairwise_dists / 2 - r_target, beta) + softmin(1 - |c_i| - r_target, beta)
    But we want to maximize the minimum, so we minimize the negative of the softmin.
    
    Actually, let's define:
    g1 = min_{i<j} (d_ij / 2 - r_target)  -> we want g1 >= 0
    g2 = min_i (1 - |c_i| - r_target)     -> we want g2 >= 0
    
    Smooth penalty = max(0, -softmin(g1_vals, beta)) + max(0, -softmin(g2_vals, beta))
    But this is still nonsmooth. Let's use a smooth max.
    
    Alternatively, minimize:
    -softmin(g1_vals, beta) - softmin(g2_vals, beta)
    This encourages both g1 and g2 to be large.
    """
    X = X_flat.reshape(-1, 2)
    dists = _pairwise_distances(X)
    np.fill_diagonal(dists, np.inf)
    g1_vals = dists / 2.0 - r_target
    g1_vals = g1_vals[~np.isinf(g1_vals)]
    
    dists_to_origin = np.linalg.norm(X, axis=1)
    g2_vals = 1.0 - dists_to_origin - r_target
    
    # Use softmin to get a smooth approximation of the minimum
    s1 = _softmin_logsumexp(g1_vals, beta)
    s2 = _softmin_logsumexp(g2_vals, beta)
    
    # We want to maximize min(g1, g2), so minimize -(s1 + s2) or just -min(s1, s2)
    # Let's use a combined softmin
    combined = np.array([s1, s2])
    s_combined = _softmin_logsumexp(combined, beta)
    
    return -s_combined

def _ring_initialization(n_rings=None):
    """
    Initialize points in concentric rings.
    """
    if n_rings is None:
        n_rings = 7
    
    points = [np.array([0.0, 0.0])]
    remaining = N - 1
    
    for ring in range(1, n_rings + 1):
        if remaining <= 0:
            break
        # Number of points in this ring
        n_pts = min(6 * ring, remaining)
        # Radius of this ring
        r = ring * 0.13
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + np.pi / (3 * ring)
        for a in angles:
            points.append(np.array([r * np.cos(a), r * np.sin(a)]))
        remaining -= n_pts
    
    # If we still need more points, add them in outer rings
    ring = n_rings + 1
    while remaining > 0:
        n_pts = min(6 * ring, remaining)
        r = ring * 0.13
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        for a in angles:
            points.append(np.array([r * np.cos(a), r * np.sin(a)]))
        remaining -= n_pts
        ring += 1
    
    X = np.array(points[:N])
    # Add small random perturbation
    X += np.random.normal(0, 0.01, X.shape)
    return X

def _hex_initialization():
    """
    Hexagonal lattice initialization.
    """
    spacing = 0.25
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _anneal_points(X0, time_budget, beta_start=50, beta_end=500):
    """
    Multi-stage smooth annealing.
    """
    X = X0.copy()
    start_time = time.time()
    
    # Stage 1: Low beta (smoother), larger steps
    # Stage 2: Higher beta, smaller steps
    # Stage 3: Even higher beta, very small steps
    
    stages = [
        {'beta': beta_start, 'step': 0.05, 'frac': 0.4},
        {'beta': beta_start * 5, 'step': 0.02, 'frac': 0.35},
        {'beta': beta_end, 'step': 0.005, 'frac': 0.25},
    ]
    
    current_time = 0
    for stage in stages:
        stage_time = stage['frac'] * time_budget
        beta = stage['beta']
        step = stage['step']
        
        # Gradient descent with smooth objective
        X_flat = X.flatten()
        best_X_flat = X_flat.copy()
        best_obj = _smooth_objective(X_flat, beta, 0.1)
        
        t = time.time()
        while time.time() - t < stage_time:
            # Compute gradient numerically
            eps = 1e-5
            grad = np.zeros_like(X_flat)
            for i in range(len(X_flat)):
                Xp = X_flat.copy()
                Xp[i] += eps
                grad[i] = (_smooth_objective(Xp, beta, 0.1) - best_obj) / eps
            
            # Gradient step
            X_flat = X_flat - step * grad
            
            # Constraint push: keep points inside unit disk
            X_temp = X_flat.reshape(-1, 2)
            dists = np.linalg.norm(X_temp, axis=1)
            mask = dists > 0.99
            if np.any(mask):
                X_temp[mask] *= 0.99 / dists[mask]
                X_flat = X_temp.flatten()
            
            obj = _smooth_objective(X_flat, beta, 0.1)
            if obj < best_obj:
                best_obj = obj
                best_X_flat = X_flat.copy()
        
        X = best_X_flat.reshape(-1, 2)
    
    return X

def _de_ring_search(X0, time_budget, n_pop=20):
    """
    Differential Evolution on ring-structured representations.
    """
    start_time = time.time()
    
    # Initial population
    pop = [X0.copy()]
    for i in range(n_pop - 1):
        X_rand = X0.copy()
        X_rand += np.random.normal(0, 0.1, X_rand.shape)
        # Ensure inside unit disk
        dists = np.linalg.norm(X_rand, axis=1)
        mask = dists > 0.95
        if np.any(mask):
            X_rand[mask] *= 0.95 / dists[mask]
        pop.append(X_rand)
    
    best_X = pop[0].copy()
    best_score = _score(pop[0], 0)
    for i in range(1, n_pop):
        s = _score(pop[i], 0)
        if s > best_score:
            best_score = s
            best_X = pop[i].copy()
    
    F = 0.8  # Scale factor
    CR = 0.9  # Crossover rate
    
    while time.time() - start_time < time_budget:
        # Create trial vector
        i = np.random.randint(n_pop)
        a, b, c = np.random.choice([j for j in range(n_pop) if j != i], 3, replace=False)
        
        v = pop[a] + F * (pop[b] - pop[c])
        
        # Crossover
        trial = pop[i].copy()
        for j in range(N):
            if np.random.rand() < CR:
                trial[j] = v[j]
        
        # Ensure inside unit disk
        dists = np.linalg.norm(trial, axis=1)
        mask = dists > 0.99
        if np.any(mask):
            trial[mask] *= 0.99 / dists[mask]
        
        # Evaluate
        trial_score = _score(trial, 0)
        if trial_score > _score(pop[i], 0):
            pop[i] = trial
            if trial_score > best_score:
                best_score = trial_score
                best_X = trial.copy()
    
    return best_X, best_score

def _constraint_push(X):
    """
    Push points that are outside the unit disk back inside.
    """
    X = X.copy()
    dists = np.linalg.norm(X, axis=1)
    mask = dists > 0.99
    if np.any(mask):
        X[mask] *= 0.99 / dists[mask]
    return X

def _optimize_positions(X0, time_budget):
    """
    Combine annealing and DE search.
    """
    start_time = time.time()
    
    # Phase 1: Annealing
    t1 = time.time()
    X_annealed = _anneal_points(X0, min(time_budget * 0.4, 30))
    score_annealed = _score(X_annealed, 0)
    
    # Phase 2: DE search
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        X_de, score_de = _de_ring_search(X_annealed, remaining)
        if score_de > score_annealed:
            X_annealed = X_de
            score_annealed = score_de
    
    return X_annealed, score_annealed

def _get_hex_initial_points():
    return _hex_initialization()

def construct():
    start_time = time.time()
    total_time = 150.0
    safety_margin = 10.0
    available_time = total_time - safety_margin
    
    best_X = None
    best_radius = -np.inf
    
    # Multiple initializations
    inits = []
    
    # Hex
    X_hex = _get_hex_initial_points()
    inits.append(X_hex)
    
    # Rings
    X_rings = _ring_initialization()
    inits.append(X_rings)
    
    # Random
    for _ in range(2):
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        X_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(X_rand)
    
    time_per_init = available_time / len(inits)
    
    for i, X0 in enumerate(inits):
        if time.time() - start_time > available_time:
            break
        
        X_opt, r_opt = _optimize_positions(X0, time_per_init)
        
        if r_opt > best_radius:
            best_radius = r_opt
            best_X = X_opt
    
    # Final polish with L-BFGS on the best solution
    remaining_time = available_time - (time.time() - start_time)
    if remaining_time > 5.0 and best_X is not None:
        # Use scipy minimize to polish
        X_flat = best_X.flatten()
        
        def obj_to_minimize(X_flat):
            X = X_flat.reshape(-1, 2)
            return -_score(X, 0)
        
        result = minimize(obj_to_minimize, X_flat, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if result.success:
            X_polished = result.x.reshape(-1, 2)
            X_polished = _constraint_push(X_polished)
            r_polished = _score(X_polished, 0)
            if r_polished > best_radius:
                best_radius = r_polished
                best_X = X_polished
    
    # Ensure the result is valid
    if best_X is None:
        best_X = _ring_initialization()
    
    # Final constraint push
    best_X = _constraint_push(best_X)
    
    return best_X

# EVOLVE-BLOCK-END
