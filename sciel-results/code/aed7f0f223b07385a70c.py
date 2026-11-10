import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
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

def _concentric_packing():
    """Generate initial concentric ring packing."""
    centers = []
    # Center
    centers.append([0.0, 0.0])
    rings = [1, 6, 12, 18, 24, 30]
    # We need 50 points. 1 + 6 + 12 + 18 = 37. 1+6+12+18+24 = 61. So use 4 rings.
    # Let's distribute roughly.
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # Ring 4: 14 points (remaining)
    
    radii = [0.3, 0.6, 0.85, 1.0] # Approximate radii for rings
    counts = [6, 12, 18, 14]
    
    for r, count in zip(radii, counts):
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        for a in angles:
            centers.append([r * np.cos(a), r * np.sin(a)])
            
    centers = np.array(centers[:N])
    return centers

def _random_packing():
    """Generate random initial packing."""
    centers = np.random.uniform(-0.8, 0.8, size=(N, 2))
    return centers

def _soft_min_value_and_grad(values, p=4):
    """
    Compute soft minimum of values and its gradient.
    soft_min(v) = -(1/p) * sum(v_i^p)^(1/p) is not quite right for min.
    Actually, soft_min(v) approx min(v).
    A common approximation is:
    m = min(v)
    But for differentiability, we can use:
    s(v) = - (1/k) log sum exp(-k v_i)
    This is the "LogSumExp" trick for minimum.
    Gradient: ds/dv_i = exp(-k v_i) / sum exp(-k v_j)
    """
    k = 20.0
    # Shift to avoid overflow
    v_shifted = -k * values
    v_max = np.max(v_shifted)
    exps = np.exp(v_shifted - v_max)
    sum_exps = np.sum(exps)
    val = -(v_max + np.log(sum_exps)) / k
    grad = exps / sum_exps / k
    return val, grad

def _optimize_lagrangian(centers_init, deadline):
    """
    Optimize centers using Lagrangian relaxation with soft constraints.
    Objective: Maximize r
    Constraints:
    1. |c_i| + r <= 1  => 1 - |c_i| - r >= 0
    2. |c_i - c_j| - 2r >= 0
    
    We use a penalty method:
    L(c, r) = r - lambda * sum( max(0, r + |c_i| - 1)^2 + max(0, 2r - |c_i - c_j|)^2 )
    
    We alternate between updating r (via line search or gradient) and c (via gradient ascent on L).
    """
    centers = centers_init.copy()
    r = 0.1 # Initial guess
    lambda_pen = 10.0
    
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Precompute pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    lr_c = 0.01 # Learning rate for centers
    lr_r = 0.001 # Learning rate for radius
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Compute distances
        dx = centers[i_idx, 0] - centers[j_idx, 0]
        dy = centers[i_idx, 1] - centers[j_idx, 1]
        d_pairs = np.sqrt(dx**2 + dy**2)
        
        # Boundary distances
        d_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
        
        # Violations
        # Boundary violation: r + |c_i| - 1 > 0
        viol_b = np.maximum(0, r - d_boundary)
        # Pair violation: 2r - d_ij > 0
        viol_p = np.maximum(0, 2*r - d_pairs)
        
        # Penalty term
        penalty = np.sum(viol_b**2) + np.sum(viol_p**2)
        
        # Objective to maximize: r - lambda * penalty
        # We want to increase r and decrease penalty.
        
        # Gradients
        # d(r)/dr = 1
        # d(penalty)/dr = 2 * sum(viol_b) + 2 * sum(viol_p)
        grad_r = 1.0 - lambda_pen * (2 * np.sum(viol_b) + 2 * np.sum(viol_p))
        
        # Gradients for centers
        grad_c = np.zeros_like(centers)
        
        # Boundary gradient
        # d(viol_b_i)/d c_i = 0 if viol_b_i == 0
        # If viol_b_i > 0, d(viol_b_i^2)/d c_i = 2 * viol_b_i * d(r - d_b_i)/d c_i
        # d_b_i = 1 - |c_i|. d(d_b_i)/d c_i = - c_i / |c_i|
        # So d(r - d_b_i)/d c_i = c_i / |c_i|
        # Wait, viol_b = r - d_b. If viol_b > 0, we want to increase d_b (move center inward).
        # d(viol_b^2)/d c_i = 2 * viol_b * (1 * 0 - (- c_i/|c_i|)) = 2 * viol_b * (c_i/|c_i|)
        # This points outward. But we want to reduce violation, so we subtract this gradient?
        # L = r - lambda * P. grad L = grad r - lambda * grad P.
        # grad P w.r.t c_i is in direction of increasing violation.
        # So grad L w.r.t c_i is in direction of decreasing violation.
        
        norms_c = np.linalg.norm(centers, axis=1, keepdims=True)
        # Avoid div by 0
        norms_c_safe = np.where(norms_c < 1e-6, 1.0, norms_c)
        unit_c = centers / norms_c_safe
        
        # Only for active boundary violations
        active_b = viol_b > 1e-6
        if np.any(active_b):
            # grad P_b = 2 * viol_b * unit_c (points outward, increasing violation)
            # We want to move inward, so -grad P_b
            # grad L_c = - lambda * (2 * viol_b * unit_c)
            grad_c[active_b] -= lambda_pen * 2 * viol_b[active_b, None] * unit_c[active_b]
            
        # Pair gradients
        # viol_p_k = max(0, 2r - d_k)
        # d(viol_p_k^2)/d c_i = 2 * viol_p_k * d(2r - d_k)/d c_i
        # d_k = |c_i - c_j|. d(d_k)/d c_i = (c_i - c_j) / d_k
        # d(2r - d_k)/d c_i = - (c_i - c_j) / d_k
        # So grad P_p w.r.t c_i = 2 * viol_p_k * (-(c_i - c_j) / d_k)
        # grad L_c w.r.t c_i = - lambda * grad P_p = lambda * 2 * viol_p_k * (c_i - c_j) / d_k
        # This pushes i away from j. Correct.
        
        active_p = viol_p > 1e-6
        if np.any(active_p):
            # We need to accumulate gradients for each i
            # For each active pair k (i, j):
            # grad_c[i] += lambda * 2 * viol_p[k] * (c_i - c_j) / d_k
            # grad_c[j] += lambda * 2 * viol_p[k] * (c_j - c_i) / d_k
            
            # Vectorized accumulation is tricky with numpy without loops or advanced indexing.
            # N=50, n_pairs=1225. Loop is fine.
            for k in range(n_pairs):
                if active_p[k]:
                    i = i_idx[k]
                    j = j_idx[k]
                    d_k = d_pairs[k]
                    if d_k < 1e-6:
                        # If coincident, push apart randomly
                        vec = np.random.normal(size=2)
                        vec /= np.linalg.norm(vec)
                    else:
                        vec = (centers[i] - centers[j]) / d_k
                    
                    force = lambda_pen * 2 * viol_p[k] * vec
                    grad_c[i] += force
                    grad_c[j] -= force
                    
        # Update centers
        centers += lr_c * grad_c
        
        # Update radius
        # We can use a simple line search or gradient step for r.
        # Since r is scalar, we can try to increase it if grad_r > 0.
        # But we must ensure constraints are not violated too much.
        # A safer approach: set r to the current feasible radius (min of slacks).
        # r_feasible = min(min(d_boundary), min(d_pairs)/2)
        # If we set r = r_feasible, we are on the boundary of feasibility.
        # Then we optimize positions to increase r_feasible.
        
        r_feasible = min(np.min(d_boundary), np.min(d_pairs)/2)
        if r_feasible > 0:
            r = r_feasible
            
        # Check improvement
        current_r = _compute_radius(centers)
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            no_improve = 0
            # Increase penalty weight slightly to enforce constraints better
            lambda_pen *= 1.1
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Reheat / Perturb
                centers += np.random.normal(0, 0.05, centers.shape)
                # Project inside?
                norms = np.linalg.norm(centers, axis=1, keepdims=True)
                centers = centers / np.maximum(norms, 0.9) * 0.9
                no_improve = 0
                lambda_pen = 10.0
                lr_c = 0.01
                # Reset r
                r = _compute_radius(centers)
                
        # Adapt learning rate
        if no_improve % 10 == 0:
            lr_c *= 0.9
            
    return best_centers, best_r

def construct():
    deadline = time.time() + 110.0
    
    # Multi-start
    inits = [
        _hexagonal_packing(),
        _concentric_packing(),
        _random_packing(),
        _random_packing()
    ]
    
    best_centers = inits[0]
    best_r = 0.0
    
    for init in inits:
        if time.time() > deadline - 20: # Reserve time
            break
            
        time_per_init = (deadline - time.time()) / len(inits)
        # Ensure at least 5 seconds per init
        time_per_init = max(5.0, time_per_init)
        
        c, r = _optimize_lagrangian(init, time.time() + time_per_init)
        
        if r > best_r:
            best_r = r
            best_centers = c
            
    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(_compute_radius(centers))
