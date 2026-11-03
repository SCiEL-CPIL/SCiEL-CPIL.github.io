"""Simulated annealing packing with soft-min objective and polish phase."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
    best_points = None
    for s in [0.10, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.115, 0.12]:
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if len(pts) == N:
                best_points = pts
                break
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback: very fine grid
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]

def _score(points):
    """Compute the maximum radius r for given centers."""
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _soft_min_objective(points, temp):
    """
    Compute a differentiable approximation of the negative radius.
    We want to maximize min(min_boundary, min_pair/2).
    This is equivalent to minimizing -min(min_boundary, min_pair/2).
    
    Let g1 = -min_boundary = min( - (1 - r_i) ) = min( r_i - 1 ) ... wait.
    min_boundary = min(1 - r_i). So -min_boundary = -min(1-r_i) = max(r_i - 1).
    Let's define penalties that are 0 if constraint is satisfied and positive otherwise.
    
    Constraint 1: r_i <= 1 - r.  =>  r_i + r <= 1.
    Constraint 2: dist_ij >= 2r.  =>  dist_ij - 2r >= 0.
    
    We want to maximize r.
    Let's define a function that penalizes violations.
    Let's use the log-sum-exp trick for the minimum.
    min(x) = -logsumexp(-x) / T  (approx)
    
    We want to maximize r such that:
    1 - r >= max(r_i)  =>  r <= 1 - max(r_i)
    2r <= min(dist_ij) => r <= min(dist_ij) / 2
    
    So r_max = min(1 - max(r_i), min(dist_ij)/2).
    
    Let A = 1 - max(r_i).
    Let B = min(dist_ij) / 2.
    We want to maximize min(A, B).
    
    This is a non-smooth function.
    Let's use a soft-min for A and B?
    Actually, A and B are themselves mins/maxs.
    
    Let's define:
    S1 = logsumexp(r_i / T) * T  (approximates max(r_i))
    A_soft = 1 - S1
    
    S2 = -logsumexp(-dist_ij / T) * T (approximates min(dist_ij))
    B_soft = S2 / 2
    
    Objective to minimize: -min(A_soft, B_soft)
    min(A_soft, B_soft) = -logsumexp([-A_soft, -B_soft] / T2) * T2
    
    This is getting complex. Let's try a different formulation.
    
    Standard approach for circle packing:
    Minimize f(x) = sum of penalty terms.
    Penalty for boundary: max(0, r_i - (1-r))^2
    Penalty for overlap: max(0, 2r - dist_ij)^2
    
    We can use soft-max for the max(0, ...) part?
    Or just use the squared violation directly, which is differentiable almost everywhere (kink at 0).
    To make it smooth, we can use a soft-plus or just accept the kink and use subgradient or just standard gradient with small steps.
    Actually, the kink at 0 is not a problem for gradient descent if we handle it carefully, but log-sum-exp is smoother.
    
    Let's stick to the "soft-min" of the radius components.
    r_est = soft_min(1 - max(r_i), min(dist_ij)/2)
    
    To compute gradients, we need the derivatives of max and min.
    d(max(r_i))/dx = r_i / sum(exp(r_j/T)) * T ... roughly.
    
    Let's implement a robust soft-min.
    """
    r_vec = np.hypot(points[:, 0], points[:, 1])
    
    # Approximate max(r_i)
    # max(r_i) ≈ T * log(sum(exp(r_i/T)))
    # To avoid overflow, subtract max
    max_r = np.max(r_vec)
    exp_r = np.exp((r_vec - max_r) / temp)
    sum_exp_r = np.sum(exp_r)
    logsumexp_r = max_r + temp * np.log(sum_exp_r)
    
    # A = 1 - max(r_i)
    A = 1.0 - logsumexp_r
    
    # Approximate min(dist_ij)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # min(dists) ≈ -T * log(sum(exp(-dist_ij/T)))
    min_d = np.min(dists)
    exp_neg_d = np.exp(-(dists - min_d) / temp)
    # Note: dists has inf on diagonal. exp(-inf) = 0. So they don't contribute.
    sum_exp_neg_d = np.sum(exp_neg_d)
    logsumexp_neg_d = -min_d + temp * np.log(sum_exp_neg_d)
    # min_d_approx = -logsumexp_neg_d
    min_d_approx = -logsumexp_neg_d
    
    # B = min(dist_ij) / 2
    B = min_d_approx / 2.0
    
    # We want to maximize min(A, B).
    # min(A, B) ≈ -T2 * log(exp(-A/T2) + exp(-B/T2))
    # Let's use a separate temperature for the final min to control sharpness.
    # Or just use the same temp?
    # Let's use a slightly higher temp for the final min to smooth it out more if needed,
    # but usually the inner mins are the bottleneck.
    # Let's just return the soft-min of A and B.
    
    # To compute gradient of this objective, we need to be careful.
    # However, for the "polish" phase, we can use a very small temp, making it close to the true min.
    # For the main SA, we use a larger temp.
    
    # Let's just return the value for now, and compute gradients analytically or via finite diff?
    # Finite diff is too slow for 600 vars.
    # Let's derive the gradient.
    
    # Objective J = - min(A, B)
    # Let M = min(A, B).
    # Gradient of M w.r.t x:
    # If A < B, M = A. grad M = grad A.
    # If B < A, M = B. grad M = grad B.
    # Soft min: M_soft = -T2 * log(exp(-A/T2) + exp(-B/T2))
    # dM_soft/dA = exp(-A/T2) / (exp(-A/T2) + exp(-B/T2)) = 1 / (1 + exp(-(B-A)/T2))
    # dM_soft/dB = exp(-B/T2) / (exp(-A/T2) + exp(-B/T2)) = 1 / (1 + exp(-(A-B)/T2))
    
    # Let w_A = 1 / (1 + exp(-(B-A)/T2))
    # Let w_B = 1 / (1 + exp(-(A-B)/T2))
    # Note w_A + w_B = 1.
    
    # grad M_soft = w_A * grad A + w_B * grad B
    
    # grad A = - grad logsumexp_r
    # logsumexp_r = max_r + T * log(sum(exp((r_i - max_r)/T)))
    # d(logsumexp_r)/dx_k = (1/T) * sum_i exp((r_i - max_r)/T) * d(r_i)/dx_k
    # d(r_i)/dx_k = (x_k - x_i) / r_i ... wait, r_i is norm of point i.
    # d(r_i)/dx = (x_i - x) / r_i? No.
    # r_i = ||p_i||. d(r_i)/dp_i = p_i / r_i.
    # So d(logsumexp_r)/dp_k = (1/T) * exp((r_k - max_r)/T) * (p_k / r_k)
    
    # grad B = 0.5 * grad min_d_approx
    # min_d_approx = -(-min_d + T * log(sum(exp(-(dists - min_d)/T))))
    # = min_d - T * log(sum(exp(-(dists - min_d)/T)))
    # Let S = sum_{i<j} exp(-(d_ij - min_d)/T)
    # min_d_approx = min_d - T * log(S)
    # d(min_d_approx)/dp_k = d(min_d)/dp_k - T * (1/S) * dS/dp_k
    # dS/dp_k = sum_{j!=k} exp(-(d_kj - min_d)/T) * (-1/T) * d(d_kj)/dp_k
    # d(d_kj)/dp_k = (p_k - p_j) / d_kj
    
    # This is implementable.
    
    T2 = temp # Use same temp for final min for simplicity, or maybe 2*temp? Let's use temp.
    
    exp_neg_A = np.exp(-A / T2)
    exp_neg_B = np.exp(-B / T2)
    denom = exp_neg_A + exp_neg_B
    
    w_A = exp_neg_A / denom
    w_B = exp_neg_B / denom
    
    # Gradient of A
    # A = 1 - logsumexp_r
    # grad_A = - grad_logsumexp_r
    # grad_logsumexp_r[k] = (1/T) * exp((r_k - max_r)/T) * (p_k / r_k)
    
    # Handle r_k = 0
    r_safe = np.where(r_vec > 1e-10, r_vec, 1.0)
    
    weights_r = (1.0 / temp) * exp_r # exp_r is exp((r_i - max_r)/temp)
    # Note: sum(weights_r) = 1.0
    
    grad_A = - (weights_r[:, np.newaxis] * (points / r_safe[:, np.newaxis]))
    
    # Gradient of B
    # B = min_d_approx / 2
    # min_d_approx = min_d - T * log(S)
    # S = sum_{i<j} exp(-(d_ij - min_d)/T)
    
    # We need dS/dp_k
    # dS/dp_k = sum_{j!=k} exp(-(d_kj - min_d)/T) * (-1/T) * (p_k - p_j) / d_kj
    
    # Let's compute this vectorized.
    # dists matrix is symmetric.
    # exp_neg_d matrix is symmetric.
    # term_kj = exp_neg_d[k,j] * (-1/T) * (p_k - p_j) / d_kj
    
    # We can compute the force for each k by summing over j.
    # F_k = sum_j term_kj
    
    # term_kj = -1/T * exp_neg_d[k,j] * (p_k - p_j) / d_kj
    # = -1/T * exp_neg_d[k,j] / d_kj * (p_k - p_j)
    
    # Let coeff_kj = -1/T * exp_neg_d[k,j] / d_kj
    # F_k = sum_j coeff_kj * (p_k - p_j)
    # = p_k * sum_j coeff_kj - sum_j coeff_kj * p_j
    
    # coeff_kj is 0 if d_kj is inf (diagonal).
    
    # Create coeff matrix
    coeff = np.zeros_like(dists)
    valid = dists < 1e10
    coeff[valid] = -(1.0 / temp) * exp_neg_d[valid] / dists[valid]
    
    # Sum coeff over j for each k
    sum_coeff = np.sum(coeff, axis=1)
    
    # Sum coeff_kj * p_j for each k
    # (coeff @ points)
    sum_coeff_p = coeff @ points
    
    grad_min_d = points * sum_coeff[:, np.newaxis] - sum_coeff_p
    
    # Wait, d(min_d)/dp_k term?
    # min_d_approx = min_d - T * log(S)
    # The derivative of min_d itself is non-differentiable (it's the min).
    # However, the soft-min approximation replaces the hard min with a smooth function.
    # The formula min_d - T * log(S) IS the smooth approximation.
    # So we just differentiate this expression.
    # The term "min_d" in the expression is a constant offset for the logsumexp calculation, 
    # but in the derivative of the logsumexp function defined as -T log sum exp(-x/T), 
    # the "min_d" subtraction inside the exp is just for numerical stability and doesn't change the value.
    # So the gradient of the smooth function is just the gradient of -T log(S).
    
    # Let's re-verify.
    # f(x) = -T log(sum exp(-x_i/T))
    # df/dx_k = -T * (1/S) * sum_j exp(-x_j/T) * (-1/T) * dx_j/dx_k
    # = (1/S) * sum_j exp(-x_j/T) * dx_j/dx_k
    # Here x_j are the distances d_j.
    # dx_j/dp_k is non-zero only if j involves k.
    
    # My previous derivation:
    # S = sum exp(-(d_ij - min_d)/T) = exp(min_d/T) * sum exp(-d_ij/T)
    # log S = min_d/T + log(sum exp(-d_ij/T))
    # -T log S = -min_d - T log(sum exp(-d_ij/T))
    # This is exactly the soft-min formula.
    # Gradient of -T log(sum exp(-d_ij/T)) w.r.t p_k:
    # = -T * (1/S') * sum_j exp(-d_j/T) * (-1/T) * dd_j/dp_k   where S' = sum exp(-d_j/T)
    # = (1/S') * sum_j exp(-d_j/T) * dd_j/dp_k
    
    # Note that exp_neg_d in my code was exp(-(dists - min_d)/T).
    # So exp_neg_d = exp(min_d/T) * exp(-dists/T).
    # S = exp(min_d/T) * S'.
    # (1/S) * exp_neg_d[j] = (1/S) * exp(min_d/T) * exp(-d_j/T) = (1/S') * exp(-d_j/T).
    # So the weights are correct.
    
    # So grad_min_d = sum_j (1/S) * exp_neg_d[j] * dd_j/dp_k
    # dd_j/dp_k for j=(k,m) is (p_k - p_m)/d_km.
    # So grad_min_d = sum_m (1/S) * exp_neg_d[k,m] * (p_k - p_m)/d_km
    
    # My previous code:
    # coeff = -(1/T) * exp_neg_d / dists
    # This has an extra -1/T.
    # Let's check the chain rule again.
    # f = -T log S
    # df/dp = -T * (1/S) * dS/dp
    # dS/dp = sum exp(...) * d(exp(...))/dp
    # d(exp(-d/T))/dp = exp(-d/T) * (-1/T) * dd/dp
    # So df/dp = -T * (1/S) * sum exp(-d/T) * (-1/T) * dd/dp
    # = (1/S) * sum exp(-d/T) * dd/dp
    
    # So the -1/T cancels out.
    # My coeff should be exp_neg_d / dists / S?
    # No, S is the sum.
    # We need weights w_j = exp_neg_d[j] / S.
    # Then grad = sum_j w_j * dd_j/dp.
    
    S_val = np.sum(exp_neg_d)
    if S_val == 0:
        return 1e10, np.zeros_like(points) # Degenerate
    
    weights_d = exp_neg_d / S_val
    
    # grad_min_d[k] = sum_m weights_d[k,m] * (p_k - p_m) / d_km
    
    # Vectorized:
    # term[k,m] = weights_d[k,m] / d_km * (p_k - p_m)
    # Sum over m.
    
    # Let's compute this.
    inv_dists = np.zeros_like(dists)
    valid_d = dists > 1e-10
    inv_dists[valid_d] = 1.0 / dists[valid_d]
    
    # term = weights_d * inv_dists
    term = weights_d * inv_dists
    
    # grad_min_d[k] = sum_m term[k,m] * (p_k - p_m)
    # = p_k * sum_m term[k,m] - sum_m term[k,m] * p_m
    
    sum_term = np.sum(term, axis=1)
    sum_term_p = term @ points
    
    grad_min_d = points * sum_term[:, np.newaxis] - sum_term_p
    
    grad_B = 0.5 * grad_min_d
    
    # Total gradient of M_soft
    grad_M = w_A * grad_A + w_B * grad_B
    
    # Objective is -M_soft
    # So we want to minimize -M_soft, which means we should move in direction -grad(-M_soft) = grad(M_soft).
    # Wait, standard gradient descent minimizes f.
    # If we define f = -M_soft, then grad f = -grad M_soft.
    # Step: x_new = x - alpha * grad f = x + alpha * grad M_soft.
    
    # So the "force" is grad_M.
    
    val = -M_soft
    # M_soft = -T2 * log(exp_neg_A + exp_neg_B)
    M_soft = -T2 * np.log(denom)
    val = -M_soft
    
    return val, grad_M

def _simulated_annealing(points, time_limit):
    """
    Simulated annealing with soft-min objective.
    """
    start_time = time.time()
    best_score = _score(points)
    best_points = points.copy()
    
    # Initial temperature
    # The scale of the objective is roughly the radius (0.05).
    # We want exp(-delta/T) to be meaningful.
    T = 0.01
    T_min = 1e-6
    cooling = 0.99
    
    current_points = points.copy()
    current_val, _ = _soft_min_objective(current_points, T)
    
    max_iter = 5000
    patience = 500
    no_improve = 0
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit * 0.9:
            break
            
        # Compute gradient
        val, grad = _soft_min_objective(current_points, T)
        
        # Normalize gradient to control step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
            
        # Step size
        step_size = T * 0.1 # Scale step with temperature?
        # Or fixed step?
        # Let's try adaptive.
        step = 0.01
        
        # Random perturbation
        noise = np.random.normal(0, T * 0.5, current_points.shape)
        
        new_points = current_points + step * grad + noise
        
        # Clamp to disk
        r = np.hypot(new_points[:, 0], new_points[:, 1])
        outside = r > 1.0
        if np.any(outside):
            new_points[outside] = new_points[outside] / r[outside, np.newaxis] * (1.0 - 1e-10)
            
        new_val, _ = _soft_min_objective(new_points, T)
        
        # Acceptance probability
        delta = new_val - val
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_points = new_points
            current_val = new_val
            
            # Check true score
            true_score = _score(current_points)
            if true_score > best_score:
                best_score = true_score
                best_points = current_points.copy()
                no_improve = 0
            else:
                no_improve += 1
                
            # Cool down
            T = max(T * cooling, T_min)
            
            if no_improve > patience:
                break
        else:
            no_improve += 1
            T = max(T * cooling, T_min)
            if no_improve > patience:
                break
                
    return best_points

def _polish(points, time_limit):
    """
    Gradient descent with very sharp soft-min (low T) to polish.
    """
    start_time = time.time()
    best_score = _score(points)
    best_points = points.copy()
    
    T = 1e-4 # Very sharp
    
    current_points = points.copy()
    
    max_iter = 1000
    no_improve = 0
    patience = 50
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit * 0.9:
            break
            
        val, grad = _soft_min_objective(current_points, T)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break
            
        grad = grad / grad_norm
        
        step_size = 0.001
        
        new_points = current_points + step_size * grad
        
        r = np.hypot(new_points[:, 0], new_points[:, 1])
        outside = r > 1.0
        if np.any(outside):
            new_points[outside] = new_points[outside] / r[outside, np.newaxis] * (1.0 - 1e-10)
            
        new_score = _score(new_points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            current_points = new_points
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                break
                
    return best_points

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * np.cos(angle)
            points[i, 1] = r * np.sin(angle)
            
    init_score = _score(points)
    
    # Phase 1: Simulated Annealing
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.7:
        remaining = time_limit - elapsed - 10.0
        if remaining > 10:
            points = _simulated_annealing(points, remaining)
            
    # Phase 2: Polish
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.9:
        remaining = time_limit - elapsed - 5.0
        if remaining > 5:
            points = _polish(points, remaining)
            
    # Final check
    r = np.hypot(points[:, 0], points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        points[mask] = points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
        
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(points[i] - points[j]) < 1e-10:
                    points[j, 0] += 1e-6
                    
    return points

# EVOLVE-BLOCK-END
