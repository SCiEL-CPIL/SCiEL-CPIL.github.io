"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Strategy: Graph Embedding with Spectral Regularization.
    # 1. Objective: Minimize (D_max / D_min)^2.
    #    Since min/max are non-smooth, we use soft approximations for gradient-based optimization.
    #    soft_max(x, beta) ~ log(sum(exp(beta*x))/beta)
    #    soft_min(x, beta) ~ -log(sum(exp(-beta*x))/beta)
    #    However, for stability and speed, we can optimize a smooth proxy:
    #    f = log(sum(exp(beta * d^2))) - log(sum(exp(-beta * d^2)))
    #    This approximates 2*beta*(d_max - d_min) which correlates with the ratio.
    #    Actually, the ratio is better approximated by minimizing the variance of distances 
    #    or using a specific smooth surrogate for the ratio.
    #    
    #    Let's use a robust smooth surrogate for the ratio R = d_max / d_min.
    #    R^2 = d_max^2 / d_min^2.
    #    Let S_max = sum(exp(beta * d^2)) and S_min = sum(exp(-beta * d^2)).
    #    log(S_max) approximates beta * d_max^2.
    #    log(S_min) approximates -beta * d_min^2.
    #    So log(S_max) - log(S_min) approximates beta * (d_max^2 + d_min^2). This is not the ratio.
    #    
    #    Better approach: Minimize the coefficient of variation of distances? No.
    #    Let's stick to the soft-min/max of the squared distances.
    #    Let u = d^2.
    #    f(u) = soft_max(u, beta) / soft_min(u, beta).
    #    This is smooth and differentiable.
    #    
    # 2. Regularization: Laplacian of the complete graph K_N.
    #    The Laplacian L of K_N has eigenvalues 0 (once) and N (N-1 times).
    #    We want the points to be isotropic. We can penalize the deviation from an isotropic distribution.
    #    A common regularizer is the trace of the covariance matrix minus its determinant (for 2D),
    #    or simply enforcing that the second moment tensor is proportional to identity.
    #    Regularizer = || Cov(pts) - lambda * I ||^2.
    #    Since we normalize the points in the objective calculation (or scale invariant?),
    #    actually the ratio is scale invariant. So we can fix the scale to avoid singularity.
    #    Let's fix the mean to 0 and the sum of squared norms to 1.
    
    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        norm_sq = np.sum(pts**2)
        if norm_sq > 1e-12:
            pts = pts / np.sqrt(norm_sq)
        return pts

    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    beta = 50.0 # Sharpness of soft-min/max
    
    def objective(x):
        pts = x.reshape(N, 2)
        pts = normalize(pts)
        d_sq = get_sq_dists(pts)
        
        # Soft max and min of squared distances
        # To avoid overflow, shift by max/min
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        # soft_max ~ max + log(sum(exp(beta*(x-max)))/beta)
        # soft_min ~ min - log(sum(exp(-beta*(x-min)))/beta)
        
        # We want to minimize soft_max / soft_min
        # Let's compute log(soft_max) - log(soft_min) to keep numbers stable?
        # Or just compute the ratio.
        
        exp_pos = np.exp(beta * (d_sq - max_d_sq))
        soft_max_val = max_d_sq + np.log(np.sum(exp_pos)) / beta
        
        exp_neg = np.exp(-beta * (d_sq - min_d_sq))
        soft_min_val = min_d_sq - np.log(np.sum(exp_neg)) / beta
        
        if soft_min_val < 1e-9:
            return 1e10
            
        ratio_sq = (soft_max_val / soft_min_val)
        
        # Spectral Regularization:
        # We want the point cloud to be as "isotropic" as possible.
        # The covariance matrix C = pts.T @ pts / N.
        # Isotropic means C ~ c * I.
        # Penalty = || C - trace(C)/2 * I ||_F^2
        C = (pts.T @ pts) / N
        trace_C = np.trace(C)
        target_C = (trace_C / 2.0) * np.eye(2)
        reg_penalty = np.sum((C - target_C)**2)
        
        # Weight the regularization
        w_reg = 0.1
        return ratio_sq + w_reg * reg_penalty

    def gradient(x):
        # Numerical gradient for simplicity and robustness, 
        # or use analytical if possible. 
        # Given time constraints and complexity of soft-min/max grads,
        # numerical gradient with L-BFGS-B might be slow if not careful.
        # L-BFGS-B can use numerical gradients, but it's O(N) evaluations per step.
        # Let's try to implement analytical gradient for the smooth part.
        
        pts = x.reshape(N, 2)
        pts = normalize(pts)
        d_sq = get_sq_dists(pts)
        
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        exp_pos = np.exp(beta * (d_sq - max_d_sq))
        sum_pos = np.sum(exp_pos)
        soft_max_val = max_d_sq + np.log(sum_pos) / beta
        
        exp_neg = np.exp(-beta * (d_sq - min_d_sq))
        sum_neg = np.sum(exp_neg)
        soft_min_val = min_d_sq - np.log(sum_neg) / beta
        
        if soft_min_val < 1e-9:
            return np.zeros_like(x)
            
        # Gradient of ratio f = A / B where A = soft_max, B = soft_min
        # df/dx = (B * dA/dx - A * dB/dx) / B^2
        
        # dA/dx:
        # A = max_d + (1/beta) * log(sum_pos)
        # dA/dx_i = (1/beta) * (1/sum_pos) * sum_j exp_pos_j * d(d_sq_j)/dx_i
        # Note: max_d_sq also depends on x, but its gradient is 0 almost everywhere (subgradient).
        # We ignore the derivative of the max/min shift for the smooth approximation,
        # treating max_d_sq and min_d_sq as constants for the gradient step,
        # which is a common approximation in soft-min/max optimizations.
        
        # d(d_sq_j)/dx_i:
        # d_sq_j is distance between point a and b.
        # d(||p_a - p_b||^2)/dp_a = 2(p_a - p_b)
        
        # This is getting complex to vectorize analytically quickly.
        # Let's use finite difference gradient with a small step, relying on L-BFGS-B's efficiency.
        # To speed up, we can use a central difference with step 1e-6.
        
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            f_plus = objective(x_plus)
            f_minus = objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
            
        return grad

    # Initialization: Use a diverse set of starting points
    # 1. 4x4 Grid
    # 2. Random
    # 3. Hexagonal lattice (rotated)
    
    best_score = 1e10
    best_x = None
    
    # Time budget for restarts
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate initial point
        if r == 0:
            # 4x4 Grid
            x_coords = np.arange(4) - 1.5
            y_coords = np.arange(4) - 1.5
            xx, yy = np.meshgrid(x_coords, y_coords)
            pts = np.column_stack([xx.ravel(), yy.ravel()])
        elif r == 1:
            # Hexagonal lattice
            pts = []
            side = 3
            for i in range(side):
                for j in range(side):
                    x = i + 0.5 * (j % 2)
                    y = j * 0.866
                    pts.append([x, y])
            # Add one more to make 16? 3x3=9. Need 16.
            # Let's do a 4x4 hex-like or just random.
            # 4x4 grid is better for r=0.
            # Let's use a perturbed circle for r=1
            angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
            radii = np.ones(16)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            pts += np.random.normal(0, 0.1, pts.shape)
        else:
            # Random
            pts = np.random.uniform(-2, 2, (16, 2))
            
        x0 = normalize(pts).ravel()
        
        try:
            res = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            final_pts = res.x.reshape(N, 2)
            final_score = score_raw(final_pts)
            
            if final_score < best_score:
                best_score = final_score
                best_x = res.x.copy()
                
        except Exception:
            continue
            
    if best_x is None:
        # Fallback
        return np.random.uniform(-1, 1, (16, 2))
        
    return best_x.reshape(N, 2)

# EVOLVE-BLOCK-END
