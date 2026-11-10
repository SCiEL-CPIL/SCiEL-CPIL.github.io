# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def true_score(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2

    def soft_score(points, beta):
        D2 = pdist(points, 'sqeuclidean')
        log_softmin = -1.0 / beta * logsumexp(-beta * D2)
        log_softmax = 1.0 / beta * logsumexp(beta * D2)
        return -(log_softmin - log_softmax)

    def objective_smooth(params, beta):
        points = params.reshape((n, 2))
        return soft_score(points, beta)

    def objective_true(params):
        points = params.reshape((n, 2))
        return -true_score(points)

    def gradient_smooth(params, beta):
        points = params.reshape((n, 2))
        D2 = pdist(points, 'sqeuclidean')
        
        # Gradient of softmin
        exp_neg_beta_D2 = np.exp(-beta * D2)
        sum_exp_neg = np.sum(exp_neg_beta_D2)
        w_min = exp_neg_beta_D2 / sum_exp_neg
        
        # Gradient of softmax
        exp_beta_D2 = np.exp(beta * D2)
        sum_exp_pos = np.sum(exp_beta_D2)
        w_max = exp_beta_D2 / sum_exp_pos
        
        # Build pairwise index matrix
        idx = np.triu_indices(n, k=1)
        i_idx, j_idx = idx
        
        # Gradient contribution from softmin
        grad_min = np.zeros((n, 2))
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            diff = points[i] - points[j]
            grad_min[i] += 2 * w_min[k] * diff
            grad_min[j] -= 2 * w_min[k] * diff
        
        # Gradient contribution from softmax
        grad_max = np.zeros((n, 2))
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            diff = points[i] - points[j]
            grad_max[i] += 2 * w_max[k] * diff
            grad_max[j] -= 2 * w_max[k] * diff
        
        # Total gradient: - (grad of log_softmin - grad of log_softmax)
        # = -(grad_min/beta - grad_max/beta) = (-grad_min + grad_max) / beta
        grad = (grad_max - grad_min) / beta
        return grad.flatten()

    def refine_local(points, time_limit=5.0):
        """Use Powell method for robust local refinement on non-smooth objective."""
        if time.time() > deadline - 10:
            return points
        
        def obj(p):
            return -true_score(p.reshape((n, 2)))
        
        start_t = time.time()
        result = minimize(obj, points.flatten(), method='Powell',
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if time.time() - start_t < time_limit and result.success:
            final_score = -result.fun
            if final_score > true_score(points):
                return result.x.reshape((n, 2))
        return points

    # Generate diverse initializations
    def generate_initializations():
        inits = []
        
        # 4x4 grid
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        inits.append(np.column_stack([X.flatten(), Y.flatten()]))
        
        # 4x4 grid with different scaling
        x = np.linspace(-1.0, 1.0, 4)
        y = np.linspace(-1.0, 1.0, 4)
        X, Y = np.meshgrid(x, y)
        inits.append(np.column_stack([X.flatten(), Y.flatten()]))
        
        # Hexagonal-ish arrangement
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        r1 = 0.5
        r2 = 1.0
        pts = []
        pts.append([0, 0])
        for a in angles:
            pts.append([r1 * np.cos(a), r1 * np.sin(a)])
        for a in angles[::2]:
            pts.append([r2 * np.cos(a), r2 * np.sin(a)])
        # Add 7 more points
        for a in angles[1::2]:
            pts.append([r2 * np.cos(a), r2 * np.sin(a)])
        inits.append(np.array(pts))
        
        # Random initializations
        rng = np.random.default_rng(42)
        for _ in range(5):
            pts = rng.uniform(-2, 2, size=(n, 2))
            inits.append(pts)
        
        # Concentric circles
        angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles2 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        pts = []
        for a in angles1:
            pts.append([0.5 * np.cos(a), 0.5 * np.sin(a)])
        for a in angles2:
            pts.append([1.0 * np.cos(a), 1.0 * np.sin(a)])
        inits.append(np.array(pts))
        
        return inits

    initializations = generate_initializations()
    best_points = initializations[0].copy()
    best_score = true_score(best_points)

    # For each initialization, run smooth optimization with increasing beta, then local refinement
    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline - 60:
            break
        
        current_points = init_points.flatten()
        init_score = true_score(init_points)
        
        # Smooth optimization with increasing beta
        betas = [20, 50, 100, 200, 500]
        
        for beta in betas:
            if time.time() > deadline - 30:
                break
            
            def obj_func(p):
                return objective_smooth(p, beta)
            
            def grad_func(p):
                return gradient_smooth(p, beta)
            
            try:
                result = minimize(obj_func, current_points, jac=grad_func, method='L-BFGS-B',
                                 options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if result.success:
                    candidate_points = result.x.reshape((n, 2))
                    candidate_score = true_score(candidate_points)
                    
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_points
                        current_points = candidate_points
                    else:
                        current_points = result.x
            except Exception:
                continue
        
        # Local refinement with Powell
        if time.time() < deadline - 20:
            refined = refine_local(best_points)
            refined_score = true_score(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined

    # Final refinement on the best point found
    if time.time() < deadline - 10:
        final_refined = refine_local(best_points, time_limit=10.0)
        final_score = true_score(final_refined)
        if final_score > best_score:
            best_points = final_refined

    return best_points

# EVOLVE-BLOCK-END
