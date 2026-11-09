"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Initialize with a Fibonacci spiral for a good starting configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(x0, axis=1, keepdims=True)
    x0 = x0 / norms
    
    # We will use projected gradient ascent to maximize the minimum pairwise dot product
    # The covering radius is related to the maximum circumradius of the convex hull facets.
    # Maximizing the minimum angular separation (minimizing max dot product of antipodes, or maximizing min dot product for repulsion)
    # A common proxy for covering radius minimization is to maximize the minimum distance between points.
    # Let's use a soft-max penalty on the closest neighbors.
    
    def objective_and_grad(params):
        """
        Objective: We want to maximize the minimum pairwise angular distance.
        This is equivalent to minimizing the maximum pairwise dot product (since cos is decreasing in [0, pi]).
        However, covering radius is more about the farthest point on the sphere from the set.
        A better formulation: Minimize the covering radius.
        The covering radius is the max over y on sphere of min over x in X of angle(y, x).
        
        Let's use a different approach: 
        We can define a function that measures how "uniform" the distribution is.
        A common method is to minimize the sum of exp(-k * min_dist) or similar.
        
        Let's try to minimize the maximum circumradius of the convex hull.
        This is hard to compute directly.
        
        Alternative: Use the fact that if we maximize the minimum pairwise distance, we get a good covering.
        Let's define the objective as -min_pairwise_distance.
        But min is non-differentiable. We can use a soft-min: -log(sum(exp(-beta * d_ij)))
        
        Let's use the approach from the directive: projected gradient ascent to maximize the minimum pairwise dot product.
        Wait, maximizing the minimum dot product means making points closer? No, dot product close to 1 means close.
        We want to maximize the minimum angular separation, which means minimizing the maximum dot product (for points not antipodal).
        
        Let's re-read the directive: "maximize the minimum pairwise dot product".
        If we maximize the minimum dot product, we are pushing points apart?
        No, dot product = cos(angle). If angle is large, cos is small (negative).
        If we want large angles, we want small cos values.
        So we want to minimize the maximum cos value? Or maximize the minimum cos value?
        
        If points are well distributed, the minimum angle is large, so the minimum cos is small (negative).
        Maximizing the minimum cos would make the minimum angle small (points close). That's the opposite.
        
        Maybe the directive means: maximize the minimum distance.
        Distance d = arccos(dot). Maximize min(d_ij) is equivalent to minimize max(dot_ij).
        
        Let's implement: minimize f(X) = max_{i != j} dot(x_i, x_j)
        Using soft-max: f(X) ≈ (1/beta) * log(sum_{i != j} exp(beta * dot(x_i, x_j)))
        
        This is a standard technique.
        """
        X = params.reshape(N, 3)
        # Normalize to ensure they are on the sphere (projection)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / norms
        
        # Compute all pairwise dot products
        dots = X @ X.T
        # Mask out diagonal
        mask = ~np.eye(N, dtype=bool)
        dots = dots[mask]
        
        # Soft-max to approximate max
        beta = 20.0  # Temperature parameter
        # To avoid overflow, subtract max
        max_dot = np.max(dots)
        log_sum_exp = max_dot + np.log(np.sum(np.exp(beta * (dots - max_dot))))
        obj = log_sum_exp / beta
        
        # Gradient computation
        # grad_obj = d/dX [ (1/beta) log(sum exp(beta * dot_ij)) ]
        # = (1/beta) * (1/sum) * sum (beta * exp(beta * dot_ij) * d(dot_ij)/dX)
        # = (1/sum) * sum (exp(beta * dot_ij) * d(dot_ij)/dX)
        
        # Let's compute the weights
        exp_dots = np.exp(beta * (dots - max_dot))
        sum_exp = np.sum(exp_dots)
        weights = exp_dots / sum_exp  # Shape: (N*(N-1),)
        
        # We need to map these weights back to pairs
        # This is tricky. Let's use a vectorized approach for the gradient.
        
        # Alternative: Use automatic differentiation or a simpler gradient approximation.
        # Given time constraints, let's use a finite difference or a simplified gradient.
        
        # Actually, let's use the analytical gradient for the soft-max objective.
        # Let S = sum_{i!=j} exp(beta * x_i . x_j)
        # dS/dx_k = sum_{j != k} beta * exp(beta * x_k . x_j) * x_j + sum_{i != k} beta * exp(beta * x_i . x_k) * x_i
        # = beta * sum_{j != k} exp(beta * x_k . x_j) * (x_j + x_j)  [since dot is symmetric]
        # Wait, x_i . x_k = x_k . x_i
        # So dS/dx_k = 2 * beta * sum_{j != k} exp(beta * x_k . x_j) * x_j
        
        # grad_obj = (1/beta) * (1/S) * dS/dx_k
        # = (1/S) * 2 * sum_{j != k} exp(beta * x_k . x_j) * x_j
        
        # Let's compute this efficiently
        # Create a matrix of exp(beta * dots)
        dots_matrix = X @ X.T
        np.fill_diagonal(dots_matrix, 0)  # Ignore self
        exp_matrix = np.exp(beta * (dots_matrix - max_dot))
        
        # Normalize by sum_exp
        # Note: sum_exp is sum over all i!=j
        # The sum in the gradient for point k is sum_{j != k} exp_matrix[k,j] * x_j
        # This is (exp_matrix @ X)[k]
        
        grad_S = 2.0 * beta * (exp_matrix @ X)
        grad_obj = grad_S / sum_exp
        
        # Project gradient onto the tangent space of the sphere
        # The constraint is ||x_k|| = 1.
        # Projected gradient: grad - (grad . x_k) * x_k
        proj_grad = grad_obj - np.sum(grad_obj * X, axis=1, keepdims=True) * X
        
        # We are minimizing obj, so we use -proj_grad for ascent?
        # No, minimize means we go in direction of negative gradient.
        # But we want to minimize the max dot product, which pushes points apart.
        # So we should minimize obj.
        
        return obj, proj_grad.flatten()
    
    # Use L-BFGS-B or similar? No, we have sphere constraints.
    # We can use a custom projected gradient method.
    
    x = x0.copy()
    lr = 0.01
    n_iter = 500
    
    for _ in range(n_iter):
        obj, grad = objective_and_grad(x)
        # Update
        x = x - lr * grad.reshape(N, 3)
        # Project back to sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
    
    return x

# EVOLVE-BLOCK-END
