"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    
    Approach: Non-convex optimization via projected gradient ascent on a
    soft-max penalty targeting the closest neighbors. We maximize the minimum
    pairwise separation (which correlates with minimizing covering radius) using
    a smooth approximation of the min function.
    
    The objective is to maximize the minimum angular distance, which we approximate
    by minimizing a soft-max of the negative dot products (i.e., maximizing the
    minimum dot product). We use projected gradient descent on the sphere.
    """
    np.random.seed(42)
    
    # Initialize with Fibonacci spiral for a good starting point
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    
    # Parameters for soft-max approximation
    beta = 50.0  # Higher beta -> sharper approximation to min
    lr = 0.01    # Learning rate
    n_iter = 5000
    
    for iteration in range(n_iter):
        # Compute pairwise dot products
        # D[i,j] = dot(X[i], X[j])
        D = X @ X.T
        np.fill_diagonal(D, -np.inf)  # Exclude self
        
        # For each point i, find the minimum dot product with other points
        # We want to maximize the minimum dot product (minimize covering radius)
        # Using soft-min: min_j D[i,j] ≈ -logsumexp(-beta * D[i,j]) / beta
        
        # Soft-min of D[i,j] over j != i
        neg_beta_D = -beta * D
        # Subtract max for numerical stability
        neg_beta_D_max = np.max(neg_beta_D, axis=1, keepdims=True)
        logsumexp = neg_beta_D_max + np.log(np.sum(np.exp(neg_beta_D - neg_beta_D_max), axis=1, keepdims=True))
        soft_min = -logsumexp / beta
        
        # Gradient of soft_min w.r.t. D[i,j]
        # d(soft_min_i)/d(D[i,j]) = -exp(-beta * D[i,j]) / sum_k exp(-beta * D[i,k])
        weights = np.exp(neg_beta_D - neg_beta_D_max)
        weights /= np.sum(weights, axis=1, keepdims=True)
        
        # Gradient of the objective (sum of soft_min) w.r.t. X
        # We want to maximize sum_i soft_min_i, so we do gradient ascent
        # d(soft_min_i)/d(X) involves chain rule through D[i,j] = X[i] · X[j]
        
        # For point i, gradient contribution from soft_min_i:
        # d(soft_min_i)/d(X[i]) = -sum_j weights[i,j] * beta * X[j]  (from D[i,j] terms where i is first)
        #                         -sum_j weights[j,i] * beta * X[i]  (from D[j,i] terms where i is second)
        # Wait, let me be more careful.
        
        # soft_min_i = -1/beta * log(sum_j exp(-beta * D[i,j]))
        # d(soft_min_i)/d(D[i,j]) = -1/beta * (-beta) * exp(-beta * D[i,j]) / sum_k exp(-beta * D[i,k])
        #                          = exp(-beta * D[i,j]) / sum_k exp(-beta * D[i,k])
        #                          = weights[i,j]
        
        # Now D[i,j] = X[i] · X[j]
        # d(D[i,j])/d(X[i]) = X[j]
        # d(D[i,j])/d(X[j]) = X[i]
        
        # So d(soft_min_i)/d(X[i]) = sum_j weights[i,j] * X[j]
        # And for the second index: d(soft_min_j)/d(X[i]) = sum_k weights[j,k] * (if k==i) X[j] ... 
        # Actually, X[i] appears in D[j,i] for all j != i, and in D[i,j] for all j != i.
        
        # Let's compute gradient properly:
        # grad_i = sum_j weights[i,j] * X[j]  (from D[i,j] where i is first index)
        #        + sum_j weights[j,i] * X[j]  (from D[j,i] where i is second index, d(D[j,i])/d(X[i]) = X[j])
        
        # Wait, d(D[j,i])/d(X[i]) = X[j], and d(soft_min_j)/d(D[j,i]) = weights[j,i]
        # So contribution to grad from soft_min_j through D[j,i] is weights[j,i] * X[j]
        
        # So total gradient for X[i]:
        # grad_X[i] = sum_j weights[i,j] * X[j] + sum_j weights[j,i] * X[j]
        
        # Let's compute this efficiently
        # First term: weights @ X  (each row i: sum_j weights[i,j] * X[j])
        grad_first = weights @ X
        
        # Second term: for each i, sum_j weights[j,i] * X[j]
        # This is (weights.T @ X) but transposed... 
        # Actually, (weights.T @ X)[i] = sum_j weights.T[i,j] * X[j] = sum_j weights[j,i] * X[j]
        grad_second = weights.T @ X
        
        grad = grad_first + grad_second
        
        # We want to maximize the sum of soft_min, so we do gradient ascent
        # But we need to project back to the sphere
        
        # Adaptive learning rate
        if iteration % 100 == 0:
            # Decay learning rate
            lr = 0.01 * (0.995 ** (iteration // 100))
        
        X_new = X + lr * grad
        X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
        
        # Check convergence
        if iteration % 100 == 0:
            min_dot = np.min(D)
            #print(f"Iter {iteration}, min dot: {min_dot:.6f}, angle: {np.degrees(np.arccos(min_dot)):.4f}")
        
        X = X_new
    
    # Final normalization
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    return X
# EVOLVE-BLOCK-END
