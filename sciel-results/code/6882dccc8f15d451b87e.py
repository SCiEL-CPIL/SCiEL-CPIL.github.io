"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math


def _log_energy_vectorized(pts):
    """Compute logarithmic energy using vectorized operations."""
    # pts shape (N, 3)
    # Compute pairwise distances
    # diff shape (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid log(0)
    mask = ~np.eye(N, dtype=bool)
    # We only want i < j to avoid double counting and self-interaction
    # But for energy, sum over i<j.
    # Let's just sum over all i!=j and divide by 2? No, log energy is sum i<j.
    # Using upper triangle indices
    i_upper, j_upper = np.triu_indices(N, k=1)
    d = dists[i_upper, j_upper]
    
    # Handle very small distances
    d = np.maximum(d, 1e-12)
    
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())


def _gradient_vectorized(pts):
    """Compute projected gradient of logarithmic energy using vectorized operations."""
    # pts shape (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    
    # Gradient term for pair (i, j) on x_i is -(x_i - x_j) / d^2
    # We need sum over j != i of -(x_i - x_j) / d_ij^2
    # This is equivalent to sum over j != i of (x_j - x_i) / d_ij^2
    
    # Create weight matrix W where W[i,j] = 1/d_ij^2 if i!=j else 0
    W = np.zeros((N, N))
    mask = ~np.eye(N, dtype=bool)
    W[mask] = 1.0 / (dists_safe[mask] ** 2)
    
    # grad[i] = sum_j W[i,j] * (pts[j] - pts[i])
    # grad = W @ pts - W.sum(axis=1, keepdims=True) * pts
    # Let's verify: sum_j W[i,j] * pts[j] - sum_j W[i,j] * pts[i]
    # = (W @ pts)[i] - (sum_j W[i,j]) * pts[i]
    
    sum_W = W.sum(axis=1, keepdims=True)
    grad = W @ pts - sum_W * pts
    
    # Project to tangent space: grad_tan = grad - (grad . pts) * pts
    dot_prod = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot_prod * pts
    
    return grad


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms

    # Gradient Descent with Momentum
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(pts)
    
    # Adaptive learning rate / Line search might be better
    # Let's do a simple step with decay
    # Since we have 120s, we can run many iterations
    
    # Pre-compute energy for monitoring? No, just run.
    
    # We will run for a fixed number of iterations or until convergence
    # N=100 is small enough for vectorized ops to be fast (~ms per iter)
    
    num_iters = 20000
    
    for k in range(num_iters):
        grad = _gradient_vectorized(pts)
        
        # Update velocity
        v = momentum * v - lr * grad
        pts_new = pts + v
        
        # Normalize back to sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts_new = pts_new / norms
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(pts_new)):
            break
            
        pts = pts_new
        
        # Decay learning rate slowly
        if k % 1000 == 0 and k > 0:
            lr *= 0.95
            
    # Final normalization
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    return pts
# EVOLVE-BLOCK-END
