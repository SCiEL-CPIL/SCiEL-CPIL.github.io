"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _fibonacci_spiral(N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(v):
    """Normalize rows of v to unit length."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _covering_radius_smooth(pts, grid, beta):
    """
    Compute a smooth approximation of the covering radius (in radians)
    using log-sum-exp over a dense spherical grid.
    
    pts: (N, 3) configuration points (unit vectors)
    grid: (M, 3) grid points on the unit sphere
    beta: temperature for soft-max (larger = closer to hard max)
    
    Returns: (value, gradient w.r.t. pts)
    """
    # Compute all pairwise dot products: (M, N)
    # dot[i, j] = grid[i] . pts[j]
    dots = grid @ pts.T  # (M, N)
    
    # Angular distance: arccos(dot). We want to maximize the minimum distance,
    # i.e., minimize the maximum of arccos(dot) over grid points.
    # arccos(x) is decreasing in x, so maximizing distance = minimizing dot.
    # So we want to minimize max_i arccos(dots[i, :])
    # = minimize max_i (-dots[i, :]) ... no, arccos is nonlinear.
    
    # Better: work with the angles directly.
    # angle[i, j] = arccos(dots[i, j])
    # For each grid point i, nearest distance = min_j angle[i, j]
    # Covering radius = max_i (min_j angle[i, j])
    
    # This is a max-min problem. We can use:
    # For each i, let m_i = min_j angle[i, j]
    # We want to minimize max_i m_i.
    
    # Use soft-min for the inner min, and soft-max for the outer max.
    # soft_min_j angle[i,j] = -1/beta * log(sum_j exp(-beta * angle[i,j]))
    # soft_max_i soft_min_j = -1/beta * log(sum_i exp(-beta * soft_min_j[i]))
    
    # But this is expensive. Alternative: since arccos is monotonic, 
    # min_j arccos(d) = arccos(max_j d)
    # So m_i = arccos(max_j dots[i, j])
    # And we want min over configuration of max_i arccos(max_j dots[i,j])
    
    # Let dmax[i] = max_j dots[i, j]
    # m_i = arccos(dmax[i])
    # Covering = max_i m_i = arccos(min_i dmax[i])  ... no, arccos is decreasing
    # max_i arccos(x_i) = arccos(min_i x_i)
    
    # So covering radius = arccos(min_i dmax[i])
    # We want to minimize this = maximize min_i dmax[i]
    
    # So the objective is: minimize -min_i dmax[i], or equivalently
    # maximize min_i dmax[i]
    
    # Use soft-min: soft_min_i dmax[i] = -1/beta * log(sum_i exp(-beta * dmax[i]))
    # But dmax[i] = max_j dots[i,j] is also non-smooth.
    
    # Two-level soft-max:
    # soft_dmax[i] = 1/beta * log(sum_j exp(beta * dots[i, j]))
    # soft_min = -1/beta * log(sum_i exp(-beta * soft_dmax[i]))
    
    # This is fully differentiable. Let's compute it.
    
    # For numerical stability, use max trick
    # Level 1: for each i, soft_dmax[i] = 1/beta * log(sum_j exp(beta * dots[i,j]))
    exp_dots = np.exp(beta * dots - beta * np.max(dots, axis=1, keepdims=True))  # (M, N)
    logsumexp_dots = np.log(np.sum(exp_dots, axis=1)) + beta * np.max(dots, axis=1)  # (M,)
    soft_dmax = logsumexp_dots / beta  # (M,)
    
    # Level 2: soft_min = -1/beta * log(sum_i exp(-beta * soft_dmax[i]))
    exp_neg = np.exp(-beta * soft_dmax - (-beta * np.min(soft_dmax)))  # (M,)
    logsumexp_neg = np.log(np.sum(exp_neg)) + (-beta * np.min(soft_dmax))
    soft_min = -logsumexp_neg / beta  # scalar
    
    # Covering radius (smooth) = arccos(soft_min)
    # Clamp soft_min to [-1, 1]
    soft_min_clamped = np.clip(soft_min, -1.0 + 1e-12, 1.0 - 1e-12)
    value = np.arccos(soft_min_clamped)
    
    # Gradient computation:
    # value = arccos(soft_min)
    # dvalue/dsoft_min = -1 / sqrt(1 - soft_min^2)
    
    # soft_min = -1/beta * log(S) where S = sum_i exp(-beta * soft_dmax[i])
    # dsoft_min/dsoft_dmax[i] = exp(-beta * soft_dmax[i]) / S
    
    # soft_dmax[i] = 1/beta * log(T_i) where T_i = sum_j exp(beta * dots[i,j])
    # dsoft_dmax[i]/ddots[i,j] = exp(beta * dots[i,j]) / T_i
    
    # ddots[i,j]/dpts[j] = grid[i]  (since dots[i,j] = grid[i] . pts[j])
    
    # So dvalue/dpts[j] = sum_i [dvalue/dsoft_min * dsoft_min/dsoft_dmax[i] * dsoft_dmax[i]/ddots[i,j] * grid[i]]
    
    # Compute weights
    # w1[i] = exp(-beta * soft_dmax[i]) / S
    S = np.sum(exp_neg)
    w1 = exp_neg / S  # (M,)
    
    # w2[i, j] = exp(beta * dots[i,j]) / T_i
    T = np.sum(exp_dots, axis=1)  # (M,)
    w2 = exp_dots / T[:, np.newaxis]  # (M, N)
    
    # dvalue/dsoft_min
    dsdm = -1.0 / np.sqrt(np.maximum(1e-12, 1.0 - soft_min_clamped * soft_min_clamped))
    
    # Gradient w.r.t. dots: grad_dots[i,j] = dsdm * w1[i] * w2[i,j] / beta * beta
    # Wait, let me redo the chain rule carefully.
    
    # value = arccos(soft_min)
    # dvalue/dsoft_min = -1/sqrt(1-soft_min^2)
    
    # soft_min = -1/beta * log(S)
    # S = sum_i exp(-beta * soft_dmax[i])
    # dsoft_min/dsoft_dmax[i] = -1/beta * (1/S) * (-beta * exp(-beta * soft_dmax[i]))
    #                          = exp(-beta * soft_dmax[i]) / S = w1[i]
    
    # soft_dmax[i] = 1/beta * log(T_i)
    # T_i = sum_j exp(beta * dots[i,j])
    # dsoft_dmax[i]/ddots[i,j] = 1/beta * (1/T_i) * (beta * exp(beta * dots[i,j]))
    #                          = exp(beta * dots[i,j]) / T_i = w2[i,j]
    
    # dots[i,j] = grid[i] . pts[j]
    # ddots[i,j]/dpts[j,k] = grid[i,k]
    
    # So dvalue/dpts[j,k] = sum_i [dsdm * w1[i] * w2[i,j] * grid[i,k]]
    
    # = dsdm * sum_i [w1[i] * w2[i,j] * grid[i,k]]
    
    # Let's compute: for each j, grad_pts[j] = dsdm * sum_i [w1[i] * w2[i,j] * grid[i]]
    
    # grad_pts[j] = dsdm * (grid.T @ (w1 * w2[:, j])) ... let me think
    # sum_i w1[i] * w2[i,j] * grid[i,k] for each k
    # = (grid.T @ (w1 * w2[:, j]))[k]
    
    # So grad_pts = dsdm * grid.T @ (w1[:, np.newaxis] * w2)  ... shape (N, M) @ (M, N) ... no
    # w1 is (M,), w2 is (M, N)
    # w1[:, np.newaxis] * w2 is (M, N)
    # grid.T is (3, M)
    # grid.T @ (w1[:, np.newaxis] * w2) is (3, N)
    # We want (N, 3), so transpose.
    
    weighted = w1[:, np.newaxis] * w2  # (M, N)
    grad_pts = dsdm * (grid.T @ weighted).T  # (N, 3)
    
    # Project gradient to tangent space of sphere (remove radial component)
    # grad_tangent = grad_pts - (grad_pts . pts) * pts
    radial = np.sum(grad_pts * pts, axis=1, keepdims=True)
    grad_tangent = grad_pts - radial * pts
    
    return value, grad_tangent


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    # Dense spherical grid for evaluation
    # Use a grid of ~10000 points
    n_theta = 100
    n_phi = 100
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    grid = np.column_stack([
        np.sin(THETA) * np.cos(PHI).flatten(),
        np.sin(THETA) * np.sin(PHI).flatten(),
        np.cos(THETA).flatten()
    ])
    grid = _normalize(grid)
    M = len(grid)
    
    # Initialize with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Optimize using L-BFGS-B
    # We'll use a moderate beta for the soft-max
    beta = 20.0
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _normalize(p)
        val, grad = _covering_radius_smooth(p, grid, beta)
        return val, grad.flatten()
    
    # Use L-BFGS-B
    result = minimize(
        objective_and_grad,
        pts.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    pts_opt = _normalize(result.x.reshape(N, 3))
    
    # Refinement: do a second pass with a higher beta (closer to hard max)
    beta2 = 50.0
    
    def objective_and_grad2(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _normalize(p)
        val, grad = _covering_radius_smooth(p, grid, beta2)
        return val, grad.flatten()
    
    result2 = minimize(
        objective_and_grad2,
        pts_opt.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    pts_final = _normalize(result2.x.reshape(N, 3))
    
    return pts_final
# EVOLVE-BLOCK-END
