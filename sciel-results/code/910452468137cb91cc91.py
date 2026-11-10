# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    deadline = time.time() + 620  # safety margin before 630s kill

    def score(points):
        """Compute R = (d_min / d_max)^2"""
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(flat):
        """Negative score for minimization"""
        pts = flat.reshape(n, 2)
        # Center to remove translational invariance
        pts_centered = pts - np.mean(pts, axis=0)
        return -score(pts_centered)

    def objective_and_grad(flat):
        """Objective with numerical gradient"""
        pts = flat.reshape(n, 2)
        pts_centered = pts - np.mean(pts, axis=0)
        val = -score(pts_centered)
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(flat)
        for i in range(len(flat)):
            orig = flat[i]
            flat[i] = orig + eps
            f_plus = -score(flat.reshape(n, 2) - np.mean(flat.reshape(n, 2), axis=0))
            flat[i] = orig - eps
            f_minus = -score(flat.reshape(n, 2) - np.mean(flat.reshape(n, 2), axis=0))
            flat[i] = orig
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return val, grad

    # Generate diverse initial configurations
    def generate_grid_init(scale=1.0):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        pts = np.column_stack((X.ravel(), Y.ravel())) * scale
        return pts

    def generate_ring_init(r1, r2, n1=8, n2=8):
        """Two concentric rings"""
        angles1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + np.pi/n2  # offset
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def generate_hex_init():
        """Hexagonal-like arrangement"""
        # Center + 6 inner + 9 outer
        center = np.array([[0, 0]])
        r1 = 1.0
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        ring1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        r2 = 1.732
        angles2 = np.linspace(0, 2*np.pi, 9, endpoint=False) + np.pi/9
        ring2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([center, ring1, ring2])

    def generate_random_init(seed):
        np.random.seed(seed)
        return np.random.randn(n, 2)

    # List of initial configurations
    inits = []
    inits.append(generate_grid_init(1.0))
    inits.append(generate_grid_init(0.8))
    inits.append(generate_grid_init(1.2))
    inits.append(generate_ring_init(0.8, 1.5))
    inits.append(generate_ring_init(1.0, 1.8))
    inits.append(generate_ring_init(0.6, 1.2))
    inits.append(generate_hex_init())
    for seed in range(10):
        inits.append(generate_random_init(seed))

    best_score = -np.inf
    best_points = inits[0].copy()

    for init in inits:
        if time.time() > deadline:
            break

        # Center the initial configuration
        init_centered = init - np.mean(init, axis=0)
        flat_init = init_centered.flatten()

        # Try L-BFGS-B optimization
        try:
            result = minimize(
                objective,
                flat_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_pts = result.x.reshape(n, 2)
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            s = score(opt_pts)

            if s > best_score:
                best_score = s
                best_points = opt_pts.copy()

        except Exception:
            pass

        # Also try Nelder-Mead for robustness
        if time.time() > deadline:
            break
        try:
            result_nm = minimize(
                objective,
                flat_init,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_pts_nm = result_nm.x.reshape(n, 2)
            opt_pts_nm = opt_pts_nm - np.mean(opt_pts_nm, axis=0)
            s_nm = score(opt_pts_nm)

            if s_nm > best_score:
                best_score = s_nm
                best_points = opt_pts_nm.copy()

        except Exception:
            pass

    # Final refinement: try a few more random perturbations from best
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            perturbed = best_points + 0.01 * np.random.randn(n, 2)
            perturbed = perturbed - np.mean(perturbed, axis=0)
            flat_p = perturbed.flatten()
            try:
                result = minimize(
                    objective,
                    flat_p,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                opt_pts = result.x.reshape(n, 2)
                opt_pts = opt_pts - np.mean(opt_pts, axis=0)
                s = score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_points = opt_pts.copy()
            except Exception:
                pass

    return best_points

# EVOLVE-BLOCK-END
