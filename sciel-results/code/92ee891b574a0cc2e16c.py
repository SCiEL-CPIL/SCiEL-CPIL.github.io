# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_score(polygon: np.ndarray) -> float:
    """Compute the task score for a (20,2) polygon. Higher is better."""
    n = len(polygon)
    if n != 20:
        return 0.0
    
    # Check finite
    if not np.all(np.isfinite(polygon)):
        return 0.0
    
    # Normalize: subtract centroid
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    
    # Mean distance to centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    centered = centered / mean_dist
    
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(centered[i] - centered[j]) < 1e-10:
                return 0.0
    
    # Check strictly convex: every turn nonzero and same sign, total turning +-2*pi
    # Compute edge vectors
    edges = np.roll(centered, -1, axis=0) - centered
    # Check no zero-length edges
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return 0.0
    
    # Compute cross products of consecutive edges (turn angles)
    turns = np.cross(edges, np.roll(edges, -1, axis=0))
    # All turns must be nonzero and same sign
    if np.any(np.abs(turns) < 1e-10):
        return 0.0
    signs = np.sign(turns)
    if not np.all(signs == signs[0]):
        return 0.0
    
    # Total turning should be +-2*pi (for a simple convex polygon, sum of exterior angles = 2*pi)
    # For a convex polygon traversed in order, the sum of signed turns should be 2*pi (or -2*pi)
    total_turn = np.sum(turns)
    # Actually, for convex polygon, the sum of exterior angles is 2*pi.
    # The cross product sum relates to area, not directly to total turning.
    # Let's compute the actual turning angles.
    angles = np.zeros(n)
    for i in range(n):
        e1 = edges[i]
        e2 = edges[(i+1) % n]
        cos_a = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        cos_a = np.clip(cos_a, -1.0, 1.0)
        sin_a = np.cross(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        angles[i] = np.arctan2(sin_a, cos_a)
    
    total_turning = np.sum(angles)
    if abs(abs(total_turning) - 2 * np.pi) > 1e-6:
        return 0.0
    
    # Now compute vertex scores
    min_score = 1.0
    for i in range(n):
        v = centered[i]
        # Distances to all other vertices
        dists = np.zeros(n)
        for j in range(n):
            if i == j:
                dists[j] = 0.0
            else:
                dists[j] = np.linalg.norm(v - centered[j])
        
        # Sort distances to other vertices (exclude self, which is 0)
        other_dists = np.sort([dists[j] for j in range(n) if j != i])
        # other_dists has 19 elements, d_0 <= ... <= d_18
        
        # Check if any < 0.05
        if np.any(other_dists < 0.05):
            min_score = 0.0
            break
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = other_dists[1:] - other_dists[:-1]  # 18 elements, diffs[i] = d_{i+1} - d_i
        k = np.argmin(diffs)
        
        D = (other_dists[k] + other_dists[k+1]) / 2.0
        if D < 0.1:
            min_score = 0.0
            break
        
        # Take four distances
        if k == 0:
            selected = other_dists[0:4]
        elif k == 17:
            selected = other_dists[15:19]
        else:
            selected = other_dists[k-1:k+3]
        
        # raw = mean over these four d of max(d/D, D/d)
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        vertex_score = min(1.0, 1.0 / raw)
        min_score = min(min_score, vertex_score)
    
    return min_score


def _fourier_to_polygon(params: np.ndarray) -> np.ndarray:
    """
    Convert Fourier parameters to polygon vertices.
    params: array of shape (num_modes * 2,) where for each mode k,
            we have amplitude and phase for the angular deviation.
    The polygon is defined by:
      r_i = 1 + sum_k a_k * cos(k * theta_i + phi_k)
      where theta_i = 2*pi*i/n
    This ensures smoothness and symmetry.
    """
    n = 20
    num_modes = len(params) // 2
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    r = np.ones(n)
    for m in range(num_modes):
        a = params[2*m]
        phi = params[2*m + 1]
        k = m + 1  # mode number (1, 2, 3, ...)
        r += a * np.cos(k * theta + phi)
    
    # Ensure r > 0 for all vertices (to keep polygon star-shaped)
    r = np.maximum(r, 0.1)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    polygon = np.stack([x, y], axis=1)
    return polygon


def _nelder_mead(f, x0, bounds=None, max_iter=2000, tol=1e-8):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x = x0.copy()
        x[i] += 0.05 if x[i] == 0 else 0.05 * x[i]
        simplex.append(x)
    
    simplex_f = [f(x) for x in simplex]
    
    for it in range(max_iter):
        # Sort simplex
        order = np.argsort(simplex_f)
        simplex = [simplex[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]
        
        if abs(simplex_f[-1] - simplex_f[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        # Reflection
        alpha = 1.0
        x_ref = centroid + alpha * (centroid - worst)
        f_ref = f(x_ref)
        
        if f_ref < simplex_f[0]:
            # Expansion
            gamma = 2.0
            x_exp = centroid + gamma * (centroid - worst)
            f_exp = f(x_exp)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                simplex_f[-1] = f_exp
            else:
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
        elif f_ref < simplex_f[-2]:
            simplex[-1] = x_ref
            simplex_f[-1] = f_ref
        else:
            # Contraction
            rho = 0.5
            x_con = centroid + rho * (worst - centroid)
            f_con = f(x_con)
            if f_con < simplex_f[-1]:
                simplex[-1] = x_con
                simplex_f[-1] = f_con
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex_f[i] = f(simplex[i])
    
    return simplex[0], simplex_f[0]


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with 20 vertices using Fourier series parameterization
    and Nelder-Mead optimization to maximize the minimum vertex score.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit
    
    n = 20
    num_modes = 3  # Use modes 1, 2, 3
    num_params = num_modes * 2
    
    # Initial guess: small perturbations
    np.random.seed(42)
    best_params = np.zeros(num_params)
    best_score = _compute_score(_fourier_to_polygon(best_params))
    
    # Try multiple restarts with different initial conditions
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 60:  # Safety margin
            break
        
        # Random initial parameters
        if restart == 0:
            params0 = np.zeros(num_params)
        else:
            params0 = 0.1 * np.random.randn(num_params)
            # Ensure mode 1 amplitude is small to keep polygon close to circle
            params0[0] *= 0.5
        
        # Objective: negative of score (we want to maximize score)
        def objective(params):
            if time.time() > deadline - 30:
                return 0.0
            polygon = _fourier_to_polygon(params)
            return -_compute_score(polygon)
        
        # Run Nelder-Mead
        x_opt, f_opt = _nelder_mead(objective, params0, max_iter=500, tol=1e-7)
        
        score_opt = -f_opt
        if score_opt > best_score:
            best_score = score_opt
            best_params = x_opt.copy()
    
    # Final result
    polygon = _fourier_to_polygon(best_params)
    return polygon


# EVOLVE-BLOCK-END
