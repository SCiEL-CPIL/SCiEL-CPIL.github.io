# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 590  # Leave 40s margin from 630s

    # Precompute indices for all triples of points
    idx = np.array(list((i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)))

    # Spectral basis: Fourier modes for representing point positions
    # We use a low-rank basis to constrain points to a smooth manifold
    # Basis: constant + first few harmonics in x and y
    # This gives us a parameterization: point_i = sum_k c_k * phi_k(i)
    # where phi_k are basis functions evaluated at point index i

    def spectral_to_points(coeffs):
        """
        Convert spectral coefficients to point coordinates.
        coeffs: (num_modes,) array of coefficients
        Returns: (n, 2) array of point coordinates

        We use a basis of:
        - DC (mean position)
        - First harmonic (x, y) - circular mode
        - Second harmonic (x, y) - for hexagonal symmetry
        - Third harmonic (x, y) - for triangular symmetry
        - Small random modes for flexibility
        """
        num_coeffs = len(coeffs)

        # Point indices for basis evaluation
        # We'll use a circular arrangement as the "natural" index
        # point i is at angle 2*pi*i/n on the unit circle
        point_angles = 2 * np.pi * np.arange(n) / n

        # Build basis matrix: (n, num_coeffs)
        # Each column is a basis function evaluated at all n points
        basis = np.zeros((n, num_coeffs))

        # DC mode (indices 0, 1 for x, y)
        if num_coeffs > 0:
            basis[:, 0] = 1.0  # x offset
        if num_coeffs > 1:
            basis[:, 1] = 1.0  # y offset

        # First harmonic (indices 2, 3 for x, y)
        if num_coeffs > 2:
            basis[:, 2] = np.cos(point_angles)
        if num_coeffs > 3:
            basis[:, 3] = np.sin(point_angles)

        # Second harmonic (indices 4, 5 for x, y)
        if num_coeffs > 4:
            basis[:, 4] = np.cos(2 * point_angles)
        if num_coeffs > 5:
            basis[:, 5] = np.sin(2 * point_angles)

        # Third harmonic (indices 6, 7 for x, y)
        if num_coeffs > 6:
            basis[:, 6] = np.cos(3 * point_angles)
        if num_coeffs > 7:
            basis[:, 7] = np.sin(3 * point_angles)

        # Fourth harmonic (indices 8, 9 for x, y)
        if num_coeffs > 8:
            basis[:, 8] = np.cos(4 * point_angles)
        if num_coeffs > 9:
            basis[:, 9] = np.sin(4 * point_angles)

        # Fifth harmonic (indices 10, 11 for x, y)
        if num_coeffs > 10:
            basis[:, 10] = np.cos(5 * point_angles)
        if num_coeffs > 11:
            basis[:, 11] = np.sin(5 * point_angles)

        # Sixth harmonic (indices 12, 13 for x, y)
        if num_coeffs > 12:
            basis[:, 12] = np.cos(6 * point_angles)
        if num_coeffs > 13:
            basis[:, 13] = np.sin(6 * point_angles)

        # Compute points: (n, num_coeffs) @ (num_coeffs, 2) -> (n, 2)
        # But our coeffs are interleaved: [x0, y0, x1, y1, ...]
        # Let's restructure: separate x and y coefficients
        x_coeffs = coeffs[0::2]
        y_coeffs = coeffs[1::2]

        # For each mode, we have cos and sin components
        # Reconstruct: for mode k, x gets c_{2k}*cos(k*theta) + c_{2k+1}*sin(k*theta)
        # This is getting complex. Let's use a simpler approach.

        # Simpler: just use the first num_coeffs//2 + 1 modes
        # and compute directly
        points = np.zeros((n, 2))

        # DC
        if num_coeffs >= 2:
            points[:, 0] += coeffs[0]
            points[:, 1] += coeffs[1]

        # For each harmonic k (starting from 1)
        for k in range(1, (num_coeffs // 2) + 1):
            x_idx = 2 * k
            y_idx = 2 * k + 1
            if x_idx >= num_coeffs or y_idx >= num_coeffs:
                break

            # x component: coeffs[x_idx] * cos(k*theta) + coeffs[x_idx+1] * sin(k*theta) if available
            # Actually let's use a cleaner parameterization
            # For harmonic k: amplitude_a * cos(k*theta + phase)
            # = amplitude_a * cos(phase) * cos(k*theta) - amplitude_a * sin(phase) * sin(k*theta)

            # Use two coefficients per harmonic: a_k for cos, b_k for sin
            a_k = coeffs[x_idx]
            b_k = coeffs[y_idx]

            points[:, 0] += a_k * np.cos(k * point_angles)
            points[:, 1] += b_k * np.sin(k * point_angles)

        return points

    def min_triangle_area_vectorized(points):
        """
        Calculate the minimum triangle area among all triples of points using vectorization.
        points: (n, 2)
        """
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        return np.min(areas)

    def all_triangle_areas(points):
        """
        Calculate all triangle areas among all triples of points using vectorization.
        points: (n, 2)
        Returns: array of shape (286,)
        """
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        return areas

    def hull_area(points):
        """Calculate the convex hull area of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def soft_min(areas, beta):
        """
        Smooth approximation of min using log-sum-exp trick.
        soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
        As beta -> inf, soft_min -> min
        """
        shifted = -beta * areas
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        return -log_sum / beta

    def objective_spectral(coeffs, beta=50.0):
        """
        Objective in spectral coefficient space.
        Converts coefficients to points, then computes smooth objective.
        """
        points = spectral_to_points(coeffs)

        # Check if points are valid
        if not np.all(np.isfinite(points)):
            return 1e10

        areas = all_triangle_areas(points)
        A_min_smooth = soft_min(areas, beta)

        if A_min_smooth < 1e-15:
            return 1e10

        # Use bounding box area as a fast proxy for hull area
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        bbox_area = (x_max - x_min) * (y_max - y_min)

        if bbox_area < 1e-15:
            return 1e10

        return -A_min_smooth / bbox_area

    def objective_exact(points):
        """
        Exact objective: minimize -A_min / A_hull.
        Uses actual ConvexHull area.
        """
        A_hull = hull_area(points)
        if A_hull <= 1e-10:
            return 0.0
        
        A_min = min_triangle_area_vectorized(points)
        
        if A_min < 1e-12:
            return 0.0
            
        return -A_min / A_hull

    def objective_bbox(points):
        """
        Fast objective using bounding box area as proxy for hull area.
        This is much faster than ConvexHull and provides a smooth landscape.
        The ratio A_min/A_bbox is correlated with A_min/A_hull.
        """
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        bbox_area = (x_max - x_min) * (y_max - y_min)
        
        if bbox_area < 1e-15:
            return 0.0
        
        A_min = min_triangle_area_vectorized(points)
        
        if A_min < 1e-12:
            return 0.0
            
        return -A_min / bbox_area

    def score(points):
        """Calculate the actual score: A_min / A_hull"""
        A_hull = hull_area(points)
        if A_hull <= 1e-10:
            return 0.0
        A_min = min_triangle_area_vectorized(points)
        if A_min < 1e-12:
            return 0.0
        return A_min / A_hull

    # Generate diverse initial configurations in spectral space
    rng = np.random.default_rng(seed=42)

    # Number of spectral modes to use
    # We'll try different numbers of modes
    num_modes_list = [6, 8, 10, 12, 14, 16]

    best_points = None
    best_score = -float('inf')

    # Phase 1: Optimize in spectral space with various numbers of modes
    for num_modes in num_modes_list:
        if time.time() - start_time > time_limit - 120:
            break

        num_coeffs = num_modes  # Each mode has 2 coefficients (cos, sin)

        # Generate initial coefficient vectors
        init_coeffs_list = []

        # 1. Regular polygon: only first harmonic
        coeffs_reg = np.zeros(num_coeffs)
        coeffs_reg[2] = 1.0  # cos(1*theta) for x
        coeffs_reg[3] = 1.0  # sin(1*theta) for y
        init_coeffs_list.append(coeffs_reg)

        # 2. Hexagonal: first + second harmonics
        coeffs_hex = np.zeros(num_coeffs)
        coeffs_hex[2] = 1.0
        coeffs_hex[3] = 1.0
        if num_coeffs > 5:
            coeffs_hex[4] = 0.3
            coeffs_hex[5] = 0.3
        init_coeffs_list.append(coeffs_hex)

        # 3. Triangular: first + third harmonics
        coeffs_tri = np.zeros(num_coeffs)
        coeffs_tri[2] = 1.0
        coeffs_tri[3] = 1.0
        if num_coeffs > 7:
            coeffs_tri[6] = 0.2
            coeffs_tri[7] = 0.2
        init_coeffs_list.append(coeffs_tri)

        # 4. Random small coefficients
        for _ in range(5):
            coeffs_rand = rng.normal(0, 0.3, num_coeffs)
            # Ensure first harmonic is dominant
            coeffs_rand[2:4] += 0.5
            init_coeffs_list.append(coeffs_rand)

        # 5. Perturbed regular polygon
        for _ in range(3):
            coeffs_pert = np.zeros(num_coeffs)
            coeffs_pert[2] = 1.0 + rng.normal(0, 0.1)
            coeffs_pert[3] = 1.0 + rng.normal(0, 0.1)
            for k in range(4, num_coeffs):
                coeffs_pert[k] = rng.normal(0, 0.05)
            init_coeffs_list.append(coeffs_pert)

        # Optimize each initial configuration
        for beta in [30.0, 80.0]:
            if time.time() - start_time > time_limit - 100:
                break

            for init_coeffs in init_coeffs_list:
                if time.time() - start_time > time_limit - 90:
                    break

                try:
                    result = minimize(
                        lambda c: objective_spectral(c, beta),
                        init_coeffs,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 2000,
                            'xatol': 1e-9,
                            'fatol': 1e-11,
                            'adaptive': True,
                            'initial_simplex': 0.1
                        }
                    )

                    if result.fun < 1e9:  # Valid result
                        points = spectral_to_points(result.x)
                        if np.all(np.isfinite(points)):
                            exact_score = score(points)
                            if exact_score > best_score:
                                best_score = exact_score
                                best_points = points.copy()

                except Exception:
                    pass

    # Phase 2: Refine best points in coordinate space
    if best_points is not None and time.time() - start_time < time_limit - 60:
        try:
            result = minimize(
                objective_exact,
                best_points,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-11,
                    'fatol': 1e-13,
                    'adaptive': True,
                    'initial_simplex': 0.03
                }
            )

            if result.fun < 0:
                exact_score = -result.fun
                if exact_score > best_score:
                    best_score = exact_score
                    best_points = result.x.copy()
        except Exception:
            pass

        # Multiple small perturbations to escape local minima
        if time.time() - start_time < time_limit - 40:
            for pert_scale in [0.02, 0.01, 0.005]:
                if time.time() - start_time > time_limit - 35:
                    break
                rng_local = np.random.default_rng(seed=12345)
                for _ in range(3):
                    if time.time() - start_time > time_limit - 35:
                        break
                    perturbed = best_points + rng_local.normal(0, pert_scale, (n, 2))
                    try:
                        result = minimize(
                            objective_bbox,
                            perturbed,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 3000,
                                'xatol': 1e-10,
                                'fatol': 1e-12,
                                'adaptive': True,
                                'initial_simplex': 0.01
                            }
                        )

                        if result.fun < 0:
                            exact_score = score(result.x)
                            if exact_score > best_score:
                                best_score = exact_score
                                best_points = result.x.copy()
                    except Exception:
                        pass

    # Phase 3: High-precision polish with multiple simplex sizes
    if best_points is not None and time.time() - start_time < time_limit - 40:
        for simplex_size in [0.01, 0.005, 0.002, 0.001]:
            if time.time() - start_time > time_limit - 30:
                break
            try:
                result = minimize(
                    objective_bbox,
                    best_points,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-13,
                        'fatol': 1e-15,
                        'adaptive': True,
                        'initial_simplex': simplex_size
                    }
                )

                if result.fun < 0:
                    exact_score = score(result.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
            except Exception:
                pass

    # Phase 4: Final ultra-precise polish with exact objective
    if best_points is not None and time.time() - start_time < time_limit - 20:
        try:
            result = minimize(
                objective_exact,
                best_points,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True,
                    'initial_simplex': 0.0005
                }
            )

            if result.fun < 0:
                exact_score = -result.fun
                if exact_score > best_score:
                    best_score = exact_score
                    best_points = result.x.copy()
        except Exception:
            pass

        # One more bbox polish if time remains
        if best_points is not None and time.time() - start_time < time_limit - 10:
            try:
                result = minimize(
                    objective_bbox,
                    best_points,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-14,
                        'fatol': 1e-16,
                        'adaptive': True,
                        'initial_simplex': 0.0003
                    }
                )

                if result.fun < 0:
                    exact_score = score(result.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
            except Exception:
                pass

    # Final fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-END
