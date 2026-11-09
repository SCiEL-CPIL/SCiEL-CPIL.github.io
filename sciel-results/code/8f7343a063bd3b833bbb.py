# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _normalize(polygon):
    """Normalize polygon: subtract centroid, divide by mean distance to centroid."""
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    distances = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(distances)
    if mean_dist <= 0.01:
        return None
    normalized = centered / mean_dist
    return normalized

def _is_strictly_convex(polygon):
    """Check if polygon is strictly convex."""
    n = len(polygon)
    if n < 3:
        return False
    
    # Check for repeated points
    for i in range(n):
        if np.linalg.norm(polygon[i] - polygon[(i+1)%n]) < 1e-9:
            return False
            
    # Check turns
    turns = []
    for i in range(n):
        p0 = polygon[(i-1)%n]
        p1 = polygon[i]
        p2 = polygon[(i+1)%n]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        # Cross product in 2D
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if np.abs(cross) < 1e-9:
            return False
        turns.append(cross)
        
    # All turns must have the same sign
    if not np.all(np.sign(turns) == np.sign(turns[0])):
        return False
        
    # Total turning must be +-2*pi (approx)
    # For a simple convex polygon, the sum of exterior angles is 2*pi.
    # The cross product magnitude relates to area, but the sign consistency is the main check for convexity.
    # A more rigorous check involves summing angles, but sign consistency of cross products for a closed loop 
    # usually implies convexity if no self-intersections. 
    # Let's do a quick self-intersection check for safety? 
    # Actually, the task says "strictly convex: no repeated points, every turn ... nonzero and of one sign".
    # It does not explicitly demand non-self-intersection in the text provided, but "convex polygon" implies it.
    # However, if turns are all same sign, it's a convex curve.
    
    return True

def _vertex_score(polygon, i):
    """Calculate score for vertex i."""
    n = len(polygon)
    v = polygon[i]
    
    # Distances to all other vertices
    dists = np.zeros(n-1)
    idx = 0
    for j in range(n):
        if j != i:
            dists[idx] = np.linalg.norm(v - polygon[j])
            idx += 1
            
    # Sort distances
    sorted_dists = np.sort(dists)
    
    # Check min distance
    if sorted_dists[0] < 0.05:
        return 0.0
        
    # Find k: first index minimizing d_{k+1} - d_k
    diffs = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(diffs)
    # k is index in diffs, so d_{k} and d_{k+1} in sorted_dists
    # The problem says: "Let k be the first index minimizing d_{k+1} - d_k"
    # Usually indices are 0-based. d_0...d_18.
    # diffs[0] is d_1 - d_0. If k=0, it refers to d_0, d_1.
    # If k is index in diffs (0 to 17), then the pair is d_k, d_{k+1}.
    
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    # Determine which 4 distances to take
    if k == 0:
        # d_0..d_3
        target_dists = sorted_dists[0:4]
    elif k == 17:
        # d_15..d_18
        target_dists = sorted_dists[15:19]
    else:
        # d_{k-1}..d_{k+2}
        # Indices: k-1, k, k+1, k+2
        target_dists = sorted_dists[k-1:k+3]
        
    # Calculate raw score
    ratios = []
    for d in target_dists:
        if D == 0:
            return 0.0
        ratio = max(d/D, D/d)
        ratios.append(ratio)
        
    raw = np.mean(ratios)
    if raw <= 0:
        return 0.0
        
    score = min(1.0, 1.0/raw)
    return score

def _polygon_score(polygon):
    """Calculate overall score: min vertex score."""
    if polygon is None:
        return 0.0
    if not _is_strictly_convex(polygon):
        return 0.0
        
    min_score = 1.0
    for i in range(len(polygon)):
        s = _vertex_score(polygon, i)
        if s < min_score:
            min_score = s
        if min_score == 0.0:
            return 0.0
    return min_score

def _build_polygon_from_params(params):
    """
    Construct a convex polygon from edge lengths and turning angles.
    params: [l_0, ..., l_19, a_0, ..., a_19]
    Constraints:
    - l_i > 0
    - a_i > 0
    - sum(a_i) = 2*pi (for closed convex polygon, exterior angles sum to 2pi)
    
    We enforce sum(a_i) = 2*pi by normalizing the angle part.
    We start at (0,0) with angle 0.
    """
    n = 20
    lengths = params[:n]
    angles = params[n:]
    
    # Ensure lengths are positive
    lengths = np.maximum(lengths, 1e-6)
    
    # Normalize angles to sum to 2*pi
    angle_sum = np.sum(angles)
    if angle_sum < 1e-6:
        angles = np.ones(n) * (2*np.pi/n)
    else:
        angles = angles / angle_sum * (2*np.pi)
        
    # Ensure angles are positive (they are by construction if inputs > 0)
    # But differential evolution might give negative? 
    # We will bound them in the optimizer.
    
    points = np.zeros((n+1, 2))
    points[0] = [0.0, 0.0]
    current_angle = 0.0
    
    for i in range(n):
        points[i+1] = points[i] + lengths[i] * np.array([np.cos(current_angle), np.sin(current_angle)])
        # Turn by exterior angle a_i
        # For counter-clockwise convex polygon, we turn left by a_i
        current_angle += angles[i]
        
    # Check closure
    # Due to floating point, it might not close perfectly.
    # We can force closure by adjusting the last point? 
    # Or just return the first n points and hope the score function handles it?
    # The score function expects a closed polygon implicitly (distances to "other vertices").
    # If it doesn't close, it's not a polygon.
    # To ensure closure, we can solve for the last edge vector?
    # Simpler: Use the points as is. If it doesn't close, _is_strictly_convex might fail or score 0.
    # Actually, _is_strictly_convex checks turns. If it doesn't close, the last turn to first point is undefined in my current loop.
    # My _is_strictly_convex loop uses (i-1), i, (i+1). For i=0, it uses n-1, 0, 1.
    # So it assumes the polygon is closed.
    # If the generated points don't close, the "polygon" is actually an open chain.
    # The distance calculation in _vertex_score uses all n points.
    # If it's not closed, it's not a valid polygon.
    
    # To guarantee closure, we can define the polygon by n-1 edges and 1 closing edge?
    # Or better: Use the fact that sum of vectors must be 0.
    # This is hard to enforce with simple params.
    
    # Alternative approach: 
    # The standard way to generate a convex polygon with given side lengths and angles is tricky if they don't sum correctly.
    # However, if we normalize angles to sum to 2pi, the direction of the last edge is determined.
    # The position of the last point is determined.
    # It is NOT guaranteed to be (0,0).
    
    # Let's try a different parameterization that guarantees closure:
    # Use n-1 edge vectors? No.
    
    # Let's stick to the radial modulation idea but with a twist?
    # The directive asks for a STRUCTURAL change.
    # Edge/angle param is good, but closure is the issue.
    
    # How to fix closure?
    # We can project the last point to (0,0)? No, that breaks convexity.
    
    # Let's use a different structural change:
    # **Optimization of the angles of the vertices on a circle?** No, that's just regular polygon.
    # **Optimization of radii with a smooth constraint?**
    
    # Let's try **Spherical Harmonics** on the boundary?
    # Or **Fourier Series** for the radius r(theta)?
    # r(theta) = 1 + sum(a_k cos(k theta) + b_k sin(k theta))
    # This guarantees a closed curve.
    # If we ensure r(theta) > 0 and the curve is convex, we are good.
    # Convexity of a star-shaped curve r(theta) requires r + r'' > 0.
    
    # This is a very different representation from Cartesian coordinates or simple jittering.
    # It allows global shapes (ellipses, etc.) while maintaining closure.
    
    # Let's implement the Fourier Series approach.
    
    # We will return the points calculated from the Fourier series.
    # The "params" passed to this function will be the Fourier coefficients.
    
    # Wait, I am inside _build_polygon_from_params which I defined for edge/angle.
    # Let's rewrite the strategy in the code.
    
    # I will discard the edge/angle logic in this function and use Fourier.
    
    # params: [a_1, b_1, ..., a_M, b_M]
    # r(theta) = 1 + sum_{k=1}^M (a_k cos(k theta) + b_k sin(k theta))
    # We need to ensure r(theta) > 0 and convexity.
    # If we keep coefficients small, it will be close to a circle and convex.
    
    # Let's use M=5 harmonics. 10 params.
    # We will evaluate the polygon on 20 equally spaced angles.
    # This guarantees the vertices are in cyclic order.
    # We just need to check convexity.
    
    M = 5
    if len(params) < 2*M:
        # Pad if necessary
        params = np.concatenate([params, np.zeros(2*M - len(params))])
    coeffs = params[:2*M]
    
    thetas = np.linspace(0, 2*np.pi, 20, endpoint=False)
    r = np.ones(20)
    
    for k in range(1, M+1):
        a_k = coeffs[2*(k-1)]
        b_k = coeffs[2*(k-1)+1]
        r += a_k * np.cos(k * thetas) + b_k * np.sin(k * thetas)
        
    # Ensure r is positive
    r = np.maximum(r, 0.5) # Prevent collapse
    
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    
    polygon = np.stack([x, y], axis=1)
    return polygon

def _objective(params):
    """
    Objective function for minimization (we want to maximize score, so minimize -score).
    Includes penalty for non-convexity or invalidity.
    """
    polygon = _build_polygon_from_params(params)
    
    # Check validity
    if not _is_strictly_convex(polygon):
        # Penalty: return a high value (low score)
        return 10.0 - 0.0 # Score is max 1, so -score is min -1. Penalty 10 is safe.
        
    score = _polygon_score(polygon)
    
    # If score is 0, it's bad.
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices using Fourier series parameterization
    of the radius, optimized via Differential Evolution and Local Polish.
    """
    start_time = time.time()
    deadline = start_time + 580.0 # 50s margin
    
    n = 20
    M = 5 # Number of Fourier harmonics
    n_params = 2 * M
    
    # Bounds for Fourier coefficients
    # Keep them small to ensure convexity (r > 0 and r+r'' > 0)
    # Max deviation from 1 should be small.
    bounds = [(-0.1, 0.1)] * n_params
    
    best_params = None
    best_score = -1.0
    best_polygon = None
    
    # 1. Differential Evolution
    # Use a small population for speed, rely on multiple restarts if needed
    # Maxiter 100, popsize 15
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            init='latinhypercube'
        )
        
        if time.time() > deadline:
            # Timeout during DE? Unlikely with small bounds, but check.
            pass
            
        if result.fun < best_score:
            best_score = -result.fun
            best_params = result.x
            best_polygon = _build_polygon_from_params(best_params)
            
    except Exception:
        pass
        
    if time.time() > deadline:
        # Return best found so far (likely regular polygon if no improvement)
        if best_polygon is None:
             angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
             x = np.cos(angles)
             y = np.sin(angles)
             best_polygon = np.stack([x, y], axis=1)
        return _normalize(best_polygon)

    # 2. Local Polish using L-BFGS-B on the Fourier coefficients
    # Start from the best DE result
    
    if best_params is not None:
        try:
            result_local = minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500}
            )
            
            if result_local.fun < best_score:
                best_score = -result_local.fun
                best_params = result_local.x
                best_polygon = _build_polygon_from_params(best_params)
                
        except Exception:
            pass
            
    if time.time() > deadline:
        if best_polygon is None:
             angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
             x = np.cos(angles)
             y = np.sin(angles)
             best_polygon = np.stack([x, y], axis=1)
        return _normalize(best_polygon)

    # 3. Try a few random restarts with larger coefficients to escape local optima?
    # If we have time left.
    
    remaining_time = deadline - time.time()
    if remaining_time > 50:
        # Try 2 more restarts with different seeds and slightly larger bounds
        for seed in [123, 456]:
            if time.time() > deadline:
                break
                
            # Larger bounds might risk non-convexity, but DE penalty handles it.
            bounds_large = [(-0.2, 0.2)] * n_params
            
            try:
                result_restart = differential_evolution(
                    _objective, 
                    bounds_large, 
                    maxiter=50, 
                    popsize=10, 
                    tol=1e-4, 
                    seed=seed,
                    init='latinhypercube'
                )
                
                if result_restart.fun < best_score:
                    best_score = -result_restart.fun
                    best_params = result_restart.x
                    best_polygon = _build_polygon_from_params(best_params)
                    
                    # Quick polish
                    result_local2 = minimize(
                        _objective, 
                        best_params, 
                        method='L-BFGS-B', 
                        bounds=bounds_large,
                        options={'maxiter': 200}
                    )
                    if result_local2.fun < best_score:
                        best_score = -result_local2.fun
                        best_params = result_local2.x
                        best_polygon = _build_polygon_from_params(best_params)
                        
            except Exception:
                pass

    # Fallback to regular polygon if no valid polygon found
    if best_polygon is None:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        best_polygon = np.stack([x, y], axis=1)
    else:
        # Ensure we return a valid polygon
        # If the optimization resulted in an invalid polygon (score 0), 
        # check if the regular polygon is better.
        regular_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        regular_polygon = np.stack([np.cos(regular_angles), np.sin(regular_angles)], axis=1)
        
        score_opt = _polygon_score(best_polygon)
        score_reg = _polygon_score(regular_polygon)
        
        if score_reg > score_opt:
            best_polygon = regular_polygon

    # Final normalization
    normalized = _normalize(best_polygon)
    if normalized is None:
        # Should not happen if we started with valid ones
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        normalized = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
    return normalized

# EVOLVE-BLOCK-END
