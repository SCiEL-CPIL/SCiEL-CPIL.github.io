# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _vertex_score(vertices, i):
    """
    Compute the score for vertex i of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    """
    n = len(vertices)
    # Get all other vertices relative to vertex i
    # We want distances from v_i to all v_j for j != i
    # It's easier to rotate the array so vertex i is at index 0
    rotated = np.roll(vertices, -i, axis=0)
    diffs = rotated - rotated[0]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    other_dists = dists[1:]
    d = np.sort(other_dists)
    
    if d[0] < 0.05:
        return 0.0
    
    gaps = d[1:] - d[:-1]
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k-1:k+3]
        
    raw = np.mean(np.maximum(sel / D, D / sel))
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score over all vertices of a polygon.
    polygon: (n, 2) array.
    """
    n = len(polygon)
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return 0.0
        
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
        
    min_score = 1.0
    for i in range(n):
        v_score = _vertex_score(normalized, i)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _polygon_from_side_lengths(lengths):
    """
    Construct a convex polygon from given side lengths.
    For a polygon to be closed with given side lengths, the largest side
    must be less than or equal to the sum of the others.
    To ensure a specific shape (e.g., convex, cyclic order), we can use the fact
    that any set of side lengths satisfying the polygon inequality can form a convex polygon.
    
    A simple way to generate the vertices is to assume the polygon is cyclic (inscribed in a circle).
    If we assume it's cyclic, we can solve for the radius R and angles.
    However, solving for R and angles for 20 sides might be complex if not symmetric.
    
    Alternative: Use the "chain" approach.
    Place vertex 0 at (0,0).
    Place vertex 1 at (l0, 0).
    The remaining vertices are determined by the angles.
    But we have 20 degrees of freedom (angles) and 2 constraints (closure).
    Actually, a convex polygon with fixed side lengths is a mechanism with 20-2 = 18 DOF?
    No, a polygon with N sides has N-2 internal DOF.
    Wait, if we fix side lengths, the shape is still flexible.
    
    Let's use a different representation that guarantees convexity and closure:
    Use the support function or radial function, but we tried Fourier.
    
    Let's try representing the polygon by its **turning angles** (exterior angles).
    For a convex polygon, the exterior angles $\alpha_i$ must be in $(0, \pi)$ and sum to $2\pi$.
    If we specify side lengths $l_i$ and exterior angles $\alpha_i$, the polygon is determined up to rotation and translation.
    Closure is NOT guaranteed for arbitrary $l_i$ and $\alpha_i$.
    
    So we need a representation that guarantees closure.
    
    **Better Representation**: 
    Use the **radial distances** $r_i$ from the centroid (or origin) to the vertices, assuming the vertices are at fixed angular positions $\theta_i = 2\pi i / 20$.
    This is the Fourier approach with only the 20th harmonic (or just the radial modulation).
    $r_i = 1 + \epsilon_i$.
    This guarantees the vertices are in cyclic order around the origin.
    Convexity is guaranteed if the modulation is small enough.
    This is a 20-dimensional search space (or less if we enforce symmetry).
    
    Let's stick to the radial representation $r_i$ but optimize it directly.
    We can use symmetry to reduce dimensions.
    The regular polygon is symmetric. The optimal shape might retain some symmetry (e.g., D10 or D20).
    Let's assume the optimal shape has the same symmetry as the regular polygon (D20) or maybe just D10?
    Actually, the score function depends on local distances.
    
    Let's try a **20-dimensional radial modulation** with the constraint that the polygon remains convex.
    $x_i = r_i \cos(\theta_i)$, $y_i = r_i \sin(\theta_i)$ with $\theta_i = 2\pi i / 20$.
    We optimize $r_i$.
    To ensure convexity, we can penalize non-convexity in the objective function.
    
    However, the previous program used Fourier modes 1-5. This allows for "wavy" shapes.
    Pure radial modulation $r_i$ with fixed angles $\theta_i$ is equivalent to a specific subset of Fourier modes?
    No, $r_i$ is just the radius.
    
    Let's try a **different structural change**:
    Optimize the **side lengths** $l_i$ directly, and assume the polygon is **cyclic** (inscribed in a circle).
    A cyclic polygon is determined by its side lengths (up to congruence) IF we fix the circle radius.
    Actually, for a given set of side lengths $l_i$, there is a unique cyclic polygon (if it exists).
    The condition for existence is that the sum of arcs is $2\pi$.
    $2 \arcsin(l_i / (2R)) = \theta_i$.
    $\sum \theta_i = 2\pi \implies \sum \arcsin(l_i / (2R)) = \pi$.
    This is a 1D equation for $R$ given $l_i$.
    So we can:
    1. Define parameters as side lengths $l_i$ (normalized so sum is constant, or just optimize raw lengths).
    2. For a given set of $l_i$, solve for $R$ such that the polygon closes on a circle.
    3. Construct the vertices.
    4. Compute score.
    
    This representation guarantees:
    - Cyclic order (by construction of angles).
    - Convexity (cyclic polygons with $l_i < 2R$ are convex).
    - Closure (by solving for R).
    
    This is a very robust parameterization.
    The dimension is 20 (side lengths). We can reduce it by assuming symmetry or just optimizing all 20.
    Let's optimize all 20 side lengths.
    To avoid trivial scaling issues, we can fix the sum of side lengths or the radius.
    Actually, the score is scale-invariant (normalization step).
    So we can fix $\sum l_i = 1$ or similar.
    
    Let's use parameters $u_i \in (0, 1)$ such that $l_i = u_i / \sum u_j$.
    Then solve for $R$.
    Equation: $f(R) = \sum_{i=0}^{19} 2 \arcsin(l_i / (2R)) - 2\pi = 0$.
    Note: $\arcsin(x)$ is defined for $x \in [-1, 1]$. So we need $l_i \le 2R$.
    As $R \to \infty$, $\arcsin(l_i/2R) \approx l_i/2R$, so sum $\approx \sum l_i / R \to 0$.
    As $R \to \max(l_i)/2$, one term goes to $\pi/2$, others are smaller.
    Actually, the minimum R is determined by the largest side.
    If $l_{max} > \sum_{j \ne max} l_j$, no solution.
    If $l_{max} \le \sum_{j \ne max} l_j$, a solution exists.
    
    Let's implement this.
    """
    n = 20
    # Normalize lengths
    total_len = np.sum(lengths)
    if total_len <= 0:
        return None
    l = lengths / total_len
    
    # Solve for R
    # f(R) = sum(2 * arcsin(l_i / (2R))) - 2*pi
    # We need R > max(l_i)/2
    R_min = np.max(l) / 2.0 * 1.01
    R_max = 10.0 # Should be enough
    
    # Bisection or Newton
    # f(R) is decreasing in R.
    # f(R_min) > 0?
    # If R is very close to l_max/2, arcsin(l_max/2R) -> pi/2.
    # Sum -> pi/2 + sum(others).
    # If sum(others) is small, f(R_min) < 2*pi?
    # Wait. If the polygon is degenerate (flat), sum of angles is 2*pi?
    # For a cyclic polygon, the sum of central angles is 2*pi.
    # Central angle $\theta_i = 2 \arcsin(l_i / 2R)$.
    # So we need $\sum \theta_i = 2\pi$.
    
    def f(R):
        # Check validity
        if np.any(l / (2*R) > 1.0):
            return -1.0 # Invalid, R too small
        angles = 2.0 * np.arcsin(l / (2.0 * R))
        return np.sum(angles) - 2.0 * np.pi
    
    # Check if solution exists
    # If f(R_min) < 0, then even at minimum R, sum of angles < 2*pi.
    # This happens if the polygon is "too small" relative to the circle?
    # No, as R decreases, angles increase.
    # So f(R) is decreasing?
    # d/dR arcsin(l/2R) = 1/sqrt(1-(l/2R)^2) * (-l/2R^2) < 0.
    # So as R increases, angles decrease, sum decreases.
    # So f(R) is decreasing.
    # We want f(R) = 0.
    # We need f(R_min) >= 0 and f(R_max) <= 0.
    
    # R_min is slightly larger than l_max/2.
    # If l_max is close to sum of others, R_min is large?
    # No, R_min is determined by l_max.
    
    # Let's just use a robust solver or bisection.
    lo = R_min
    hi = 100.0
    
    # Check f(lo)
    if f(lo) < 0:
        # No solution? This implies the polygon cannot be cyclic with these sides?
        # Or my R_min is not small enough?
        # If l_max is very large, R_min is large.
        # If the polygon is flat, R -> infinity.
        # If the polygon is "closed" in a small circle, R is small.
        # Actually, if the sides are very short, R must be large to make the angles sum to 2*pi?
        # No. If sides are short, they subtend small angles. To sum to 2*pi, R must be such that angles are not too small.
        # Wait. $\theta \approx l/R$. Sum $\theta \approx \sum l / R = 1/R$.
        # We want 1/R = 2*pi => R = 1/(2*pi).
        # So for small l, R is small.
        # For large l (close to diameter), R is large.
        
        # Let's re-evaluate f(R).
        # If R is small, l/2R is large. arcsin is large. Sum is large.
        # If R is large, l/2R is small. arcsin is small. Sum is small.
        # So f(R) is DECREASING.
        # We want f(R) = 0.
        # We need f(lo) >= 0 and f(hi) <= 0.
        
        # If f(lo) < 0, it means even at the smallest valid R, the sum of angles is < 2*pi.
        # This implies the sides are too short to close a circle of radius R_min?
        # No, if sides are short, they need a smaller R to subtend larger angles.
        # But R cannot be smaller than l_max/2.
        # So if l_max is large, R_min is large.
        # If l_max is large, the angle for that side is close to pi.
        # The other sides must sum to pi.
        # If the other sides are too short, they can't sum to pi even with a small R?
        # But R is fixed by l_max.
        # So if the other sides are too short, there is no cyclic polygon.
        
        # In that case, we can just return a regular polygon or penalize heavily.
        return None
        
    # Bisection
    for _ in range(100):
        mid = (lo + hi) / 2.0
        val = f(mid)
        if val > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
            
    R = (lo + hi) / 2.0
    
    # Construct vertices
    angles = 2.0 * np.arcsin(l / (2.0 * R))
    # Cumulative angles
    theta = np.cumsum(angles)
    theta = np.concatenate([[0.0], theta[:-1]]) # Start at 0
    
    # Vertices on circle
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    
    # The polygon is inscribed in a circle of radius R.
    # It is strictly convex.
    # It is in cyclic order.
    
    return np.stack([x, y], axis=1)

def _objective_side_lengths(params):
    """
    Objective function for side length optimization.
    params: (20,) array of side lengths.
    """
    try:
        poly = _polygon_from_side_lengths(params)
        if poly is None:
            return 1.0
        score = _polygon_min_score(poly)
        return -score
    except Exception:
        return 1.0

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    n = 20
    
    # Initial best: Regular polygon
    # For a regular polygon, all side lengths are equal.
    l_reg = np.ones(n)
    best_poly = _polygon_from_side_lengths(l_reg)
    if best_poly is None:
        # Fallback
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_poly = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    best_score = _polygon_min_score(best_poly)
    best_params = l_reg.copy()
    
    # Optimization
    # Bounds for side lengths.
    # To ensure a valid polygon, l_max < sum(others).
    # If we use uniform bounds [0.1, 1.0], it's likely valid.
    bounds = [(0.1, 1.0)] * n
    
    try:
        result_de = differential_evolution(
            _objective_side_lengths, 
            bounds, 
            maxiter=200, 
            popsize=40, 
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        if result_de.fun < -best_score:
            candidate_params = result_de.x
            candidate_poly = _polygon_from_side_lengths(candidate_params)
            if candidate_poly is not None:
                score = _polygon_min_score(candidate_poly)
                if score > best_score:
                    best_score = score
                    best_poly = candidate_poly
                    best_params = candidate_params
                    
    except Exception as e:
        print(f"DE failed: {e}")
        
    if time.time() > deadline - 120.0:
        return best_poly

    # Local refinement
    x0 = best_params.copy()
    
    try:
        options = {
            'maxiter': 2000,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }
        
        result_nm = minimize(
            _objective_side_lengths, 
            x0, 
            method='Nelder-Mead', 
            options=options
        )
        
        if result_nm.fun < -best_score:
            candidate_params = result_nm.x
            candidate_poly = _polygon_from_side_lengths(candidate_params)
            if candidate_poly is not None:
                score = _polygon_min_score(candidate_poly)
                if score > best_score:
                    best_score = score
                    best_poly = candidate_poly
                    best_params = candidate_params
                    
    except Exception as e:
        print(f"NM failed: {e}")
        
    # Restarts
    if time.time() < deadline - 60.0:
        for restart in range(5):
            if time.time() > deadline - 60.0:
                break
            
            perturbation_scale = 0.1 * (1.0 / (restart + 1))
            x0_perturbed = best_params + np.random.normal(0, perturbation_scale, size=n)
            # Clip to bounds
            x0_perturbed = np.clip(x0_perturbed, 0.1, 1.0)
            
            try:
                options = {
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
                result_nm = minimize(
                    _objective_side_lengths, 
                    x0_perturbed, 
                    method='Nelder-Mead', 
                    options=options
                )
                
                if result_nm.fun < -best_score:
                    candidate_params = result_nm.x
                    candidate_poly = _polygon_from_side_lengths(candidate_params)
                    if candidate_poly is not None:
                        score = _polygon_min_score(candidate_poly)
                        if score > best_score:
                            best_score = score
                            best_poly = candidate_poly
                            best_params = candidate_params
                            
            except Exception:
                pass

    return best_poly

# EVOLVE-BLOCK-END
