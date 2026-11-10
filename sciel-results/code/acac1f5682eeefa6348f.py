# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon that maximizes the minimum per-vertex equidistance score.
    
    Strategy:
    1. Start from a regular polygon.
    2. Use a parameterization that guarantees convexity: 
       - Center at origin (by symmetry/normalization)
       - Represent vertices by angles (sorted) and radii.
       - To ensure strict convexity and closure, we can optimize the angles and radii directly, 
         but we must ensure the polygon is simple and convex.
         
    A more robust approach for convex polygons:
    Represent the polygon by its edge vectors. Let the edges be $e_i = (l_i \cos \theta_i, l_i \sin \theta_i)$ 
    where $\sum e_i = 0$ for closure.
    However, enforcing closure and convexity in edge-length/angle space is tricky.
    
    Alternative: 
    Use the fact that a convex polygon can be defined by its support function or by vertices on a curve.
    A common parameterization for convex polygons centered at the origin is:
    $v_i = r_i (\cos \phi_i, \sin \phi_i)$
    where $\phi_i$ are sorted angles and $r_i > 0$.
    For the polygon to be strictly convex, the cross product of consecutive edge vectors must be positive.
    Edge $i$ goes from $v_i$ to $v_{i+1}$.
    This constraint is non-linear.
    
    Let's use a simpler, very effective heuristic search:
    1. Initialize with regular polygon.
    2. Randomly perturb angles and radii.
    3. Check convexity.
    4. If convex, compute score.
    5. Keep best.
    6. Use local optimization (Nelder-Mead or BFGS) on the best found parameters.
    
    Parameters:
    - Angles $\phi_i$ (20 values, sorted, sum to 2*pi effectively by closure of the shape, but actually just 20 angles in [0, 2*pi))
    - Radii $r_i$ (20 values, positive)
    
    To handle the sorting constraint on angles, we can optimize the *differences* between angles.
    Let $\Delta \phi_i = \phi_{i+1} - \phi_i > 0$, with $\sum \Delta \phi_i = 2\pi$.
    So we have 19 free parameters for angles (the 20th is determined by the sum) and 20 for radii.
    Total 39 parameters.
    
    Objective function to maximize:
    Min over vertices of the vertex score.
    This is a non-smooth function (min, and the 'k' selection is discrete).
    We can smooth it or just use a derivative-free optimizer like Nelder-Mead, or use a stochastic search.
    
    Given the time limit (630s), we can afford a few thousand evaluations.
    Let's implement a score function first.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Leave 30s buffer
    
    n = 20
    
    def compute_score(vertices):
        """
        Computes the metric for a given set of vertices.
        Returns 0 if invalid.
        """
        if not np.all(np.isfinite(vertices)):
            return 0.0
        
        # Normalize
        centroid = np.mean(vertices, axis=0)
        verts = vertices - centroid
        dists_to_centroid = np.linalg.norm(verts, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return 0.0
        verts = verts / mean_dist
        
        # Check convexity
        # 1. No repeated points
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(verts[i] - verts[j]) < 1e-10:
                    return 0.0
        
        # 2. Strictly convex: every turn nonzero and same sign
        # Compute cross products of consecutive edges
        edges = np.roll(verts, -1, axis=0) - verts
        cross_prods = np.cross(edges, np.roll(edges, -1, axis=0))
        
        if np.any(np.abs(cross_prods) < 1e-10):
            return 0.0
        
        # All cross products must have the same sign
        if not (np.all(cross_prods > 0) or np.all(cross_prods < 0)):
            return 0.0
            
        # Total turning +-2*pi is implied by convexity for a closed polygon in 2D?
        # Yes, for a simple closed convex curve, total curvature is 2*pi.
        # The check above ensures it's a convex polygon.
        
        # Compute vertex scores
        min_score = 1.0
        
        for i in range(n):
            v = verts[i]
            # Distances to other vertices
            d = np.linalg.norm(verts - v, axis=1)
            d[i] = np.inf  # Exclude self
            d_sorted = np.sort(d)
            
            # d_0 ... d_18
            if d_sorted[0] < 0.05:
                min_score = 0.0
                break
            
            # Find k: first index minimizing d_{k+1} - d_k
            diffs = d_sorted[1:] - d_sorted[:-1]
            k = np.argmin(diffs)
            
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            if D < 0.1:
                min_score = 0.0
                break
            
            # Select 4 distances
            if k == 0:
                selected = d_sorted[0:4]
            elif k == 17:
                selected = d_sorted[15:19]
            else:
                selected = d_sorted[k-1:k+3]
            
            # raw = mean of max(d/D, D/d)
            ratios = np.maximum(selected / D, D / selected)
            raw = np.mean(ratios)
            
            score = min(1.0, 1.0 / raw)
            
            if score < min_score:
                min_score = score
                
        return min_score

    def params_to_vertices(params):
        """
        Convert parameters (dphi, log_r) to vertices.
        dphi: 19 values, positive.
        log_r: 20 values.
        """
        dphi = params[:19]
        log_r = params[19:]
        
        # Ensure dphi are positive
        # Use softplus or just exp to ensure positivity, but we need sum to be 2*pi.
        # Let's use a softmax-like normalization or just scale.
        # To maintain flexibility, let's optimize dphi directly but constrain them to be positive.
        # A simple way: dphi_i = exp(params[:19]), then normalize to sum 2*pi.
        raw_dphi = np.exp(params[:19])
        dphi = raw_dphi / np.sum(raw_dphi) * 2 * np.pi
        
        # Add the last dphi
        dphi_full = np.concatenate([dphi, [2*np.pi - np.sum(dphi)]])
        # Note: if sum(dphi) > 2*pi, the last one is negative. 
        # The exp normalization ensures sum(raw) > 0, so dphi sums to 2*pi.
        # The last element is calculated to make the sum exactly 2*pi.
        # It is guaranteed to be positive because dphi (first 19) are positive and sum to < 2*pi?
        # No, they sum to 2*pi * (sum(raw)/sum(raw)) = 2*pi.
        # Wait, dphi = raw / sum(raw) * 2*pi. So sum(dphi) = 2*pi.
        # Then the last dphi is 2*pi - sum(dphi) = 0. That's bad.
        
        # Correction:
        # We have 20 angles. Let's define 20 free parameters for angles, but fix the sum.
        # Better: Use 19 parameters for the gaps.
        # g_i = exp(params[i]) for i in 0..18.
        # g_19 = 2*pi - sum(g_0..g_18).
        # To ensure g_19 > 0, we need sum(g_0..g_18) < 2*pi.
        # This is a constraint.
        
        # Alternative: Use angles directly.
        # Let's stick to the radius/angle parameterization but optimize the angles directly.
        # Angles must be sorted.
        # Let's use the differences (gaps) but ensure the last one is positive.
        # We can use a transformation that maps R^19 to the simplex of positive numbers summing to 2*pi.
        # Let x = params[:19].
        # y_i = exp(x_i).
        # g_i = y_i / sum(y) * 2*pi for i=0..18.
        # g_19 = 2*pi - sum(g_0..g_18).
        # Since sum(g_0..g_18) < 2*pi (strictly, because y_19 is not included in the sum? No, the sum is over 19 items).
        # Wait, if we have 19 gaps, the 20th is determined.
        # If we use softmax on 19 values, their sum is 1. So sum(g_0..g_18) = 2*pi.
        # Then g_19 = 0. This is degenerate.
        
        # We need 20 gaps.
        # Let's use 20 parameters for gaps, but fix the sum.
        # Let x = params[:20].
        # y = exp(x).
        # g = y / sum(y) * 2*pi.
        # This ensures all g_i > 0 and sum(g_i) = 2*pi.
        # Then angles phi_i are cumulative sums of g.
        # phi_0 = 0 (WLOG, rotation invariance).
        # phi_i = sum(g_0..g_{i-1}).
        
        # Radii: r_i = exp(params[20:40]).
        
        g = np.exp(params[:20])
        g = g / np.sum(g) * 2 * np.pi
        
        phi = np.zeros(20)
        phi[1:] = np.cumsum(g[:-1])
        
        r = np.exp(params[20:40])
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        return np.stack([x, y], axis=1)

    def objective(params):
        """
        Objective function for minimization (we want to maximize score, so minimize -score).
        """
        verts = params_to_vertices(params)
        score = compute_score(verts)
        return -score

    # Initial parameters
    # Regular polygon:
    # g_i = 2*pi/20
    # r_i = 1
    # x_i = log(g_i) = log(2*pi/20)
    # y_i = log(1) = 0
    
    x0_g = np.full(20, np.log(2*np.pi/20))
    x0_r = np.zeros(20)
    x0 = np.concatenate([x0_g, x0_r])
    
    best_params = x0.copy()
    best_score = compute_score(params_to_vertices(best_params))
    
    # Local optimization
    # Nelder-Mead is robust for non-smooth functions.
    # We will run multiple random restarts or just one long run.
    # Given the complexity, let's try a few restarts.
    
    n_restarts = 5
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            x_start = x0
        else:
            # Perturb the best found so far
            x_start = best_params + np.random.normal(0, 0.1, size=40)
            
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline - 60:
            break

    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass

    final_verts = params_to_vertices(best_params)
    
    # Final validation and normalization (though compute_score does it, we need to return the normalized verts? 
    # The task says "The polygon is normalized... It must then be strictly convex".
    # The function returns the array. The evaluator normalizes it.
    # But we should return a valid polygon.
    return final_verts


# EVOLVE-BLOCK-END
