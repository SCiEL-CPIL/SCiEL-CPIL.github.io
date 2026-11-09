import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

def construct():
    # 1. Initialization: Triangular Lattice Subset
    # We want 13 points. A triangular lattice is a good starting point for mesh-like distribution.
    # Let's construct a specific set of 13 points from a triangular lattice.
    # Lattice vectors: e1 = (1, 0), e2 = (0.5, sqrt(3)/2)
    # We can take a 4x4 grid and remove corners or specific points to get 13.
    # Or simply generate a compact cluster.
    # Let's use a 4x4 grid of indices (i,j) where i,j in 0..3.
    # Total 16 points. Remove 3 to get 13?
    # Better: Define a specific set of 13 points that forms a convex-ish shape.
    # Let's take a 4x4 grid and remove the 3 "outermost" corners to make it more compact/convex?
    # Actually, a regular 13-gon is convex. A lattice subset might not be.
    # The evaluator normalizes by convex hull area.
    # If the points are not in convex position, the hull area is smaller, but the min triangle area
    # is determined by internal triangles too.
    # The task says "Place 13 points in a planar convex region...".
    # Usually this implies the points themselves define the region or are within it.
    # "normalized by the point set convex hull area" suggests we just need to place them.
    
    # Let's stick to the directive: "Triangular Lattice Initialization".
    # Generate a 4x4 grid of triangular lattice points.
    # Points: (i + 0.5*j, j*sqrt(3)/2) for i,j in 0..3.
    # This gives 16 points.
    # We need 13. Let's remove the 3 corners that make the shape least "circular" or just pick 13.
    # To be safe and "mesh-like", let's take the 13 points closest to the centroid of the 16.
    
    sqrt3 = np.sqrt(3)
    lattice_points = []
    for i in range(4):
        for j in range(4):
            x = i + 0.5 * j
            y = j * sqrt3 / 2
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    
    # Center the lattice
    centroid = lattice_points.mean(axis=0)
    lattice_points -= centroid
    
    # Calculate distances to centroid
    dists = np.linalg.norm(lattice_points, axis=1)
    # Sort by distance and take the first 13 (most compact)
    indices = np.argsort(dists)
    points = lattice_points[indices[:13]]
    
    # Add a tiny random perturbation to break symmetry
    rng = np.random.RandomState(42)
    points += rng.uniform(-0.01, 0.01, points.shape)
    
    # 2. Define the objective function and gradient
    # We want to maximize min(triangle_area)
    # We use soft-min: -1/k * log(sum(exp(-k * A_i)))
    # Maximizing this is equivalent to minimizing -softmin
    
    def get_delaunay_triangles(pts):
        try:
            tri = Delaunay(pts)
            return tri.simplices
        except Exception:
            # If triangulation fails (e.g., collinear points), return None
            return None

    def soft_min_area(pts, k_temp):
        triangles = get_delaunay_triangles(pts)
        if triangles is None:
            return 0.0, np.zeros_like(pts)
        
        areas = np.zeros(len(triangles))
        # Compute areas
        for i, tri_idx in enumerate(triangles):
            p1 = pts[tri_idx[0]]
            p2 = pts[tri_idx[1]]
            p3 = pts[tri_idx[2]]
            # Signed area
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            areas[i] = 0.5 * abs(cross)
            
        # Soft min
        # To avoid overflow/underflow, we can subtract max or use log-sum-exp trick
        # log(sum(exp(-k*A))) = -k*max(A) + log(sum(exp(-k*(A - max(A)))))
        # Actually, we want to maximize min(A), so we minimize -softmin.
        # softmin(A) = -1/k * log(sum(exp(-k*A)))
        # -softmin(A) = 1/k * log(sum(exp(-k*A)))
        
        # We want the gradient of -softmin(A)
        # Let S = sum(exp(-k*A_i))
        # f = 1/k * log(S)
        # df/dp = 1/k * (1/S) * dS/dp
        # dS/dp = sum( -k * exp(-k*A_i) * dA_i/dp )
        # df/dp = 1/k * 1/S * -k * sum( exp(-k*A_i) * dA_i/dp )
        # df/dp = - 1/S * sum( w_i * dA_i/dp ) where w_i = exp(-k*A_i)
        
        # Use log-sum-exp trick for stability
        # Let m = max(A)
        # S = exp(-k*m) * sum(exp(-k*(A_i - m)))
        # log(S) = -k*m + log(sum(exp(-k*(A_i - m))))
        # f = 1/k * log(S) = -m + 1/k * log(sum(exp(-k*(A_i - m))))
        
        m = np.max(areas)
        exp_vals = np.exp(-k_temp * (areas - m))
        S = np.sum(exp_vals)
        
        if S == 0:
            return 0.0, np.zeros_like(pts)
            
        weights = exp_vals / S
        
        grad = np.zeros_like(pts)
        
        for i, tri_idx in enumerate(triangles):
            p1 = pts[tri_idx[0]]
            p2 = pts[tri_idx[1]]
            p3 = pts[tri_idx[2]]
            
            # Gradients of signed area A = 0.5 * cross
            # grad_p1 = 0.5 * perp(p3 - p2)
            # grad_p2 = 0.5 * perp(p1 - p3)
            # grad_p3 = 0.5 * perp(p2 - p1)
            # where perp(v) = (-v_y, v_x)
            
            v32 = p3 - p2
            v13 = p1 - p3
            v21 = p2 - p1
            
            g1 = 0.5 * np.array([-v32[1], v32[0]])
            g2 = 0.5 * np.array([-v13[1], v13[0]])
            g3 = 0.5 * np.array([-v21[1], v21[0]])
            
            # Note: These are gradients of the *signed* area.
            # We used abs(area) in the objective. 
            # If the triangle is oriented CW, signed area is negative.
            # The soft-min uses A_i > 0.
            # If we use signed area in the gradient, and the triangle flips, the gradient direction flips.
            # Since we are maximizing the minimum area, and we start with a valid mesh,
            # we assume the orientation stays consistent.
            # If we want to be robust, we should multiply the gradient by the sign of the area.
            # sign = 1 if signed_area > 0 else -1.
            # But we computed `areas` using abs().
            # Let's compute signed area to get the sign.
            
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            sign = 1.0 if cross >= 0 else -1.0
            
            # The gradient of |A| is sign(A) * grad(A_signed)
            # So we multiply g1, g2, g3 by sign.
            
            grad[tri_idx[0]] += weights[i] * sign * g1
            grad[tri_idx[1]] += weights[i] * sign * g2
            grad[tri_idx[2]] += weights[i] * sign * g3
            
        # The function value to minimize is -softmin = 1/k * log(S)
        # We return the value and the gradient.
        # Wait, the formula derived was:
        # df/dp = - 1/S * sum( w_i * dA_i/dp )
        # where w_i = exp(-k*A_i)
        # My code above accumulated `weights[i] * grad_A_i`.
        # `weights` here is `exp_vals / S`.
        # So `sum(weights[i] * grad_A_i)` is `1/S * sum(exp(-k*A_i) * grad_A_i)`.
        # The derivative of `1/k * log(S)` is `1/k * (1/S) * dS/dp`.
        # `dS/dp = sum( -k * exp(-k*A_i) * grad_A_i )`.
        # So `df/dp = 1/k * 1/S * -k * sum( exp(-k*A_i) * grad_A_i ) = - sum( (exp(-k*A_i)/S) * grad_A_i )`.
        # So the gradient is `-1 * accumulated_grad`.
        
        grad = -grad
        
        val = (1.0 / k_temp) * np.log(S)
        
        return val, grad

    # 3. Optimization
    # We minimize the negative soft-min area.
    # Initial points
    x0 = points.flatten()
    
    # Multi-stage optimization
    # Start with low k (smooth landscape) and increase k (sharp min)
    k_values = [5.0, 10.0, 20.0, 50.0]
    
    current_x = x0
    
    for k_temp in k_values:
        # Define objective and jac for this k
        def objective(x, k=k_temp):
            return soft_min_area(x.reshape(N, 2), k)[0]
        
        def jac(x, k=k_temp):
            return soft_min_area(x.reshape(N, 2), k)[1].flatten()
        
        res = minimize(
            objective,
            current_x,
            jac=jac,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if res.success:
            current_x = res.x
        else:
            # If it fails, keep the previous points
            pass
            
    final_points = current_x.reshape(N, 2)
    
    # 4. Final processing
    # Normalize to unit area?
    # The evaluator normalizes by convex hull area.
    # It is safer to return the points as is, or normalized to have area 1.
    # If we normalize, we might change the relative positions if the hull area is not 1.
    # The evaluator does: score = min_tri_area / hull_area.
    # This ratio is scale-invariant.
    # So we can return the points at any scale.
    # However, if the points are very large or small, numerical issues might arise in the evaluator.
    # Let's normalize the final points to have a convex hull area of 1.
    
    try:
        hull = Delaunay(final_points).convex_hull
        hull_pts = final_points[hull]
        # Shoelace formula for area
        x = hull_pts[:, 0]
        y = hull_pts[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area > 0:
            scale = 1.0 / np.sqrt(area)
            final_points = final_points * scale
    except Exception:
        pass
        
    return final_points

# EVOLVE-BLOCK-END
