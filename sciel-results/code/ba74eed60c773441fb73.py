import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _compute_covering_radius_and_grad(points):
    """
    Computes the covering radius in radians and the subgradient with respect to the points.
    The covering radius is the maximum over facets of the circumradius.
    For the worst facet, the subgradient points in the direction that increases the circumradius.
    To minimize, we move points in the opposite direction.
    
    Returns:
    radius: float, the covering radius in radians
    grad: np.array of shape (N, 3), the subgradient (direction to move to increase radius)
    worst_indices: tuple of 3 indices or None
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_cr = 0.0
        worst_facet = None
        worst_circumcenter = None
        
        # Vectorized computation of circumcenters and radii for all facets
        # p_i, p_j, p_k
        pi = points[facets[:, 0]]
        pj = points[facets[:, 1]]
        pk = points[facets[:, 2]]
        
        # Circumcenter direction is normalized sum of vertices (for unit sphere, if triangle is small/acute)
        # More robust: The circumcenter c is the intersection of the sphere with the plane defined by the 3 points?
        # Actually, for a spherical triangle, the circumcenter is the normalized vector (pi + pj + pk) 
        # ONLY IF the triangle is acute and doesn't contain the antipode. 
        # A more robust way to find the point on the sphere equidistant from pi, pj, pk:
        # It lies in the plane spanned by pi, pj, pk.
        # Let c = a*pi + b*pj + c*pk. 
        # But simpler: The normal to the plane of the triangle is n = cross(pi, pj).
        # The circumcenter is in the direction of the projection of the origin onto the plane? No.
        # The circumcenter of a triangle on the sphere is the unit vector in the direction of the 
        # intersection of the sphere with the line passing through the origin and the circumcenter of the planar triangle?
        # No, the circumcenter of the spherical triangle is the unit vector c such that c . pi = c . pj = c . pk.
        # This c is parallel to pi + pj + pk ONLY IF pi, pj, pk form an acute triangle with respect to the center.
        # If the triangle is obtuse, the circumcenter is the vertex with the largest angle? No, that's for Euclidean.
        # On the sphere, if the triangle contains the antipode of the "center", the circumcenter is defined differently?
        # Standard formula: c = (pi + pj + pk) / |pi + pj + pk| works for the "small" circumcenter.
        # Let's use this and check validity.
        
        sums = pi + pj + pk
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        
        # Avoid division by zero
        valid = norms > 1e-10
        centers = np.zeros_like(sums)
        centers[valid] = sums[valid] / norms[valid]
        
        # Compute angular distance from center to pi (should be same for pj, pk)
        dots = np.einsum('ij,ij->i', centers[valid], pi[valid])
        dots = np.clip(dots, -1.0, 1.0)
        radii = np.arccos(dots)
        
        if len(radii) == 0:
            return 0.0, np.zeros((N, 3)), None
            
        local_max_idx = np.argmax(radii)
        max_cr = radii[local_max_idx]
        
        # Get the global index of the worst facet
        # We need to map local_max_idx back to the original facets array index
        # valid is a boolean mask of same length as facets
        global_facet_idx = np.where(valid)[0][local_max_idx]
        
        worst_facet = facets[global_facet_idx]
        worst_circumcenter = centers[global_facet_idx]
        
        # Construct Gradient
        # We want to minimize the radius. 
        # The radius R is determined by the distance from c to pi.
        # dR/dpi = ?
        # R = arccos(c . pi).
        # c depends on pi, pj, pk.
        # This is complex. A simpler heuristic:
        # Move the vertices of the worst facet towards the circumcenter to shrink the triangle?
        # No, moving vertices towards the center reduces the circumradius?
        # If we move pi towards c, the distance c-pi decreases.
        # So the gradient of R w.r.t pi points AWAY from c?
        # Let's test: if pi moves towards c, dot(c, pi) increases, arccos decreases. So R decreases.
        # So to MINIMIZE R, we should move pi TOWARDS c.
        # The "gradient" for minimization is (c - pi) projected onto the tangent space.
        
        grad = np.zeros((N, 3))
        i, j, k = worst_facet
        pi_v = points[i]
        pj_v = points[j]
        pk_v = points[k]
        
        # Tangent projection of (c - p)
        def tangent_proj(v, p):
            return v - np.dot(v, p) * p
        
        # We want to move points to reduce the circumradius.
        # Moving a vertex towards the circumcenter reduces the circumradius.
        # So the direction of decrease is towards c.
        # grad_min[i] = tangent_proj(c - pi, pi)
        # grad_min[j] = tangent_proj(c - pj, pj)
        # grad_min[k] = tangent_proj(c - pk, pk)
        
        # However, moving points might create new larger facets elsewhere.
        # A common strategy is to move the worst facet's vertices towards the center,
        # and also add a small repulsive force from all other points to maintain uniformity.
        
        dir_i = tangent_proj(worst_circumcenter - pi_v, pi_v)
        dir_j = tangent_proj(worst_circumcenter - pj_v, pj_v)
        dir_k = tangent_proj(worst_circumcenter - pk_v, pk_v)
        
        # Normalize directions to have unit step magnitude potential
        n_i = np.linalg.norm(dir_i)
        n_j = np.linalg.norm(dir_j)
        n_k = np.linalg.norm(dir_k)
        
        if n_i > 1e-10: grad[i] += dir_i / n_i
        if n_j > 1e-10: grad[j] += dir_j / n_j
        if n_k > 1e-10: grad[k] += dir_k / n_k
        
        return max_cr, grad, worst_facet

    except Exception:
        return np.pi, np.zeros((N, 3)), None

def _repulsive_force(points, k=10):
    """
    Computes a repulsive force for each point based on its k-nearest neighbors.
    This helps to prevent points from collapsing and maintains a valid hull.
    """
    forces = np.zeros_like(points)
    # Simple O(N^2) repulsion is fine for N=130
    for i in range(N):
        # Distance to all other points
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf # Ignore self
        
        # Get indices of k nearest neighbors
        nearest_idx = np.argsort(dists)[:k]
        
        for j in nearest_idx:
            if i == j: continue
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6: continue
            # Force is proportional to 1/dist^2 or similar
            # We want to push i away from j
            forces[i] += diff / (dist * dist)
            
    return forces

def construct():
    """
    Constructs a set of N points on the unit sphere using Gradient Descent on the Manifold.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initialization: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = _compute_covering_radius_and_grad(points)[0]
    
    # Learning rate schedule
    # Start with a moderate step size
    lr = 0.05
    min_lr = 1e-5
    
    step = 0
    stall_count = 0
    current_radius = best_radius
    
    while time.time() - start_time < time_limit:
        radius, grad, worst_facet = _compute_covering_radius_and_grad(points)
        
        if not np.isfinite(radius):
            # Handle invalid hull by adding noise
            noise = rng.normal(0, 0.1, (N, 3))
            points = points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            continue
        
        # If radius improved, update best
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            stall_count = 0
        else:
            stall_count += 1
            # Reduce learning rate if stalled
            if stall_count > 20:
                lr *= 0.95
                stall_count = 0
                if lr < min_lr:
                    # If learning rate is too small, try a random restart or perturbation
                    # Perturb the worst facet points
                    if worst_facet is not None:
                        for idx in worst_facet:
                            noise = rng.normal(0, 0.05, 3)
                            points[idx] += noise
                            points[idx] /= np.linalg.norm(points[idx])
                    else:
                        break
        
        # Compute repulsive force to maintain spread
        # We blend the gradient descent step with the repulsive force
        # The gradient from _compute... is the direction to MINIMIZE radius (towards center)
        # Wait, my grad calculation in _compute... was:
        # "grad_min[i] = tangent_proj(c - pi, pi)" -> This is the direction to move to DECREASE radius.
        # So we want to move points ALONG this grad.
        
        # Add repulsion
        rep_force = _repulsive_force(points)
        
        # Blend: primarily follow the covering radius gradient, secondarily repel
        # We need to project rep_force to tangent space too? 
        # rep_force is already in 3D. Projecting is safer.
        tangent_rep = np.zeros_like(rep_force)
        for i in range(N):
            p = points[i]
            tangent_rep[i] = rep_force[i] - np.dot(rep_force[i], p) * p
        
        # Step
        # Scale repulsion to be smaller than the main gradient
        rep_scale = 0.01 * lr 
        combined_grad = grad + rep_scale * tangent_rep
        
        # Normalize combined gradient? 
        # The grad components are unit vectors (from my implementation).
        # Let's just use lr as the step size.
        
        new_points = points + lr * combined_grad
        
        # Normalize to sphere
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check validity
        new_radius, _, _ = _compute_covering_radius_and_grad(new_points)
        
        if np.isfinite(new_radius):
            # Accept step if it doesn't increase radius too much, or always accept if it decreases
            # Since we are doing gradient descent, we expect it to decrease.
            # But non-smoothness can cause issues.
            # If new_radius < radius: accept
            # Else: maybe reject and reduce lr?
            if new_radius < radius * 1.01: # Allow small increases to escape local minima
                points = new_points
                current_radius = new_radius
            else:
                # Revert and reduce lr
                lr *= 0.9
                # Try a different direction? Random perturbation
                if rng.random() < 0.1:
                    idx = rng.integers(0, N)
                    noise = rng.normal(0, 0.02, 3)
                    points[idx] += noise
                    points[idx] /= np.linalg.norm(points[idx])
        
        step += 1
        
        # Periodic global check / restart if stuck for a long time
        if step % 500 == 0:
            # Check if we are in a local minimum by checking if gradient is small
            # Or just rely on the stall_count logic above
            pass

    # Final cleanup
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    # Use a simpler radius calc for printing to avoid grad overhead
    hull = ConvexHull(pts)
    facets = hull.simplices
    max_cr = 0
    for i, j, k in facets:
        c = pts[i] + pts[j] + pts[k]
        c /= np.linalg.norm(c)
        dot = np.clip(np.dot(c, pts[i]), -1, 1)
        theta = np.arccos(dot)
        if theta > max_cr: max_cr = theta
    print(f"Covering Radius: {max_cr * 180 / np.pi:.6f} degrees")
