import numpy as np
import time

N = 50
R_SPHERE = 1.0 / np.sqrt(2.0)

def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[:, 0:1].T
    dy = centers[:, 1:2] - centers[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r

def _hexagonal_packing():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _stereo_forward(p2d):
    """
    Map 2D points in unit disk to sphere of radius R_SPHERE.
    Center (0,0) maps to North Pole (0,0,R).
    Edge (|p|=1) maps to Equator (z=0).
    """
    r2 = np.sum(p2d**2, axis=1)
    denom = 1.0 + r2
    # x, y scale by 2R, z scales by R(1-r2)
    x = 2.0 * R_SPHERE * p2d[:, 0] / denom
    y = 2.0 * R_SPHERE * p2d[:, 1] / denom
    z = R_SPHERE * (1.0 - r2) / denom
    return np.column_stack((x, y, z))

def _stereo_backward(p3d):
    """
    Map 3D points on sphere of radius R_SPHERE back to 2D plane.
    Inverse of _stereo_forward.
    """
    x, y, z = p3d[:, 0], p3d[:, 1], p3d[:, 2]
    denom = 1.0 + z / R_SPHERE
    # Handle z close to -R (South Pole) which maps to infinity.
    # In our domain, z >= 0, so denom >= 1. Safe.
    x2d = x / denom
    y2d = y / denom
    return np.column_stack((x2d, y2d))

def _riemannian_optimize(p2d_init, deadline):
    """
    Optimize packing using Riemannian gradient ascent on the sphere.
    Objective: Maximize min(geodesic_distance, boundary_constraint).
    On the sphere, the boundary constraint is z >= 0 (since |p2d| <= 1 <=> z >= 0).
    Actually, the constraint is that the mapped points must lie in the hemisphere z >= 0.
    If a point goes to z < 0, it corresponds to |p2d| > 1, which is invalid.
    So we must ensure z >= 0.
    """
    # Initialize on sphere
    p3d = _stereo_forward(p2d_init)
    
    # Ensure valid start
    # If any z < 0, project back? 
    # _stereo_forward of valid points always has z >= 0.
    
    current_r2d = _compute_radius(p2d_init)
    best_p2d = p2d_init.copy()
    best_r = current_r2d
    
    lr = 0.05
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Compute geodesic distances
        # cos(theta_ij) = (p_i . p_j) / R^2
        dots = p3d @ p3d.T
        dots /= (R_SPHERE * R_SPHERE)
        np.clip(dots, -1.0, 1.0, out=dots)
        thetas = np.arccos(dots)
        np.fill_diagonal(thetas, np.inf)
        
        # Geodesic distance
        geo_dists = R_SPHERE * thetas
        
        # Find minimum geodesic distance
        min_geo_dist = np.min(geo_dists)
        
        # Active constraints: pairs with distance close to min
        # Also consider boundary constraint: z_i close to 0
        # If z_i is small, the circle is close to the boundary.
        # The "radius" in 2D is related to z.
        # For small r2d, z ~ R. For r2d ~ 1, z ~ 0.
        # Actually, the 2D radius r is determined by min(geo_dist/2 mapped to 2D?, 1-|c|).
        # It's easier to just evaluate the 2D radius and use that for acceptance.
        # But for the gradient, we need to know which constraints are active in the manifold.
        
        # Let's define the objective on the manifold as:
        # f(P) = min( min_{i<j} geo_dist(i,j), min_i z_i )
        # Wait, is z_i proportional to the "boundary slack"?
        # 1 - |c| = 1 - 2x/denom.
        # z = R(1-r2)/denom.
        # They are not identical, but correlated.
        # If we maximize min(geo_dist, z), does it maximize the 2D radius?
        # The 2D radius r is bounded by min_pair_dist/2 and min_boundary_dist.
        # The mapping is monotonic.
        # Let's use the 2D radius for acceptance, but compute gradient based on active 2D constraints mapped to 3D.
        
        # Simpler approach:
        # Compute 2D radius r.
        # Determine active pairs in 2D.
        # Determine active boundary points in 2D.
        # Compute Riemannian gradient of the "distance" function for active pairs.
        # For boundary, push towards North Pole (increase z).
        
        # Compute 2D centers to check constraints
        p2d = _stereo_backward(p3d)
        r_current = _compute_radius(p2d)
        
        if r_current > best_r:
            best_r = r_current
            best_p2d = p2d.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > max_no_improve:
            # Reheat
            # Add random noise on sphere
            noise = np.random.normal(0, 0.01, p3d.shape)
            p3d += noise
            # Renormalize
            norms = np.linalg.norm(p3d, axis=1, keepdims=True)
            p3d = p3d / norms * R_SPHERE
            # Ensure z >= 0?
            # If z < 0, reflect to z > 0? Or just project to equator?
            # Reflection might be discontinuous.
            # Let's just ensure validity by checking 2D radius.
            # If invalid, revert.
            p2d_check = _stereo_backward(p3d)
            if _compute_radius(p2d_check) <= 0:
                p3d = _stereo_forward(best_p2d)
            
            no_improve = 0
            lr = max(0.001, lr * 0.5)
            continue
        
        # Compute Gradient
        # Active pairs in 2D
        dx = p2d[:, 0:1] - p2d[:, 0:1].T
        dy = p2d[:, 1:2] - p2d[:, 1:2].T
        d2d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d2d, np.inf)
        
        min_d2d = np.min(d2d)
        # Threshold for active pairs
        thresh = min_d2d * 1.05
        active_pairs = d2d < thresh
        
        # Boundary active
        dists_boundary = 1.0 - np.hypot(p2d[:, 0], p2d[:, 1])
        min_boundary = np.min(dists_boundary)
        active_boundary = dists_boundary < min_boundary + 1e-4
        
        # We want to increase the minimum.
        # Gradient of distance d_ij with respect to p_i (in 2D) is (p_i - p_j)/d_ij.
        # We need the Riemannian gradient on the sphere.
        # The Riemannian gradient of a function f on the sphere is the projection of the Euclidean gradient onto the tangent space.
        # Tangent space at p: {v | v . p = 0}.
        # Projection: grad_R = grad_E - (grad_E . p / |p|^2) p.
        
        # How to get Euclidean gradient of the 2D distance from 3D positions?
        # This is complex. 
        # Alternative: Use the geodesic distance directly.
        # Gradient of geodesic distance d(i,j) with respect to p_i on the sphere:
        # It is the unit vector along the geodesic from j to i, projected to tangent space of p_i?
        # Actually, the gradient of the geodesic distance function d(p, q) with respect to p is the unit vector pointing from q to p along the geodesic, tangent to the sphere at p.
        # Let u_i be the unit vector from p_j to p_i along the geodesic.
        # u_i = (p_i - p_j) / ||p_i - p_j|| is NOT correct on sphere.
        # Correct: u_i = (p_i - cos(theta) p_j) / sin(theta) is the tangent vector at p_i pointing towards p_j?
        # No, the gradient of d(p,q) w.r.t p is the unit tangent vector at p pointing AWAY from q?
        # To maximize d, we want to move p away from q.
        # So grad d = - (unit tangent vector from p to q).
        # Or simply: grad d = (p_i - cos(theta_ij) p_j) / sin(theta_ij) ?
        # Let's check. If p_i = p_j, undefined.
        # If we move p_i in direction v, d changes by v . grad d.
        # The geodesic from p_j to p_i is p(t) = (sin((1-t)theta)/sin(theta)) p_j + (sin(t theta)/sin(theta)) p_i.
        # The tangent vector at p_i is proportional to p_i - cos(theta) p_j.
        # So grad_{p_i} d(i,j) = (p_i - cos(theta_ij) p_j) / sin(theta_ij).
        # This vector is tangent to the sphere at p_i (dot with p_i is 0).
        
        grad_3d = np.zeros_like(p3d)
        
        # Sum gradients from active pairs
        for i in range(N):
            for j in range(N):
                if i == j or not active_pairs[i, j]:
                    continue
                # Calculate gradient contribution from j to i
                dot_ij = np.dot(p3d[i], p3d[j]) / (R_SPHERE * R_SPHERE)
                cos_theta = np.clip(dot_ij, -1.0, 1.0)
                sin_theta = np.sqrt(1.0 - cos_theta**2)
                if sin_theta > 1e-6:
                    # Vector pointing from j to i along geodesic
                    # grad d_ij / d p_i
                    v = (p3d[i] - cos_theta * p3d[j]) / sin_theta
                    # We want to maximize distance, so move in direction v
                    grad_3d[i] += v
                    
        # Boundary gradient
        # We want to maximize z_i (to keep inside).
        # Gradient of z_i w.r.t p_i on sphere is (0,0,1) projected to tangent space.
        # proj(0,0,1) = (0,0,1) - (z_i/R^2) p_i
        for i in range(N):
            if active_boundary[i]:
                e_z = np.array([0.0, 0.0, 1.0])
                proj_e_z = e_z - (p3d[i, 2] / (R_SPHERE * R_SPHERE)) * p3d[i]
                grad_3d[i] += proj_e_z
        
        # Normalize gradient?
        norm_grad = np.linalg.norm(grad_3d)
        if norm_grad > 1e-6:
            grad_3d /= norm_grad
        
        # Update
        p3d_new = p3d + lr * grad_3d
        
        # Retraction: Project back to sphere
        norms = np.linalg.norm(p3d_new, axis=1, keepdims=True)
        p3d_new = p3d_new / norms * R_SPHERE
        
        # Check validity
        p2d_new = _stereo_backward(p3d_new)
        r_new = _compute_radius(p2d_new)
        
        if r_new > r_current:
            p3d = p3d_new
        else:
            # If no improvement, maybe reduce lr?
            # We handle stagnation above.
            pass
            
    return best_p2d, best_r

def construct():
    deadline = time.time() + 110.0
    
    # Start with hexagonal packing
    centers = _hexagonal_packing()
    
    # Run Riemannian optimization
    best_centers, best_r = _riemannian_optimize(centers, deadline)
    
    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(_compute_radius(centers))
