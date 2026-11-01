"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s kill

    # 1. Generate initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(P):
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return P / norms

    pts = normalize(pts)

    # 2. Compute covering radius (max angular distance to nearest point)
    def covering_radius(P):
        # P: (N, 3)
        # For each pair, compute angular distance
        # We want max over sphere of min distance to any point.
        # This is equivalent to max over all pairs (i,j) of the circumradius of the triangle
        # formed by the three nearest neighbors? No, it's the max over all points on the sphere.
        # The max distance to nearest point occurs at a vertex of the Voronoi diagram on the sphere,
        # which is the circumcenter of some set of 3 (or more) points.
        # A simpler upper bound / approximation: for each point p_i, look at its neighbors.
        # Actually, the covering radius is the max over all points y on sphere of min_i angle(y, p_i).
        # This is equal to the max over all "Voronoi vertices" (circumcenters of empty caps).
        # An efficient way: for each point, compute distances to all others.
        # The covering radius is at least half the max nearest-neighbor distance? No.
        # Let's use the fact that the covering radius is the max over all pairs (i,j) such that
        # there is no point in the cap centered at the circumcenter of p_i, p_j, and a third point?
        # Actually, the standard way: the covering radius is the maximum, over all points on the sphere,
        # of the distance to the nearest point. This maximum occurs at a point equidistant from at least 3 points.
        # So we can iterate over all triples, compute the circumcenter, check if the cap is empty, and take the max radius.
        # But that's O(N^3) which is 130^3 = 2.2M, which is fine.
        
        # Alternative: use scipy.spatial.ConvexHull on the points projected to 3D.
        # The covering radius is related to the inradius of the convex hull? No, it's the max distance from the origin to the facets?
        # Actually, the covering radius is the max over all facets of the convex hull of the distance from the origin to the facet plane.
        # Wait, no. The covering radius is the max over all points y on the sphere of min_i |y - p_i| (angular).
        # This is equal to the max over all "empty caps". The largest empty cap is centered at a point equidistant from at least 3 points.
        # So the covering radius is the max over all triples (i,j,k) of the circumradius of the triangle (i,j,k) on the sphere,
        # provided that the cap of that radius centered at the circumcenter contains no other points.
        
        # A faster approach: for each point p_i, find its k-nearest neighbors. The covering radius is at least half the max nearest-neighbor distance.
        # But to get the exact covering radius, we need to check the Voronoi vertices.
        
        # Let's use a simpler but accurate method: 
        # For each point, compute the distance to all other points.
        # The covering radius is the max over all points y of min_i d(y, p_i).
        # This is the same as the max over all "Voronoi cells" of the max distance in that cell.
        # The max distance in a Voronoi cell occurs at a vertex of the cell.
        # So we need to find all Voronoi vertices and compute their distance to the nearest point.
        
        # Since N=130, we can afford O(N^3) or even O(N^2 * log N).
        # Let's compute all pairwise angular distances.
        P = normalize(P)
        # Compute dot products
        dots = np.dot(P, P.T)
        dots = np.clip(dots, -1.0, 1.0)
        # Angular distances
        dists = np.arccos(dots)
        
        # For each point, sort distances to others
        # The covering radius is the max over all y of min_i d(y, p_i).
        # We can approximate this by checking the midpoints between pairs? No, that's for packing.
        # For covering, the worst case is at a point equidistant from 3 points.
        
        # Let's use the convex hull approach.
        # The covering radius is the max over all facets of the convex hull of the distance from the origin to the facet.
        # Wait, no. The distance from the origin to the facet plane is the inradius of that facet's cap?
        # Actually, the covering radius is the max over all points y on the sphere of the distance to the nearest point.
        # This is equal to the max over all "empty caps". The largest empty cap has its center at a Voronoi vertex.
        # The radius of that cap is the distance from the center to the nearest point.
        
        # Let's use scipy.spatial.ConvexHull to find the facets.
        from scipy.spatial import ConvexHull
        hull = ConvexHull(P)
        
        # For each facet, compute the distance from the origin to the facet plane.
        # The facet is defined by 3 points. The plane equation is n . x = d, where n is the normal.
        # The distance from the origin to the plane is |d| / |n|.
        # But we need the angular distance.
        # The covering radius is the max over all facets of the angle between the origin and the facet plane.
        # Actually, the distance from the origin to the facet plane is cos(theta), where theta is the angular distance.
        # So the covering radius is the max over all facets of arccos(distance from origin to facet plane).
        
        # Let's compute this.
        max_cov = 0.0
        for simplex in hull.simplices:
            # simplex is a set of 3 indices
            p1, p2, p3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
            # Compute the normal to the plane
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            # Distance from origin to plane
            d = np.dot(n, p1)
            # The angular distance is arccos(d)
            # But we need to check if the cap is empty. The convex hull ensures that the origin is on one side,
            # so the cap on the other side is empty.
            cov = np.arccos(np.clip(abs(d), -1.0, 1.0))
            if cov > max_cov:
                max_cov = cov
        
        return np.degrees(max_cov)

    # 3. Optimization: repulsive force / gradient descent
    # We want to minimize the covering radius.
    # The covering radius is a nonsmooth function, but we can use a smooth approximation.
    # Alternatively, we can use a repulsive force model: each point repels others, and we also have a force
    # that pushes points away from the "worst" directions.
    
    # Let's use a simple gradient descent on a smooth surrogate.
    # The covering radius is the max over all y of min_i d(y, p_i).
    # A smooth approximation: use the log-sum-exp of the distances.
    # But we need to evaluate this over a grid of y points.
    
    # Let's use a grid of points on the sphere to approximate the covering radius.
    # Generate a grid of M points on the sphere.
    M = 2000  # number of grid points
    # Use a Fibonacci spiral for the grid
    j = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * j / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_grid = np.sqrt(np.maximum(0.0, 1.0 - z_grid * z_grid))
    grid = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    grid = normalize(grid)
    
    def smooth_covering_radius(P, temperature=0.1):
        P = normalize(P)
        # Compute distances from grid to all points
        # dists_grid: (M, N)
        dots = np.dot(grid, P.T)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        # For each grid point, find the min distance
        min_dists = np.min(dists, axis=1)
        # Smooth max: log-sum-exp
        # max_i min_dists_i ≈ temperature * log(sum(exp(min_dists_i / temperature)))
        # But we want the max, so we use the log-sum-exp of the min_dists.
        # Actually, the max of a set of numbers x_i is approximated by temperature * log(sum(exp(x_i / temperature))).
        # So the smooth covering radius is temperature * log(sum(exp(min_dists / temperature))).
        # But we want to minimize this, so we can use it as the objective.
        # However, the log-sum-exp is a smooth approximation to the max.
        # Let's compute it.
        scaled = min_dists / temperature
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return temperature * log_sum_exp

    # Gradient descent
    learning_rate = 0.01
    for step in range(5000):
        if time.time() > deadline - 5.0:
            break
        
        P = normalize(pts)
        # Compute gradient of the smooth covering radius
        # The gradient with respect to p_i is the sum over grid points of the contribution.
        # This is complex, so let's use a finite difference or a simpler approach.
        
        # Alternative: use a repulsive force model.
        # Each point repels others with a force proportional to 1/d^2.
        # Also, add a force that pushes points away from the "worst" grid points.
        
        # Let's use a simpler approach: for each point, compute the gradient of the smooth covering radius.
        # The smooth covering radius is F(P) = temperature * log(sum_j exp(min_i d(y_j, p_i) / temperature)).
        # The gradient with respect to p_i is:
        # sum_j [ exp(min_i d(y_j, p_i) / temperature) / sum_k exp(min_k d(y_j, p_k) / temperature) * gradient of min_i d(y_j, p_i) with respect to p_i ]
        
        # This is complex, so let's use a finite difference approximation.
        # Or, let's use a simpler repulsive force model.
        
        # Repulsive force: each point repels others.
        dots = np.dot(P, P.T)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        # Set diagonal to 1 to avoid division by zero
        np.fill_diagonal(dists, 1.0)
        # Force: 1/d^2
        forces = 1.0 / (dists * dists)
        np.fill_diagonal(forces, 0.0)
        # The force on point i is the sum over j of (p_i - p_j) / |p_i - p_j|^2 * (1/d^2)
        # But we need to project this onto the tangent plane.
        
        # Let's use a simpler approach: move each point in the direction of the repulsive force.
        # The repulsive force on point i is sum_j (p_i - p_j) / |p_i - p_j|^2 * (1/d^2)
        # But this is not the gradient of the covering radius.
        
        # Let's use a different approach: use the grid to compute the gradient.
        # For each grid point y_j, the min distance is min_i d(y_j, p_i).
        # Let i_j be the index of the nearest point to y_j.
        # The gradient of d(y_j, p_i) with respect to p_i is (y_j - p_i) / |y_j - p_i| * (1 / sqrt(1 - (y_j . p_i)^2))
        # But this is complex.
        
        # Let's use a simpler approach: use a random perturbation and accept if it improves.
        # This is a stochastic gradient descent.
        
        # Choose a random point
        idx = np.random.randint(N)
        # Compute the current smooth covering radius
        F_old = smooth_covering_radius(pts, temperature=0.1)
        # Perturb the point
        delta = np.random.randn(3) * 0.01
        pts_new = pts.copy()
        pts_new[idx] += delta
        pts_new[idx] /= np.linalg.norm(pts_new[idx])
        F_new = smooth_covering_radius(pts_new, temperature=0.1)
        if F_new < F_old:
            pts = pts_new
        else:
            # Try a smaller perturbation
            delta = np.random.randn(3) * 0.001
            pts_new[idx] += delta
            pts_new[idx] /= np.linalg.norm(pts_new[idx])
            F_new = smooth_covering_radius(pts_new, temperature=0.1)
            if F_new < F_old:
                pts = pts_new
    
    # 4. Final polish: use a high-precision local solve
    # Use the convex hull to find the exact covering radius and adjust points.
    for _ in range(100):
        if time.time() > deadline - 2.0:
            break
        P = normalize(pts)
        cov = covering_radius(P)
        # Find the worst facet
        from scipy.spatial import ConvexHull
        hull = ConvexHull(P)
        max_cov = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            p1, p2, p3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            d = np.dot(n, p1)
            cov_facet = np.arccos(np.clip(abs(d), -1.0, 1.0))
            if cov_facet > max_cov:
                max_cov = cov_facet
                worst_simplex = simplex
        if worst_simplex is None:
            break
        # Move the points in the worst simplex slightly away from the circumcenter
        p1, p2, p3 = P[worst_simplex[0]], P[worst_simplex[1]], P[worst_simplex[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            break
        n = n / n_norm
        # The circumcenter is in the direction of n
        # Move the points slightly away from n
        for idx in worst_simplex:
            pts[idx] -= 0.001 * n
            pts[idx] /= np.linalg.norm(pts[idx])
    
    return normalize(pts)
# EVOLVE-BLOCK-END
