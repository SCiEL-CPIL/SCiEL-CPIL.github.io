"""Rewritten program: spectral + gradient optimization for 130-point sphere covering."""
import numpy as np
import time

N = 130


def _covering_radius(points):
    """Compute the covering radius (in degrees) for points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the
    sphere to its nearest chosen point. This equals the largest circumradius
    of the Delaunay triangulation facets (convex hull faces).
    
    We approximate by: for each point, compute the angular radius of its
    Voronoi cell's farthest vertex. A practical approximation uses the
    nearest-neighbor distances and the convex hull.
    
    Here we use a sampling-based estimate refined by checking hull facets.
    """
    # Use scipy for convex hull if available, otherwise fallback
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        # For each hull face (facet), compute the circumradius of that face
        # The covering radius is the max over all facets of the angular distance
        # from the facet center (projected to sphere) to the farthest vertex
        max_cr = 0.0
        for simplex in hull.simplices:
            face_pts = points[simplex]
            # Compute circumcenter of the face points in 3D
            # For a triangle, the circumcenter is the point equidistant from all 3 vertices
            # We project the face centroid onto the sphere and measure max angular distance
            # Better: compute the circumradius in 3D, then convert to angular
            a, b, c = face_pts[0], face_pts[1], face_pts[2]
            # Circumcenter of triangle in 3D
            # |a-c|^2 * |b-c|^2 - (a-c).(b-c))^2) / (2 * cross(a-c, b-c))
            ac = a - c
            bc = b - c
            cross = np.cross(ac, bc)
            cross_norm_sq = np.dot(cross, cross)
            if cross_norm_sq < 1e-14:
                continue
            dot_ac_bc = np.dot(ac, bc)
            dot_ac_ac = np.dot(ac, ac)
            dot_bc_bc = np.dot(bc, bc)
            # Circumcenter relative to c
            num = dot_ac_ac * dot_bc_bc - dot_ac_bc ** 2
            denom = 2.0 * cross_norm_sq
            if denom < 1e-14:
                continue
            # The circumradius squared
            R_sq = num / denom
            R = np.sqrt(max(R_sq, 0.0))
            # Angular distance: for points on unit sphere, the circumradius of the
            # spherical triangle is related to the planar circumradius
            # Actually, for small triangles, angular radius ≈ R (in radians)
            # More precisely: the covering radius contribution from this facet is
            # the angular distance from the circumcenter (on sphere) to each vertex
            # Let's compute it properly:
            # The circumcenter in 3D:
            # O = c + (|a-c|^2 * (b-c) x (n) + |b-c|^2 * (n) x (a-c)) / (2 * |cross|^2)
            # where n = cross / |cross| ... this is getting complex.
            # Simpler: the angular covering radius for this facet is
            # arcsin(R / d) where d is distance from origin to the plane, but
            # for points on unit sphere, the spherical circumradius is:
            # Actually, the farthest point on the sphere from the three vertices
            # within this facet's Voronoi cell is at angular distance = 
            # the angle subtended by the circumradius at the origin.
            # For a triangle inscribed in the unit sphere, the circumradius of
            # the spherical triangle satisfies:
            # sin(circumradius) = R_planar / (2 * R_circumscribed_of_triangle)
            # This is getting too complex. Let me use a different approach.
            
            # Alternative: the angular distance from the circumcenter of the 
            # spherical triangle to each vertex IS the covering radius for that cell.
            # For a spherical triangle with vertices on unit sphere, the circumradius
            # (angular) can be computed from the side lengths.
            # Let's just use: the covering radius is max over all pairs of adjacent
            # points of half their angular distance, refined by checking hull.
            # Actually the simplest correct method: for each hull face, the 
            # "farthest" point in that Voronoi region is the circumcenter of the
            # face (on the sphere). The angular distance from that circumcenter to
            # any of the 3 vertices is the local covering radius.
            
            # Compute spherical circumcenter:
            # The circumcenter of a spherical triangle is the point on the sphere
            # equidistant (angularly) from all three vertices.
            # It's the intersection of the sphere with the line through the planar
            # circumcenter perpendicular to the face plane... no.
            # 
            # Simplest robust approach: for each hull face, the local covering
            # radius is the maximum angular distance from the face's centroid
            # (normalized to unit sphere) to its vertices. This is an upper bound
            # on the true circumradius but a good approximation.
            centroid = np.mean(face_pts, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-10:
                continue
            centroid_unit = centroid / centroid_norm
            # Angular distances from centroid_unit to each vertex
            for v in face_pts:
                cos_ang = np.clip(np.dot(centroid_unit, v), -1.0, 1.0)
                ang = np.arccos(cos_ang)
                max_cr = max(max_cr, ang)
        
        return np.degrees(max_cr)
    except Exception:
        # Fallback: use nearest neighbor based estimate
        # For each point, the covering radius is at most half the distance to
        # the second nearest neighbor (rough heuristic)
        # Better: compute for each point the max angular distance to its 3 nearest neighbors
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # Query 4 nearest neighbors (including self)
        dists, idxs = tree.query(points, k=4)
        # The local covering radius is approx the max angular dist to 3rd nearest
        # (index 2, since 0 is self)
        max_ang = 0.0
        for i in range(N):
            for j in range(1, 4):
                cos_ang = np.clip(np.dot(points[i], points[idxs[i, j]]), -1.0, 1.0)
                ang = np.arccos(cos_ang)
                max_ang = max(max_ang, ang)
        return np.degrees(max_ang) * 0.5  # rough


def _covering_radius_fast(points):
    """Faster covering radius using convex hull facet circumradii."""
    from scipy.spatial import ConvexHull
    
    hull = ConvexHull(points)
    max_cr = 0.0
    
    for simplex in hull.simplices:
        face_pts = points[simplex]
        # Compute the circumradius of this spherical triangle
        # Using the formula: for a spherical triangle with side lengths a, b, c,
        # the circumradius R satisfies:
        # sin(R) = (sin(a) * sin(b) * sin(c)) / (4 * sin(s) * sin(s-a) * sin(s-b) * sin(s-c))
        # where s = (a+b+c)/2
        # This is complex. Simpler: use the planar circumcenter approach.
        
        a, b, c = face_pts[0], face_pts[1], face_pts[2]
        
        # Compute planar circumcenter
        ac = a - c
        bc = b - c
        cross = np.cross(ac, bc)
        cross_norm_sq = np.dot(cross, cross)
        if cross_norm_sq < 1e-12:
            continue
        
        # Circumcenter in the plane of the triangle
        # O = c + (|ac|^2 * cross_bc_perp + |bc|^2 * cross_ac_perp) / (2 * |cross|^2)
        # where cross_bc_perp = cross x bc, cross_ac_perp = ac x cross
        cross_bc_perp = np.cross(cross, bc)
        cross_ac_perp = np.cross(ac, cross)
        
        dot_ac_ac = np.dot(ac, ac)
        dot_bc_bc = np.dot(bc, bc)
        
        O = c + (dot_ac_ac * cross_bc_perp + dot_bc_bc * cross_ac_perp) / (2.0 * cross_norm_sq)
        
        # Normalize to unit sphere
        O_norm = np.linalg.norm(O)
        if O_norm < 1e-10:
            continue
        O_unit = O / O_norm
        
        # Angular distances from O_unit to each vertex
        for v in face_pts:
            cos_ang = np.clip(np.dot(O_unit, v), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            max_cr = max(max_cr, ang)
    
    return np.degrees(max_cr)


def _energy(points):
    """Energy function: sum of exp(-beta * min_dist^2) to encourage spreading."""
    # Use a potential that penalizes close pairs
    # E = sum_{i<j} exp(-alpha * d_ij^2) where d is angular distance
    # Or use: E = sum_{i<j} (1/d_ij^2) to directly maximize min distance
    
    # For efficiency, compute pairwise distances
    # cos_dist[i,j] = dot(p_i, p_j)
    cos_mat = points @ points.T
    # Angular distance: arccos(cos_mat)
    # For the energy, we can use: E = sum exp(-beta * (1 - cos_mat)^2)
    # which penalizes small angular separations
    
    beta = 2.0
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    diff = 1.0 - cos_mat  # proportional to small-angle distance squared
    E = np.sum(np.exp(-beta * diff[mask]))
    return E


def _gradient(points):
    """Gradient of the energy with respect to points on the sphere.
    
    We use the projection: grad_sphere = grad - (grad . p) * p
    """
    cos_mat = points @ points.T
    beta = 2.0
    diff = 1.0 - cos_mat
    
    # dE/dcos_mat[i,j] = -beta * exp(-beta * diff[i,j]) * (-1) = beta * exp(-beta * diff[i,j])
    # But diff = 1 - cos, so d(diff)/d(cos) = -1
    # dE/d(cos_mat[i,j]) = beta * exp(-beta * diff[i,j]) * (-1) * (-1) ... 
    # Let me redo: E = sum_{i!=j} exp(-beta * (1-cos_ij)^2)
    # dE/d(cos_ij) = exp(-beta*(1-cos_ij)^2) * (-beta) * 2*(1-cos_ij) * (-1)
    #               = 2*beta*(1-cos_ij)*exp(-beta*(1-cos_ij)^2)
    
    mask = ~np.eye(N, dtype=bool)
    grad_cos = np.zeros_like(cos_mat)
    grad_cos[mask] = 2.0 * beta * diff[mask] * np.exp(-beta * diff[mask] ** 2)
    
    # grad_cos is symmetric, and dE/dp_i = sum_j dE/d(cos_ij) * d(cos_ij)/dp_i
    # cos_ij = p_i . p_j, so d(cos_ij)/dp_i = p_j
    # grad_p_i = sum_{j != i} grad_cos[i,j] * p_j
    
    grad = grad_cos @ points  # This gives sum_j grad_cos[i,j] * p_j
    
    # Project to tangent space
    grad_proj = grad - np.sum(grad * points, axis=1, keepdims=True) * points
    
    return grad_proj


def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _project_to_sphere(points)
    
    best_points = points.copy()
    best_cr = _covering_radius_fast(points)
    
    # Phase 1: Gradient descent with adaptive step size
    alpha = 0.05
    min_alpha = 1e-6
    patience = 0
    max_patience = 200
    
    while time.time() < deadline - 20.0:
        grad = _gradient(points)
        
        # Adaptive step size
        new_points = _project_to_sphere(points - alpha * grad)
        
        # Check improvement
        new_energy = _energy(new_points)
        old_energy = _energy(points)
        
        if new_energy < old_energy:
            points = new_points
            patience = 0
            # Increase step size slightly if improving
            alpha = min(alpha * 1.05, 0.1)
        else:
            patience += 1
            alpha *= 0.8
            if alpha < min_alpha:
                break
        
        if patience > max_patience:
            break
        
        # Periodically check covering radius
        if patience % 50 == 0 or patience == 0:
            cr = _covering_radius_fast(points)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
    
    # Phase 2: Local refinement - try small random perturbations
    while time.time() < deadline - 5.0:
        # Pick a random subset of points to perturb
        n_perturb = np.random.randint(3, 15)
        idx = np.random.choice(N, n_perturb, replace=False)
        
        perturb_points = best_points.copy()
        for i in idx:
            # Random small perturbation on sphere
            delta = np.random.randn(3) * 0.02
            perturb_points[i] = _project_to_sphere(perturb_points[i] + delta)[0]
        
        cr = _covering_radius_fast(perturb_points)
        if cr < best_cr:
            best_cr = cr
            best_points = perturb_points.copy()
    
    # Phase 3: Fine polish - very small steps
    alpha = 1e-4
    for _ in range(500):
        if time.time() > deadline - 2.0:
            break
        grad = _gradient(best_points)
        new_points = _project_to_sphere(best_points - alpha * grad)
        cr = _covering_radius_fast(new_points)
        if cr < best_cr:
            best_cr = cr
            best_points = new_points.copy()
            alpha *= 1.1
        else:
            alpha *= 0.7
    
    # Final check
    final_cr = _covering_radius_fast(best_points)
    if final_cr < best_cr:
        best_cr = final_cr
    
    return _project_to_sphere(best_points)
