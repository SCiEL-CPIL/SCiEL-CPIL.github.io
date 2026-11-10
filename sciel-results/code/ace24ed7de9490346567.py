import numpy as np
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _angular_dist(a, b):
    """Angular distance between unit vectors a (N,3) and b (M,3) -> (N,M)"""
    dots = np.clip(a @ b.T, -1.0, 1.0)
    return np.arccos(dots)

def _covering_radius(pts):
    """Compute covering radius in degrees using convex hull facets."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            face_pts = pts[simplex]
            # Circumradius of the spherical triangle
            # For a spherical triangle, the circumradius is the angular distance
            # from the sphere center to the circumcenter
            # Use formula: for triangle with side lengths a,b,c (angular),
            # circumradius R satisfies sin(R) = ... 
            # Simpler: find the point on the sphere that maximizes min distance to face vertices
            # The circumcenter of a spherical triangle:
            a = face_pts[0]
            b = face_pts[1]
            c = face_pts[2]
            # Circumcenter via cross product method
            # The circumcenter lies along the direction a x b + b x c + c x a normalized? No.
            # Use: circumcenter = normalize(a*sin(A) + b*sin(B) + c*sin(C)) where A,B,C are angles
            # Simpler: solve for point p on sphere minimizing max angular dist to a,b,c
            # For a triangle, the circumradius is arccos((a·b + b·c + c·a - 1) / (2*sqrt((1-(a·b)^2)(1-(b·c)^2)(1-(c·a)^2)) * ... ))
            # Actually, let me use a different approach: the circumradius of a spherical triangle
            # with vertices a,b,c is the angle from the origin to the circumcenter.
            # The circumcenter p satisfies p·a = p·b = p·c = cos(R)
            # So p is in the span of a,b,c and p·a = p·b = p·c
            # Let p = alpha*a + beta*b + gamma*c
            # p·a = alpha + beta*(b·a) + gamma*(c·a) = cos(R)
            # p·b = alpha*(a·b) + beta + gamma*(c·b) = cos(R)
            # p·c = alpha*(a·c) + beta*(b·c) + gamma = cos(R)
            ab = a @ b
            bc = b @ c
            ca = c @ a
            A = np.array([[1.0, ab, ca],
                           [ab, 1.0, bc],
                           [ca, bc, 1.0]])
            b_vec = np.ones(3)
            try:
                coeffs = np.linalg.solve(A, b_vec)
                p = coeffs[0]*a + coeffs[1]*b + coeffs[2]*c
                pn = np.linalg.norm(p)
                if pn < 1e-15:
                    continue
                p = p / pn
                R = np.arccos(np.clip(p @ a, -1.0, 1.0))
                if R > max_r:
                    max_r = R
            except np.linalg.LinAlgError:
                continue
        return np.degrees(max_r)
    except Exception:
        return 180.0

def _soft_covering_radius(pts, sample_count=2000, temperature=0.05):
    """Soft approximation of covering radius using random sample points."""
    # Generate sample points on sphere
    theta = np.random.uniform(0, 2*np.pi, sample_count)
    z = np.random.uniform(-1, 1, sample_count)
    r = np.sqrt(1 - z*z)
    samples = np.column_stack([r*np.cos(theta), r*np.sin(theta), z])
    
    # For each sample, find min angular distance to any point in pts
    dists = _angular_dist(samples, pts)  # (sample_count, N)
    min_dists = np.min(dists, axis=1)
    
    # Soft max
    scaled = min_dists / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    soft_max = temperature * (max_scaled + np.log(np.sum(exp_scaled)))
    
    return soft_max

def _gradient_soft_covering(pts, sample_count=1000, temperature=0.05):
    """Compute gradient of soft covering radius w.r.t. points."""
    theta = np.random.uniform(0, 2*np.pi, sample_count)
    z = np.random.uniform(-1, 1, sample_count)
    r = np.sqrt(1 - z*z)
    samples = np.column_stack([r*np.cos(theta), r*np.sin(theta), z])
    
    dots = np.clip(samples @ pts.T, -1.0, 1.0)  # (S, N)
    dists = np.arccos(dots)  # (S, N)
    min_idx = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(sample_count), min_idx]
    
    # Soft max weights
    scaled = min_dists / temperature
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # Gradient: d/d(pts) of soft_max
    # For each point j, gradient = sum_s weights_s * d/d(pts_j) min_dist_s
    # min_dist_s is distance from sample_s to pts[min_idx_s]
    # d/d(pts_j) arccos(sample @ pts_j) = -sample / sqrt(1 - (sample@pts_j)^2)
    grad = np.zeros_like(pts)
    for s in range(sample_count):
        j = min_idx[s]
        sample = samples[s]
        dot = np.clip(sample @ pts[j], -1.0, 1.0)
        denom = np.sqrt(max(1.0 - dot*dot, 1e-12))
        grad[j] += weights[s] * (-sample / denom)
    
    return grad

def _local_refine(pts, max_iter=200, tol=1e-10):
    """Local refinement using facet circumradius minimization."""
    from scipy.spatial import ConvexHull
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        # Find the facet with largest circumradius
        max_r = 0.0
        max_face = None
        for simplex in hull.simplices:
            face_pts = best_pts[simplex]
            a, b, c = face_pts[0], face_pts[1], face_pts[2]
            ab = a @ b
            bc = b @ c
            ca = c @ a
            A = np.array([[1.0, ab, ca],
                           [ab, 1.0, bc],
                           [ca, bc, 1.0]])
            b_vec = np.ones(3)
            try:
                coeffs = np.linalg.solve(A, b_vec)
                p = coeffs[0]*a + coeffs[1]*b + coeffs[2]*c
                pn = np.linalg.norm(p)
                if pn < 1e-15:
                    continue
                p = p / pn
                R = np.arccos(np.clip(p @ a, -1.0, 1.0))
                if R > max_r:
                    max_r = R
                    max_face = (simplex, p, R)
            except np.linalg.LinAlgError:
                continue
        
        if max_face is None or max_r < tol:
            break
        
        simplex, circumcenter, R = max_face
        
        # Move the vertex farthest from circumcenter toward it
        # Actually, move all three vertices toward the circumcenter to shrink the triangle
        # Better: move the vertex that is farthest from the circumcenter
        face_pts = best_pts[simplex]
        dists_to_cc = np.array([np.arccos(np.clip(fp @ circumcenter, -1.0, 1.0)) for fp in face_pts])
        worst_idx = np.argmax(dists_to_cc)
        global_idx = simplex[worst_idx]
        
        # Move this point slightly toward the circumcenter
        step = 0.01
        new_pt = best_pts[global_idx] + step * (circumcenter - best_pts[global_idx])
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        new_pts = best_pts.copy()
        new_pts[global_idx] = new_pt
        
        new_radius = _covering_radius(new_pts)
        if new_radius < best_radius - 1e-12:
            best_pts = new_pts
            best_radius = new_radius
        else:
            # Try smaller step
            step = 0.001
            new_pt = best_pts[global_idx] + step * (circumcenter - best_pts[global_idx])
            new_pt = new_pt / np.linalg.norm(new_pt)
            new_pts = best_pts.copy()
            new_pts[global_idx] = new_pt
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius - 1e-12:
                best_pts = new_pts
                best_radius = new_radius
            else:
                break
    
    return best_pts, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_radius = np.inf
    
    # Phase 1: Generate diverse initial configurations and run soft-gradient optimization
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Generate initial configuration
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 1:
            # Different golden angle
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (3.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 2:
            # Random with repulsion
            pts = np.random.randn(N, 3)
            pts = _normalize(pts)
            # Simple repulsion iterations
            for _ in range(50):
                dots = np.clip(pts @ pts.T, -1.0, 1.0)
                np.fill_diagonal(dots, 0)
                dists = np.arccos(dots)
                # Repulsion: move points away from close neighbors
                forces = np.zeros_like(pts)
                for i in range(N):
                    for j in range(N):
                        if i != j and dists[i,j] < 0.5:
                            forces[i] += (pts[i] - pts[j]) / (dists[i,j] + 1e-10)
                pts = _normalize(pts + 0.01 * forces)
        else:
            # Random start
            pts = np.random.randn(N, 3)
            pts = _normalize(pts)
        
        pts = _normalize(pts)
        
        # Soft-gradient optimization
        temperature = 0.1
        lr = 0.005
        for it in range(300):
            if time.time() > deadline - 20.0:
                break
            
            grad = _gradient_soft_covering(pts, sample_count=500, temperature=temperature)
            new_pts = pts - lr * grad
            new_pts = _normalize(new_pts)
            
            # Check improvement
            old_cr = _soft_covering_radius(pts, 500, temperature)
            new_cr = _soft_covering_radius(new_pts, 500, temperature)
            if new_cr < old_cr:
                pts = new_pts
            else:
                lr *= 0.8
                if lr < 1e-6:
                    break
            
            # Anneal temperature
            if it % 50 == 0 and it > 0:
                temperature *= 0.7
        
        # Local refinement
        refined_pts, refined_radius = _local_refine(pts, max_iter=100)
        
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_pts = refined_pts
    
    if best_pts is None:
        # Fallback: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return _normalize(best_pts)
