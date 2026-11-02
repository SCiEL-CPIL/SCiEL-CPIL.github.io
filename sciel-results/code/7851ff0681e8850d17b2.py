"""Sphere covering with 130 equal caps - combined facet and density optimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.array(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-14] = 1e-14
    return pts / norms[:, None]

def _covering_radius(points):
    """Compute covering radius in degrees via convex hull facets."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
    except Exception:
        return _covering_radius_grid(points, 360, 720)
    
    max_r = 0.0
    for face in faces:
        p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-14:
            continue
        n = n / nn
        # Circumradius: angle from face normal to vertex
        cos_r = np.abs(n @ p1)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    return np.degrees(max_r)

def _covering_radius_grid(points, n_theta=360, n_phi=720):
    """Grid-based covering radius for fallback."""
    points = _normalize(points)
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    grid = np.column_stack([
        (np.sin(TH)*np.cos(PH)).ravel(),
        (np.sin(TH)*np.sin(PH)).ravel(),
        np.cos(TH).ravel()
    ])
    max_d = 0.0
    for s in range(0, len(grid), 20000):
        e = min(s+20000, len(grid))
        dots = np.clip(grid[s:e] @ points.T, -1, 1)
        max_d = max(max_d, np.arccos(dots.max(axis=1)).max())
    return np.degrees(max_d)

def _worst_facet(points):
    """Return (worst_face, radius) or (None, 0)."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
    except Exception:
        return None, 0.0
    
    max_r = 0.0
    wf = None
    for face in faces:
        p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-14:
            continue
        n = n / nn
        cos_r = np.abs(n @ p1)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
            wf = face
    return wf, max_r

def _fibonacci_init(n):
    """Fibonacci spiral initial placement."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0*i/n
    phi = np.pi*(1.0+np.sqrt(5.0))*i
    r = np.sqrt(np.maximum(0, 1.0 - z*z))
    pts = np.column_stack([r*np.cos(phi), r*np.sin(phi), z])
    return _normalize(pts)

def _lloyd_refine(points, n_iter=80):
    """Spherical Lloyd's algorithm for balanced Voronoi cells."""
    pts = _normalize(points.copy())
    n_theta, n_phi = 240, 480
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    grid = np.column_stack([
        (np.sin(TH)*np.cos(PH)).ravel(),
        (np.sin(TH)*np.sin(PH)).ravel(),
        np.cos(TH).ravel()
    ])
    
    for it in range(n_iter):
        dots = grid @ pts.T
        nearest = dots.argmax(axis=1)
        new_pts = np.zeros_like(pts)
        counts = np.zeros(N)
        for k in range(N):
            m = nearest == k
            if m.any():
                new_pts[k] = grid[m].mean(axis=0)
                counts[k] = m.sum()
        new_pts[counts == 0] = pts[counts == 0]
        new_pts = _normalize(new_pts)
        if np.linalg.norm(new_pts - pts) < 1e-12:
            pts = new_pts
            break
        pts = new_pts
    return pts

def _facet_push(points, max_time=20.0, lr_init=0.03):
    """Iteratively push worst-facet vertices toward their circumcenter."""
    pts = _normalize(points.copy())
    t0 = time.time()
    best = pts.copy()
    best_r = _covering_radius(pts)
    lr = lr_init
    stall = 0
    
    while time.time() - t0 < max_time and stall < 60:
        wf, r = _worst_facet(pts)
        if wf is None:
            break
        
        p1, p2, p3 = pts[wf[0]], pts[wf[1]], pts[wf[2]]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-14:
            break
        n = n / nn
        c = n if (n @ p1) > 0 else -n  # circumcenter direction
        
        for idx in wf:
            p = pts[idx]
            d = c - p
            d -= (d @ p) * p  # project to tangent plane
            dn = np.linalg.norm(d)
            if dn > 1e-14:
                pts[idx] += lr * d / dn
        pts = _normalize(pts)
        
        nr = _covering_radius(pts)
        if nr < best_r - 1e-9:
            best_r = nr
            best = pts.copy()
            stall = 0
        else:
            pts = best.copy()
            stall += 1
            lr *= 0.88
    
    return best, best_r

def _repulsion_refine(points, max_iter=120, alpha_init=0.004):
    """Repulsion-based spreading to reduce clustering."""
    pts = _normalize(points.copy())
    alpha = alpha_init
    
    for it in range(max_iter):
        dots = np.clip(pts @ pts.T, -1.0, 1.0)
        np.fill_diagonal(dots, 1.0)
        d2 = 2.0 - 2.0*dots
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 1e-10)
        
        w = 1.0 / (d2**2)
        np.fill_diagonal(w, 0.0)
        gw = 2.0 / (d2**2.5)
        np.fill_diagonal(gw, 0.0)
        
        grad = gw @ pts
        gn = np.linalg.norm(grad)
        if gn < 1e-14:
            break
        pts += alpha * grad / gn
        pts = _normalize(pts)
        alpha *= 0.995
    return pts

def _local_search(points, max_time=20.0):
    """Perturbation-based local search on facet objective."""
    pts = _normalize(points.copy())
    t0 = time.time()
    best = pts.copy()
    best_r = _covering_radius(pts)
    step = 0.02
    stall = 0
    
    while time.time() - t0 < max_time and stall < 80:
        # Pick random subset of points to perturb
        n_pert = np.random.randint(1, 6)
        idxs = np.random.choice(N, n_pert, replace=False)
        
        new_pts = pts.copy()
        for idx in idxs:
            # Random tangent perturbation
            p = new_pts[idx]
            d = np.random.randn(3)
            d -= (d @ p) * p
            dn = np.linalg.norm(d)
            if dn > 1e-14:
                new_pts[idx] += step * d / dn
        new_pts = _normalize(new_pts)
        
        nr = _covering_radius(new_pts)
        if nr < best_r - 1e-9:
            best_r = nr
            best = new_pts.copy()
            pts = new_pts.copy()
            stall = 0
            step = min(step * 1.1, 0.05)
        else:
            stall += 1
            step *= 0.92
    
    return best, best_r

def _density_refine(points, max_iter=150, alpha_init=0.003):
    """
    Density-based refinement: minimize a smooth surrogate that approximates
    the covering radius using a density estimation approach.
    
    This uses a kernel density estimator on the sphere and optimizes points
    to maximize the minimum density (which corresponds to minimizing covering radius).
    """
    pts = _normalize(points.copy())
    
    # Create a fine grid for density estimation
    n_theta, n_phi = 120, 240
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    grid = np.column_stack([
        (np.sin(TH)*np.cos(PH)).ravel(),
        (np.sin(TH)*np.sin(PH)).ravel(),
        np.cos(TH).ravel()
    ])
    M = len(grid)
    
    # Kernel bandwidth (angular distance)
    sigma = 0.15  # ~8.6 degrees
    
    alpha = alpha_init
    best = pts.copy()
    best_r = _covering_radius(pts)
    
    for it in range(max_iter):
        if time.time() > 145.0:
            break
            
        # Compute density at each grid point
        # Density_j = sum_k exp(-d(j,k)^2 / (2*sigma^2))
        dots = np.clip(grid @ pts.T, -1.0, 1.0)  # M x N
        d2 = 2.0 - 2.0 * dots  # M x N, squared chord distance ~ 2*sin(d/2)^2
        # Use angular distance squared approximation for small angles
        # For better accuracy, use arccos
        ang_dists = np.arccos(dots)  # M x N
        d2_ang = ang_dists * ang_dists
        
        kernel = np.exp(-d2_ang / (2.0 * sigma * sigma))  # M x N
        density = kernel.sum(axis=1)  # M
        
        # We want to maximize the minimum density
        # Use soft-min: -1/T * log(sum exp(-T * density))
        T = 30.0
        exp_neg_T_d = np.exp(-T * density)
        sum_exp = exp_neg_T_d.sum()
        if sum_exp < 1e-300:
            sum_exp = 1e-300
        soft_min_density = -1.0 / T * np.log(sum_exp)
        
        # Gradient of soft_min_density w.r.t. pts[k]
        # d(soft_min)/d(pts[k]) = (1/T) * (1/sum_exp) * sum_j [exp(-T*density_j) * d(density_j)/d(pts[k])]
        # d(density_j)/d(pts[k]) = sum over the kernel contribution
        # d(kernel_jk)/d(pts[k]) = kernel_jk * d(-d2_ang_jk / (2*sigma^2)) / d(pts[k])
        # d(d2_ang_jk)/d(pts[k]) = 2 * ang_dists_jk * d(ang_dists_jk)/d(pts[k])
        # d(ang_dists_jk)/d(pts[k]) = -grid_j / sqrt(1 - dots_jk^2)  (for the k-th point)
        
        # Simplified gradient computation
        # w_j = exp(-T * density_j) / sum_exp  (M,)
        w = exp_neg_T_d / sum_exp  # M
        
        # For each point k, gradient contribution:
        # sum_j w_j * d(density_j)/d(pts[k])
        # d(density_j)/d(pts[k]) = d(kernel_jk)/d(pts[k])
        # d(kernel_jk)/d(pts[k]) = kernel_jk * (-1/(2*sigma^2)) * d(d2_ang_jk)/d(pts[k])
        # d(d2_ang_jk)/d(pts[k]) = 2 * ang_dists_jk * (-grid_j / sqrt(1-dots_jk^2))
        # So d(kernel_jk)/d(pts[k]) = kernel_jk * (1/(sigma^2)) * ang_dists_jk * grid_j / sqrt(1-dots_jk^2)
        
        sqrt_term = np.sqrt(np.clip(1.0 - dots*dots, 1e-10, 1.0))  # M x N
        grad_kernel = kernel * (ang_dists / (sigma * sigma)) * (grid[:, None] / sqrt_term)  # M x N x 3
        
        # Gradient w.r.t. pts[k]: sum_j w_j * grad_kernel[j, k, :]
        grad_pts = np.einsum('j,jki->i', w, grad_kernel)  # N x 3
        
        # Project to tangent plane
        for k in range(N):
            grad_pts[k] -= (grad_pts[k] @ pts[k]) * pts[k]
        
        # Update: we want to maximize soft_min_density, so move in +grad direction
        gn = np.linalg.norm(grad_pts)
        if gn < 1e-14:
            break
        
        pts = _normalize(pts + alpha * grad_pts / max(gn, 1e-14))
        
        nr = _covering_radius(pts)
        if nr < best_r:
            best_r = nr
            best = pts.copy()
        
        alpha *= 0.995
        T = max(20.0, T * 0.998)
    
    return best

def _combined_refine(points, max_time=30.0):
    """
    Combined refinement that alternates between facet push and density refinement.
    This helps escape local minima that one method alone might get stuck in.
    """
    pts = _normalize(points.copy())
    t0 = time.time()
    best = pts.copy()
    best_r = _covering_radius(pts)
    
    use_facet = True
    stall = 0
    
    while time.time() - t0 < max_time and stall < 100:
        if use_facet:
            # Facet push for a few iterations
            wf, r = _worst_facet(pts)
            if wf is not None:
                p1, p2, p3 = pts[wf[0]], pts[wf[1]], pts[wf[2]]
                n = np.cross(p2 - p1, p3 - p1)
                nn = np.linalg.norm(n)
                if nn > 1e-14:
                    n = n / nn
                    c = n if (n @ p1) > 0 else -n
                    
                    lr = 0.02
                    for idx in wf:
                        p = pts[idx]
                        d = c - p
                        d -= (d @ p) * p
                        dn = np.linalg.norm(d)
                        if dn > 1e-14:
                            pts[idx] += lr * d / dn
                    pts = _normalize(pts)
            
            nr = _covering_radius(pts)
            if nr < best_r - 1e-9:
                best_r = nr
                best = pts.copy()
                stall = 0
            else:
                pts = best.copy()
                stall += 1
            use_facet = False
        else:
            # Density refinement for a few iterations
            # Quick density step
            n_theta, n_phi = 60, 120
            theta = np.linspace(0, np.pi, n_theta)
            phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
            TH, PH = np.meshgrid(theta, phi, indexing='ij')
            grid = np.column_stack([
                (np.sin(TH)*np.cos(PH)).ravel(),
                (np.sin(TH)*np.sin(PH)).ravel(),
                np.cos(TH).ravel()
            ])
            M = len(grid)
            
            sigma = 0.15
            dots = np.clip(grid @ pts.T, -1.0, 1.0)
            ang_dists = np.arccos(dots)
            d2_ang = ang_dists * ang_dists
            kernel = np.exp(-d2_ang / (2.0 * sigma * sigma))
            density = kernel.sum(axis=1)
            
            T = 30.0
            exp_neg_T_d = np.exp(-T * density)
            sum_exp = exp_neg_T_d.sum()
            if sum_exp < 1e-300:
                sum_exp = 1e-300
            w = exp_neg_T_d / sum_exp
            
            sqrt_term = np.sqrt(np.clip(1.0 - dots*dots, 1e-10, 1.0))
            grad_kernel = kernel * (ang_dists / (sigma * sigma)) * (grid[:, None] / sqrt_term)
            grad_pts = np.einsum('j,jki->i', w, grad_kernel)
            
            for k in range(N):
                grad_pts[k] -= (grad_pts[k] @ pts[k]) * pts[k]
            
            gn = np.linalg.norm(grad_pts)
            if gn > 1e-14:
                alpha = 0.002
                pts = _normalize(pts + alpha * grad_pts / gn)
            
            nr = _covering_radius(pts)
            if nr < best_r - 1e-9:
                best_r = nr
                best = pts.copy()
                stall = 0
            else:
                pts = best.copy()
                stall += 1
            use_facet = True
    
    return best, best_r

def construct():
    """Construct 130 points on unit sphere minimizing covering radius."""
    t_start = time.time()
    deadline = t_start + 138  # safety margin before 150s kill
    
    # Stage 1: Fibonacci init
    pts = _fibonacci_init(N)
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    
    # Stage 2: Lloyd's for global balance
    if time.time() < deadline - 100:
        lpts = _lloyd_refine(pts, n_iter=60)
        lr = _covering_radius(lpts)
        if lr < best_r:
            best_pts, best_r = lpts, lr
    
    # Stage 3: Repulsion spread
    if time.time() < deadline - 70:
        rpts = _repulsion_refine(best_pts, max_iter=100)
        rr = _covering_radius(rpts)
        if rr < best_r:
            best_pts, best_r = rpts, rr
    
    # Stage 4: Facet push (primary refinement)
    if time.time() < deadline - 45:
        fpts, fr = _facet_push(best_pts, max_time=min(30, deadline - time.time() - 15))
        if fr < best_r:
            best_pts, best_r = fpts, fr
    
    # Stage 5: Density refinement (new)
    if time.time() < deadline - 35:
        dpts = _density_refine(best_pts, max_iter=100)
        dr = _covering_radius(dpts)
        if dr < best_r:
            best_pts, best_r = dpts, dr
    
    # Stage 6: Combined refinement (alternating facet and density)
    if time.time() < deadline - 20:
        cpts, cr = _combined_refine(best_pts, max_time=min(20, deadline - time.time() - 10))
        if cr < best_r:
            best_pts, best_r = cpts, cr
    
    # Stage 7: Local search with perturbations
    if time.time() < deadline - 10:
        spts, sr = _local_search(best_pts, max_time=min(10, deadline - time.time() - 5))
        if sr < best_r:
            best_pts, best_r = spts, sr
    
    # Final facet polish
    if time.time() < deadline - 3:
        fpts2, fr2 = _facet_push(best_pts, max_time=min(5, deadline - time.time() - 2))
        if fr2 < best_r:
            best_pts, best_r = fpts2, fr2
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
