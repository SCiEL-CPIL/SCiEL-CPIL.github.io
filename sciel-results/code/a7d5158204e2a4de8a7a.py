"""Sphere covering with 130 equal caps - facet-first optimization with gradient refinement."""
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
        n_pert = np.random.randint(1, 6)
        idxs = np.random.choice(N, n_pert, replace=False)
        
        new_pts = pts.copy()
        for idx in idxs:
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

def _gradient_refine(points, max_iter=200, alpha_init=0.006):
    """Gradient descent on a smooth surrogate of covering radius using soft-min/soft-max."""
    pts = _normalize(points.copy())
    
    # Build dense grid for smooth objective
    n_theta, n_phi = 180, 360
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    grid = np.column_stack([
        (np.sin(TH)*np.cos(PH)).ravel(),
        (np.sin(TH)*np.sin(PH)).ravel(),
        np.cos(TH).ravel()
    ])
    
    alpha = alpha_init
    T = 40.0
    best = pts.copy()
    best_r = _covering_radius(pts)
    
    for it in range(max_iter):
        dots = np.clip(grid @ pts.T, -0.9999, 0.9999)
        ang = np.arccos(dots)  # M x N
        
        # Soft-min per grid point (find nearest point)
        exp_neg = np.exp(-T * ang)
        sum_exp = exp_neg.sum(axis=1, keepdims=True)
        sum_exp[sum_exp < 1e-300] = 1e-300
        w = exp_neg / sum_exp  # M x N
        soft_min = -(1.0/T) * np.log(sum_exp).ravel()  # M
        
        # Soft-max over grid (find worst grid point)
        mx = soft_min.max()
        exp_s = np.exp(T * (soft_min - mx))
        s_exp = exp_s.sum()
        if s_exp < 1e-300:
            s_exp = 1e-300
        w_max = exp_s / s_exp  # M
        val = mx + (1.0/T) * np.log(s_exp)
        
        # Gradient: d(val)/d(pts[k]) = -sum_j w_max[j] * w[j,k] * grid[j] / sqrt(1-dots[j,k]^2)
        sqrt_t = np.sqrt(np.clip(1.0 - dots*dots, 1e-12, 1.0))  # M x N
        G = (w_max[:, None] * w / sqrt_t)  # M x N
        grad = -(G.T @ grid)  # N x 3
        
        # Project to tangent
        for k in range(N):
            grad[k] -= (grad[k] @ pts[k]) * pts[k]
        
        gn = np.linalg.norm(grad)
        if gn < 1e-14:
            break
        
        pts = _normalize(pts - alpha * grad / max(gn, 1e-14))
        nr = _covering_radius(pts)
        if nr < best_r:
            best_r = nr
            best = pts.copy()
        alpha *= 0.995
        T = max(15.0, T * 0.998)
    
    return best

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
    
    # Stage 5: Local search with perturbations
    if time.time() < deadline - 30:
        spts, sr = _local_search(best_pts, max_time=min(20, deadline - time.time() - 10))
        if sr < best_r:
            best_pts, best_r = spts, sr
    
    # Stage 6: Facet push again
    if time.time() < deadline - 15:
        fpts2, fr2 = _facet_push(best_pts, max_time=min(15, deadline - time.time() - 5))
        if fr2 < best_r:
            best_pts, best_r = fpts2, fr2
    
    # Stage 7: Gradient refinement if time allows
    remaining = deadline - time.time()
    if remaining > 25:
        gpts = _gradient_refine(best_pts, max_iter=min(150, int(remaining * 5)))
        gr = _covering_radius(gpts)
        if gr < best_r:
            best_pts, best_r = gpts, gr
        
        # Final facet polish
        if time.time() < deadline - 8:
            fpts3, fr3 = _facet_push(best_pts, max_time=min(8, deadline - time.time() - 3))
            if fr3 < best_r:
                best_pts, best_r = fpts3, fr3
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
