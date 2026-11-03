"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using nearest-neighbor circumradius approximation."""
    n = len(pts)
    # Compute pairwise angular distances
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, 0.0)
    
    # For each point, find its 3 nearest neighbors and compute circumradius of the triangle
    # The covering radius is approximated by the max circumradius
    cr = 0.0
    for i in range(n):
        # Get 3 nearest neighbors
        order = np.argsort(dists[i])
        order = order[order != i][:3]
        if len(order) < 3:
            continue
        a, b, c = pts[i], pts[order[0]], pts[order[1]]
        # Compute circumradius of triangle abc on unit sphere
        # Using the formula: circumradius = (a*b*c) / (4 * area) where a,b,c are side lengths
        # But on sphere, we need spherical trig
        # Simpler: find circumcenter numerically
        # The circumcenter is the point equidistant from a, b, c
        # Solve: |x - a|^2 = |x - b|^2 = |x - c|^2, |x| = 1
        # This gives x . (b - a) = |b|^2 - |a|^2 / 2 = 0 (since on unit sphere)
        # So x is orthogonal to (b-a) and (c-a), i.e., x is parallel to (b-a) x (c-a)
        v = np.cross(b - a, c - a)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            continue
        x = v / norm_v
        # Check which direction: x or -x
        d1 = np.arccos(np.clip(np.dot(x, a), -1, 1))
        d2 = np.arccos(np.clip(np.dot(-x, a), -1, 1))
        if d2 < d1:
            x = -x
        cr_i = np.arccos(np.clip(np.dot(x, a), -1, 1))
        cr = max(cr, cr_i)
    return cr

def _energy(pts, k=10.0):
    """Smooth approximation of covering radius using log-sum-exp."""
    n = len(pts)
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)
    # For each point, compute log-sum-exp of exp(k * dists)
    # This approximates k * max(dist) + log(n)
    with np.errstate(over='ignore'):
        exp_k_d = np.exp(k * dists)
    # Subtract max for numerical stability
    max_d = np.max(dists, axis=1, keepdims=True)
    exp_k_d_shifted = np.exp(k * (dists - max_d))
    lse = max_d[:, 0] + np.log(np.sum(exp_k_d_shifted, axis=1)) / k
    return np.mean(lse)

def _gradient(pts, k=10.0):
    """Gradient of the smooth energy."""
    n = len(pts)
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # For each point i, the gradient of log-sum-exp of exp(k * d_ij) w.r.t. pts[i]
    # d/dpts[i] d_ij = -pts[j] / sin(d_ij) for j != i
    grad = np.zeros_like(pts)
    for i in range(n):
        weights = np.zeros(n)
        for j in range(n):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-10:
                continue
            # Weight for log-sum-exp
            with np.errstate(over='ignore'):
                w = np.exp(k * (d - dists[i, :].max()))
            weights[j] = w
        total_w = np.sum(weights)
        if total_w < 1e-15:
            continue
        weights /= total_w
        # Gradient contribution from point i's perspective
        for j in range(n):
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-10:
                continue
            # d/dpts[i] arccos(pts[i] . pts[j]) = -pts[j] / sin(arccos(pts[i].pts[j]))
            sin_d = np.sin(d)
            if sin_d < 1e-10:
                continue
            grad[i] += weights[j] * (-pts[j] / sin_d) * k
    return grad

def _gradient_descent(pts, k_schedule, max_iter=2000, lr=0.01, deadline=None):
    """Perform gradient descent with adaptive k."""
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        k = k_schedule[it] if it < len(k_schedule) else k_schedule[-1]
        
        e = _energy(pts, k)
        g = _gradient(pts, k)
        
        # Project gradient to tangent space
        for i in range(n):
            g[i] -= np.dot(g[i], pts[i]) * pts[i]
        
        # Adaptive step size
        step = lr / (1.0 + it * 0.001)
        
        new_pts = pts - step * g
        new_pts = _normalize(new_pts)
        
        # Check if energy decreased
        new_e = _energy(new_pts, k)
        if new_e < e:
            pts = new_pts
        else:
            # Try smaller step
            step *= 0.5
            new_pts = pts - step * g
            new_pts = _normalize(new_pts)
            new_e = _energy(new_pts, k)
            if new_e < e:
                pts = new_pts
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # K schedule: start small, increase
    k_schedule = np.linspace(2.0, 20.0, 500)
    
    # Phase 1: Coarse optimization with smaller k
    if time.time() < deadline - 30.0:
        pts = _gradient_descent(pts, k_schedule[:200], max_iter=200, lr=0.05, deadline=deadline)
    
    # Phase 2: Finer optimization with larger k
    if time.time() < deadline - 15.0:
        k_fine = np.linspace(10.0, 30.0, 300)
        pts = _gradient_descent(pts, k_fine, max_iter=300, lr=0.01, deadline=deadline)
    
    # Phase 3: Final polish
    if time.time() < deadline - 5.0:
        k_polish = np.full(100, 30.0)
        pts = _gradient_descent(pts, k_polish, max_iter=100, lr=0.001, deadline=deadline)
    
    return _normalize(pts)
# EVOLVE-BLOCK-END
