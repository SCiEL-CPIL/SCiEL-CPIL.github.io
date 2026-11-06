"""Covering the sphere with 130 equal caps: convex-hull circumradius minimization."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_TIME_LIMIT = 115.0  # safety margin under the 150s wall-clock kill
_T0 = time.time()


def _time_left():
    return _TIME_LIMIT - (time.time() - _T0)


def _normalize(P):
    n = np.linalg.norm(P, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return P / n


def _hull_circumradii(P):
    """Return (radii, centers, simplices) for each facet of the convex hull.
    radii[i] is the circumradius (in radians) of facet i.
    centers[i] is the cap-center direction (unit vector) for facet i.
    """
    try:
        hull = ConvexHull(P)
    except Exception:
        return None, None, None
    simplices = hull.simplices
    m = len(simplices)
    radii = np.zeros(m)
    centers = np.zeros((m, 3))
    centroid = P.mean(axis=0)
    for k, tri in enumerate(simplices):
        a = P[tri[0]]
        b = P[tri[1]]
        c = P[tri[2]]
        # Circumcenter direction: unit vector u with u.a = u.b = u.c
        # u is parallel to (a-b) x (a-c)
        d = np.cross(a - b, a - c)
        nd = np.linalg.norm(d)
        if nd < 1e-12:
            u = a.copy()
        else:
            u = d / nd
        # Determine outward direction
        n_out = np.cross(b - a, c - a)
        if np.dot(n_out, centroid) > 0:
            n_out = -n_out
        if np.dot(u, n_out) < 0:
            u = -u
        u = u / np.linalg.norm(u)
        centers[k] = u
        cosang = np.clip(np.dot(u, a), -1.0, 1.0)
        radii[k] = np.arccos(cosang)
    return radii, centers, simplices


def _covering_radius(P):
    radii, centers, simplices = _hull_circumradii(P)
    if radii is None:
        return np.inf
    return float(radii.max())


def _repulsion_minimize(P0, iterations=200, cooling_rate=0.995):
    """Minimize pairwise repulsion to spread points evenly."""
    P = _normalize(P0.copy())
    temp = 1.0
    for it in range(iterations):
        if _time_left() < 5.0:
            break
        # Compute repulsive forces
        forces = np.zeros_like(P)
        for i in range(N):
            for j in range(i + 1, N):
                d = P[i] - P[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                # Repulsion: force proportional to 1/dist^2 in direction d
                f = d / (dist * dist * dist)  # d / dist^3 = unit_dir / dist^2
                forces[i] += f
                forces[j] -= f
        # Normalize forces to tangent space
        for i in range(N):
            forces[i] -= np.dot(forces[i], P[i]) * P[i]
        # Move points
        step = 0.01 * temp
        P = _normalize(P + step * forces)
        temp *= cooling_rate
    return P


def _gradient_step(P, beta=50.0, lr=0.005, max_iter=100):
    """Gradient descent on soft-max of facet circumradii."""
    P = _normalize(P.copy())
    bestP = P.copy()
    bestVal = _covering_radius(P)
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        
        radii, centers, simplices = _hull_circumradii(P)
        if radii is None:
            break
        
        # Soft-max weights
        rmax = radii.max()
        w = np.exp(beta * (radii - rmax))
        wsum = w.sum()
        if wsum < 1e-30:
            break
        w = w / wsum
        
        # Gradient of soft-max w.r.t. P
        grad = np.zeros_like(P)
        for k, tri in enumerate(simplices):
            wk = w[k]
            if wk < 1e-12:
                continue
            u = centers[k]
            for idx in tri:
                p = P[idx]
                cosang = np.clip(np.dot(u, p), -1.0, 1.0)
                sinang = np.sqrt(max(1.0 - cosang * cosang, 1e-12))
                grad[idx] += wk * (-u / sinang)
        
        # Project to tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], P[i]) * P[i]
        
        # Check if gradient is small
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-10:
            break
        
        # Gradient step
        Pnew = _normalize(P - lr * grad / (gnorm + 1e-15))
        
        # Evaluate
        val = _covering_radius(Pnew)
        if val < bestVal:
            bestVal = val
            bestP = Pnew.copy()
            P = Pnew
        else:
            # Reduce learning rate
            lr *= 0.9
            if lr < 1e-8:
                break
    
    return bestP, bestVal


def _local_refine(P, n_iter=50):
    """Refine by moving the worst facet's vertices to reduce its circumradius."""
    P = _normalize(P.copy())
    for _ in range(n_iter):
        if _time_left() < 2.0:
            break
        
        radii, centers, simplices = _hull_circumradii(P)
        if radii is None:
            break
        
        # Find worst facet
        worst_k = np.argmax(radii)
        worst_r = radii[worst_k]
        
        # If all radii are close, we're near optimal
        if np.max(radii) - np.min(radii) < 1e-6:
            break
        
        tri = simplices[worst_k]
        u = centers[worst_k]
        
        # Move vertices of worst facet in direction that reduces circumradius
        # The circumradius is arccos(u . a), so to reduce it, increase u . a
        # Move a in direction of u (projected to tangent space)
        for idx in tri:
            p = P[idx]
            # Direction to move: component of u in tangent space
            dir_vec = u - np.dot(u, p) * p
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-10:
                continue
            dir_vec = dir_vec / dir_norm
            # Small step
            P[idx] = _normalize((p + 0.01 * dir_vec).reshape(1, 3))[0]
    
    return P


def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(seed):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(N, 3))
    return _normalize(P)


def construct():
    global _T0
    _T0 = time.time()
    
    # Generate multiple starting configurations
    starts = []
    starts.append(_fibonacci_start())
    for s in range(8):
        starts.append(_random_start(42 + s * 137))
    
    bestP = None
    bestVal = np.inf
    
    for P0 in starts:
        if _time_left() < 10.0:
            break
        
        # Phase 1: Repulsion to spread points
        P1 = _repulsion_minimize(P0, iterations=150)
        
        # Phase 2: Gradient descent on covering radius
        P2, val2 = _gradient_step(P1, beta=50.0, lr=0.005, max_iter=80)
        
        # Phase 3: Local refinement
        P3 = _local_refine(P2, n_iter=30)
        
        # Final evaluation
        exact = _covering_radius(P3)
        if exact < bestVal:
            bestVal = exact
            bestP = P3.copy()
        
        # Also check P2 in case refinement made it worse
        exact2 = _covering_radius(P2)
        if exact2 < bestVal:
            bestVal = exact2
            bestP = P2.copy()
    
    if bestP is None:
        bestP = _fibonacci_start()
    
    return _normalize(bestP)
