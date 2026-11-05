"""Morse cluster global minima: Structured templates + Basin Hopping + Local Polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 290.0  # Safety margin


def _remaining():
    return _DEADLINE - time.time()


def _morse_energy(x, rho):
    n = len(x)
    if n < 2:
        return 0.0
    d = x[:, None, :] - x[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    e = np.exp(rho * (1.0 - r))
    return float(np.sum(e * (e - 2.0)))


def _min_distance(x):
    n = len(x)
    if n < 2:
        return np.inf
    d = x[:, None, :] - x[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    return float(np.min(r))


def _enforce_min_dist(x, min_d=0.5):
    """Nudge apart any pair closer than min_d. Returns valid x."""
    x = np.array(x, dtype=float)
    n = len(x)
    for _ in range(100):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = x[i] - x[j]
                r = np.linalg.norm(d)
                if r < min_d:
                    if r < 1e-8:
                        d = np.array([0.001, 0.002, 0.003])
                    push = (min_d - r) / r * d * 0.5
                    x[i] += push
                    x[j] -= push
                    moved = True
        if not moved:
            break
    return x


def _local_polish(x, rho, maxiter=100):
    """Local minimization using L-BFGS-B. We assume repulsion prevents r<0.5 if started well."""
    n = len(x)
    x = np.asarray(x, dtype=float)

    def fun(v):
        return _morse_energy(v.reshape(n, 3), rho)

    def jac(v):
        g = morse_energy_gradient(v.reshape(n, 3), rho)
        return g.reshape(-1)

    try:
        res = minimize(fun, x.reshape(-1), jac=jac, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6})
        if res.success:
            return res.x.reshape(n, 3), float(res.fun)
    except Exception:
        pass
    return x, _morse_energy(x, rho)


def _generate_icosahedron(n, rng):
    """Generate Mackay icosahedron-like cluster."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, 1, -t], [0, -1, t], [0, -1, -t],
        [1, t, 0], [1, -t, 0], [-1, t, 0], [-1, -t, 0],
        [t, 0, 1], [t, 0, -1], [-t, 0, 1], [-t, 0, -1],
    ])
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    if n > 1:
        pts.extend((verts * 1.0).tolist())
        
    # Shell 2: 42 edge midpoints
    if n > 13:
        edge_pts = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = verts[i] - verts[j]
                if 1.0 < np.linalg.norm(d) < 1.6: # Edge length approx 1.05 in unit radius icosahedron? No, edge length is 1/sqrt(3+t^2/2)?
                    # Actually, for unit radius, edge length is ~1.05.
                    # Let's just take midpoints of short vectors
                    edge_pts.append((verts[i] + verts[j]) / 2.0)
        # Sort by distance to center? They are all same.
        # We need 42. Icosahedron has 30 edges. Wait.
        # Mackay icosahedron layers: 1, 12, 42, 100, 170...
        # Layer 2 has 42 points. 30 edge midpoints + 12 face centers?
        # Yes, 30 edges + 12 faces = 42.
        face_pts = []
        seen = set()
        for i in range(12):
            for j in range(i + 1, 12):
                f = np.cross(verts[i], verts[j])
                fn = f / (np.linalg.norm(f) + 1e-12)
                key = tuple(np.round(fn, 3))
                if key not in seen:
                    seen.add(key)
                    face_pts.append(fn)
        # Normalize face centers to radius 1?
        # In Mackay, they are at the same radius as vertices? No, they are slightly inside or out?
        # Standard Mackay: vertices on shell r, edge midpoints on shell r, face centers on shell r.
        # Actually, for a regular icosahedron, face centers are closer to center.
        # Let's project them to radius 1.0 to keep it simple, or scale.
        # Let's just add them.
        pts.extend(edge_pts)
        pts.extend(face_pts)
        
    # Shell 3: 100 points
    if n > 55:
        # 60 face centers of the 20 faces? No.
        # Layer 3 has 100 points.
        # This is getting complex. Let's use a simpler "spherical shell" approximation for higher layers
        # or just fill with random points on the surface if we exceed simple templates.
        # For now, let's add random points on a sphere of radius 1.8 if we need more.
        pass

    pts = np.array(pts[:n])
    if len(pts) < n:
        # Fill remaining with random points on a sphere
        need = n - len(pts)
        extra = rng.normal(size=(need, 3))
        extra /= (np.linalg.norm(extra, axis=1, keepdims=True) + 1e-12)
        extra *= 1.8
        pts = np.vstack([pts, extra])
        
    pts = _enforce_min_dist(pts)
    return pts


def _generate_fcc(n, rng):
    """Generate FCC cluster."""
    pts = []
    a = 1.0 # Lattice constant
    # Generate points in a large cube, keep those within radius
    R = 2.0 * n ** (1/3)
    for i in range(int(-R), int(R)+1):
        for j in range(int(-R), int(R)+1):
            for k in range(int(-R), int(R)+1):
                # FCC basis
                for b in [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]:
                    p = np.array([i,j,k]) + np.array(b)
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    pts = np.array(pts[:n])
    if len(pts) < n:
        need = n - len(pts)
        extra = rng.normal(size=(need, 3))
        extra /= (np.linalg.norm(extra, axis=1, keepdims=True) + 1e-12)
        extra *= 2.0
        pts = np.vstack([pts, extra])
    pts -= pts.mean(axis=0)
    pts = _enforce_min_dist(pts)
    return pts


def _generate_hcp(n, rng):
    """Generate HCP cluster."""
    pts = []
    a = 1.0
    c = np.sqrt(8.0/3.0) * a
    R = 2.0 * n ** (1/3)
    for layer in range(int(-R/c*2), int(R/c*2)+1):
        z = layer * c / 2.0
        offset = (layer % 2) * 0.5
        for i in range(int(-R), int(R)+1):
            for j in range(int(-R), int(R)+1):
                x = (i + offset) * a
                y = j * a * np.sqrt(3.0)/2.0
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= R:
                    pts.append(p)
    pts = np.array(pts[:n])
    if len(pts) < n:
        need = n - len(pts)
        extra = rng.normal(size=(need, 3))
        extra /= (np.linalg.norm(extra, axis=1, keepdims=True) + 1e-12)
        extra *= 2.0
        pts = np.vstack([pts, extra])
    pts -= pts.mean(axis=0)
    pts = _enforce_min_dist(pts)
    return pts


def _basin_hop(x0, rho, max_iter=50, step=0.5, rng=np.random.default_rng(42)):
    """Simple Basin Hopping."""
    n = len(x0)
    x_cur = x0.copy()
    e_cur = _morse_energy(x_cur, rho)
    
    # Local minimize
    x_cur, e_cur = _local_polish(x_cur, rho)
    
    for _ in range(max_iter):
        if _remaining() < 10:
            break
            
        # Perturb
        x_new = x_cur + rng.normal(scale=step, size=(n, 3))
        
        # Local minimize
        x_new, e_new = _local_polish(x_new, rho)
        
        # Accept if better
        if e_new < e_cur:
            x_cur = x_new
            e_cur = e_new
            
    return x_cur, e_cur


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size)

    templates = []
    # Generate diverse templates
    templates.append(_generate_icosahedron(size, rng))
    templates.append(_generate_fcc(size, rng))
    templates.append(_generate_hcp(size, rng))
    
    # Add random compact
    r = 1.5 * size ** (1/3)
    pts = [np.zeros(3)]
    while len(pts) < size:
        p = rng.uniform(-r, r, 3)
        if np.linalg.norm(p) <= r and all(np.linalg.norm(p - q) >= 0.6 for q in pts):
            pts.append(p)
    templates.append(np.array(pts))

    best_x = None
    best_e = np.inf

    # 1. Local polish each template
    for t in templates:
        if _remaining() < 15:
            break
        try:
            x, e = _local_polish(t, rho)
            if e < best_e:
                best_e = e
                best_x = x
        except Exception:
            continue

    # 2. Basin Hopping from the best
    if best_x is not None and _remaining() > 20:
        try:
            # Adjust step size based on N
            step = 0.5
            bh_x, bh_e = _basin_hop(best_x, rho, max_iter=30, step=step, rng=rng)
            if bh_e < best_e:
                best_e = bh_e
                best_x = bh_x
        except Exception:
            pass

    # 3. Final polish
    if best_x is not None and _remaining() > 5:
        try:
            fx, fe = _local_polish(best_x, rho, maxiter=200)
            if fe < best_e:
                best_e = fe
                best_x = fx
        except Exception:
            pass

    if best_x is None:
        best_x = templates[0]
        
    # Final validity check
    best_x = _enforce_min_dist(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
