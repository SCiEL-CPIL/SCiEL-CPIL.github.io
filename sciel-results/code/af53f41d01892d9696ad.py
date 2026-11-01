"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 200
_DEADLINE = 145.0  # safety margin before 150s kill


def _score(c):
    """Compute the packing score (radius) for centre array c of shape (N,2)."""
    # Ensure finite
    if not np.all(np.isfinite(c)):
        return 0.0
    # Boundary constraint
    boundary = 0.5 - np.max(np.abs(c), axis=1)
    r_bound = np.min(boundary)
    # Pairwise distances
    tree = cKDTree(c)
    pairs, dists = tree.query_pairs(r=1.0, output_type='ndarray')
    if len(pairs) == 0:
        r_pair = 0.5
    else:
        diff = c[pairs[:, 0]] - c[pairs[:, 1]]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        r_pair = np.min(d) / 2.0
    return min(r_bound, r_pair)


def _init_grid(n):
    """Square grid initialisation."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n].copy()


def _init_hex(n, side=1.0):
    """Hexagonal packing initialisation inside unit square."""
    # Hex lattice: rows offset by half spacing
    # spacing s: vertical distance between rows = s * sqrt(3)/2
    # We want to fit as many as possible in [-0.5, 0.5]^2
    # Try a range of spacings and pick the one that fits ~n circles
    best_c = None
    best_score = -1
    for s in np.linspace(0.02, 0.15, 300):
        h = s * np.sqrt(3) / 2
        # Number of rows and cols
        n_rows = int(np.floor((1.0 - 2 * s) / h)) + 1 if h > 0 else 1
        n_cols = int(np.floor((1.0 - 2 * s) / s)) + 1 if s > 0 else 1
        if n_rows < 1 or n_cols < 1:
            continue
        pts = []
        for i in range(n_rows):
            x0 = -0.5 + s + (0.5 * s if i % 2 == 1 else 0.0)
            y = -0.5 + s + i * h
            for j in range(n_cols):
                x = x0 + j * s
                if -0.5 + s <= x <= 0.5 - s:
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) < n:
            continue
        # Take the n points closest to center
        dists_c = np.sum(pts ** 2, axis=1)
        idx = np.argsort(dists_c)[:n]
        c = pts[idx]
        sc = _score(c)
        if sc > best_score:
            best_score = sc
            best_c = c
    if best_c is None:
        return _init_grid(n)
    return best_c


def _force_relax(c, max_iter=5000, dt=0.001, temp=1.0):
    """Simple force-directed relaxation to improve packing."""
    c = c.copy()
    n = len(c)
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        forces = np.zeros_like(c)
        # Repulsion between close pairs
        tree = cKDTree(c)
        pairs, dists = tree.query_pairs(r=0.15, output_type='ndarray')
        if len(pairs) > 0:
            diff = c[pairs[:, 0]] - c[pairs[:, 1]]
            d = np.sqrt(np.sum(diff * diff, axis=1))
            # Avoid division by zero
            d = np.maximum(d, 1e-12)
            # Force magnitude: stronger when closer
            f_mag = 1.0 / (d * d)
            f_vec = diff / d[:, None] * f_mag[:, None]
            # Accumulate forces
            for k in range(len(pairs)):
                i, j = pairs[k]
                forces[i] += f_vec[k]
                forces[j] -= f_vec[k]
        # Boundary repulsion
        for i in range(n):
            for dim in range(2):
                if c[i, dim] > 0.5 - 0.05:
                    forces[i, dim] -= (c[i, dim] - (0.5 - 0.05)) * 10
                if c[i, dim] < -0.5 + 0.05:
                    forces[i, dim] += (-0.5 + 0.05 - c[i, dim]) * 10
        # Update
        c += dt * forces
        # Clamp to square
        c = np.clip(c, -0.5, 0.5)
        # Anneal
        temp *= 0.999
        dt *= 0.999
    return c


def _local_optimize(c, max_iter=2000):
    """Local optimization using gradient of min-distance (subgradient method)."""
    c = c.copy()
    n = len(c)
    step = 0.005
    best_c = c.copy()
    best_sc = _score(c)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        # Compute current score and identify constraining pairs
        tree = cKDTree(c)
        boundary = 0.5 - np.max(np.abs(c), axis=1)
        r_bound = np.min(boundary)
        
        # Find closest pair
        pairs, dists = tree.query_pairs(r=0.12, output_type='ndarray')
        if len(pairs) == 0:
            r_pair = 0.5
            pair_idx = None
        else:
            diff = c[pairs[:, 0]] - c[pairs[:, 1]]
            d = np.sqrt(np.sum(diff * diff, axis=1))
            min_idx = np.argmin(d)
            r_pair = d[min_idx] / 2.0
            pair_idx = pairs[min_idx]
        
        r = min(r_bound, r_pair)
        if r > best_sc:
            best_sc = r
            best_c = c.copy()
        
        # Gradient: push apart the constraining pair, push inward for boundary
        grad = np.zeros_like(c)
        if r_bound <= r_pair:
            # Boundary is constraining: find which point
            min_b_idx = np.argmin(boundary)
            # Push that point toward center
            for dim in range(2):
                if c[min_b_idx, dim] > 0:
                    grad[min_b_idx, dim] -= 1.0
                else:
                    grad[min_b_idx, dim] += 1.0
        else:
            i, j = pair_idx
            diff = c[i] - c[j]
            d = np.linalg.norm(diff)
            if d > 1e-12:
                unit = diff / d
                grad[i] += unit
                grad[j] -= unit
        
        # Also add small repulsion from all nearby pairs for stability
        if len(pairs) > 0:
            for k in range(min(len(pairs), 20)):
                i, j = pairs[k]
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d > 1e-12:
                    unit = diff / d
                    w = 0.1 / max(d, 0.01)
                    grad[i] += w * unit
                    grad[j] -= w * unit
        
        # Update
        c += step * grad
        c = np.clip(c, -0.5, 0.5)
        step *= 0.998
    return best_c


def _bundle_method(c, max_iter=100):
    """
    Bundle method for maximizing r = min(constraints).
    Uses subgradients of the min function.
    """
    c = c.copy()
    n = len(c)
    best_c = c.copy()
    best_r = _score(c)
    
    # Bundle: list of (x_k, g_k, f_k) where f_k = r(x_k), g_k = subgradient
    bundle = [(c.copy(), None, best_r)]
    
    step_size = 0.002
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        # Evaluate at current point
        r_curr = _score(c)
        if r_curr > best_r:
            best_r = r_curr
            best_c = c.copy()
        
        # Compute subgradient of r(x) = min_k g_k(x)
        tree = cKDTree(c)
        boundary = 0.5 - np.max(np.abs(c), axis=1)
        r_bound = np.min(boundary)
        
        pairs, dists = tree.query_pairs(r=0.12, output_type='ndarray')
        if len(pairs) == 0:
            r_pair = 0.5
        else:
            diff = c[pairs[:, 0]] - c[pairs[:, 1]]
            d = np.sqrt(np.sum(diff * diff, axis=1))
            r_pair = np.min(d) / 2.0
        
        r_curr = min(r_bound, r_pair)
        
        # Build subgradient
        g = np.zeros_like(c)
        active = []
        
        if r_bound <= r_pair:
            # Boundary active
            min_b = np.argmin(boundary)
            for dim in range(2):
                if c[min_b, dim] > 0:
                    g[min_b, dim] -= 1.0
                else:
                    g[min_b, dim] += 1.0
            active.append(('b', min_b))
        else:
            # Find closest pair
            min_d_idx = np.argmin(d)
            i, j = pairs[min_d_idx]
            diff = c[i] - c[j]
            dn = np.linalg.norm(diff)
            if dn > 1e-12:
                unit = diff / dn
                g[i] += unit
                g[j] -= unit
                active.append(('p', (i, j)))
        
        # Normalize gradient
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            g /= gn
        
        # Add to bundle
        bundle.append((c.copy(), g.copy(), r_curr))
        if len(bundle) > 50:
            bundle.pop(0)
        
        # Compute search direction from bundle (simple averaging)
        # Use the most recent subgradients
        recent_g = [b[1] for b in bundle[-10:] if b[1] is not None]
        if recent_g:
            avg_g = np.mean(recent_g, axis=0)
            an = np.linalg.norm(avg_g)
            if an > 1e-12:
                avg_g /= an
                c += step_size * avg_g
            else:
                c += step_size * g
        else:
            c += step_size * g
        
        c = np.clip(c, -0.5, 0.5)
        step_size *= 0.995
    
    return best_c


def _perturb(c, sigma=0.01):
    """Add random perturbation."""
    c = c.copy()
    c += np.random.normal(0, sigma, c.shape)
    c = np.clip(c, -0.5, 0.5)
    return c


def construct():
    start_time = time.time()
    
    # Multiple initializations
    inits = [
        _init_grid(N),
        _init_hex(N),
    ]
    
    best_c = inits[0].copy()
    best_sc = _score(best_c)
    
    # Evaluate all inits
    for c_init in inits:
        sc = _score(c_init)
        if sc > best_sc:
            best_sc = sc
            best_c = c_init.copy()
    
    # Iterate: force relaxation + local optimization + bundle
    for round_num in range(5):
        if time.time() > _DEADLINE - 20:
            break
        
        # Force relaxation
        c_relaxed = _force_relax(best_c, max_iter=3000)
        sc_relaxed = _score(c_relaxed)
        if sc_relaxed > best_sc:
            best_sc = sc_relaxed
            best_c = c_relaxed.copy()
        
        # Local optimization
        c_local = _local_optimize(best_c, max_iter=1500)
        sc_local = _score(c_local)
        if sc_local > best_sc:
            best_sc = sc_local
            best_c = c_local.copy()
        
        # Bundle method
        c_bundle = _bundle_method(best_c, max_iter=80)
        sc_bundle = _score(c_bundle)
        if sc_bundle > best_sc:
            best_sc = sc_bundle
            best_c = c_bundle.copy()
        
        # Try perturbed restart
        if time.time() < _DEADLINE - 30:
            for trial in range(3):
                if time.time() > _DEADLINE - 15:
                    break
                c_pert = _perturb(best_c, sigma=0.005 * (trial + 1))
                c_opt = _local_optimize(c_pert, max_iter=800)
                sc_opt = _score(c_opt)
                if sc_opt > best_sc:
                    best_sc = sc_opt
                    best_c = c_opt.copy()
    
    # Final polish
    if time.time() < _DEADLINE - 5:
        c_final = _local_optimize(best_c, max_iter=2000)
        sc_final = _score(c_final)
        if sc_final > best_sc:
            best_c = c_final
    
    # Ensure validity: clip to square, check finite
    best_c = np.clip(best_c, -0.5, 0.5)
    if not np.all(np.isfinite(best_c)):
        return _init_grid(N)
    
    return best_c
