"""Rewritten solver: Voronoi-Lloyd equalization + multi-scale soft-min polish + repulsion + local search.
Entry point: construct() -> (N,3) array of sphere centres in [-0.5,0.5]^3."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

DEADLINE_MARGIN = 15.0  # safety margin before 150s kill


def _compute_radius(c):
    """Exact common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    return min(min_pair / 2.0, min_wall)


def _pair_dist_matrix(c):
    """(N,N) pairwise distance matrix, diagonal = inf."""
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def _lloyd_step(c, deadline, steps=200, damping=0.5):
    """Voronoi-based Lloyd's algorithm: move each center toward the centroid
    of its Voronoi cell, clipped to stay inside the cube. This naturally
    equalizes clearance around each sphere."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    
    for it in range(steps):
        if time.time() > deadline:
            break
        
        try:
            # Build Delaunay triangulation in 3D
            tri = Delaunay(c)
        except Exception:
            # Fallback: simple repulsion step
            d = _pair_dist_matrix(c)
            inv = 1.0 / np.maximum(d, 1e-6) ** 2
            diff = c[:, None, :] - c[None, :, :]
            F = np.sum(diff * inv[:, :, None], axis=1)
            c = np.clip(c + 0.01 * F, -0.49, 0.49)
            continue
        
        # For each point, find its Voronoi cell vertices
        # In 3D, we use the dual: each Delaunay tetrahedron containing point i
        # contributes a vertex to i's Voronoi cell
        
        # Get all tetrahedra
        simplices = tri.simplices  # (M, 4)
        
        # For each point, collect the circumcenters of tetrahedra that contain it
        cell_centroids = np.zeros((N, 3))
        cell_counts = np.zeros(N)
        
        # Compute circumcenters of all tetrahedra
        # Circumcenter of a tetrahedron with vertices v0,v1,v2,v3
        def _circumcenter(v0, v1, v2, v3):
            # Solve for point equidistant from all 4 vertices
            # |x-v0|^2 = |x-v1|^2 = |x-v2|^2 = |x-v3|^2
            # 2*x.(v1-v0) = |v1|^2 - |v0|^2
            # etc.
            A = np.array([
                v1 - v0,
                v2 - v0,
                v3 - v0
            ])
            b = np.array([
                np.dot(v1, v1) - np.dot(v0, v0),
                np.dot(v2, v2) - np.dot(v0, v0),
                np.dot(v3, v3) - np.dot(v0, v0)
            ])
            try:
                x = np.linalg.solve(2.0 * A, b)
                return x
            except np.linalg.LinAlgError:
                return np.mean([v0, v1, v2, v3])
        
        # This is expensive; do it vectorized-ish
        # For each simplex, compute circumcenter and add to all 4 vertices' cells
        for idx in range(len(simplices)):
            si = simplices[idx]
            v0, v1, v2, v3 = c[si[0]], c[si[1]], c[si[2]], c[si[3]]
            cc = _circumcenter(v0, v1, v2, v3)
            # Check if circumcenter is inside the tetrahedron (acute tet)
            # For simplicity, always add it
            for vi in si:
                cell_centroids[vi] += cc
                cell_counts[vi] += 1
        
        # Average
        valid = cell_counts > 0
        if np.any(valid):
            targets = cell_centroids[valid] / cell_counts[valid, None]
            # Damped move toward Voronoi centroid
            c[valid] = c[valid] + damping * (targets - c[valid])
        
        c = np.clip(c, -0.49, 0.49)
    
    return c


def _lloyd_optimize(c, deadline):
    """Iterate Lloyd's algorithm with decreasing damping."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    
    for damp in (0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05):
        if time.time() > deadline:
            break
        steps = max(50, int(300 * damp))
        c = _lloyd_step(c, deadline, steps=steps, damping=damp)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best = c.copy()
    
    return best, best_r


def _softmin_radius(c, beta):
    """Smooth approximation of the true radius via log-sum-exp over all
    constraints (pair/2 and wall). beta is the inverse temperature."""
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    m = np.min(vals)
    s = np.sum(np.exp(-(vals - m) * beta))
    return m - np.log(s) / beta


def _epigraph_obj(x, beta):
    """Epigraph formulation: minimize t such that all constraints >= t.
    Uses smooth max-min via log-sum-exp. x = [c_flat, t] where c is (N,3) and t is scalar.
    Objective: t + penalty for violating smooth constraints.
    """
    c = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    d = _pair_dist_matrix(c)
    pair = d / 2.0
    wall = 0.5 - np.abs(c)
    vals = np.concatenate([pair.ravel(), wall.ravel()])
    vals = vals[np.isfinite(vals)]
    
    # Smooth max-min: approximate min(vals) - t
    m = np.min(vals)
    s = np.sum(np.exp(-(vals - m) * beta))
    smooth_min = m - np.log(s) / beta
    
    # Penalty for smooth_min < t (i.e., t > smooth_min)
    violation = np.maximum(0.0, t - smooth_min)
    
    # Boundary penalty
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-6) ** 2)
    
    # We want to maximize t, so minimize -t + violation + penalty
    return -t + 10.0 * violation + 1e3 * pen


def _smooth_polish(c, deadline):
    """Polish with epigraph formulation via L-BFGS-B, annealing beta."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    best = c.copy()
    best_r = _compute_radius(c)
    
    # Initial t estimate
    t0 = _compute_radius(c)
    
    for beta in (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0):
        if time.time() > deadline:
            break
        try:
            x0 = np.concatenate([c.flatten(), [t0]])
            bounds = [(-0.49, 0.49)] * (N * 3) + [(0.0, 0.5)]
            
            res = minimize(
                _epigraph_obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(beta,),
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11,
                         'maxls': 100},
            )
            c_new = np.clip(res.x[:N*3].reshape(N, 3), -0.49, 0.49)
            t_new = res.x[N*3]
            r = _compute_radius(c_new)
            if r > best_r:
                best_r = r
                best = c_new.copy()
                t0 = max(t_new, r)
            c = c_new
        except Exception:
            pass
    return best, best_r


def _repulsion_anneal(c, deadline, steps=3000):
    """Vectorized repulsion + wall-attraction gradient descent."""
    c = c.copy()
    c = np.clip(c, -0.49, 0.49)
    
    for it in range(steps):
        if time.time() > deadline:
            break
        d = _pair_dist_matrix(c)
        inv = 1.0 / np.maximum(d, 1e-6) ** 2
        diff = c[:, None, :] - c[None, :, :]
        F = np.sum(diff * inv[:, :, None], axis=1)
        
        wall_force = np.zeros_like(c)
        for k in range(3):
            sgn = np.sign(c[:, k])
            sgn[sgn == 0] = 1.0
            wall_force[:, k] = sgn * (0.5 - np.abs(c[:, k]))
        
        frac = it / max(steps - 1, 1)
        rep_scale = 1e-3 * (1.0 - 0.7 * frac)
        wall_scale = 1e-3 * (0.3 + 1.0 * frac)
        lr = 1e-2 * (1.0 - 0.5 * frac)
        c = c + lr * (rep_scale * F + wall_scale * wall_force)
        c = np.clip(c, -0.49, 0.49)
    
    return c


def _metropolis_mc(c, deadline, iters=5000):
    """Metropolis Monte Carlo: random displacements with acceptance based on radius.
    Allows occasional worsening moves to escape local minima."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(123)
    
    # Adaptive temperature that cools over time
    T0 = 0.002
    T_min = 0.0001
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        frac = it / max(iters - 1, 1)
        T = T0 * (T_min / T0) ** frac
        
        # Random displacement
        trial = c.copy()
        trial += rng.normal(scale=T, size=(N, 3))
        trial = np.clip(trial, -0.499, 0.499)
        
        r_trial = _compute_radius(trial)
        r_cur = _compute_radius(c)
        
        # Accept if better, or with probability exp(-(r_cur - r_trial)/T) if worse
        if r_trial >= r_cur:
            c = trial
        else:
            delta = r_cur - r_trial
            if delta < 1e-12:
                c = trial
            else:
                prob = np.exp(-delta / max(T, 1e-8))
                if rng.random() < prob:
                    c = trial
        
        if _compute_radius(c) > best_r:
            best_r = _compute_radius(c)
            best_c = c.copy()
    
    return best_c, best_r


def _delaunay_equalize(c, deadline, iters=500):
    """Delaunay-based edge equalization: for each Delaunay edge, nudge endpoints
    toward equalizing edge length with neighbors. This drives the packing toward
    mechanical equilibrium where all critical constraints are balanced."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        try:
            tri = Delaunay(c)
            edges = set()
            for simplex in tri.simplices:
                for i in range(4):
                    for j in range(i+1, 4):
                        a, b = int(simplex[i]), int(simplex[j])
                        edges.add((min(a,b), max(a,b)))
            
            # For each edge, compute target length (average of incident edge lengths)
            edge_list = list(edges)
            n_edges = len(edge_list)
            if n_edges == 0:
                break
            
            # Compute current edge lengths
            edge_lens = np.zeros(n_edges)
            for ei, (i, j) in enumerate(edge_list):
                edge_lens[ei] = np.linalg.norm(c[i] - c[j])
            
            # For each vertex, compute average incident edge length
            vert_avg = np.zeros(N)
            vert_count = np.zeros(N)
            for ei, (i, j) in enumerate(edge_list):
                vert_avg[i] += edge_lens[ei]
                vert_count[i] += 1
                vert_avg[j] += edge_lens[ei]
                vert_count[j] += 1
            valid = vert_count > 0
            vert_avg[valid] /= vert_count[valid]
            
            # Move each vertex slightly toward its average neighbor distance
            # This is a simplified version of Lloyd's on the Delaunay graph
            forces = np.zeros_like(c)
            for ei, (i, j) in enumerate(edge_list):
                d = np.linalg.norm(c[i] - c[j])
                if d < 1e-10:
                    continue
                # Target: average of the two vertices' incident averages
                target = 0.5 * (vert_avg[i] + vert_avg[j])
                # Force proportional to (d - target)
                f = (d - target) / d
                direction = (c[i] - c[j]) / d
                forces[i] += f * direction
                forces[j] -= f * direction
            
            lr = 0.05
            c = c + lr * forces
            c = np.clip(c, -0.499, 0.499)
            
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best = c.copy()
                
        except Exception:
            break
    
    return best, best_r


def _local_search(c, deadline, iters=4000):
    """Local search: perturb binding constraints with adaptive steps, accept if radius improves."""
    c = c.copy()
    c = np.clip(c, -0.499, 0.499)
    best_r = _compute_radius(c)
    best_c = c.copy()
    rng = np.random.default_rng(42)
    
    # Adaptive step size
    step_size = 0.005
    no_improve_count = 0
    
    for it in range(iters):
        if time.time() > deadline:
            break
        
        d = _pair_dist_matrix(c)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]
        
        wall = 0.5 - np.abs(c)
        min_wall_idx = np.unravel_index(np.argmin(wall), wall.shape)
        min_wall_val = wall[min_wall_idx]
        
        improved = False
        
        if min_pair_dist / 2.0 < min_wall_val:
            i, j = min_pair_idx
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
            
            # Try multiple step sizes around current
            for step in (step_size, step_size * 0.5, step_size * 2.0, step_size * 0.25):
                trial = c.copy()
                trial[i] += step * direction
                trial[j] -= step * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    improved = True
                    break
        else:
            i, k = min_wall_idx
            sgn = np.sign(c[i, k])
            if sgn == 0:
                sgn = 1.0
            direction = np.zeros(3)
            direction[k] = sgn
            
            for step in (step_size, step_size * 0.5, step_size * 2.0, step_size * 0.25):
                trial = c.copy()
                trial[i] += step * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    improved = True
                    break
        
        if improved:
            no_improve_count = 0
            step_size = min(step_size * 1.1, 0.01)
        else:
            no_improve_count += 1
            if no_improve_count % 50 == 0:
                step_size = max(step_size * 0.8, 0.0001)
        
        # Random perturbation to escape local optima
        if no_improve_count >= 100 and it % 50 == 0:
            for step in (0.01, 0.005, 0.002, 0.001):
                trial = c.copy()
                trial += rng.normal(scale=step, size=(N, 3))
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    no_improve_count = 0
                    break
    
    return best_c, best_r


def _seed_fcc():
    """FCC lattice supercell, then farthest-point sample to N points."""
    pts = []
    a = 1.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    pts = np.array(pts)
    pts = pts - pts.mean(axis=0)
    me = np.max(np.abs(pts))
    if me > 0:
        pts = pts / me * 0.45
    
    # Farthest point sampling
    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = list(range(len(pts)))
    rem.remove(sel[0])
    while len(sel) < N and rem:
        # Min distance from each remaining point to any selected point
        dm = np.full(len(rem), np.inf)
        for ri, idx in enumerate(rem):
            dmin = np.min(np.linalg.norm(pts[idx] - pts[sel], axis=1))
            dm[ri] = dmin
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
    return pts[sel]


def _seed_grid():
    m = int(np.ceil(N ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N]


def _seed_random(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def _seed_shell_core():
    """Points on a spherical shell plus a central cluster."""
    rng = np.random.default_rng(7)
    shell_n = 35
    core_n = N - shell_n
    idx = np.arange(shell_n)
    phi = np.arccos(1 - 2 * (idx + 0.5) / shell_n)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    R = 0.32
    shell = np.column_stack([
        R * np.sin(phi) * np.cos(theta),
        R * np.sin(phi) * np.sin(theta),
        R * np.cos(phi),
    ])
    shell = np.clip(shell, -0.45, 0.45)
    ticks = (np.arange(3) + 0.5) / 3 - 0.5
    core = np.array([(x * 0.2, y * 0.2, z * 0.2)
                     for z in ticks for y in ticks for x in ticks])[:core_n]
    return np.vstack([shell, core])


def _seed_bcc():
    """BCC lattice points, scaled to fit in cube."""
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float)
    pts = (pts - 1.0)  # center at origin
    pts = pts / np.max(np.abs(pts)) * 0.45
    # Farthest point sampling
    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = list(range(len(pts)))
    rem.remove(sel[0])
    while len(sel) < N and rem:
        dm = np.full(len(rem), np.inf)
        for ri, idx in enumerate(rem):
            dmin = np.min(np.linalg.norm(pts[idx] - pts[sel], axis=1))
            dm[ri] = dmin
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
    return pts[sel]


def construct():
    """Maximize the common radius of 50 equal spheres in [-0.5,0.5]^3."""
    start = time.time()
    deadline = start + (150.0 - DEADLINE_MARGIN)

    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_shell_core(),
        _seed_bcc(),
        _seed_random(1),
        _seed_random(2),
    ]

    best_c = None
    best_r = -1.0

    for si, s in enumerate(seeds):
        if time.time() > deadline:
            break
        s = np.clip(s, -0.49, 0.49)

        # Stage A: repulsion annealing
        c_an = _repulsion_anneal(s, deadline, steps=2000)
        r_an = _compute_radius(c_an)

        # Stage B: Lloyd's algorithm (Voronoi equalization)
        c_lloyd, r_lloyd = _lloyd_optimize(c_an, deadline)

        # Stage C: smooth softmin polish on the best of A and B
        c_cur = c_lloyd if r_lloyd >= r_an else c_an
        r_cur = max(r_lloyd, r_an)
        
        if time.time() < deadline:
            c_pol, r_pol = _smooth_polish(c_cur, deadline)
            if r_pol > r_cur:
                r_cur = r_pol
                c_cur = c_pol

        # Stage D: Metropolis MC to escape local minima
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(c_cur, deadline, iters=3000)
            if r_mc > r_cur:
                r_cur = r_mc
                c_cur = c_mc
        
        # Stage E: local search
        if time.time() < deadline:
            c_ls, r_ls = _local_search(c_cur, deadline, iters=2000)
            if r_ls > r_cur:
                r_cur = r_ls
                c_cur = c_ls

        if r_cur > best_r:
            best_r = r_cur
            best_c = c_cur.copy()

    # Final extra polish on the global best if time remains
    if best_c is not None and time.time() < deadline:
        # Delaunay-based equalization
        if time.time() < deadline:
            c_deq, r_deq = _delaunay_equalize(best_c, deadline, iters=300)
            if r_deq > best_r:
                best_r = r_deq
                best_c = c_deq
        
        # High-precision epigraph polish
        def _final_polish(c, deadline):
            c = c.copy()
            c = np.clip(c, -0.499, 0.499)
            best = c.copy()
            best_r = _compute_radius(c)
            t0 = _compute_radius(c)
            
            for beta in (5000.0, 10000.0, 20000.0, 50000.0):
                if time.time() > deadline:
                    break
                try:
                    x0 = np.concatenate([c.flatten(), [t0]])
                    bounds = [(-0.499, 0.499)] * (N * 3) + [(0.0, 0.5)]
                    
                    res = minimize(
                        _epigraph_obj,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        args=(beta,),
                        options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12,
                                 'maxls': 200},
                    )
                    c_new = np.clip(res.x[:N*3].reshape(N, 3), -0.499, 0.499)
                    t_new = res.x[N*3]
                    r = _compute_radius(c_new)
                    if r > best_r:
                        best_r = r
                        best = c_new.copy()
                        t0 = max(t_new, r)
                    c = c_new
                except Exception:
                    pass
            return best, best_r
        
        c_pol, r_pol = _final_polish(best_c, deadline)
        if r_pol > best_r:
            best_r = r_pol
            best_c = c_pol
        
        if time.time() < deadline:
            c_mc, r_mc = _metropolis_mc(best_c, deadline, iters=2000)
            if r_mc > best_r:
                best_r = r_mc
                best_c = c_mc
        
        if time.time() < deadline:
            c_ls, r_ls = _local_search(best_c, deadline, iters=1500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls
        
        # One more Delaunay equalization pass after local search
        if time.time() < deadline:
            c_deq2, r_deq2 = _delaunay_equalize(best_c, deadline, iters=200)
            if r_deq2 > best_r:
                best_r = r_deq2
                best_c = c_deq2

    if best_c is None:
        best_c = _seed_grid()

    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END
