"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150 s)


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def _covering_radius_hull(pts):
    """Exact covering radius (radians) via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    best = 0.0
    for f in hull.simplices:
        tri = pts[f]
        a, b, c = tri[0], tri[1], tri[2]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        cc = n / nn
        cosv = np.clip(np.dot(cc, a), -1.0, 1.0)
        r = np.arccos(cosv)
        if r > best:
            best = r
    if best == 0.0:
        G = np.clip(pts @ pts.T, -1.0, 1.0)
        np.fill_diagonal(G, -1.0)
        best = np.arccos(G.min())
    return best


def _covering_radius_fast(pts, n_grid=200):
    """Fast approximate covering radius using a grid on the sphere."""
    pts = _normalize(pts)
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    G = grid @ pts.T
    G = np.clip(G, -1.0, 1.0)
    max_cos = G.max(axis=1)
    angles = np.arccos(max_cos)
    return angles.max()


def _build_hss50():
    """Build the Hardin-Sloane-Smith 50-point configuration."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico = _normalize(ico)
    
    D = 1.0 - ico @ ico.T
    min_d = D.min()
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_d) < 1e-8:
                edges.append((i, j))
    
    verts = []
    for (i, j) in edges:
        verts.append(ico[i] * (2.0 / 3.0) + ico[j] * (1.0 / 3.0))
        verts.append(ico[i] * (1.0 / 3.0) + ico[j] * (2.0 / 3.0))
    verts = np.array(verts)
    verts = _normalize(verts)
    
    vmap = []
    for (i, j) in edges:
        vmap.append((i, j))
        vmap.append((i, j))
    
    G = ico @ ico.T
    np.fill_diagonal(G, -2.0)
    pair_i, pair_j = np.unravel_index(np.argmin(G), G.shape)
    
    remove = set()
    for idx, (i, j) in enumerate(vmap):
        if i == pair_i or j == pair_i or i == pair_j or j == pair_j:
            remove.add(idx)
    
    keep = [k for k in range(len(verts)) if k not in remove]
    out = verts[keep]
    if len(out) != 50:
        out = out[:50]
    return _normalize(out)


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _local_optimize_fast(pts, time_budget):
    """Local optimization using fast covering radius evaluation."""
    pts = _normalize(pts).copy()
    best_r = _covering_radius_fast(pts)
    start = time.time()
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_step = 0
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(N):
            for axis in range(3):
                for sgn in (1.0, -1.0):
                    for ss in step_sizes[current_step:current_step+2]:
                        cand = pts.copy()
                        cand[i, axis] += sgn * ss
                        norm = np.linalg.norm(cand[i])
                        if norm < 1e-10:
                            continue
                        cand[i] /= norm
                        r = _covering_radius_fast(cand)
                        if r < best_r - 1e-12:
                            pts = cand
                            best_r = r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            current_step += 1
            if current_step >= len(step_sizes):
                break
    
    return pts, best_r


def _repulsive_force_optimize(pts, time_budget):
    """Repulsive force-based dynamics: points flow away from crowded regions."""
    pts = _normalize(pts).copy()
    best_r = _covering_radius_fast(pts)
    best_pts = pts.copy()
    start = time.time()
    
    lr = 0.01
    decay = 0.999
    min_lr = 1e-5
    
    while time.time() - start < time_budget and lr > min_lr:
        G = pts @ pts.T
        G = np.clip(G, -1.0, 1.0)
        np.fill_diagonal(G, 1.0)
        
        k = 8
        dists = 1.0 - G
        np.fill_diagonal(dists, np.inf)
        
        nn_idx = np.argpartition(dists, k, axis=1)[:, :k]
        local_dists = np.take_along_axis(dists, nn_idx, axis=1)
        local_density = local_dists.mean(axis=1)
        
        d_min, d_max = local_density.min(), local_density.max()
        if d_max - d_min > 1e-10:
            local_density_norm = (local_density - d_min) / (d_max - d_min)
        else:
            local_density_norm = np.ones(N) * 0.5
        
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # Repulsion: direction away from j, magnitude ~ 1/(1+d^2) * (1 - density_norm[i])
                # Points in sparse regions (high density_norm) get less repulsion
                mag = 1.0 / (1.0 + d * d) * (1.0 - local_density_norm[i])
                forces[i] += mag * (pts[i] - pts[j])
        
        # Normalize forces to unit vectors
        f_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        f_norms[f_norms == 0] = 1.0
        forces = forces / f_norms
        
        # Update points
        pts = pts + lr * forces
        pts = _normalize(pts)
        
        r = _covering_radius_fast(pts)
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
        
        lr *= decay
    
    return best_pts, best_r


def _socp_covering_optimize(pts, time_budget, n_grid=100):
    """
    Smooth epigraph-style optimization: minimize soft-max of angular distances
    using a log-sum-exp surrogate, with adaptive temperature annealing.
    
    Instead of hard max (nonsmooth), we use:
      f_T(pts) = T * log(sum_j exp(angles_j / T))
    which is a smooth upper bound on max_j angles_j, approaching it as T -> 0.
    
    We use gradient descent on this smooth surrogate with:
    - Analytic gradient via softmax weights
    - Adaptive learning rate with cosine annealing
    - Multiple restarts with different temperatures
    """
    pts = _normalize(pts).copy()
    start = time.time()
    
    # Generate a dense grid for identifying worst points
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    n_grid_pts = len(grid)
    
    best_r = _covering_radius_fast(pts)
    best_pts = pts.copy()
    
    # Temperature schedule: start high for exploration, anneal to low for precision
    T_start = 0.1
    T_end = 0.005
    n_steps = max(1, int(time_budget / 0.05))  # estimate steps at ~50ms each
    total_steps = min(n_steps, 5000)
    
    for step in range(total_steps):
        if time.time() - start >= time_budget:
            break
        
        # Cosine annealing for temperature
        progress = step / max(total_steps - 1, 1)
        T = T_start + (T_end - T_start) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Compute angles from each grid point to nearest config point
        G = grid @ pts.T  # (n_grid_pts, N)
        G = np.clip(G, -1.0, 1.0)
        max_cos = G.max(axis=1)
        angles = np.arccos(max_cos)
        
        # Soft-max (log-sum-exp) surrogate
        # f_T = T * log(sum exp(angles/T))
        # For numerical stability, subtract max
        a_max = angles.max()
        exp_vals = np.exp((angles - a_max) / T)
        log_sum_exp = np.log(exp_vals.sum()) + a_max / T
        f_T = T * log_sum_exp
        
        if f_T < best_r:
            best_r = f_T
            best_pts = pts.copy()
        
        # Compute gradient of f_T w.r.t. each point
        # d/dpts_i [T * log sum_j exp(angles_j/T)]
        # = T * (1/sum_j exp(angles_j/T)) * sum_j exp(angles_j/T) * (1/T) * d(angles_j)/dpts_i
        # = (1/sum_j exp(angles_j/T)) * sum_j exp(angles_j/T) * d(angles_j)/dpts_i
        # where d(angles_j)/dpts_i = - (pts_i - grid_j) / |pts_i - grid_j| for the nearest point
        
        # Softmax weights: w_j = exp(angles_j/T) / sum_k exp(angles_k/T)
        w = exp_vals / exp_vals.sum()
        
        # For each grid point j, the nearest config point is nearest_idx[j]
        # Gradient contribution: w_j * d(angles_j)/dpts_nearest_idx[j]
        # d(angles_j)/dpts_i for i = nearest_idx[j]:
        #   angle_j = arccos(pts_i . grid_j)
        #   d(angle)/d(pts_i) = -(grid_j - pts_i) / sqrt(1 - (pts_i . grid_j)^2)
        #   = -(grid_j - pts_i) / |grid_j x pts_i|
        
        nearest_idx = G.argmax(axis=1)
        
        grad = np.zeros_like(pts)
        for j in range(n_grid_pts):
            if w[j] < 1e-10:
                continue
            i = nearest_idx[j]
            gp = grid[j]
            pi = pts[i]
            dot = np.dot(pi, gp)
            dot = np.clip(dot, -1.0, 1.0)
            denom = np.sqrt(max(1.0 - dot * dot, 1e-12))
            # d(angle)/d(pi) = -(gp - pi) / denom
            d_angle = -(gp - pi) / denom
            grad[i] += w[j] * d_angle
        
        # Normalize gradient and apply step
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-12:
            continue
        
        # Adaptive learning rate: larger when T is large (exploration), smaller when T is small (exploitation)
        lr = 0.02 * (T / T_start) ** 0.5
        pts = pts - lr * grad / g_norm * 0.1  # Scale down to keep on sphere
        
        # Project back to sphere
        pts = _normalize(pts)
        
        # Periodic repulsion to avoid coincident points
        if step % 10 == 0:
            G_pts = pts @ pts.T
            G_pts = np.clip(G_pts, -1.0, 1.0)
            np.fill_diagonal(G_pts, 1.0)
            dists_pts = 1.0 - G_pts
            np.fill_diagonal(dists_pts, np.inf)
            
            for i in range(N):
                nn = np.argmin(dists_pts[i])
                if dists_pts[i, nn] < 0.01:
                    push = (pts[i] - pts[nn]) / max(np.linalg.norm(pts[i] - pts[nn]), 1e-10)
                    pts[i] += 0.005 * push
                    pts[i] /= np.linalg.norm(pts[i])
    
    return best_pts, best_r


def _hybrid_optimize(pts, time_budget):
    """Combine repulsive force and SOCP-style subgradient optimization."""
    pts = _normalize(pts).copy()
    start = time.time()
    best_r = _covering_radius_fast(pts)
    best_pts = pts.copy()
    
    # Phase 1: Repulsive force to spread points
    phase1_budget = min(time_budget * 0.4, 30.0)
    pts, r1 = _repulsive_force_optimize(pts, phase1_budget)
    if r1 < best_r:
        best_r = r1
        best_pts = pts.copy()
    
    # Phase 2: SOCP-style subgradient to target worst regions
    remaining = time_budget - (time.time() - start)
    if remaining > 5.0:
        pts, r2 = _socp_covering_optimize(pts, remaining, n_grid=80)
        if r2 < best_r:
            best_r = r2
            best_pts = pts.copy()
    
    # Phase 3: Fine-tune with local coordinate descent
    remaining = time_budget - (time.time() - start)
    if remaining > 10.0:
        pts, r3 = _local_optimize_fast(best_pts, remaining)
        if r3 < best_r:
            best_r = r3
            best_pts = pts.copy()
    
    return best_pts, best_r


def construct():
    t0 = time.time()
    
    # Generate multiple initial configurations
    configs = []
    
    # 1. Hardin-Sloane-Smith based
    try:
        hss = _build_hss50()
        if len(hss) == 50 and np.all(np.isfinite(hss)):
            configs.append(hss)
    except Exception:
        pass
    
    # 2. Fibonacci spiral
    fib = _fibonacci_spiral()
    configs.append(fib)
    
    # 3. Random configurations with symmetry
    rng = np.random.RandomState(42)
    for _ in range(3):
        ico = np.array([
            [0, 1, (1+np.sqrt(5))/2], [0, -1, (1+np.sqrt(5))/2],
            [0, 1, -(1+np.sqrt(5))/2], [0, -1, -(1+np.sqrt(5))/2],
            [1, (1+np.sqrt(5))/2, 0], [-1, (1+np.sqrt(5))/2, 0],
            [1, -(1+np.sqrt(5))/2, 0], [-1, -(1+np.sqrt(5))/2, 0],
            [(1+np.sqrt(5))/2, 0, 1], [-(1+np.sqrt(5))/2, 0, 1],
            [(1+np.sqrt(5))/2, 0, -1], [-(1+np.sqrt(5))/2, 0, -1],
        ], dtype=float)
        ico = _normalize(ico)
        
        rand_pts = rng.randn(38, 3)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
        
        all_pts = np.vstack([ico, rand_pts])
        configs.append(all_pts)
    
    # Evaluate all configs and pick the best
    best_pts = None
    best_r = np.inf
    
    for cfg in configs:
        cfg = _normalize(cfg)
        r = _covering_radius_fast(cfg)
        if r < best_r:
            best_r = r
            best_pts = cfg
    
    if best_pts is None:
        best_pts = fib
        best_r = _covering_radius_fast(best_pts)
    
    # Hybrid optimization: repulsive force + SOCP-style subgradient + local descent
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 20.0:
        opt_pts, opt_r = _hybrid_optimize(best_pts, min(remaining - 10.0, 100.0))
        if opt_r < best_r:
            best_pts, best_r = opt_pts, opt_r
    
    # Final verification with exact hull
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5.0:
        try:
            exact_r = _covering_radius_hull(best_pts)
        except Exception:
            exact_r = best_r
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)) or len(best_pts) != N:
        best_pts = fib
    
    return best_pts
# EVOLVE-BLOCK-END
