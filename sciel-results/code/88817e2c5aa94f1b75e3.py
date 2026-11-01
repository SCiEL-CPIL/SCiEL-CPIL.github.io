"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _retract(x):
    """Retract to the sphere (normalize each point)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-15)

def _lbfgs_sphere(pts, max_iter=500, tol=1e-12, deadline=None, m=30):
    """
    L-BFGS on the sphere using Riemannian gradient and retraction.
    """
    n = pts.shape[0]
    x = pts.copy()
    f, g = _energy_and_grad(x)
    
    # Project gradient to tangent space
    proj = np.einsum('ij,ij->i', g, x)
    g_t = g - proj[:, np.newaxis] * x
    
    # L-BFGS memory
    s_list = []  # steps
    y_list = []  # gradient differences
    g0_t = g_t.copy()
    
    start_time = time.time()
    
    for k in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if np.linalg.norm(g_t) < tol:
            break
        
        # Two-loop recursion to compute direction
        if len(s_list) > 0:
            # Compute rho
            rhos = []
            for s, y in zip(s_list, y_list):
                sy = np.sum(s * y)
                rhos.append(1.0 / sy if sy != 0 else 0.0)
            
            # Two-loop recursion
            q = g_t.copy()
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                alpha_i = rho_i * np.sum(s_i * q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            gamma = 1.0  # initial step length
            r = -gamma * q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[i]
                beta_i = rho_i * np.sum(y_i * r)
                r = r + (alphas[i] - beta_i) * s_i
            
            direction = r
        else:
            direction = -g_t
        
        # Line search with backtracking
        step_size = 1.0
        for _ in range(40):
            new_x = _retract(x + step_size * direction)
            new_f, new_g = _energy_and_grad(new_x)
            new_proj = np.einsum('ij,ij->i', new_g, new_x)
            new_g_t = new_g - new_proj[:, np.newaxis] * new_x
            
            # Armijo condition
            if new_f <= f + 1e-4 * step_size * np.sum(g_t * direction):
                break
            step_size *= 0.5
        
        # Update L-BFGS memory
        s_k = new_x - x
        y_k = new_g_t - g_t
        sy = np.sum(s_k * y_k)
        if sy > 1e-12:
            s_list.append(s_k)
            y_list.append(y_k)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        x = new_x
        f = new_f
        g = new_g
        proj = np.einsum('ij,ij->i', g, x)
        g_t = g - proj[:, np.newaxis] * x
    
    return x, f

def _polish(pts, deadline=None, max_iter=200):
    """
    Fine-tuning with smaller steps and tighter tolerance.
    """
    return _lbfgs_sphere(pts, max_iter=max_iter, tol=1e-14, deadline=deadline, m=20)

def _deduplicate(pts, min_dist=1e-9):
    """
    Check for and fix any points that are too close together.
    """
    n = pts.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                # Move point j slightly away
                direction = pts[j] - pts[i]
                if np.linalg.norm(direction) < 1e-15:
                    direction = np.array([1.0, 0, 0])
                direction = direction / np.linalg.norm(direction)
                pts[j] = _retract(pts[j] + 1e-8 * direction)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = None
    best_energy = np.inf
    
    inits = []
    
    # 1. Golden-angle spiral (Fibonacci lattice) - good global structure
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(pts_fib)
    
    # 2. Multiple random starts with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]:
        np.random.seed(seed)
        pts_rand = np.random.randn(N, 3)
        norms = np.linalg.norm(pts_rand, axis=1, keepdims=True)
        pts_rand = pts_rand / norms
        inits.append(pts_rand)
    
    # 3. Perturbed Fibonacci
    for seed in [11, 22, 33, 44, 55]:
        np.random.seed(seed)
        noise = np.random.randn(N, 3) * 0.05
        pts_pert = pts_fib + noise
        norms = np.linalg.norm(pts_pert, axis=1, keepdims=True)
        pts_pert = pts_pert / norms
        inits.append(pts_pert)
    
    # 4. Structured starts: bands with rotational offset
    np.random.seed(777)
    for n_bands in [8, 10, 12, 14, 16]:
        pts_bands = []
        for band in range(n_bands):
            z = 1.0 - 2.0 * (band + 0.5) / n_bands
            r = np.sqrt(max(1.0 - z*z, 0.0))
            n_pts_in_band = N // n_bands
            if band < N % n_bands:
                n_pts_in_band += 1
            phis = 2.0 * np.pi * np.arange(n_pts_in_band) / n_pts_in_band + band * 0.3
            for ph in phis:
                pts_bands.append([r * np.cos(ph), r * np.sin(ph), z])
        pts_bands = np.array(pts_bands[:N])
        if len(pts_bands) == N:
            inits.append(pts_bands)
    
    # 5. Icosahedral-based structured start
    def icosahedron():
        t = (1.0 + np.sqrt(5.0)) / 2.0
        verts = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        return verts / norms
    
    ico = icosahedron()
    # Create a denser structure based on icosahedral symmetry
    # Use icosahedron vertices as seeds and add points in their vicinity
    np.random.seed(999)
    ico_based = []
    for v in ico:
        for k in range(N // 12):
            noise = np.random.randn(3) * 0.1
            ico_based.append(v + noise)
    ico_based = np.array(ico_based[:N])
    norms = np.linalg.norm(ico_based, axis=1, keepdims=True)
    ico_based = ico_based / norms
    inits.append(ico_based)
    
    # Optimize each initialization, keeping the best result
    n_inits = len(inits)
    
    # Sort inits by initial energy to prioritize better starting points
    init_energies = []
    for init_pts in inits:
        e, _ = _energy_and_grad(init_pts)
        init_energies.append(e)
    
    order = np.argsort(init_energies)
    inits = [inits[i] for i in order]
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - 10.0 - time.time()
        time_per_init = remaining / max(1, n_inits - idx)
        init_deadline = time.time() + time_per_init
        
        pts_opt, energy = _lbfgs_sphere(init_pts, max_iter=400, tol=1e-12, deadline=init_deadline, m=30)
        
        # Polish step
        if time.time() < deadline - 5.0:
            pts_opt, energy = _polish(pts_opt, deadline=deadline - 5.0, max_iter=100)
        
        # Ensure points are on the sphere and deduplicate
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        pts_opt = pts_opt / np.maximum(norms, 1e-15)
        pts_opt = _deduplicate(pts_opt)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt.copy()
    
    # Use the best result found
    if best_pts is None:
        best_pts = pts_fib
    
    # Final verification
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-15)
    best_pts = _deduplicate(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
