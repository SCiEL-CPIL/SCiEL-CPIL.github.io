"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(c):
    """
    Compute the actual score (largest common radius) for a configuration.
    c: (N, 3) array of centers.
    Returns the radius r.
    """
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check boundaries: min distance to each face
    face_dists = 0.5 - np.abs(c)
    min_face = np.min(face_dists)
    if min_face < -1e-12:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_pair / 2.0, min_face)
    return r


def _soft_min_logsumexp(d, beta):
    """
    Smooth surrogate for min(d) using log-sum-exp.
    Returns -log(sum(exp(-beta*d)))/beta which approximates min(d) as beta->inf.
    """
    d_min = np.min(d)
    terms = np.exp(-beta * (d - d_min))
    s = np.sum(terms)
    return d_min - np.log(s) / beta


def _soft_min_grad(d, beta):
    """
    Gradient of soft_min_logsumexp with respect to d.
    Returns array of same shape as d.
    """
    d_min = np.min(d)
    terms = np.exp(-beta * (d - d_min))
    s = np.sum(terms)
    return -beta * terms / s


def _objective_and_grad(c_flat, beta):
    """
    Objective and gradient for optimization.
    c_flat: 1D array of length 3*N
    Returns (obj, grad) where obj = -soft_min(all distances)
    """
    c = c_flat.reshape(N, 3)
    
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero on diagonal
    dists_sq[np.diag_indices(N)] = 1.0
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Face distances
    face_dists = 0.5 - np.abs(c)
    face_flat = face_dists.flatten()
    
    # Combine
    all_dists = np.concatenate([pair_dists, face_flat])
    
    # Soft min
    r_smooth = _soft_min_logsumexp(all_dists, beta)
    obj = -r_smooth
    
    # Gradient
    grad_d = _soft_min_grad(all_dists, beta)
    # grad of obj w.r.t. all_dists is -grad_d
    grad_pair = -grad_d[:len(pair_dists)]
    grad_face = -grad_d[len(pair_dists):]
    
    # Now compute gradient w.r.t. c
    grad_c = np.zeros((N, 3))
    
    # Pairwise part: for each pair (i,j), d_ij/2 = 0.5 * ||c_i - c_j||
    # d(d_ij/2)/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d(d_ij/2)/d(c_j) = 0.5 * (c_j - c_i) / ||c_i - c_j||
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        d_ij = dists[i, j]
        if d_ij < 1e-15:
            continue
        diff_ij = c[i] - c[j]
        coeff = grad_pair[idx] * 0.5 / d_ij
        grad_c[i] += coeff * diff_ij
        grad_c[j] -= coeff * diff_ij
    
    # Face part: f_ik = 0.5 - |c_ik|
    # d f_ik / d c_ik = -sign(c_ik)
    face_grad = grad_face.reshape(N, 3)
    signs = np.sign(c)
    # Where c is exactly 0, sign is 0; use subgradient 0
    grad_c -= face_grad * signs
    
    return obj, grad_c.flatten()


def _generate_bcc_init():
    """Generate BCC lattice initialization."""
    # BCC: simple cubic lattice + body center
    # For N=50, try to find a good BCC-based arrangement
    # Simple cubic with spacing s, plus body centers
    # In a cube of side 1, if we use n points per side in SC, that's n^3 points
    # Plus body centers: n^3 more, total 2*n^3
    # For N=50: 2*2^3=16, 2*3^3=54. So 3^3 SC + 3^3 body = 54 points, take 50
    
    n = 3
    s = 1.0 / (n + 1)  # spacing to keep inside [-0.5, 0.5]
    # Actually, let's use spacing that fits n+1 intervals
    # Points at -0.5 + (k+0.5)*s for k=0..n, but that's n+1 points
    # For n=3, we want 4 points per axis? No, let's think differently.
    
    # Let's use a grid of n x n x n for SC corners and n x n x n for body centers
    # Total 2*n^3
    # For n=3: 54 points, we need 50
    # Spacing: place n points per axis in [-0.5+margin, 0.5-margin]
    
    # Use n=3: 3 points per axis for SC, 3 for body
    # SC points: x in {-a, 0, a} where a = 0.5 - eps
    # Body center points: x in {-a/2, 0, a/2}... no, BCC body centers are at (0.5,0.5,0.5) relative to SC
    
    # Let me use a proper BCC lattice:
    # SC lattice points at (i*s, j*s, k*s) for i,j,k in 0..n-1
    # Body center points at ((i+0.5)*s, (j+0.5)*s, (k+0.5)*s)
    # Total: 2*n^3
    # For n=3: 54 points
    
    n = 3
    # Choose s so that all points are in [-0.5, 0.5]
    # SC points: 0 to (n-1)*s, so (n-1)*s <= 0.5 => s <= 0.5/(n-1) = 0.25
    # Body points: 0.5*s to (n-0.5)*s, so (n-0.5)*s <= 0.5 => s <= 0.5/2.5 = 0.2
    # Use s = 0.2
    s = 0.2
    
    points = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                points.append([(i - (n-1)/2.0) * s, (j - (n-1)/2.0) * s, (k - (n-1)/2.0) * s])
                points.append([(i - (n-1)/2.0 + 0.5) * s, (j - (n-1)/2.0 + 0.5) * s, (k - (n-1)/2.0 + 0.5) * s])
    
    points = np.array(points[:N])
    return points


def _generate_fcc_init():
    """Generate FCC lattice initialization."""
    # FCC: face-centered cubic
    # Lattice points: (i,j,k), (i+0.5,j+0.5,k), (i+0.5,j,k+0.5), (i,j+0.5,k+0.5) * s
    # For a unit cell, 4 points per cell
    # We need 50 points, so about 12-13 unit cells
    
    # Let's generate FCC points in a region and pick the best 50
    s = 0.25  # lattice parameter
    n = 5  # number of unit cells per axis
    
    points = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                base = np.array([i, j, k]) * s
                points.append(base)
                points.append(base + np.array([0.5, 0.5, 0]) * s)
                points.append(base + np.array([0.5, 0, 0.5]) * s)
                points.append(base + np.array([0, 0.5, 0.5]) * s)
    
    points = np.array(points)
    # Center the points
    points -= points.mean(axis=0)
    
    # Filter to those inside [-0.45, 0.45]
    mask = np.all(np.abs(points) <= 0.45, axis=1)
    points = points[mask]
    
    # If we have more than N, pick the ones closest to center
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    
    # If we have fewer, pad with random
    if len(points) < N:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-0.4, 0.4, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    
    return points[:N]


def _generate_hcp_init():
    """Generate HCP-like initialization."""
    # Hexagonal close packing layers
    # Layer 1: hexagonal grid
    # Layer 2: offset
    
    a = 0.28  # lattice parameter
    c = a * np.sqrt(8.0 / 3.0)  # height between layers
    
    points = []
    n_layers = 6
    for layer in range(n_layers):
        z = -2.5 * c / 2 + layer * c
        if z > 0.45 or z < -0.45:
            continue
        # Hexagonal grid in xy plane
        n = 4
        for j in range(n):
            for i in range(n):
                x = i * a + (j % 2) * a / 2
                y = j * a * np.sqrt(3) / 2
                # Center
                x -= (n - 1) * a / 2
                y -= (n - 1) * a * np.sqrt(3) / 4
                points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        rng = np.random.default_rng(42)
        extra = rng.uniform(-0.4, 0.4, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    
    return points[:N]


def _local_polish(c, beta_start=100.0, n_iters=5, time_limit=30.0):
    """
    Local polish using high beta values.
    """
    start_time = time.time()
    best_c = c.copy()
    best_score = _score(best_c)
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit:
            break
        beta = beta_start * (2.0 ** it)
        
        try:
            res = minimize(
                _objective_and_grad,
                best_c.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_opt = res.x.reshape(N, 3)
            c_opt = np.clip(c_opt, -0.49999, 0.49999)
            
            score = _score(c_opt)
            if score > best_score:
                best_score = score
                best_c = c_opt.copy()
        except Exception:
            pass
    
    return best_c, best_score


def _optimize_from_initial(c_init, beta, time_budget=10.0):
    """
    Optimize from an initial configuration.
    """
    start_time = time.time()
    
    try:
        res = minimize(
            _objective_and_grad,
            c_init.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        c_opt = res.x.reshape(N, 3)
        c_opt = np.clip(c_opt, -0.4999, 0.4999)
        return c_opt
    except Exception:
        return c_init


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Generate multiple initializations
    inits = []
    
    # BCC initialization
    try:
        c_bcc = _generate_bcc_init()
        inits.append(c_bcc)
    except Exception:
        pass
    
    # FCC initialization
    try:
        c_fcc = _generate_fcc_init()
        inits.append(c_fcc)
    except Exception:
        pass
    
    # HCP initialization
    try:
        c_hcp = _generate_hcp_init()
        inits.append(c_hcp)
    except Exception:
        pass
    
    # Random initialization
    rng = np.random.default_rng(42)
    c_rand = rng.uniform(-0.4, 0.4, size=(N, 3))
    inits.append(c_rand)
    
    # Grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c_grid = grid[:N].copy()
    inits.append(c_grid)
    
    best_c = inits[0].copy()
    best_score = _score(best_c)
    
    # Optimize from each initialization with moderate beta
    beta_starts = [50.0, 100.0, 200.0]
    
    for beta in beta_starts:
        if time.time() > deadline - 30.0:
            break
        
        for c_init in inits:
            if time.time() > deadline - 30.0:
                break
            
            c_opt = _optimize_from_initial(c_init, beta, time_budget=15.0)
            score = _score(c_opt)
            if score > best_score:
                best_score = score
                best_c = c_opt.copy()
    
    # Local polish with high beta
    if time.time() < deadline - 15.0:
        best_c, best_score = _local_polish(best_c, beta_start=200.0, n_iters=4, time_limit=20.0)
    
    # Final very high precision polish
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _objective_and_grad,
                best_c.flatten(),
                args=(1000.0,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            c_polished = res.x.reshape(N, 3)
            c_polished = np.clip(c_polished, -0.499999, 0.499999)
            
            score = _score(c_polished)
            if score > best_score:
                best_score = score
                best_c = c_polished.copy()
        except Exception:
            pass
    
    return best_c
# EVOLVE-BLOCK-END
