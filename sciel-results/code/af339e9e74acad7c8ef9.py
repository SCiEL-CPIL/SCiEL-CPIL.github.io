"""LJ38 cluster global minimum search."""
import numpy as np
import time

N = 38

# Deadline guard: 150s wall-clock limit, keep 10s safety margin
DEADLINE = 150.0
SAFETY_MARGIN = 15.0

def lj_energy(pos):
    """Compute total Lennard-Jones energy and optionally gradient."""
    # pos: (N, 3)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff ** 2, axis=2)  # (N, N)
    # Set diagonal to large value to avoid division by zero
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3  # r^6
    r12 = r6 ** 2  # r^12
    # Energy: 4 * sum over pairs (r^-12 - r^-6), but we count each pair twice, so divide by 2
    E = 4.0 * np.sum(1.0 / r12 - 1.0 / r6) / 2.0
    return E

def lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff ** 2, axis=2)  # (N, N)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Energy
    E = 4.0 * np.sum(1.0 / r12 - 1.0 / r6) / 2.0
    # Gradient: dE/dpos_i = 4 * sum_j [ -12/r^13 + 6/r^7 ] * (pos_i - pos_j) / r
    # More precisely: dE/dr = 4*(-12/r^13 + 6/r^7), and dE/dpos_i = 4*sum_j [ -12/r^13 + 6/r^7 ] * (pos_i - pos_j)/r
    # Let's compute the force on each atom
    # F_ij = 4 * (12/r^13 - 6/r^7) * (pos_i - pos_j) / r  ... wait, let me be careful
    
    # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dpos_i = 4 * sum_{j != i} (-12 * r_ij^-14 * (pos_i - pos_j) + 6 * r_ij^-8 * (pos_i - pos_j))
    #           = 4 * sum_{j != i} (pos_i - pos_j) * (-12 * r_ij^-14 + 6 * r_ij^-8)
    #           = 4 * sum_{j != i} (pos_i - pos_j) * (6 * r_ij^-8 - 12 * r_ij^-14)
    #           = 24 * sum_{j != i} (pos_i - pos_j) * (r_ij^-8 - 2 * r_ij^-14)
    
    inv_r8 = 1.0 / r2 ** 4  # r^-8
    inv_r14 = 1.0 / r2 ** 7  # r^-14
    coeff = 24.0 * (inv_r8 - 2.0 * inv_r14)  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)  # (N, 3)
    return E, grad

def l_bfgs(pos, max_iter=2000, tol=1e-10):
    """Simple L-BFGS implementation for local minimization."""
    m = 10  # memory
    n = len(pos) * 3
    pos_flat = pos.flatten().copy()
    
    s_hist = []
    y_hist = []
    
    E, g = lj_energy_and_grad(pos.reshape(-1, 3))
    g_flat = g.flatten()
    pk = -g_flat
    pk_norm = np.linalg.norm(pk)
    
    for it in range(max_iter):
        if pk_norm < tol:
            break
        if it > 0:
            # L-BFGS update
            # Compute alpha via line search (backtracking)
            alpha = 1.0
            new_pos_flat = pos_flat + alpha * pk
            new_E, new_g = lj_energy_and_grad(new_pos_flat.reshape(-1, 3))
            new_g_flat = new_g.flatten()
            # Backtracking line search
            while new_E > E + 1e-4 * alpha * np.dot(g_flat, pk):
                alpha *= 0.5
                new_pos_flat = pos_flat + alpha * pk
                new_E, new_g = lj_energy_and_grad(new_pos_flat.reshape(-1, 3))
                new_g_flat = new_g.flatten()
                if alpha < 1e-12:
                    break
        else:
            # First iteration: simple steepest descent step
            alpha = 1.0 / (1.0 + np.linalg.norm(g_flat))
            new_pos_flat = pos_flat + alpha * pk
            new_E, new_g = lj_energy_and_grad(new_pos_flat.reshape(-1, 3))
            new_g_flat = new_g.flatten()
        
        # Update history
        s = new_pos_flat - pos_flat
        y = new_g_flat - g_flat
        sy = np.dot(s, y)
        if sy > 1e-12:
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
        
        pos_flat = new_pos_flat
        E = new_E
        g_flat = new_g_flat
        g = new_g
        
        # Compute new pk using L-BFGS two-loop recursion
        if len(s_hist) > 0:
            q = g_flat.copy()
            alphas = []
            for i in range(len(s_hist) - 1, -1, -1):
                s_i = s_hist[i]
                y_i = y_hist[i]
                sy_i = np.dot(s_i, y_i)
                if sy_i > 1e-12:
                    beta = np.dot(s_i, q) / sy_i
                    alphas.append(beta)
                    q = q - beta * y_i
                else:
                    alphas.append(0.0)
            r = q / np.dot(g_flat, pk)  # approximate
            for i in range(len(s_hist)):
                s_i = s_hist[i]
                y_i = y_hist[i]
                sy_i = np.dot(s_i, y_i)
                if sy_i > 1e-12:
                    gamma = np.dot(y_i, r) / sy_i
                    r = r + (alphas[i] - gamma) * s_i
                else:
                    r = r + alphas[i] * s_i
            pk = -r
        else:
            pk = -g_flat
        
        pk_norm = np.linalg.norm(pk)
    
    return pos_flat.reshape(-1, 3), E

def fcc_truncated_octahedron():
    """Generate FCC truncated octahedron coordinates for 38 atoms."""
    # FCC lattice points for a truncated octahedron cluster
    # The FCC truncated octahedron has 38 atoms in shells:
    # Shell 0: 1 atom at center
    # Shell 1: 12 atoms (cuboctahedron vertices)
    # Shell 2: 14 atoms
    # Shell 3: 11 atoms
    # Total: 1 + 12 + 14 + 11 = 38
    
    # Let's use the known FCC structure
    # FCC unit cell with a = 1.0 (in reduced units, but we need to scale)
    # Actually, let's generate FCC points within a truncated octahedron shape
    
    # FCC lattice: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
    # For a cluster, we take FCC points within a certain radius
    
    a = 1.0  # lattice constant, will be optimized
    points = []
    # Generate FCC points in a cube and filter by distance from center
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # FCC basis
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    r = np.sqrt(x*x + y*y + z*z)
                    points.append((x, y, z, r))
    
    # Sort by distance from center
    points.sort(key=lambda p: p[3])
    
    # Take the 38 closest points
    selected = points[:38]
    coords = np.array([(p[0], p[1], p[2]) for p in selected])
    
    # Center the cluster
    coords -= np.mean(coords, axis=0)
    
    return coords

def random_start(seed=None):
    """Generate a random starting configuration."""
    if seed is not None:
        np.random.seed(seed)
    # Random positions in a sphere of radius ~5
    r = np.random.uniform(0, 5.0, (N, 3))
    theta = np.random.uniform(0, 2*np.pi, N)
    phi = np.random.uniform(0, np.pi, N)
    pos = np.column_stack([
        r[:, 0] * np.sin(phi) * np.cos(theta),
        r[:, 0] * np.sin(phi) * np.sin(theta),
        r[:, 0] * np.cos(phi)
    ])
    return pos

def check_minimum_separation(pos, min_dist=0.5):
    """Check if all atoms are at least min_dist apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    return min_r >= min_dist

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_pos = None
    best_E = np.inf
    
    # Strategy 1: Start from FCC truncated octahedron
    try:
        pos = fcc_truncated_octahedron()
        # Scale to reasonable size
        # The FCC lattice constant should be around 1.0-1.2 for LJ
        # Let's try a=1.12 (pair potential minimum)
        pos = pos * 1.12
        E = lj_energy(pos)
        if E < best_E and check_minimum_separation(pos):
            best_E = E
            best_pos = pos.copy()
        
        # Local relaxation
        if time.time() < deadline:
            pos_relaxed, E_relaxed = l_bfgs(pos, max_iter=500, tol=1e-10)
            if E_relaxed < best_E and check_minimum_separation(pos_relaxed):
                best_E = E_relaxed
                best_pos = pos_relaxed.copy()
    except Exception:
        pass
    
    # Strategy 2: Multiple random starts with local relaxation
    n_random = 20
    for i in range(n_random):
        if time.time() >= deadline:
            break
        seed = 42 + i * 1000
        pos = random_start(seed)
        # Scale to reasonable size
        pos = pos * 1.0
        try:
            pos_relaxed, E_relaxed = l_bfgs(pos, max_iter=300, tol=1e-9)
            if E_relaxed < best_E and check_minimum_separation(pos_relaxed):
                best_E = E_relaxed
                best_pos = pos_relaxed.copy()
        except Exception:
            pass
    
    # Strategy 3: Perturb best solution and re-relax
    if best_pos is not None and time.time() < deadline:
        for i in range(10):
            if time.time() >= deadline:
                break
            perturbation = np.random.normal(0, 0.1, best_pos.shape)
            pos_perturbed = best_pos + perturbation
            try:
                pos_relaxed, E_relaxed = l_bfgs(pos_perturbed, max_iter=300, tol=1e-10)
                if E_relaxed < best_E and check_minimum_separation(pos_relaxed):
                    best_E = E_relaxed
                    best_pos = pos_relaxed.copy()
            except Exception:
                pass
    
    # Final polish with tighter tolerance
    if best_pos is not None and time.time() < deadline:
        try:
            pos_polished, E_polished = l_bfgs(best_pos, max_iter=1000, tol=1e-12)
            if E_polished < best_E and check_minimum_separation(pos_polished):
                best_E = E_polished
                best_pos = pos_polished.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: return FCC structure
        pos = fcc_truncated_octahedron() * 1.12
        return pos
    
    return best_pos
