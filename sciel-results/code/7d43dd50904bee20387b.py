"""Rewritten LJ38 global minimum search with deadline guard and multi-start relaxation."""
import numpy as np
import time

N = 38
DEADLINE_MARGIN = 10.0  # seconds before the 150s kill to stop and return best

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions (N,3)."""
    n = pos.shape[0]
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    # Energy: 4 * sum over i<j of (r^-12 - r^-6)
    # Sum over all pairs (i,j) with i!=j, then divide by 2
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    # Gradient: dE/dr_i = 4 * sum_j ( -12 r^-13 + 6 r^-7 ) * (r_i - r_j)/r
    # Using r2: dE/dx_i = 4 * sum_j ( -12 * r2^(-7) + 6 * r2^(-4) ) * (x_i - x_j)
    # Because d(r^-12)/dr = -12 r^-13, d(r^-6)/dr = -6 r^-7
    # dE/dx_i = 4 * sum_j [ -12 * r^-13 * (x_i - x_j)/r + 6 * r^-7 * (x_i - x_j)/r ]
    #         = 4 * sum_j [ -12 * r^-14 + 6 * r^-8 ] * (x_i - x_j)
    #         = 4 * sum_j [ -12 * r2^-7 + 6 * r2^-4 ] * (x_i - x_j)
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _relax_lbfgs(pos, maxiter=2000, deadline=None):
    """L-BFGS relaxation with deadline check."""
    from scipy.optimize import minimize
    pos = pos.copy()
    
    def fun_and_grad(x):
        x = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(x)
        return e, g.ravel()
    
    try:
        res = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B',
                       jac=True, options={'maxiter': maxiter, 'gtol': 1e-8})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]

def _make_fcc_truncated_octahedron():
    """Construct FCC truncated octahedron for N=38.
    
    FCC lattice with a=1.122 (2*(1/2)^(1/6) for LJ minimum).
    Truncated octahedron has 38 sites in FCC.
    """
    a = 1.122  # FCC lattice constant for LJ minimum
    # FCC primitive vectors
    a1 = a * np.array([0.5, 0.5, 0.0])
    a2 = a * np.array([0.5, 0.0, 0.5])
    a3 = a * np.array([0.0, 0.5, 0.5])
    
    # FCC basis: (0,0,0) and (a/2, a/2, a/2)
    basis = [np.array([0.0, 0.0, 0.0]), a1]  # second basis point is (a/2,a/2,0) in cubic, but in FCC it's (a/2,a/2,a/2)/2... 
    # Actually FCC basis in conventional cubic cell: (0,0,0) and (a/2, a/2, a/2)
    # Let me use conventional cubic coordinates
    # FCC atoms in a cubic cell of side a: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
    # But for truncated octahedron, we need to select 38 atoms.
    
    # Generate FCC lattice points in a bounding box
    # FCC points: (i*a, j*a, k*a) + (a/2)(i+j, j+k, k+i) for integer i,j,k
    # More simply: points where (x,y,z) are all integers or all half-integers (times a/2)
    
    points = []
    # Generate candidate FCC sites in a region
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # Type 1: integer lattice
                p1 = a * np.array([float(i), float(j), float(k)])
                # Type 2: face centers
                p2 = a * np.array([float(i) + 0.5, float(j) + 0.5, float(k)])
                p3 = a * np.array([float(i) + 0.5, float(j), float(k) + 0.5])
                p4 = a * np.array([float(i), float(j) + 0.5, float(k) + 0.5])
                points.extend([p1, p2, p3, p4])
    
    points = np.array(points)
    # Remove duplicates
    points = np.unique(np.round(points, decimals=10), axis=0)
    
    # Select 38 closest to center
    center = np.mean(points, axis=0)
    dists = np.sum((points - center) ** 2, axis=1)
    idx = np.argsort(dists)[:N]
    pos = points[idx]
    # Center it
    pos -= np.mean(pos, axis=0)
    return pos

def _make_icosahedral():
    """Construct icosahedral structure for 38 atoms."""
    # Icosahedral shells: 1 + 12 + 30 = 43, too many. 
    # For 38: 1 + 12 + 25? Or use a known icosahedral arrangement.
    # Let's use 1 center + 12 icosahedron vertices + 25 more
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    r = 1.122  # nearest neighbor distance
    # Icosahedron vertices (unit radius)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * r
    
    pos = [np.zeros(3)]
    pos.extend(list(verts))
    # Add 25 more atoms in a shell
    # Use octahedral directions and face centers
    # For simplicity, add atoms on a spherical shell
    theta = np.linspace(0, np.pi, 6)
    phi_arr = np.linspace(0, 2*np.pi, 5)
    count = 13
    for t in theta:
        for p in phi_arr:
            if count <= 0:
                break
            v = r * 2 * np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
            pos.append(v)
            count -= 1
        if count <= 0:
            break
    
    pos = np.array(pos[:N])
    return pos

def construct():
    t_start = time.time()
    deadline = t_start + 150.0 - DEADLINE_MARGIN
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy 1: FCC truncated octahedron + relax
    try:
        pos_fcc = _make_fcc_truncated_octahedron()
        pos_relaxed, e = _relax_lbfgs(pos_fcc, maxiter=5000)
        if e < best_energy:
            best_energy = e
            best_pos = pos_relaxed.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_pos is not None:
            return best_pos
        # Fallback to simple lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        return np.array(points[:N], dtype=float) * 1.12
    
    # Strategy 2: Icosahedral + relax
    try:
        pos_ico = _make_icosahedral()
        pos_relaxed, e = _relax_lbfgs(pos_ico, maxiter=5000)
        if e < best_energy:
            best_energy = e
            best_pos = pos_relaxed.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_pos is not None:
            return best_pos
        return np.array([(i, j, k) for i in range(4) for j in range(4) for k in range(3)][:N], dtype=float) * 1.12
    
    # Strategy 3: Multi-start random with relaxation
    rng = np.random.RandomState(42)
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline:
            break
        # Random positions in a sphere of radius ~4
        pos_rand = rng.randn(N, 3) * 3.0
        # Ensure minimum distance constraint approximately
        pos_relaxed, e = _relax_lbfgs(pos_rand, maxiter=2000)
        if e < best_energy:
            best_energy = e
            best_pos = pos_relaxed.copy()
    
    if best_pos is None:
        # Fallback
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        return np.array(points[:N], dtype=float) * 1.12
    
    # Verify minimum distance constraint
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0.5:
        # Scale up if needed
        scale = 0.5 / min_dist * 1.01
        best_pos = best_pos * scale
    
    return best_pos
