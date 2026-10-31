"""LJ38 global minimum search: FCC truncated octahedron + local relaxation."""
import numpy as np
import time

N = 38

# Deadline guard: kill after 145s wall-clock (5s safety margin before 150s limit)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _time_left():
    return _DEADLINE - time.time()

def _lj_energy(pos):
    """Compute total LJ energy for positions of shape (N, 3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r6 = 1.0 / r2 ** 3
    r12 = r6 ** 2
    # Sum over upper triangle only (each pair once)
    iu = np.triu_indices(n, k=1)
    e = 4.0 * (r12[iu] - r6[iu]).sum()
    return e

def _lj_grad(pos):
    """Compute gradient of LJ energy."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r6 = 1.0 / r2 ** 3
    r12 = r6 ** 2
    # Gradient: dE/dr_i = 4*(-12*r^-13 + 6*r^-7) * (r_i - r_j) for each pair
    # In vectorized form: grad_i = 4 * sum_j (12*r12 - 6*r6) / r2 * (pos_i - pos_j)
    coeff = 4.0 * (12.0 * r12 - 6.0 * r6) / r2  # shape (n, n)
    np.fill_diagonal(coeff, 0.0)
    grad = coeff @ diff  # (n, 3)
    return grad

def _relax(pos, max_iter=500):
    """L-BFGS local relaxation using scipy."""
    from scipy.optimize import minimize
    result = minimize(
        _lj_energy,
        pos.ravel(),
        jac=lambda x: _lj_grad(x.reshape(-1, 3)).ravel(),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-8, 'maxls': 50}
    )
    return result.x.reshape(-1, 3)

def _enforce_min_separation(pos, min_dist=0.5):
    """Ensure all pairs are at least min_dist apart by nudging close pairs."""
    n = pos.shape[0]
    for _ in range(100):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_dist:
                    if dist < 1e-10:
                        d = np.array([1.0, 0.1, 0.01])
                        dist = np.linalg.norm(d)
                    push = (min_dist - dist + 0.01) / dist
                    pos[i] += push * d
                    pos[j] -= push * d
                    moved = True
        if not moved:
            break
    return pos

def _fcc_truncated_octahedron():
    """
    Generate FCC lattice points forming a truncated octahedron cluster of 38 atoms.
    
    FCC lattice: points at (i, j, k) where i+j+k is even, scaled by 1/sqrt(2)
    so nearest-neighbor distance = 1.0. Then scale by 2^(1/6) ≈ 1.12246 so
    nearest-neighbor distance = 2^(1/6) (the LJ minimum).
    
    The truncated octahedron for LJ38 consists of:
    - 1 center
    - 12 face centers (octahedral)
    - 24 edge centers (cubic)
    Total: 37... we need 38.
    
    Actually, the standard LJ38 global minimum is an FCC truncated octahedron
    with 38 atoms. Let me construct it properly.
    
    FCC unit cell has 4 atoms. For a cluster, we take FCC lattice points
    within a certain radius.
    
    Let's use the known structure: it's an FCC cluster. The 38 atoms are
    arranged as a truncated octahedron. Let me generate FCC points and
    select the 38 closest to center.
    """
    # FCC lattice vectors: basis at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # in units where nearest neighbor = 1/sqrt(2) * sqrt(2) = 1... 
    # Actually let's use the standard FCC: points (i,j,k) with i+j+k even
    # in a cubic lattice of spacing a, nearest neighbor = a*sqrt(2)/2 = a/sqrt(2)
    # Set a/sqrt(2) = 1 => a = sqrt(2)
    
    # Generate FCC points in a cubic region
    # For 38 atoms, we need a region of about 4x4x4 in the FCC lattice
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    # FCC position: (i, j, k) * a where a = sqrt(2)
                    # But actually FCC points are at integer coords where i+j+k even
                    # with the lattice constant being the cubic cell size
                    # Nearest neighbor in this representation: distance sqrt(2) * a/2... 
                    # Let me just use: position = (i, j, k) * (sqrt(2)/2) * 2^(1/6)
                    # No, let me think more carefully.
                    
                    # Standard FCC: primitive vectors a/2(1,1,0), a/2(1,0,1), a/2(0,1,1)
                    # Nearest neighbor distance = a/sqrt(2)
                    # In the cubic representation, FCC sites are at (n1, n2, n3) where
                    # n1+n2+n3 is even, with cubic lattice constant a.
                    # NN distance = a*sqrt(2)/2 = a/sqrt(2)
                    # We want NN = 2^(1/6), so a = 2^(1/6) * sqrt(2) = 2^(2/3)
                    
                    pos = np.array([i, j, k]) * (2.0 ** (1.0/3.0))
                    points.append(pos)
    
    points = np.array(points)
    
    # Sort by distance from center
    dists = np.sum(points ** 2, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    return points

def _random_start(seed):
    """Generate a random valid starting configuration."""
    rng = np.random.RandomState(seed)
    # Start with a rough spherical arrangement
    pos = rng.randn(N, 3) * 2.0
    # Add some structure: place atoms on a rough shell
    for attempt in range(1000):
        pos = rng.randn(N, 3) * 3.0
        pos = _enforce_min_separation(pos, 0.5)
        if np.min(np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=-1)[np.triu_indices(N, k=1)]) >= 0.5:
            break
    return pos

def _best_start():
    """Try to find a good starting configuration using multiple strategies."""
    candidates = []
    energies = []
    
    # 1. FCC truncated octahedron
    fcc = _fcc_truncated_octahedron()
    candidates.append(fcc)
    energies.append(_lj_energy(fcc))
    
    # 2. Relaxed FCC
    if _time_left() > 30:
        fcc_relaxed = _relax(fcc.copy(), max_iter=200)
        candidates.append(fcc_relaxed)
        energies.append(_lj_energy(fcc_relaxed))
    
    # 3. Icosahedral-like start (known local minimum, but good starting point)
    if _time_left() > 30:
        # 13-atom icosahedron + 25 shell atoms
        # Center
        pos = [np.zeros(3)]
        # 12 icosahedral vertices
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        ico_verts /= np.linalg.norm(ico_verts[0])
        pos.extend(ico_verts)
        # Add more atoms in shells
        rng = np.random.RandomState(42)
        for i in range(N - len(pos)):
            while True:
                p = rng.randn(3) * 3.5
                if all(np.linalg.norm(p - q) >= 0.6 for q in pos):
                    break
            pos.append(p)
        ico_start = np.array(pos)
        ico_start = _enforce_min_separation(ico_start, 0.5)
        candidates.append(ico_start)
        energies.append(_lj_energy(ico_start))
        
        if _time_left() > 30:
            ico_relaxed = _relax(ico_start.copy(), max_iter=200)
            candidates.append(ico_relaxed)
            energies.append(_lj_energy(ico_relaxed))
    
    # 4. Random starts with relaxation
    for seed in range(5):
        if _time_left() < 20:
            break
        rand = _random_start(100 + seed)
        candidates.append(rand)
        energies.append(_lj_energy(rand))
        
        if _time_left() > 15:
            rand_relaxed = _relax(rand.copy(), max_iter=300)
            candidates.append(rand_relaxed)
            energies.append(_lj_energy(rand_relaxed))
    
    # Return the best candidate
    best_idx = np.argmin(energies)
    return candidates[best_idx]

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 145.0
    
    best_pos = None
    best_energy = np.inf
    
    # Phase 1: Get a good initial configuration
    init_pos = _best_start()
    best_energy = _lj_energy(init_pos)
    best_pos = init_pos.copy()
    
    # Phase 2: Multiple relaxation attempts from perturbed copies
    rng = np.random.RandomState(12345)
    n_attempts = 15
    
    for attempt in range(n_attempts):
        if _time_left() < 10:
            break
        
        # Perturb the best configuration
        if attempt == 0:
            pos = best_pos.copy()
        else:
            # Mix: sometimes perturb best, sometimes try new random
            if attempt % 3 == 0:
                pos = _random_start(1000 + attempt)
            else:
                pos = best_pos + rng.randn(N, 3) * 0.3
        
        # Relax
        if _time_left() > 5:
            relaxed = _relax(pos.copy(), max_iter=500)
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    # Phase 3: Final fine relaxation
    if _time_left() > 5:
        final = _relax(best_pos.copy(), max_iter=1000)
        e_final = _lj_energy(final)
        if e_final < best_energy:
            best_energy = e_final
            best_pos = final
    
    # Ensure minimum separation
    best_pos = _enforce_min_separation(best_pos, 0.5)
    
    return best_pos
