import numpy as np
from scipy.optimize import minimize
import time

N = 38

def _lj_energy_vec(x):
    """Calculate LJ energy for a flat array of positions."""
    x = x.reshape(-1, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    iu = np.triu_indices(N, 1)
    r6 = dist_sq[iu] ** (-3)
    r12 = r6 ** 2
    return 4 * np.sum(r12 - r6)

def _icosahedral_structure():
    """Generate 1+12+25 icosahedral cluster."""
    # 1 center
    pts = [np.array([0.0, 0.0, 0.0])]
    
    # 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    # Standard icosahedron vertices
    # (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi)
    # (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0)
    # (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    
    verts = []
    for s in [-1, 1]:
        verts.append([0, s, s*phi])
        verts.append([0, s, -s*phi])
        verts.append([s, s*phi, 0])
        verts.append([s, -s*phi, 0])
        verts.append([s*phi, 0, s])
        verts.append([s*phi, 0, -s])
        
    # Normalize to distance 1
    for v in verts:
        v = np.array(v)
        v = v / np.linalg.norm(v)
        pts.append(v)
        
    # 25 second shell
    # In icosahedral symmetry, the second shell is 25 atoms.
    # 12 at distance d2 (above the 12 vertices)
    # 12 at distance d3 (above the 20 faces)
    # 1 at distance d4 (center of the icosahedron? No, center is already taken)
    # Actually, the 25 atoms are:
    # 12 "cap" atoms above each vertex.
    # 12 "face" atoms above each face.
    # 1 "center" atom? No, 12+12+1 = 25.
    
    # This is getting complex to generate analytically.
    # Instead, let's just use a known good start: 
    # The icosahedral structure is a local minimum.
    # I will skip the exact 25-atom generation and use a simpler heuristic:
    # Start with 1+12, then add 25 random points near the surface?
    # No, that's bad.
    
    # Let's stick to the FCC-like structure which is easier to define by lattice vectors.
    return np.array(pts) # Only 13 points so far. This function is incomplete.

def _fcc_structure():
    """Generate 1+12+6+19 FCC-like cluster."""
    pts = [np.array([0.0, 0.0, 0.0])]
    
    # 12 Nearest Neighbors (<110>)
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            pts.append([s1, s2, 0])
            pts.append([s1, 0, s2])
            pts.append([0, s1, s2])
            
    # 6 Next Nearest (<100>)
    pts.append([1, 0, 0])
    pts.append([-1, 0, 0])
    pts.append([0, 1, 0])
    pts.append([0, -1, 0])
    pts.append([0, 0, 1])
    pts.append([0, 0, -1])
    
    # 24 Next-Next Nearest (<221>)
    # We need 19 of these.
    # Generate all 24
    all_221 = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                # Permutations of (2,2,1)
                # Signs applied to the coordinates
                # (2,2,1) -> (s1*2, s2*2, s3*1)
                all_221.append([s1*2, s2*2, s3*1])
                all_221.append([s1*2, s2*1, s3*2])
                all_221.append([s1*1, s2*2, s3*2])
                
    # Normalize all points to unit distance?
    # No, we need relative distances.
    # NN dist: sqrt(2)
    # NNN dist: 1
    # 221 dist: sqrt(4+4+1) = 3
    
    # Scale factor s.
    # We will scale the whole array by s later.
    
    # Pick 19 from all_221.
    # To be symmetric, we can't just pick random.
    # But for a start point, symmetry isn't required, just validity.
    # Let's pick the first 19.
    for p in all_221[:19]:
        pts.append(p)
        
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_energy = np.inf
    best_pos = None
    
    # Candidate 1: FCC-like
    pts_fcc = _fcc_structure()
    # Normalize to average distance ~1.12
    # Distances are mixed. Let's just scale by 1.12 / sqrt(2) approx?
    # NN dist is sqrt(2). We want NN dist ~1.12.
    # Scale = 1.12 / sqrt(2)
    scale = 1.12 / np.sqrt(2)
    pts_fcc_scaled = pts_fcc * scale
    
    # Candidate 2: Random perturbation of FCC-like
    np.random.seed(42)
    pts_rand = pts_fcc_scaled + np.random.normal(0, 0.1, pts_fcc_scaled.shape)
    
    # Candidate 3: Simple Cubic (from seed, but improved)
    # Just in case
    
    candidates = [pts_fcc_scaled, pts_rand]
    
    for init_pos in candidates:
        if time.time() > deadline:
            break
            
        # Flatten
        x0 = init_pos.flatten()
        
        # Optimize
        try:
            res = minimize(_lj_energy_vec, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success or res.fun < best_energy:
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_pos = res.x.reshape(-1, 3)
        except Exception:
            pass
            
    # If we still have time, try a few random restarts
    if time.time() < deadline - 10.0:
        for i in range(5):
            if time.time() > deadline:
                break
            # Random sphere
            pts = np.random.normal(0, 1, (N, 3))
            # Scale to reasonable size
            pts *= 1.1
            x0 = pts.flatten()
            try:
                res = minimize(_lj_energy_vec, x0, method='L-BFGS-B', options={'maxiter': 500})
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_pos = res.x.reshape(-1, 3)
            except Exception:
                pass

    if best_pos is None:
        # Fallback to seed
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
        best_energy = _lj_energy_vec(best_pos.flatten())

    return best_pos
