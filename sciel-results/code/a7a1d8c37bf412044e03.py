import time
import numpy as np
from scipy.optimize import minimize

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient: F_i = sum_j 4*(6/r_ij^8 - 12/r_ij^14) * (r_i - r_j)
    # Note: standard LJ force is -dV/dr = 4*(12/r^13 - 6/r^7)
    # dV/dr = 4*(-12/r^13 + 6/r^7)
    # grad_i = sum_j dV/dr * (r_i - r_j)/r = sum_j (4*(-12/r^13 + 6/r^7)/r) * (r_i - r_j)
    # = sum_j 4*(-12/r^14 + 6/r^8) * (r_i - r_j)
    # = sum_j 4*(6/r^8 - 12/r^14) * (r_i - r_j)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff[i,j] * (pos[i] - pos[j])
    # = sum_j coeff[i,j] * pos[i] - sum_j coeff[i,j] * pos[j]
    # = pos[i] * sum_j coeff[i,j] - (coeff @ pos)[i]
    
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry explicitly.
    Structure: 1 (center) + 12 (1st shell) + 14 (2nd shell) + 11 (3rd shell).
    """
    pts = []
    
    # 1. Center atom
    pts.append([0.0, 0.0, 0.0])
    
    # 2. 12 atoms in 1st shell (edges of cuboctahedron / face centers of cube)
    # These are at distance 1 in FCC units (before scaling to LJ sigma)
    # FCC lattice vectors: (1,1,0), (1,0,1), (0,1,1) and permutations with signs
    # Actually, the 1st shell of an FCC cluster is the 12 vertices of a cuboctahedron?
    # No, for FCC, the 1st shell is 12 atoms at distance sqrt(2) * a/2? 
    # Let's use the standard FCC lattice points.
    # Lattice constant a=1. Points are (i,j,k) with i+j+k even.
    # Distance from origin:
    # (1,1,0) -> dist sqrt(2). There are 12 such points.
    # (1,0,0) -> dist 1. But (1,0,0) is not in FCC if we define FCC as i+j+k even?
    # Wait, standard FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0).
    # If we set lattice constant such that nearest neighbor distance is 1 (for generation),
    # then (0,0,0) is center.
    # Nearest neighbors are at (0.5, 0.5, 0) and permutations. Distance = sqrt(0.5^2+0.5^2) = sqrt(0.5) = 1/sqrt(2).
    # Let's generate points in a box and select. This is safer.
    
    # Generate FCC points in a box large enough for 38 atoms.
    # Radius of 38-atom cluster is approx 2.5 lattice units (if a=1).
    # Let's use a=1 for generation.
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([float(i), float(j), float(k)])
                    
    pts = np.array(pts)
    
    # Calculate distances from origin
    dists = np.linalg.norm(pts, axis=1)
    
    # We want the 38 closest points to the origin.
    # However, the "truncated octahedron" is a specific shape.
    # Simply taking the 38 closest in a cubic box might distort the shape if the box is small.
    # Let's ensure the box is large enough. max_c=3 gives 7x7x7=343 points.
    # The 38th closest point should be well within the box.
    
    order = np.argsort(dists)
    pts = pts[order][:N]
    
    # Center the cluster
    pts -= np.mean(pts, axis=0)
    
    # Scale to LJ equilibrium distance r_min = 2^(1/6)
    # In FCC lattice with a=1, nearest neighbor distance is sqrt(2)/2? 
    # No, for FCC, nearest neighbors are face diagonals.
    # If lattice points are (i,j,k) with i+j+k even, and a=1,
    # (0,0,0) to (1,1,0) is distance sqrt(2).
    # So min_r in generated coords is sqrt(2).
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale
        
    return pts

def _make_icosahedral_cluster():
    """
    Generate a 38-atom icosahedral cluster.
    1 center + 12 icosahedron + 20 octahedron + 5? 
    Standard icosahedral clusters: 13, 55, 147...
    38 is not a standard magic number for icosahedra (13, 55).
    We will construct a distorted icosahedron or a "bi-icosahedron" fragment.
    For now, let's create a 13-atom icosahedron and add 25 atoms in a shell.
    """
    # 13-atom icosahedron
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (normalized)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Scale to LJ
    # Nearest neighbor distance in icosahedron is edge length.
    # Edge length of unit icosahedron (radius 1) is 2/sqrt(phi^2+1)? 
    # Let's just compute min distance and scale.
    center = np.array([0.0, 0.0, 0.0])
    pts = np.vstack([center, verts])
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    scale = target_r / min_r
    pts *= scale
    
    # Add 25 more atoms to reach 38.
    # Place them in a shell around the icosahedron.
    # This is a rough heuristic to provide a different basin.
    rng = np.random.default_rng(123)
    # Generate random points on a sphere with radius slightly larger than icosahedron radius
    # Icosahedron radius is 1 * scale.
    # Let's add atoms in a "quasi-shell".
    
    # Actually, a better 38-atom icosahedral-like structure is the "Mackay" 55-atom truncated.
    # Since 38 is specific, let's just use the 13-atom icosahedron and 25 random points in a shell,
    # or better, just return the 13-atom one padded with duplicates? No, must be distinct.
    # Let's stick to the FCC as primary and use random restarts for the rest.
    # But having an icosahedral seed is good for testing if we escape the funnel.
    
    # Let's just return the 13-atom icosahedron scaled up? No, N must be 38.
    # We will generate 25 additional points uniformly on a sphere of radius 1.5 * r_ico.
    r_ico = np.linalg.norm(verts[0]) * scale
    r_shell = 1.5 * r_ico
    
    # Generate 25 points on sphere
    u = rng.random((25, 2))
    theta = 2 * np.pi * u[:, 0]
    phi_angle = np.arccos(2 * u[:, 1] - 1)
    
    shell_pts = np.column_stack([
        r_shell * np.sin(phi_angle) * np.cos(theta),
        r_shell * np.sin(phi_angle) * np.sin(theta),
        r_shell * np.cos(phi_angle)
    ])
    
    pts = np.vstack([pts, shell_pts])
    
    # Center
    pts -= np.mean(pts, axis=0)
    
    return pts

def _relax(pos, max_iter=2000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    # Use L-BFGS-B with reasonable settings
    maxcor = 50 if gtol < 1e-12 else 20
    
    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol, "maxcor": maxcor},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin (evaluator kills at 150s)
    
    best_pos = None
    best_energy = np.inf
    
    # 1. Initialize with multiple diverse seeds
    seeds = []
    try:
        seeds.append(_make_fcc_cluster())
    except Exception:
        pass
    try:
        seeds.append(_make_icosahedral_cluster())
    except Exception:
        pass
    
    # Relax all seeds
    for s in seeds:
        try:
            p, e = _relax(s, max_iter=5000, gtol=1e-12, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass
            
    # Check time
    if time.time() > deadline - 20.0:
        # Out of time for search, just polish and return
        if best_pos is not None:
             _relax(best_pos, max_iter=5000, gtol=1e-14, ftol=1e-18)
             best_pos = _ensure_min_sep(best_pos)
             return best_pos
        return _make_fcc_cluster()

    # 2. Basin Hopping
    rng = np.random.default_rng(42)
    current_best = best_pos.copy() if best_pos is not None else seeds[0]
    
    # Adaptive Basin Hopping
    # We will run until time limit or convergence
    # Use a mix of local perturbations and global restarts
    
    perturbation_scale = 0.2
    stagnant_count = 0
    max_stagnant = 5
    
    while time.time() < deadline - 15.0:
        # Choose base
        # 80% chance to perturb current best, 20% to restart from a seed or random
        if rng.random() < 0.8:
            base = current_best
        else:
            # Restart from a seed or random
            if rng.random() < 0.5:
                base = seeds[rng.integers(0, len(seeds))].copy()
            else:
                # Random cluster
                base = rng.standard_normal((N, 3)) * 2.0
                base = _ensure_min_sep(base)
                
        # Perturb
        # Scale decreases if we are finding improvements, increases if stagnant
        if stagnant_count > max_stagnant:
             perturbation_scale = min(0.5, perturbation_scale * 1.5)
             stagnant_count = 0
        else:
             perturbation_scale = max(0.05, perturbation_scale * 0.95)
             
        noise = rng.standard_normal(base.shape) * perturbation_scale
        perturbed = base + noise
        perturbed = _ensure_min_sep(perturbed)
        
        # Relax
        try:
            p, e = _relax(perturbed, max_iter=2000, gtol=1e-12, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
                current_best = p
                stagnant_count = 0
            else:
                stagnant_count += 1
        except Exception:
            continue

    # 3. Final High-Precision Polish
    if best_pos is not None:
        remaining = deadline - time.time() - 2.0
        if remaining > 0.5:
            try:
                p, e = _relax(best_pos, max_iter=10000, gtol=1e-15, ftol=1e-19)
                if e < best_energy:
                    best_energy = e
                    best_pos = p
            except Exception:
                pass
            
            # Second polish if time remains
            remaining = deadline - time.time() - 1.0
            if remaining > 0.5:
                try:
                    p, e = _relax(best_pos, max_iter=5000, gtol=1e-16, ftol=1e-20)
                    if e < best_energy:
                        best_energy = e
                        best_pos = p
                except Exception:
                    pass

    if best_pos is None:
        best_pos = _make_fcc_cluster()
        
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
