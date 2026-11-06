import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import scipy.optimize

def _morse_energy(pos, rho):
    """Compute Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    triu_idx = np.triu_indices(n, 1)
    r = dists[triu_idx]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return energy

def _min_dist_check(pos, min_d=0.5):
    """Check minimum distance constraint."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d

def _generate_icosahedral_shells(n):
    """
    Generate shell indices for an icosahedral-like structure.
    Returns a list of lists, where each inner list contains indices of atoms in that shell.
    This is a heuristic packing that respects high symmetry.
    """
    # Icosahedron has 12 vertices. 
    # We will build shells of increasing size.
    # Typical shell sizes for icosahedral growth: 1, 12, 42, 92...
    # But we need exactly n atoms.
    # Let's use a more flexible "nested icosahedra" or "cubic" approach if n doesn't match.
    
    # For now, let's try a generic "nested spheres" approach with specific high-symmetry points.
    # Actually, for 38, 50, 75, 80, specific structures are known.
    # 38: Icosahedron (12) + inner (26)? No.
    # Let's use a parameterized shell model.
    # We define K shells. Atoms are placed on shells of radii r_k.
    # The angular positions are fixed to high-symmetry points (icosahedral vertices/faces/edges).
    
    # This is complex to generalize perfectly for any n.
    # Alternative: Use a "Cubic" symmetry model which is easier to parameterize.
    # Or, use the known Doye-Wales structures as seeds if we can hardcode their topologies?
    # No, we must construct.
    
    # Let's try a different reduced coordinate strategy:
    # Parameterize the cluster as a set of concentric shells with adjustable radii.
    # The angular coordinates are fixed to a dense, symmetric distribution (e.g., Fibonacci sphere) 
    # but we allow radial breathing and angular jitter?
    # No, "symmetry-adapted" implies fixing the symmetry group.
    
    # Let's implement a specific "Icosahedral" parameterization for n <= 120.
    # We assign atoms to shells:
    # Shell 0: 1 atom (center) - optional
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 30 atoms (icosahedron edges midpoints? or face centers?)
    # This is getting complicated.
    
    # Let's switch to a simpler, robust strategy: 
    # **Multi-start L-BFGS-B with Symmetry-Broken FCC/HCP Seeds**
    # The previous program used basin hopping which is slow. 
    # If we just do many random starts + L-BFGS-B, we might find the global min faster 
    # because L-BFGS-B is very efficient for smooth Morse potentials.
    # The key is to provide good seeds.
    
    pass

def _fcc_seed(n, rng):
    """Generate FCC-like seed."""
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    side = int(np.ceil((0.25 * n) ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append(p)
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    return cluster

def _hcp_seed(n, rng):
    """Generate HCP-like seed."""
    # Hexagonal lattice
    a = 1.0
    c = 1.633 # ideal c/a
    pts = []
    # Generate a hexagonal prism of sufficient size
    side = int(np.ceil((n / 2.0) ** (1.0/3.0))) + 2
    for layer in range(-side, side + 1):
        z = layer * c
        # AB stacking
        offset = 0 if layer % 2 == 0 else 0.5
        for i in range(-side, side + 1):
            for j in range(-side, side + 1):
                x = i * a + (0.5 if j % 2 != 0 else 0.0) + offset
                y = j * a * np.sqrt(3) / 2
                pts.append([x, y, z])
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    return cluster

def _icosahedral_seed(n, rng):
    """
    Generate an icosahedral seed.
    This is specific to n values that fit icosahedral shells.
    For other n, we pad with random atoms on the surface.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    pos = verts.copy()
    count = 12
    
    if n > 12:
        # Add inner atoms? Or outer shell?
        # Doye-Wales 38-atom cluster is an icosahedron with 26 inner atoms? 
        # Actually, 38 = 12 (outer) + 26 (inner).
        # Let's try to fill the interior with a smaller icosahedron or random points.
        
        # For simplicity, let's add points on a smaller sphere.
        # 26 points.
        # Or just use a dense random packing inside the icosahedron.
        
        # Let's use a "nested" approach.
        # If n is close to 38, 50, 75, 80, we can hardcode better seeds.
        # But let's try a generic "shell" filling.
        
        # Add 2nd shell (30 atoms) if n >= 42?
        # 12 + 30 = 42.
        # 12 + 30 + 20 (inner?) = 62?
        
        # Let's just use random points within the convex hull of the icosahedron for the remaining atoms.
        # This breaks symmetry but provides a good local structure.
        
        remaining = n - count
        # Generate random points in a sphere of radius 1
        for _ in range(remaining):
            # Random point in sphere
            u = rng.random()
            v = rng.random()
            theta = 2 * np.pi * u
            phi_ = np.arccos(2 * v - 1)
            r = 0.8 * (0.5 + 0.5 * rng.random()) # Radius between 0.4 and 0.8
            
            x = r * np.sin(phi_) * np.cos(theta)
            y = r * np.sin(phi_) * np.sin(theta)
            z = r * np.cos(phi_)
            pos = np.vstack([pos, [x, y, z]])
            
    # Center
    pos -= np.mean(pos, axis=0)
    # Scale to typical distance 1
    # The icosahedron edge length is 2*phi/sqrt(phi^2+1) ? 
    # Let's just scale so that the mean distance is ~1 or max distance is ~2-3.
    # For n=12, edge is ~1.05 if normalized this way.
    # Let's scale by 1.0.
    
    # Add noise
    pos += rng.normal(0, 0.02, size=pos.shape)
    return pos

def _local_minimize(pos, rho, max_iter=2000):
    """
    Perform local minimization using L-BFGS-B.
    Returns energy and positions.
    """
    def objective(x):
        p = x.reshape(-1, 3)
        # Soft constraint for min distance to help optimizer
        # If dist < 0.5, add penalty
        # Actually, just use the energy. The optimizer might find a local min with close atoms.
        # We check validity later.
        e, g = morse_energy_gradient(p, rho)
        return e, g.flatten()

    x0 = pos.flatten()
    
    # Use L-BFGS-B
    try:
        res = scipy.optimize.minimize(
            objective, 
            x0, 
            jac=True, 
            method='L-BFGS-B', 
            options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10, 'maxls': 20}
        )
        if res.success or res.fun < 1e9:
            return res.fun, res.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Fallback: return initial
    return _morse_energy(pos, rho), pos

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Strategy: Multi-start local optimization (L-BFGS-B) with diverse structural seeds
    (FCC, HCP, Icosahedral, Random). This avoids the slow convergence of basin hopping
    and leverages the efficiency of gradient-based methods on smooth Morse potentials.
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 330s total / 6 instances ~ 55s each
    
    rng = np.random.default_rng(12345 + n * 100 + int(rho * 10))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate seeds
    seeds = []
    
    # 1. FCC
    try:
        s_fcc = _fcc_seed(n, rng)
        if _min_dist_check(s_fcc):
            seeds.append(s_fcc)
    except Exception:
        pass
    
    # 2. HCP
    try:
        s_hcp = _hcp_seed(n, rng)
        if _min_dist_check(s_hcp):
            seeds.append(s_hcp)
    except Exception:
        pass
        
    # 3. Icosahedral
    try:
        s_ico = _icosahedral_seed(n, rng)
        if _min_dist_check(s_ico):
            seeds.append(s_ico)
    except Exception:
        pass
        
    # 4. Random Compact
    try:
        radius = (n / 4.0) ** (1.0/3.0) * 1.2
        s_rand = rng.uniform(-radius, radius, size=(n, 3))
        # Ensure min dist by scaling up if needed
        for scale in [1.0, 1.1, 1.2, 1.5]:
            if _min_dist_check(s_rand * scale):
                seeds.append(s_rand * scale)
                break
    except Exception:
        pass
        
    # 5. Random FCC-like (perturbed FCC)
    try:
        s_fcc2 = _fcc_seed(n, rng)
        s_fcc2 += rng.normal(0, 0.1, size=s_fcc2.shape)
        if _min_dist_check(s_fcc2):
            seeds.append(s_fcc2)
    except Exception:
        pass

    if not seeds:
        # Fallback
        seeds.append(rng.uniform(-2, 2, size=(n, 3)))

    # Optimize each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 5.0:
            break
            
        # Local minimize
        try:
            e, pos = _local_minimize(seed, rho)
            if e < best_energy and _min_dist_check(pos):
                best_energy = e
                best_positions = pos
        except Exception:
            continue

    # If we have time, do a few more random starts
    while time.time() < deadline - 5.0:
        # Generate a new random seed
        radius = (n / 4.0) ** (1.0/3.0) * 1.2
        s_rand = rng.uniform(-radius, radius, size=(n, 3))
        # Try to fix min dist
        if not _min_dist_check(s_rand):
             # Scale up
             for scale in [1.1, 1.2, 1.5]:
                 if _min_dist_check(s_rand * scale):
                     s_rand = s_rand * scale
                     break
        if not _min_dist_check(s_rand):
             continue
             
        try:
            e, pos = _local_minimize(s_rand, rho)
            if e < best_energy and _min_dist_check(pos):
                best_energy = e
                best_positions = pos
        except Exception:
            continue

    if best_positions is None:
        # Return first seed if no valid result
        return seeds[0]
        
    # Final polish with tighter tolerance
    if time.time() < deadline - 2.0:
        try:
            e, pos = _local_minimize(best_positions, rho, max_iter=5000)
            if e < best_energy and _min_dist_check(pos):
                best_energy = e
                best_positions = pos
        except Exception:
            pass

    # Final validation
    if not _min_dist_check(best_positions):
        # Try to fix
        for scale in [1.05, 1.1, 1.2]:
            if _min_dist_check(best_positions * scale):
                best_positions = best_positions * scale
                break
        else:
            # Fallback
            best_positions = seeds[0]
            
    return best_positions
