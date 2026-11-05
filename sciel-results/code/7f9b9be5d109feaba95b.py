"""
Rewritten program for Morse cluster global minima.
Uses a Coarse-to-Fine Multi-Resolution Search strategy:
1. Generate candidate structures from distorted BCC/FCC lattices (coarse grid).
2. Relax each candidate using the provided local optimizer.
3. Select the best energy.
This avoids the random-walk inefficiency of pure random packing for N>38.
"""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax
import time

def _generate_lattice_points(n, lattice_type='bcc', radius_factor=0.9, rng=None):
    """
    Generate n points on a BCC or FCC lattice within a sphere of radius R.
    R is chosen such that the density matches n atoms.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Estimate radius for n atoms in a sphere with density ~1 atom per unit volume
    # Volume V = 4/3 * pi * R^3. Density rho = n/V.
    # For close packing, density is high. Let's aim for R such that n points fit comfortably.
    # Empirical: R ~ 1.2 * n^(1/3) is often a good start for loose packing, 
    # but for minima we want them closer. 
    # Let's use R = (3*n / (4*pi * density))^(1/3).
    # For Morse, equilibrium distance is 1. Max density in BCC is ~0.68 atoms/volume unit?
    # Let's just scale R = n^(1/3) * radius_factor.
    
    R = (n ** (1.0 / 3.0)) * radius_factor
    
    # Generate grid points
    # Max coordinate is R
    # Grid step size? 
    # For BCC, primitive vectors are (0.5, 0.5, 0.5), (0.5, -0.5, 0.5), etc.
    # Simpler: Generate a cubic grid and filter, then map to BCC/FCC if needed?
    # Actually, generating a specific lattice is easier by iterating over integer coordinates.
    
    points = []
    
    if lattice_type == 'bcc':
        # BCC lattice: points at (i, j, k) and (i+0.5, j+0.5, k+0.5)
        # We need to scale the lattice constant 'a'.
        # Number of points in a cube of side L is approx (L/a)^2 * 2 (for BCC, 2 atoms per unit cell of side a)
        # We want n points. (L/a)^3 * 2 = n => a = (2L/n)^(1/3)
        # Let's fix a range for i,j,k and adjust 'a' to fit n points in sphere R.
        
        # Try a few lattice constants to get close to n points
        best_points = None
        best_diff = float('inf')
        
        # Range of lattice constants to try
        # If R is the radius, max i is R/a.
        # Let's iterate a from 0.5 to 2.0
        for a in np.linspace(0.5, 2.0, 20):
            max_idx = int(np.ceil(R / a)) + 1
            pts = []
            # Iterate over the cube
            for i in range(-max_idx, max_idx + 1):
                for j in range(-max_idx, max_idx + 1):
                    for k in range(-max_idx, max_idx + 1):
                        # BCC positions
                        # 1. (i*a, j*a, k*a)
                        p1 = np.array([i, j, k]) * a
                        if np.linalg.norm(p1) <= R:
                            pts.append(p1)
                        # 2. (i*a + a/2, j*a + a/2, k*a + a/2)
                        p2 = (np.array([i, j, k]) + 0.5) * a
                        if np.linalg.norm(p2) <= R:
                            pts.append(p2)
            
            pts = np.array(pts)
            if len(pts) == 0: continue
            
            # If we have more than n, we need to select n.
            # If we have fewer, this 'a' is too large.
            if len(pts) >= n:
                # Select n points. To maintain symmetry, we can just take the first n?
                # Or better, random sample to break symmetry?
                # For initialization, random sampling from the lattice is good.
                idx = rng.choice(len(pts), n, replace=False)
                selected = pts[idx]
                
                # Check min distance? Lattice ensures min dist a/sqrt(2) for BCC.
                # If a is small, dist is small. We want dist ~ 1.
                # So we prefer 'a' such that min_dist is close to 1.
                # Min dist in BCC is a/sqrt(2). So a ~ sqrt(2).
                # Let's score based on closeness to n and min_dist.
                
                # Calculate min distance for a subset to save time? 
                # Actually, just check if n is close.
                diff = abs(len(pts) - n)
                if diff < best_diff:
                    best_diff = diff
                    best_points = selected
            
            elif len(pts) < n:
                # Too few points, increase density (decrease a)
                # This loop increases a, so we are moving in the wrong direction if we start small?
                # np.linspace(0.5, 2.0) goes from small a (high density) to large a (low density).
                # So if len(pts) < n, we need smaller a. The loop will handle it as it goes up?
                # No, if a=0.5 gives 1000 pts and n=50, we skip.
                # If a=2.0 gives 10 pts and n=50, we skip.
                # We want the 'a' that yields exactly n.
                pass

        if best_points is not None:
            return best_points
        else:
            # Fallback: Random points in sphere
            pts = []
            while len(pts) < n:
                p = rng.uniform(-R, R, 3)
                if np.linalg.norm(p) <= R:
                    pts.append(p)
            return np.array(pts)
            
    else: # fcc
        # Similar logic for FCC
        best_points = None
        best_diff = float('inf')
        for a in np.linspace(0.5, 2.0, 20):
            max_idx = int(np.ceil(R / a)) + 1
            pts = []
            for i in range(-max_idx, max_idx + 1):
                for j in range(-max_idx, max_idx + 1):
                    for k in range(-max_idx, max_idx + 1):
                        # FCC positions: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
                        # In a cubic cell, 4 atoms.
                        base = np.array([i, j, k]) * a
                        offsets = [
                            [0, 0, 0],
                            [0, 0.5, 0.5],
                            [0.5, 0, 0.5],
                            [0.5, 0.5, 0]
                        ]
                        for off in offsets:
                            p = base + np.array(off) * a
                            if np.linalg.norm(p) <= R:
                                pts.append(p)
            
            pts = np.array(pts)
            if len(pts) == 0: continue
            
            if len(pts) >= n:
                idx = rng.choice(len(pts), n, replace=False)
                selected = pts[idx]
                diff = abs(len(pts) - n)
                if diff < best_diff:
                    best_diff = diff
                    best_points = selected

        if best_points is not None:
            return best_points
        else:
            pts = []
            while len(pts) < n:
                p = rng.uniform(-R, R, 3)
                if np.linalg.norm(p) <= R:
                    pts.append(p)
            return np.array(pts)

def construct(instance):
    """
    Construct the cluster using multi-resolution lattice-based initialization.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget per instance
    start_time = time.time()
    time_limit = 45.0  # Leave buffer for overhead
    
    rng = np.random.default_rng(size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Strategy:
    # 1. Try BCC lattice with jitter
    # 2. Try FCC lattice with jitter
    # 3. Try Random packing (fallback)
    # 4. Try "Shell" initialization (icosahedral-like)
    
    candidates = []
    
    # 1. BCC
    try:
        pts_bcc = _generate_lattice_points(size, 'bcc', radius_factor=1.1, rng=rng)
        # Jitter
        jitter = rng.normal(0, 0.1, pts_bcc.shape)
        pts_bcc_j = pts_bcc + jitter
        candidates.append(pts_bcc_j)
    except:
        pass
        
    # 2. FCC
    try:
        pts_fcc = _generate_lattice_points(size, 'fcc', radius_factor=1.1, rng=rng)
        jitter = rng.normal(0, 0.1, pts_fcc.shape)
        pts_fcc_j = pts_fcc + jitter
        candidates.append(pts_fcc_j)
    except:
        pass
        
    # 3. Random Packing (Improved)
    # Use rejection sampling with min distance 0.8
    radius = 1.0 * size ** (1.0 / 3.0) * 1.2
    pts_rand = []
    attempts = 0
    while len(pts_rand) < size and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points
            if pts_rand:
                dists = np.linalg.norm(np.array(pts_rand) - p, axis=1)
                if np.min(dists) >= 0.7:
                    pts_rand.append(p)
            else:
                pts_rand.append(p)
        attempts += 1
    if len(pts_rand) == size:
        candidates.append(np.array(pts_rand))
    else:
        # Fallback to simple random if rejection failed
        pts_rand = []
        while len(pts_rand) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                pts_rand.append(p)
        candidates.append(np.array(pts_rand))
        
    # 4. Spherical Shell (Icosahedral approximation)
    # Place atoms on a sphere of radius R_shell
    R_shell = size ** (1.0 / 3.0) * 0.8
    pts_shell = []
    # Use Fibonacci sphere for even distribution
    n = size
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
        radius_xy = np.sqrt(1 - z**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius_xy
        y = np.sin(theta) * radius_xy
        pts_shell.append([x, y, z])
    pts_shell = np.array(pts_shell) * R_shell
    # Add some inner atoms if n is large? 
    # For now, just the shell.
    jitter = rng.normal(0, 0.1, pts_shell.shape)
    pts_shell_j = pts_shell + jitter
    candidates.append(pts_shell_j)
    
    # Relax all candidates
    for i, pos in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
            
        # Ensure minimum distance constraint is roughly met before relax?
        # relax handles it, but if points are too close, it might fail or be slow.
        # The lattice/jitter should be okay.
        
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception as e:
            pass
            
    # If no candidate worked (unlikely), return the last one or random
    if best_positions is None:
        # Fallback to the first candidate relaxed? Or just return the input?
        # Must return valid positions.
        best_positions = candidates[0]
        
    return best_positions
# EVOLVE-BLOCK-END
