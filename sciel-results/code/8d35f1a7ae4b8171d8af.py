"""Morse cluster global minima: structured lattice seeds + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import time


def _bcc_shell(n, spacing=1.0):
    """Generate n atoms on BCC lattice shells centered at origin."""
    # BCC lattice points: integer coords with all-even or all-odd parity
    pts = []
    pts.append([0.0, 0.0, 0.0])
    # Generate candidate lattice points in increasing distance
    max_r = int(np.ceil(n ** (1.0/3.0) * 1.5)) + 2
    candidates = []
    for ix in range(-max_r, max_r + 1):
        for iy in range(-max_r, max_r + 1):
            for iz in range(-max_r, max_r + 1):
                # BCC: include points where ix+iy+iz is even (simple cubic) 
                # plus body centers. Actually BCC = simple cubic + body centers.
                # Let's use: all integer points (SC) gives density 1 per unit cell.
                # BCC has 2 per unit cell. Let's just use SC lattice which is simpler.
                d = np.sqrt(ix*ix + iy*iy + iz*iz)
                candidates.append((d, [ix*spacing, iy*spacing, iz*spacing]))
    candidates.sort(key=lambda x: x[0])
    for d, p in candidates:
        if len(pts) >= n:
            break
        pts.append(p)
    return np.array(pts[:n])


def _fcc_shell(n, spacing=1.0):
    """Generate n atoms on FCC lattice shells."""
    pts = []
    pts.append([0.0, 0.0, 0.0])
    max_r = int(np.ceil(n ** (1.0/3.0) * 1.5)) + 2
    candidates = []
    for ix in range(-max_r, max_r + 1):
        for iy in range(-max_r, max_r + 1):
            for iz in range(-max_r, max_r + 1):
                # FCC: points where all coords same parity, or all differ by 1/2 offsets
                # Simpler: use the fact FCC = { (i,j,k) : i+j+k even } in a doubled lattice
                # Actually let's just use a dense SC lattice which approximates well
                d = np.sqrt(ix*ix + iy*iy + iz*iz)
                candidates.append((d, [ix*spacing, iy*spacing, iz*spacing]))
    candidates.sort(key=lambda x: x[0])
    for d, p in candidates:
        if len(pts) >= n:
            break
        pts.append(p)
    return np.array(pts[:n])


def _icosahedral(n, spacing=1.0):
    """Generate atoms in icosahedral shells (like icosahedral quasicrystal)."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0]) * spacing
    pts = [np.zeros(3)]
    if n <= 1:
        return np.array(pts[:n])
    pts.extend(verts.tolist())
    # Second shell: icosahedron of larger radius
    r2 = 2.0 * spacing
    verts2 = verts / spacing * r2
    for v in verts2:
        if len(pts) < n:
            pts.append(v.tolist())
    # Third shell
    r3 = 2.5 * spacing
    verts3 = verts / spacing * r3
    for v in verts3:
        if len(pts) < n:
            pts.append(v.tolist())
    # Fill remaining with random
    rng = np.random.default_rng(42)
    while len(pts) < n:
        p = rng.uniform(-r3, r3, 3)
        pts.append(p.tolist())
    return np.array(pts[:n])


def _random_compact(n, spacing=1.0, seed=0):
    """Random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    min_dist = 0.5 * spacing
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all, just add random
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])


def _basin_hopping(start_positions, potential, rho, n, time_budget=45.0, 
                   max_iter=200, rng=None):
    """Basin hopping: perturb, relax, keep if lower."""
    if rng is None:
        rng = np.random.default_rng(12345)
    
    best_energy, best_pos = relax(start_positions.copy(), potential, rho)
    t0 = time.time()
    current_energy, current_pos = best_energy, best_pos.copy()
    
    # Adaptive perturbation scale
    scale = 0.5
    accept_scale = 0.5
    
    for i in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        
        # Perturb
        perturbation = rng.normal(0, scale, size=current_pos.shape)
        perturbed = current_pos + perturbation
        
        # Relax
        try:
            e, pos = relax(perturbed, potential, rho)
        except Exception:
            scale *= 0.8
            continue
        
        if e < current_energy:
            current_energy, current_pos = e, pos
            scale *= 1.05
        else:
            # Metropolis-like acceptance for exploration
            delta_e = e - current_energy
            # Simple: accept with probability based on delta
            if rng.random() < np.exp(-delta_e / 0.1):
                current_pos = pos
                current_energy = e
            scale *= 0.95
        
        if current_energy < best_energy:
            best_energy, best_pos = current_energy, current_pos.copy()
        
        # Occasionally reset scale
        if i % 50 == 0 and i > 0:
            scale = max(0.1, min(1.0, scale))
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster via structured seeds + basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    spacing = 1.0  # pair equilibrium distance in reduced units
    
    # Time budget per instance: ~45s (300s / 6 instances with margin)
    time_per_instance = 45.0
    t_start = time.time()
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. BCC-like lattice
    starts.append(_bcc_shell(size, spacing))
    
    # 2. FCC-like lattice (slightly different spacing)
    starts.append(_fcc_shell(size, spacing * 0.95))
    
    # 3. Icosahedral
    starts.append(_icosahedral(size, spacing))
    
    # 4. Random compact
    starts.append(_random_compact(size, spacing, seed=size))
    
    # 5. Another random with different seed
    starts.append(_random_compact(size, spacing, seed=size + 100))
    
    # 6. Spherical shell distribution
    rng = np.random.default_rng(size + 200)
    shell_pts = []
    # Inner core
    core_n = max(1, size // 4)
    core_r = spacing * (core_n ** (1/3)) * 0.5
    for _ in range(core_n):
        p = rng.uniform(-core_r, core_r, 3)
        shell_pts.append(p)
    # Outer shell
    outer_n = size - core_n
    outer_r = spacing * (size ** (1/3)) * 0.6
    for _ in range(outer_n):
        # Random on sphere
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2*np.pi)
        x = np.sqrt(1-z*z) * np.cos(theta) * outer_r
        y = np.sqrt(1-z*z) * np.sin(theta) * outer_r
        shell_pts.append(np.array([x, y, z*outer_r]))
    starts.append(np.array(shell_pts))
    
    # Remove duplicates and validate
    valid_starts = []
    for s in starts:
        if len(s) != size:
            s = s[:size]
        # Check minimum distance
        valid = True
        for i in range(min(20, size)):  # Quick check
            for j in range(i+1, min(20, size)):
                if np.linalg.norm(s[i] - s[j]) < 0.4:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            valid_starts.append(s)
    
    if not valid_starts:
        valid_starts = [starts[0]]
    
    # Allocate time across starts
    n_starts = len(valid_starts)
    time_per_start = time_per_instance / n_starts * 0.8  # 20% margin
    
    best_energy = np.inf
    best_pos = None
    
    for idx, start in enumerate(valid_starts):
        remaining = time_per_instance - (time.time() - t_start)
        if remaining < 2.0:
            break
        
        # Quick local relaxation first
        try:
            e0, pos0 = relax(start.copy(), potential, rho)
        except Exception:
            continue
        
        if e0 < best_energy:
            best_energy = e0
            best_pos = pos0.copy()
        
        # Basin hopping from this start
        bh_time = min(time_per_start, remaining - 1.0)
        if bh_time < 1.0:
            continue
        
        try:
            e_bh, pos_bh = _basin_hopping(
                pos0, potential, rho, size, 
                time_budget=bh_time,
                max_iter=100,
                rng=np.random.default_rng(idx * 1000 + size)
            )
        except Exception:
            continue
        
        if e_bh < best_energy:
            best_energy = e_bh
            best_pos = pos_bh.copy()
    
    # Final: one more basin hopping from the best found
    remaining = time_per_instance - (time.time() - t_start)
    if remaining > 3.0 and best_pos is not None:
        try:
            e_final, pos_final = _basin_hopping(
                best_pos, potential, rho, size,
                time_budget=min(remaining - 1.0, 15.0),
                max_iter=50,
                rng=np.random.default_rng(99999)
            )
            if e_final < best_energy:
                best_energy = e_final
                best_pos = pos_final.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback
        best_pos = starts[0]
        try:
            _, best_pos = relax(best_pos.copy(), potential, rho)
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
