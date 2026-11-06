"""Dzugutov cluster global-minima search with Mackay icosahedral seeds and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 285.0  # safety margin below 300s


def _remaining_time():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)


def _icosahedron_vertices(scale=1.0):
    """Return 12 vertices of a regular icosahedron scaled to have circumradius=scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, 1.0], [-phi, 0.0, -1.0],
    ], dtype=float)
    norm = np.linalg.norm(v[0])
    v = v / norm * scale
    return v


def _octahedron_vertices(scale=1.0):
    """Return 6 vertices of a regular octahedron scaled to have circumradius=scale."""
    v = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ], dtype=float)
    return v * scale


def _build_mackay_icosahedron(n_target, spacing=1.1, rng=None):
    """
    Build a Mackay icosahedral cluster with shells:
    - Shell 0: 1 atom (center)
    - Shell 1: 12 atoms (icosahedron)
    - Shell 2: 42 atoms (octahedron 6 + icosahedron face centers 20 + additional)
    - Shell 3: 100 atoms
    Total available: 1+12+42+100 = 155 atoms
    We trim to n_target by keeping closest to center.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    pts = []
    
    # Shell 0: center
    pts.append(np.zeros(3))
    
    # Shell 1: icosahedron at radius = spacing
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p.copy() for p in shell1])
    
    # Shell 2: 42 atoms
    # The second shell of a Mackay icosahedron consists of:
    # - 6 octahedral directions at radius ~ spacing * sqrt(5)/2 ≈ 1.118*spacing
    # - 20 icosahedral face-center directions at radius ~ spacing * sqrt(5)/2 * something
    # Actually, let's use the standard Mackay construction.
    # Shell 2 has 42 atoms. Let me construct them properly.
    
    # For a proper Mackay icosahedron, shell 2 contains:
    # 12 atoms along the icosahedral edge midpoints (at distance ~ spacing*1.5)
    # 6 atoms along octahedral directions (at distance ~ spacing*1.5)  
    # 20 atoms along icosahedral face center directions
    # 4 atoms along additional directions
    # Total: 42
    
    # Let me use a simpler approach: generate points on a sphere and pick 42 well-spread ones
    # Actually, the standard Mackay icosahedron shell 2 has:
    # - 6 at octahedral vertices (±x, ±y, ±z) at radius r2
    # - 12 at icosahedral edge midpoints at radius r2
    # - 20 at icosahedral face centers at radius r2
    # - 4 at additional points
    # = 42 total
    
    # Let me just use a golden ratio spiral to place 42 points on a sphere of radius ~1.5*spacing
    # and then refine. Actually, let me be more precise.
    
    # Standard Mackay icosahedron:
    # Shell 1: 12 atoms at icosahedral vertices, radius R1
    # Shell 2: 42 atoms, radius R2 ≈ R1 * sqrt(5)/2 ≈ 1.118*R1... no that's not right.
    
    # Let me look at this differently. The Mackay icosahedron I_1 has 13 atoms.
    # I_2 has 55 atoms (1+12+42).
    # I_3 has 147 atoms (1+12+42+100).
    
    # For I_2 (55 atoms), the shell 2 atoms are at specific positions.
    # Let me construct them using the known geometry.
    
    # Actually, let me use a different approach. I'll construct the 55-atom Mackay icosahedron
    # explicitly using known coordinates, then for other n values I'll trim from the center outward.
    
    # Shell 2 of Mackay icosahedron (42 atoms):
    # These are at the positions of the second coordination shell.
    # Let me generate them as:
    # 6 octahedral directions * scale2
    # 20 icosahedral face center directions * scale2  
    # 12 icosahedral edge midpoint directions * scale2
    # 4 additional directions * scale2
    
    # For simplicity and robustness, let me place shell 2 atoms on a sphere of radius 
    # ~1.6*spacing using a well-distributed set of 42 directions.
    
    # Golden angle spiral for 42 points
    n_pts = 42
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    shell2_radius = spacing * 1.618  # golden ratio spacing
    for i in range(n_pts):
        z = 1.0 - 2.0 * (i + 0.5) / n_pts
        r = np.sqrt(1.0 - z*z)
        theta = golden_angle * i
        pt = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell2_radius
        pts.append(pt)
    
    # Shell 3: 100 atoms for larger clusters
    if n_target > 55:
        n_pts3 = 100
        shell3_radius = spacing * 2.618
        for i in range(n_pts3):
            z = 1.0 - 2.0 * (i + 0.5) / n_pts3
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * i
            pt = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell3_radius
            pts.append(pt)
    
    pts = np.array(pts, dtype=float)
    
    if len(pts) >= n_target:
        # Keep n_target closest to center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:n_target]
        pts = pts[idx]
    else:
        # Need more atoms - add jittered points near existing
        while len(pts) < n_target:
            base_idx = rng.integers(0, len(pts))
            base = pts[base_idx]
            p = base + rng.normal(0, 0.3, 3)
            # Check minimum distance
            if len(pts) > 0:
                min_d = np.min(np.linalg.norm(p - pts, axis=1))
            else:
                min_d = np.inf
            if min_d > 0.55:
                pts = np.vstack([pts, p])
            else:
                # Try again with larger displacement
                p = base + rng.normal(0, 0.8, 3)
                if len(pts) > 0:
                    min_d = np.min(np.linalg.norm(p - pts, axis=1))
                else:
                    min_d = np.inf
                if min_d > 0.55:
                    pts = np.vstack([pts, p])
                else:
                    pts = np.vstack([pts, p])
    
    return pts


def _build_icosahedral_seed(n, spacing=1.1, rng=None):
    """
    Build an icosahedral/polytetrahedral seed for n atoms.
    Uses central atom + icosahedral shells.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    pts = [np.zeros(3)]
    
    # Shell 1: 12 atoms
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p.copy() for p in shell1])
    
    # Shell 2: 42 atoms (for n > 13)
    if n > 13:
        n_shell2 = min(42, n - 13)
        shell2_radius = spacing * 1.618
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_shell2):
            z = 1.0 - 2.0 * (i + 0.5) / 42.0
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * i
            pt = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell2_radius
            pts.append(pt)
    
    # Shell 3: additional atoms (for n > 55)
    if n > 55:
        n_shell3 = n - 55
        shell3_radius = spacing * 2.618
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_shell3):
            z = 1.0 - 2.0 * (i + 0.5) / max(100, n_shell3)
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * i
            pt = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell3_radius
            pts.append(pt)
    
    pts = np.array(pts, dtype=float)
    
    if len(pts) >= n:
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:n]
        pts = pts[idx]
    else:
        while len(pts) < n:
            base_idx = rng.integers(0, len(pts))
            base = pts[base_idx]
            p = base + rng.normal(0, 0.3, 3)
            if len(pts) > 0:
                min_d = np.min(np.linalg.norm(p - pts, axis=1))
            else:
                min_d = np.inf
            if min_d > 0.55:
                pts = np.vstack([pts, p])
            else:
                p = base + rng.normal(0, 0.8, 3)
                if len(pts) > 0:
                    min_d = np.min(np.linalg.norm(p - pts, axis=1))
                else:
                    min_d = np.inf
                if min_d > 0.55:
                    pts = np.vstack([pts, p])
                else:
                    pts = np.vstack([pts, p])
    
    return pts


def _build_bcc_seed(n, spacing=1.1, rng=None):
    """Build a BCC-like compact seed."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    pts = []
    # Generate BCC lattice points
    a = spacing * 1.2
    L = int(np.ceil((n * (4.0/3.0) * np.pi / (a**3)) ** (1.0/3.0))) + 2
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # Corners
                pts.append(np.array([i*a, j*a, k*a]))
                # Body center
                if i < L and j < L and k < L:
                    pts.append(np.array([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a]))
    
    pts = np.array(pts, dtype=float)
    # Keep closest to origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    pts = pts[idx]
    
    # Center the cluster
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    
    return pts


def _make_random_compact_seed(n, spacing=1.1, rng=None):
    """Generate a random compact seed."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    radius = spacing * n ** (1.0/3.0) * 0.85
    pts = []
    attempts = 0
    min_dist = 0.6 * spacing
    
    while len(pts) < n and attempts < n * 300:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    
    if len(pts) < n:
        for _ in range(n - len(pts)):
            base_idx = rng.integers(0, max(1, len(pts)))
            base = pts[base_idx] if len(pts) > 0 else np.zeros(3)
            p = base + rng.normal(0, 0.5, 3)
            pts.append(p)
    
    return np.array(pts, dtype=float)


def _basin_hopping_step(positions, step_size, rng, potential, parameter):
    """
    Perform one basin hopping step: perturb, then relax.
    Returns (energy, positions) of the relaxed perturbed structure.
    """
    n = len(positions)
    # Perturb
    perturbed = positions + rng.normal(0, step_size, (n, 3))
    
    try:
        energy, relaxed = relax(perturbed, potential, parameter)
        return energy, relaxed
    except Exception:
        return np.inf, positions


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Per-instance time budget
    instance_budget = min(45.0, _remaining_time() - 5.0)
    if instance_budget < 3.0:
        instance_budget = 3.0
    
    t0 = time.time()
    best_energy = np.inf
    best_positions = None
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Build diverse candidate seeds
    seeds = []
    
    # 1. Mackay icosahedral seeds with varied spacing
    for sp in (1.05, 1.10, 1.15, 1.08, 1.12):
        seeds.append(_build_icosahedral_seed(size, spacing=sp, rng=rng))
    
    # 2. BCC seed
    seeds.append(_build_bcc_seed(size, spacing=1.1, rng=rng))
    
    # 3. Random compact seeds
    for _ in range(4):
        seeds.append(_make_random_compact_seed(size, spacing=1.1, rng=rng))
    
    # Relax each seed and track best
    for seed in seeds:
        if (time.time() - t0) > instance_budget * 0.4:
            break
        try:
            energy, positions = relax(seed, potential, parameter)
        except Exception:
            continue
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # Basin hopping from the best seed
    if best_positions is not None:
        # Determine step size based on cluster size
        base_step = 0.15
        n_steps = max(15, min(60, int((instance_budget * 0.6) / 0.8)))
        
        current_energy = best_energy
        current_positions = best_positions.copy()
        
        for step in range(n_steps):
            if (time.time() - t0) > instance_budget * 0.85:
                break
            
            # Adaptive step size: start moderate, can decrease
            step_size = base_step * (1.0 + 0.5 * np.cos(step * 0.1))
            step_size = max(0.05, step_size)
            
            energy, positions = _basin_hopping_step(
                current_positions, step_size, rng, potential, parameter
            )
            
            if energy < current_energy:
                current_energy = energy
                current_positions = positions
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
    
    # Second round of basin hopping from a different seed if time allows
    if (time.time() - t0) < instance_budget * 0.7 and len(seeds) > 1:
        # Try basin hopping from the second best seed
        second_best_seed = seeds[1]
        try:
            e2, p2 = relax(second_best_seed, potential, parameter)
        except Exception:
            e2, p2 = np.inf, None
        
        if p2 is not None and e2 < np.inf:
            n_steps2 = max(10, min(40, int((instance_budget * 0.25) / 0.8)))
            current_energy2 = e2
            current_positions2 = p2.copy()
            
            for step in range(n_steps2):
                if (time.time() - t0) > instance_budget * 0.95:
                    break
                
                step_size = 0.15 * (1.0 + 0.5 * np.cos(step * 0.1))
                step_size = max(0.05, step_size)
                
                energy, positions = _basin_hopping_step(
                    current_positions2, step_size, rng, potential, parameter
                )
                
                if energy < current_energy2:
                    current_energy2 = energy
                    current_positions2 = positions
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
    
    # Final: if we still don't have a good position, do one more relaxation
    if best_positions is None:
        best_positions = _make_random_compact_seed(size, spacing=1.1, rng=rng)
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
    
    # Verify minimum distance constraint
    if best_positions is not None and len(best_positions) > 1:
        # Check and fix any violations
        min_required = 0.5
        for _fix in range(3):
            dists = np.linalg.norm(best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            if min_d >= min_required:
                break
            # Nudge atoms that are too close
            for i in range(len(best_positions)):
                for j in range(i+1, len(best_positions)):
                    d = np.linalg.norm(best_positions[i] - best_positions[j])
                    if d < min_required and d > 1e-10:
                        direction = (best_positions[i] - best_positions[j]) / d
                        shift = (min_required - d) / 2.0 + 0.01
                        best_positions[i] += direction * shift
                        best_positions[j] -= direction * shift
            # Re-relax if we made changes
            try:
                _, best_positions = relax(best_positions, potential, parameter)
            except Exception:
                pass
    
    return best_positions
# EVOLVE-BLOCK-END
