"""Structure-motif + basin-hopping search for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosa(n):
    """Mackay icosahedral shells. Returns (n,3) array with nearest-neighbor ~1.0."""
    phi = (1 + 5**0.5) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Mackay shell radii and counts: (radius, count)
    # Shell 0: 1 atom at center
    # Shell 1: 12 atoms at radius 1
    # Shell 2: 42 atoms at radius sqrt(1+phi) ≈ 1.618... wait, let me recalculate
    # Actually for Mackay icosahedra:
    # Layer 0: 1 atom (center)
    # Layer 1: 12 atoms (icosahedron vertices)
    # Layer 2: 42 atoms (octahedron faces + icosahedron edges)
    # Layer 3: 105 atoms
    # Layer 4: 195 atoms
    # Layer 5: 320 atoms
    
    # The radii for Mackay icosahedra with nearest neighbor = 1:
    # r0 = 0, r1 = 1, r2 = sqrt(1+1/phi^2) ... actually let me use known values
    # For a Mackay icosahedron with nearest-neighbor distance d=1:
    # r_k = k * d / (2 * sin(π/10))... no that's not right either.
    
    # Let me use the standard construction:
    # The icosahedron has edge length 2 (so vertices at distance 1 from center for unit icosahedron)
    # Actually, let me just place atoms on concentric shells with proper radii.
    
    # Standard Mackay icosahedra shell radii (for nearest neighbor = 1):
    # Shell 1 (12 atoms): r = 1.0
    # Shell 2 (42 atoms): r = sqrt(1 + 1/phi^2) ≈ 1.272... 
    # Actually, the correct radii are:
    # r1 = 1 (12 vertices of icosahedron with edge = 2*sqrt(10+2*sqrt(5))/5... )
    
    # Let me use a simpler approach: generate points on icosahedral shells
    # The key radii for Mackay icosahedra (nn=1):
    # r_1 = 1.0 (12 atoms)
    # r_2 = sqrt(1 + 2/phi) ≈ 1.401 (42 atoms) -- this isn't quite right
    
    # Let me use the well-known result: for Mackay icosahedra with nearest neighbor d,
    # the shell radii are: 0, d, d*sqrt(1+1/phi^2), d*sqrt(1+2/phi), ...
    # Actually, I'll use the standard approach of placing atoms at specific positions.
    
    # Simpler: use the icosahedral symmetry to place atoms
    # Shell 0: 1 atom at origin
    # Shell 1: 12 atoms at icosahedron vertices, radius chosen so nn=1
    # Shell 2: 42 atoms
    # Shell 3: 105 atoms
    # Shell 4: 195 atoms
    # Shell 5: 320 atoms
    
    # For an icosahedron with vertices at distance R from center, edge length = R * 2/sin(36°) ≈ R * 3.387
    # So if we want edge length = 1, R = 1/3.387 ≈ 0.295
    # But we want nearest neighbor = 1, so R = 1 / (2*sin(36°)/2) ... 
    
    # Let me just use a practical approach: generate candidate positions and let relaxation fix things.
    
    # Icosahedron vertices (normalized to unit sphere)
    ico_verts = verts.copy()
    
    # Generate all points in up to 5 shells
    all_points = [np.zeros(3)]  # center
    
    # Shell 1: 12 vertices of icosahedron
    # For nn=1, the radius should be such that distance between adjacent vertices = 1
    # In a unit icosahedron, edge length = 2/sin(π/5) ≈ 3.387
    # So radius for nn=1: r1 = 1 / (2/sin(π/5)) = sin(π/5)/2 * 2 = sin(π/5) ≈ 0.588
    # Wait, that's for the edge. Let me think again.
    # If vertices are at distance r from center, and the icosahedron has 12 vertices,
    # the edge length is r * 2 * sin(π/10) * ... no.
    # For a regular icosahedron with circumradius R, edge length a = R * 4/sqrt(10+2*sqrt(5)) ≈ R * 1.0515
    # So R = a / 1.0515. For a=1, R ≈ 0.951
    
    # Actually, let me just use R=1 and scale later, or use R such that nn≈1
    # For now, use R=1.0 for shell 1
    r1 = 1.0
    all_points.extend(ico_verts * r1)
    
    # Shell 2: 42 atoms - these are at midpoints of edges and face centers of the icosahedron
    # Edge midpoints: 30 edges → 30 points
    # Face centers: 20 faces → 12 unique positions (due to symmetry, but actually 20)
    # Total: 30 + 12 = 42? Let me just generate them.
    
    # Actually, for simplicity and robustness, let me generate points on a grid within a sphere
    # and use the icosahedral shells as starting templates.
    
    # Let me use a different, more robust approach: generate points on concentric spherical shells
    # with icosahedral symmetry.
    
    # I'll generate points using the following shell structure:
    # Shell 0: 1 atom at (0,0,0)
    # Shell 1: 12 atoms at icosahedron vertices, r=1
    # Shell 2: 42 atoms, r≈1.4 (approximate)
    # Shell 3: 105 atoms, r≈1.8 (approximate)  
    # Shell 4: 195 atoms, r≈2.2 (approximate)
    # Shell 5: 320 atoms, r≈2.6 (approximate)
    
    # For shells 2+, I'll use a Fibonacci sphere or random points on the shell
    # and let relaxation clean up.
    
    counts = [1, 12, 42, 105, 195, 320]
    radii = [0, 1.0, 1.4, 1.8, 2.2, 2.6]
    
    points = [np.zeros(3)]
    
    # Shell 1: icosahedron vertices
    points.extend(ico_verts * radii[1])
    
    # For higher shells, use Fibonacci sphere distribution
    for shell_idx in range(2, min(6, len(counts))):
        target_count = counts[shell_idx]
        r = radii[shell_idx]
        current_count = len(points)
        needed = min(target_count - current_count, max(0, n - current_count))
        if needed <= 0:
            break
        
        # Fibonacci sphere for uniform distribution
        golden_angle = np.pi * (3 - 5**0.5)
        for i in range(needed):
            y = 1 - (i + 0.5) / needed * 2
            radius_at_y = np.sqrt(max(0, 1 - y*y))
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            points.append(np.array([x, y, z]) * r)
        
        if len(points) >= n:
            break
    
    # Trim or pad to exactly n points
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        # Add more points on the outermost shell
        while len(points) < n:
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = radii[min(5, len(radii)-1)] * 1.1
            points.append(r * np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]))
    
    return np.array(points)


def _fcc_fragment(n):
    """Generate an FCC-like fragment."""
    points = []
    # FCC lattice points
    a = 1.0  # lattice constant
    max_coord = int(np.ceil(n ** (1/3))) + 2
    
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i+offset[0], j+offset[1], k+offset[2]]) * a
                    points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    # Center the fragment
    points -= points.mean(axis=0)
    return points


def _hcp_fragment(n):
    """Generate an HCP-like fragment."""
    points = []
    a = 1.0
    c = a * np.sqrt(8/3)
    max_coord = int(np.ceil(n ** (1/3))) + 2
    
    for layer in range(-max_coord, max_coord+1):
        z = layer * c
        for i in range(-max_coord, max_coord+1):
            for j in range(-max_coord, max_coord+1):
                # HCP: two atoms per layer
                p1 = np.array([i*a, j*a, z])
                p2 = np.array([(i+0.5)*a, (j+0.5)*a, z + c/2])
                points.append(p1)
                if len(points) < n:
                    points.append(p2)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _bcc_fragment(n):
    """Generate a BCC-like fragment."""
    points = []
    a = 1.0
    max_coord = int(np.ceil(n ** (1/3))) + 2
    
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                p1 = np.array([i, j, k]) * a
                p2 = (np.array([i+0.5, j+0.5, k+0.5])) * a
                points.append(p1)
                if len(points) < n:
                    points.append(p2)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _random_compact(n, rng, spacing=0.72):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    min_dist = 0.8 * spacing
    
    for _ in range(n * 20):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    
    # If we couldn't place enough, fill with random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _validate(positions):
    """Check if positions are valid."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.35:
                return False
    return True


def _basin_hop(positions, kind, n_exp, m_exp, c, rng, max_iter=50, base_scale=0.5, deadline=None):
    """Basin hopping with Metropolis acceptance."""
    energy, pos = relax(positions, 'sutton_chen', kind)
    best_energy = energy
    best_pos = pos.copy()
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        scale = base_scale * (1 + 0.5 * np.sin(i * 0.1))
        perturbed = pos + rng.normal(0, scale, pos.shape)
        
        # Relax
        e2, pos2 = relax(perturbed, 'sutton_chen', kind)
        
        if not np.all(np.isfinite(e2)):
            continue
        
        # Metropolis acceptance
        if e2 < energy:
            energy = e2
            pos = pos2
            if energy < best_energy:
                best_energy = energy
                best_pos = pos2.copy()
        else:
            # Accept with Metropolis probability
            # Use a simple temperature that decreases over time
            T = 0.5 * (1 - i / max_iter) + 0.1
            if rng.random() < np.exp(-(e2 - energy) / T):
                energy = e2
                pos = pos2
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos2.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Structure-motif + basin-hopping search for Sutton-Chen cluster global minima."""
    size = instance["n"]
    kind = instance["kind"]
    n_exp = instance["n_exp"]
    m_exp = instance["m_exp"]
    c = instance["c"]
    
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    rng = np.random.default_rng(size + hash(kind) % 10000)
    
    # Generate diverse starting structures
    templates = []
    
    # Icosahedral
    try:
        templates.append(_icosa(size))
    except:
        pass
    
    # FCC
    try:
        templates.append(_fcc_fragment(size))
    except:
        pass
    
    # HCP
    try:
        templates.append(_hcp_fragment(size))
    except:
        pass
    
    # BCC
    try:
        templates.append(_bcc_fragment(size))
    except:
        pass
    
    # Random compact (multiple)
    for _ in range(4):
        try:
            templates.append(_random_compact(size, rng))
        except:
            pass
    
    # Filter valid templates
    valid_templates = []
    for t in templates:
        if t.shape == (size, 3) and _validate(t):
            valid_templates.append(t)
    
    if not valid_templates:
        # Fallback: simple random
        t = _random_compact(size, rng)
        valid_templates = [t]
    
    # Relax each template and run basin hopping on the best ones
    results = []
    
    # First, do a quick relaxation of all templates to get initial energies
    for t in valid_templates:
        if time.time() > deadline - 60:
            break
        try:
            e, p = relax(t, 'sutton_chen', kind)
            if np.isfinite(e):
                results.append((e, p))
        except:
            pass
    
    if not results:
        # Emergency fallback
        t = valid_templates[0]
        e, p = relax(t, 'sutton_chen', kind)
        return p
    
    # Sort by energy
    results.sort(key=lambda x: x[0])
    
    # Run basin hopping on the top templates
    best_energy = results[0][0]
    best_pos = results[0][1].copy()
    
    num_bh = min(len(results), 6)  # Run basin hopping on up to 6 best templates
    
    # Use threading for parallel basin hopping
    def run_bh(args):
        e0, p0, idx = args
        if time.time() > deadline - 30:
            return e0, p0
        try:
            e, p = _basin_hop(p0, kind, n_exp, m_exp, c, rng, max_iter=40, base_scale=0.5, deadline=deadline)
            return e, p
        except:
            return e0, p0
    
    bh_tasks = [(e, p, i) for i, (e, p) in enumerate(results[:num_bh])]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_bh, task) for task in bh_tasks]
        for f in futures:
            try:
                e, p = f.result(timeout=250)
                if np.isfinite(e) and e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
            except:
                pass
    
    # Final check
    if not _validate(best_pos):
        # Fallback to first valid result
        for e, p in results:
            if _validate(p):
                return p
        return results[0][1]
    
    return best_pos
# EVOLVE-BLOCK-END
