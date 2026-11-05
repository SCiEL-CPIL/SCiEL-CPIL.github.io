# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron (unit edge length)."""
    phi = (1 + np.sqrt(5)) / 2
    pts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit edge length
    edge = np.linalg.norm(pts[0] - pts[1])
    pts = pts / edge
    return pts


def _make_icosahedral_cluster(n, rng):
    """Build a cluster by adding icosahedral shells."""
    ico = _icosahedron_vertices()
    # Shell radii and counts for icosahedral packing
    # Center: 1, then 12, then 30, then 55, etc.
    shells = [
        (1, 0.0),       # center
        (12, 1.0),      # first shell
        (30, 1.528),    # second shell (approx)
        (55, 1.932),    # third shell (approx)
        (91, 2.302),    # fourth shell (approx)
    ]
    
    points = []
    # Center atom
    points.append(np.zeros(3))
    remaining = n - 1
    
    # Add icosahedral shells
    for count, radius in shells[1:]:
        if remaining <= 0:
            break
        take = min(count, remaining)
        # Select subset of icosahedron vertices (with duplicates for larger counts)
        # For shell 2 (30 atoms): use octahedral + icosahedral directions
        if count == 12:
            shell_pts = ico
        elif count == 30:
            # 30-atom shell: 12 ico + 8 cube + 10 more
            cube = np.array([
                [1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],
                [-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]
            ], dtype=float)
            cube = cube / np.linalg.norm(cube[0])
            # 10 additional: edge midpoints of icosahedron
            edges = [(0,1),(0,3),(0,8),(0,11),(1,2),(1,4),(1,9),(2,5),(2,10),(3,6),(3,7),(4,5),(4,12),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,0)]
            # Use simpler: just scale ico and cube
            shell_pts = np.vstack([ico, cube])
            # Need 30, we have 20, add 10 more
            # Use face centers of icosahedron approx
            extra = rng.uniform(-1, 1, (10, 3))
            extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
            shell_pts = np.vstack([shell_pts, extra])
        else:
            # For larger shells, use random directions on sphere
            shell_pts = rng.uniform(-1, 1, (count, 3))
            shell_pts = shell_pts / np.linalg.norm(shell_pts, axis=1, keepdims=True)
        
        # Select subset
        idx = rng.choice(len(shell_pts), size=take, replace=False)
        for i in idx:
            p = shell_pts[i] * radius
            points.append(p)
        remaining -= take
    
    # If still need more atoms, add random points on a larger shell
    while remaining > 0:
        p = rng.uniform(-1, 1, 3)
        p = p / np.linalg.norm(p) * 2.5
        points.append(p)
        remaining -= 1
    
    return np.array(points[:n])


def _make_decahedral_cluster(n, rng):
    """Build a decahedral cluster (5 icosahedra sharing a common edge)."""
    ico = _icosahedron_vertices()
    points = [np.zeros(3)]
    remaining = n - 1
    
    # 5 icosahedral tips
    # Directions for 5-fold symmetry
    phi = (1 + np.sqrt(5)) / 2
    tips = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0]
    ], dtype=float)
    tips = tips / np.linalg.norm(tips, axis=1, keepdims=True)
    
    # Add tip atoms
    for tip in tips:
        if remaining <= 0:
            break
        points.append(tip * 1.5)
        remaining -= 1
    
    # Add atoms around each tip
    for tip in tips:
        if remaining <= 0:
            break
        # Add 5 atoms around each tip in a pentagonal arrangement
        for k in range(5):
            if remaining <= 0:
                break
            angle = 2 * np.pi * k / 5
            # Create perpendicular vectors
            up = np.array([0, 0, 1.0])
            if np.abs(np.dot(tip, up)) > 0.9:
                up = np.array([1, 0, 0.0])
            perp1 = np.cross(tip, up)
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(tip, perp1)
            
            offset = (perp1 * np.cos(angle) + perp2 * np.sin(angle)) * 0.8
            p = tip * 1.2 + offset
            points.append(p)
            remaining -= 1
    
    # Fill remaining with random points
    while remaining > 0:
        p = rng.uniform(-1, 1, 3)
        p = p / np.linalg.norm(p) * 2.0
        points.append(p)
        remaining -= 1
    
    return np.array(points[:n])


def _make_fcc_fragment(n, rng):
    """Build an FCC fragment cluster."""
    points = []
    # Generate FCC lattice points
    a = 1.0  # lattice constant
    max_coord = int(np.ceil(n ** (1/3))) + 2
    
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i*dx + j*dy + k*dz, i*dy + j*0 + k*0.5, i*0 + j*0.5 + k*0])
                    # Simpler: just use standard FCC
                    pass
    
    # Simpler approach: generate FCC points directly
    fcc_offsets = np.array([
        [0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]
    ])
    
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for off in fcc_offsets:
                    p = base + off * a
                    r = np.linalg.norm(p)
                    if r <= max_coord * a * 0.8:
                        candidates.append(p)
    
    candidates = np.array(candidates)
    if len(candidates) < n:
        # Add more
        while len(candidates) < n:
            p = rng.uniform(-3, 3, 3)
            candidates = np.vstack([candidates, [p]])
    
    # Select n closest to center
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:n]
    points = candidates[idx]
    
    # Add small random perturbation
    points = points + rng.normal(0, 0.01, points.shape)
    
    return points


def _make_random_cluster(n, rng, spacing=1.0):
    """Generate random cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = [np.zeros(3)]
    
    for _ in range(n - 1):
        for attempt in range(100):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                    points.append(p)
                    break
        else:
            # Relax constraint
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    """Find low-energy Morse cluster configurations using diverse structure generation + relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 290  # 290s safety margin (300s total, leave 10s)
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse candidate structures
    generators = [
        ("ico", lambda: _make_icosahedral_cluster(size, rng)),
        ("deca", lambda: _make_decahedral_cluster(size, rng)),
        ("fcc", lambda: _make_fcc_fragment(size, rng)),
        ("rand1", lambda: _make_random_cluster(size, rng, spacing=1.0)),
        ("rand2", lambda: _make_random_cluster(size, rng, spacing=0.9)),
    ]
    
    # Try each generator type multiple times
    max_attempts = 12
    attempt = 0
    
    while attempt < max_attempts:
        if time.time() > deadline:
            break
        
        gen_name, gen_func = generators[attempt % len(generators)]
        try:
            positions = gen_func()
            
            # Ensure minimum distance
            # Simple check and fix
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < 0.5:
                        # Push apart
                        mid = (positions[i] + positions[j]) / 2
                        diff = positions[i] - positions[j]
                        if np.linalg.norm(diff) > 1e-10:
                            diff = diff / np.linalg.norm(diff)
                        else:
                            diff = np.array([1.0, 0, 0])
                        positions[i] = mid + diff * 0.25
                        positions[j] = mid - diff * 0.25
            
            energy, relaxed = relax(positions, potential, rho)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
            
            attempt += 1
            
        except Exception:
            attempt += 1
            continue
    
    # If we have time, do additional random restarts
    while time.time() < deadline - 30:
        try:
            positions = _make_random_cluster(size, rng, spacing=1.0)
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < 0.5:
                        mid = (positions[i] + positions[j]) / 2
                        diff = positions[i] - positions[j]
                        if np.linalg.norm(diff) > 1e-10:
                            diff = diff / np.linalg.norm(diff)
                        else:
                            diff = np.array([1.0, 0, 0])
                        positions[i] = mid + diff * 0.25
                        positions[j] = mid - diff * 0.25
            
            energy, relaxed = relax(positions, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
                
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: simple random
        best_positions = _make_random_cluster(size, rng, spacing=1.0)
    
    return best_positions
# EVOLVE-BLOCK-END
