"""Dzugutov cluster global minima search using Mackay icosahedron shell optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 295.0  # Safety margin before 300s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _dzugutov_energy(pos):
    """Compute Dzugutov potential energy."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-12:
                continue
            if r < a:
                E += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                E += B * np.exp(d / (r - b))
    return E


def _get_icosahedron_vertices(scale=1.0):
    """Get 12 vertices of regular icosahedron scaled to given radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * scale
    return verts


def _get_dodecahedron_vertices(scale=1.0):
    """Get 20 vertices of regular dodecahedron scaled to given radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * scale
    return verts


def _get_cuboctahedron_vertices(scale=1.0):
    """Get 12 vertices of cuboctahedron."""
    verts = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * scale
    return verts


def _mckay_icosahedron(n, base_scale=1.0):
    """Generate Mackay icosahedron with n atoms.
    
    Mackay icosahedra have shells:
    - Shell 0: 1 atom at center (for n >= 1)
    - Shell 1: 12 atoms at icosahedron vertices (radius r1)
    - Shell 2: 42 atoms (20 dodecahedron + 12 icosahedron + 10 extra) (radius r2)
    - Shell 3: 100 atoms (radius r3)
    
    For our purposes, we use icosahedral shells with 12 atoms each plus center.
    """
    rng = np.random.default_rng(42)
    points = []
    
    # Determine shell structure
    # For Dzugutov, the key is concentric icosahedral shells
    # Typical shell radii ratios: 1, 2, 3 (Mackay) or optimized
    
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Center atom
    points.append([0.0, 0.0, 0.0])
    remaining = n - 1
    
    # Shell 1: 12 icosahedron vertices
    if remaining >= 12:
        r1 = base_scale * 1.0
        shell1 = _get_icosahedron_vertices(r1)
        points.extend(shell1.tolist())
        remaining -= 12
    
    # Shell 2: 42 atoms (20 dodeca + 12 ico + 10 mid-edges)
    if remaining >= 42:
        r2 = base_scale * 2.0
        # 20 dodecahedron vertices
        shell2a = _get_dodecahedron_vertices(r2)
        # 12 icosahedron vertices (same angular positions as shell 1)
        shell2b = _get_icosahedron_vertices(r2)
        # 10 mid-edge points (midpoints of icosahedron edges projected to sphere)
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = _get_icosahedron_vertices(1.0)
        # 30 edges of icosahedron, take 10 representative midpoints
        # Actually for 42 = 20 + 12 + 10, the 10 are face centers
        # Face centers of icosahedron (20 faces) -> use 10 pairs
        # Simpler: just add 10 more points at r2
        extra = []
        # Use dodecahedron mid-edges or just random on sphere
        for _ in range(10):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r2
            extra.append(p)
        points.extend(shell2a.tolist())
        points.extend(shell2b.tolist())
        points.extend(extra)
        remaining -= 42
    
    # Shell 3: 100 atoms
    if remaining >= 100:
        r3 = base_scale * 3.0
        # 100 points: use combination of shells
        # 20 dodeca + 12 ico + 20 dodeca (scaled) + 12 ico (scaled) + 36 more
        shell3a = _get_dodecahedron_vertices(r3)
        shell3b = _get_icosahedron_vertices(r3)
        shell3c = _get_dodecahedron_vertices(r3 * 0.95)
        shell3d = _get_icosahedron_vertices(r3 * 0.95)
        points.extend(shell3a.tolist())
        points.extend(shell3b.tolist())
        points.extend(shell3c.tolist())
        points.extend(shell3d.tolist())
        # Need 36 more
        for _ in range(36):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r3
            points.append(p)
        remaining -= 100
    
    # Fill remaining with random points in outer shell
    while len(points) < n:
        r_outer = base_scale * (1 + len(points) / 12.0 * 0.5)
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r_outer
        points.append(p)
    
    return np.array(points[:n])


def _icosahedral_shell_assembly(n, shell_radii):
    """Generate cluster from concentric icosahedral shells with given radii.
    
    shell_radii: list of (radius, count) tuples, where count is number of atoms on that shell.
    Atoms are placed at icosahedral vertices (12 per shell) or distributed.
    """
    rng = np.random.default_rng(12345)
    points = []
    
    # Center atom if space
    if n >= 1 and len(shell_radii) > 0:
        points.append([0.0, 0.0, 0.0])
        n_left = n - 1
    else:
        n_left = n
    
    for radius, count in shell_radii:
        if n_left <= 0:
            break
        actual_count = min(count, n_left)
        
        if actual_count <= 12:
            # Use icosahedron vertices
            ico = _get_icosahedron_vertices(radius)
            points.extend(ico[:actual_count].tolist())
        elif actual_count <= 24:
            # Two icosahedron shells
            ico1 = _get_icosahedron_vertices(radius)
            ico2 = _get_icosahedron_vertices(radius * 0.95)
            points.extend(ico1.tolist())
            points.extend(ico2[:actual_count - 12].tolist())
        else:
            # Mix of icosahedron, dodecahedron, and random
            ico = _get_icosahedron_vertices(radius)
            dodeca = _get_dodecahedron_vertices(radius)
            points.extend(ico.tolist())
            points.extend(dodeca[:min(20, actual_count - 12)].tolist())
            remaining = actual_count - 12 - min(20, actual_count - 12)
            for _ in range(max(0, remaining)):
                theta = rng.uniform(0, 2 * np.pi)
                z = rng.uniform(-1, 1)
                r = np.sqrt(1 - z**2)
                p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
                points.append(p)
        
        n_left -= actual_count
    
    # Fill remaining
    while len(points) < n:
        r = shell_radii[-1][0] * 1.1 if shell_radii else 1.0
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r_comp = np.sqrt(1 - z**2)
        p = np.array([r_comp * np.cos(theta), r_comp * np.sin(theta), z]) * r
        points.append(p)
    
    return np.array(points[:n])


def _validate_positions(pos, min_dist=0.5):
    """Check minimum distance constraint."""
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < min_dist:
                return False
    return True


def _kick_and_relax(pos, kick_size, potential, parameter):
    """Add random kick and relax."""
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    kicked = pos + rng.normal(0, kick_size, pos.shape)
    energy, relaxed = relax(kicked, potential, parameter)
    return energy, relaxed


def _basin_hopping(pos, potential, parameter, n_iter=5, kick_size=0.2):
    """Simple basin hopping: kick, relax, accept if lower energy."""
    best_energy = _dzugutov_energy(pos)
    best_pos = pos.copy()
    current_energy = best_energy
    current_pos = pos.copy()
    rng = np.random.default_rng(777)
    
    for _ in range(n_iter):
        if _time_remaining() < 20:
            break
        kicked = current_pos + rng.normal(0, kick_size, current_pos.shape)
        try:
            e, relaxed = relax(kicked, potential, parameter)
        except:
            continue
        if e < current_energy:
            current_energy = e
            current_pos = relaxed
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        else:
            # Metropolis-like acceptance
            if rng.random() < np.exp(-(e - current_energy) / 0.5):
                current_energy = e
                current_pos = relaxed
    
    return best_energy, best_pos


def _evolve_shell_radii(n, potential, parameter, n_gen=10, pop_size=6):
    """Evolutionary search over shell radius configurations."""
    rng = np.random.default_rng(42)
    
    # Determine number of shells based on n
    # For n atoms, typical Mackay: 1 + 12 + 42 + 100 = 155
    # For our range (38-60): 1 + 12 + 42 = 55, so up to 3 shells
    n_shells = min(4, max(2, n // 12))
    
    # Initial population of shell radius configurations
    # Each config: list of shell radii
    base_scale = 1.0
    pop = []
    for i in range(pop_size):
        # Random shell radii, increasing
        radii = []
        r = base_scale * rng.uniform(0.8, 1.2)
        for _ in range(n_shells):
            radii.append(r)
            r *= rng.uniform(1.5, 2.5)
        pop.append(radii)
    
    # Also add known good Mackay-like
    pop.append([base_scale * 1.0, base_scale * 2.0, base_scale * 3.0][:n_shells])
    pop.append([base_scale * 0.9, base_scale * 1.8, base_scale * 2.7][:n_shells])
    
    best_energy = np.inf
    best_pos = None
    
    for gen in range(n_gen):
        if _time_remaining() < 30:
            break
        
        for radii in pop:
            if _time_remaining() < 30:
                break
            
            # Generate cluster from this shell config
            shell_counts = []
            remaining = n
            # Center
            shell_counts.append((0.0, 1))
            remaining -= 1
            for i, r in enumerate(radii):
                if remaining <= 0:
                    break
                # Distribute remaining atoms across shells
                if i == len(radii) - 1:
                    count = remaining
                else:
                    count = min(12 * (i + 1), remaining // (len(radii) - i))
                shell_counts.append((r, count))
                remaining -= count
            
            # Adjust to exactly n
            total = sum(c for _, c in shell_counts)
            if total < n:
                shell_counts[-1] = (shell_counts[-1][0], shell_counts[-1][1] + (n - total))
            elif total > n:
                # Trim from last shell
                diff = total - n
                last_r, last_c = shell_counts[-1]
                shell_counts[-1] = (last_r, max(1, last_c - diff))
            
            pos = _icosahedral_shell_assembly(n, shell_counts)
            
            if not _validate_positions(pos):
                # Try to fix
                continue
            
            energy, relaxed = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        
        # Evolve population
        if len(pop) < 2:
            break
        
        # Sort by fitness (we need to track energies)
        # Simple: keep best, add mutants
        new_pop = [pop[0].copy()]  # Keep best
        for _ in range(pop_size - 1):
            parent = pop[rng.randint(len(pop))]
            mutant = parent.copy()
            # Mutate one radius
            idx = rng.randint(len(mutant))
            mutant[idx] *= rng.uniform(0.7, 1.3)
            # Ensure increasing
            for i in range(1, len(mutant)):
                if mutant[i] <= mutant[i-1] * 1.3:
                    mutant[i] = mutant[i-1] * rng.uniform(1.3, 2.0)
            new_pop.append(mutant)
        pop = new_pop
    
    return best_energy, best_pos


def construct(instance):
    """Construct low-energy Dzugutov cluster using Mackay icosahedron optimization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Mackay icosahedron with standard radii
    base_scale = 0.8
    mackay_pos = _mckay_icosahedron(size, base_scale=base_scale)
    if _validate_positions(mackay_pos):
        energy, positions = relax(mackay_pos, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 2: Evolved shell radii
    e_energy, e_pos = _evolve_shell_radii(size, potential, parameter, n_gen=8, pop_size=5)
    if e_energy < best_energy:
        best_energy = e_energy
        best_positions = e_pos
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 3: Basin hopping from best
    if best_positions is not None:
        bh_energy, bh_pos = _basin_hopping(best_positions, potential, parameter, n_iter=8, kick_size=0.15)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    if _time_remaining() < 30:
        return best_positions
    
    # Strategy 4: Different base scale
    base_scale2 = 1.0
    mackay_pos2 = _mckay_icosahedron(size, base_scale=base_scale2)
    if _validate_positions(mackay_pos2):
        energy, positions = relax(mackay_pos2, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    if _time_remaining() < 25:
        return best_positions
    
    # Strategy 5: Basin hopping with larger kicks
    if best_positions is not None:
        bh_energy2, bh_pos2 = _basin_hopping(best_positions, potential, parameter, n_iter=6, kick_size=0.3)
        if bh_energy2 < best_energy:
            best_energy = bh_energy2
            best_positions = bh_pos2
    
    if _time_remaining() < 20:
        return best_positions
    
    # Strategy 6: Random compact with icosahedral bias
    rng = np.random.default_rng(size + 100)
    radius = 1.2 * size ** (1/3) * 0.5
    points = []
    ico = _get_icosahedron_vertices(radius)
    # Start with icosahedron-like
    for v in ico:
        if len(points) < size:
            points.append(v)
    while len(points) < size:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius * 0.9
        # Check min distance
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    if points:
        pos = np.array(points[:size])
        if _validate_positions(pos):
            energy, positions = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
