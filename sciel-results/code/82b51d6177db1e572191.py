"""Sutton-Chen transition-metal cluster global minima search using basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fib_sphere(n_pts, radius, rng):
    """Generate n_pts points roughly uniformly on a sphere of given radius."""
    if n_pts == 1:
        return np.array([[0.0, 0.0, radius]])
    if n_pts == 2:
        return np.array([[0.0, 0.0, radius], [0.0, 0.0, -radius]])
    
    # Fibonacci sphere
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y = np.linspace(-1.0, 1.0, n_pts)
    radius_per = np.sqrt(1.0 - y**2)
    theta = golden_angle * np.arange(n_pts)
    
    x = radius_per * np.cos(theta)
    z = radius_per * np.sin(theta)
    
    points = np.column_stack([x, y, z]) * radius
    return points


def _icosahedron_positions(n, spacing=0.72, seed=0):
    """Generate initial positions based on icosahedral shells."""
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (12 vertices)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_vertices = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    points = []
    # Add center atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Add icosahedron vertices at first shell
    shell_radius = spacing * 1.0
    for v in ico_vertices:
        points.append(v * shell_radius)
    
    # If we need more atoms, add outer shells using Fibonacci sphere
    if n > 13:
        remaining = n - len(points)
        # Add atoms in outer shells
        shell2_radius = spacing * 1.5
        n_shell2 = min(remaining, 30)
        if n_shell2 > 0:
            shell2_pts = _fib_sphere(n_shell2, shell2_radius, rng)
            for p in shell2_pts:
                points.append(p)
                remaining -= 1
        
        if remaining > 0:
            shell3_radius = spacing * 2.0
            shell3_pts = _fib_sphere(remaining, shell3_radius, rng)
            for p in shell3_pts:
                points.append(p)
    
    return np.array(points[:n])


def _fcc_positions(n, spacing=0.72, seed=0):
    """Generate initial positions based on FCC-like structure."""
    rng = np.random.default_rng(seed)
    
    points = []
    # Generate FCC lattice points
    a = spacing * 0.8
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
                for offset in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a, j*a, k*a]) + np.array(offset)
                    if len(points) < n:
                        points.append(p)
                    else:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    # Add random points if needed
    while len(points) < n:
        p = rng.uniform(-2*a, 2*a, 3)
        points.append(p)
    
    return np.array(points[:n])


def _random_compact(n, spacing=0.72, seed=0):
    """Generate random compact cluster positions."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 50:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
    
    # If we couldn't place all atoms, just add random ones
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _growth_initialization(n, spacing=0.72, seed=0):
    """Build cluster by adding atoms one at a time to minimize local energy."""
    rng = np.random.default_rng(seed)
    
    # Start with a small cluster
    points = [np.array([0.0, 0.0, 0.0])]
    
    for target_n in range(2, n + 1):
        best_pos = None
        best_energy = np.inf
        
        # Try adding atom at various positions
        for attempt in range(20):
            # Random position near existing cluster
            if attempt == 0:
                # Place at reasonable distance from center
                angle1 = rng.uniform(0, 2*np.pi)
                angle2 = rng.uniform(0, np.pi)
                r = spacing * 0.9
                p = np.array([
                    r * np.sin(angle2) * np.cos(angle1),
                    r * np.sin(angle2) * np.sin(angle1),
                    r * np.cos(angle2)
                ])
            else:
                # Random position
                p = rng.uniform(-spacing*2, spacing*2, 3)
            
            # Check minimum distance
            if any(np.linalg.norm(p - q) < 0.4 * spacing for q in points):
                continue
            
            new_points = points + [p]
            
            # Quick energy evaluation
            try:
                energy, _ = sutton_chen_energy_gradient(np.array(new_points), 'sutton_chen')
                if energy < best_energy and np.isfinite(energy):
                    best_energy = energy
                    best_pos = p
            except:
                pass
        
        if best_pos is not None:
            points.append(best_pos)
        else:
            # Fallback: add random position
            p = rng.uniform(-spacing, spacing, 3)
            points.append(p)
    
    return np.array(points[:n])


def _basin_hopping(positions, kind, n_exp, m_exp, c, max_iters=50, kick_size=0.3):
    """Perform basin hopping optimization."""
    current_positions = positions.copy()
    current_energy = np.inf
    
    try:
        current_energy, current_positions = relax(current_positions, 'sutton_chen', kind)
    except:
        return current_energy, current_positions
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    for i in range(max_iters):
        # Kick: perturb positions
        kick = np.random.normal(0, kick_size, current_positions.shape)
        kicked_positions = current_positions + kick
        
        # Relax
        try:
            kicked_energy, kicked_positions = relax(kicked_positions, 'sutton_chen', kind)
        except:
            kicked_energy = np.inf
            kicked_positions = current_positions
        
        # Metropolis acceptance
        if kicked_energy < current_energy:
            current_energy = kicked_energy
            current_positions = kicked_positions
        else:
            delta_e = kicked_energy - current_energy
            # Simple acceptance with decreasing kick size
            if np.random.random() < np.exp(-delta_e / 0.1):
                current_energy = kicked_energy
                current_positions = kicked_positions
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_positions.copy()
        
        # Adapt kick size
        kick_size *= 0.95
    
    return best_energy, best_positions


def construct(instance):
    """Construct cluster using basin hopping with multiple initialization strategies."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    n_exp = instance.get("n_exp", 12)
    m_exp = instance.get("m_exp", 6)
    c = instance.get("c", 144.41)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s margin
    
    # Spacing based on potential
    spacing = 0.72
    
    # Generate multiple initial configurations using different strategies
    initial_configs = []
    
    # Strategy 1: Icosahedral
    try:
        ico_pos = _icosahedron_positions(size, spacing, seed=size)
        initial_configs.append(ico_pos)
    except:
        pass
    
    # Strategy 2: FCC-like
    try:
        fcc_pos = _fcc_positions(size, spacing, seed=size + 1)
        initial_configs.append(fcc_pos)
    except:
        pass
    
    # Strategy 3: Random compact
    for seed in range(3):
        try:
            rand_pos = _random_compact(size, spacing, seed=size + 10 + seed)
            initial_configs.append(rand_pos)
        except:
            pass
    
    # Strategy 4: Growth initialization (if time permits)
    if time.time() < deadline - 60 and size <= 80:
        try:
            growth_pos = _growth_initialization(size, spacing, seed=size + 20)
            initial_configs.append(growth_pos)
        except:
            pass
    
    # Relax each initial configuration and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, pos in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        try:
            energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
            if energy < best_energy and np.isfinite(energy):
                best_energy = energy
                best_positions = relaxed_pos.copy()
        except:
            pass
    
    # If we have time, do basin hopping on the best result
    if best_positions is not None and time.time() < deadline - 40:
        try:
            bh_energy, bh_positions = _basin_hopping(
                best_positions, kind, n_exp, m_exp, c,
                max_iters=30, kick_size=0.25
            )
            if bh_energy < best_energy and np.isfinite(bh_energy):
                best_energy = bh_energy
                best_positions = bh_positions
        except:
            pass
    
    # Final relaxation to ensure we're at a local minimum
    if best_positions is not None and time.time() < deadline - 10:
        try:
            final_energy, final_positions = relax(best_positions, 'sutton_chen', kind)
            if np.isfinite(final_energy):
                best_positions = final_positions
        except:
            pass
    
    if best_positions is None:
        # Fallback: random positions
        rng = np.random.default_rng(size)
        best_positions = rng.uniform(-1, 1, (size, 3))
        try:
            _, best_positions = relax(best_positions, 'sutton_chen', kind)
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
