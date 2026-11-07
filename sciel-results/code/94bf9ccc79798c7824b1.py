"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_positions(n, spacing=0.72):
    """Generate positions based on icosahedral shells."""
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_vertices = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    ico_vertices = ico_vertices / norms
    
    # Build shells: center, icosahedron (12), then more shells
    positions = [np.zeros(3)]
    # icosahedron shell at radius r
    r1 = spacing * 0.9
    positions.extend(ico_vertices * r1)
    
    # If we need more, add a second shell (20 vertices of dodecahedron or more icosahedral points)
    if n > 13:
        # Add a second icosahedral shell scaled up
        r2 = spacing * 1.8
        positions.extend(ico_vertices * r2)
    
    if n > 25:
        r3 = spacing * 2.7
        positions.extend(ico_vertices * r3)
    
    if n > 37:
        r4 = spacing * 3.6
        positions.extend(ico_vertices * r4)
    
    if n > 49:
        r5 = spacing * 4.5
        positions.extend(ico_vertices * r5)
    
    if n > 61:
        r6 = spacing * 5.4
        positions.extend(ico_vertices * r6)
    
    if n > 73:
        r7 = spacing * 6.3
        positions.extend(ico_vertices * r7)
    
    if n > 85:
        r8 = spacing * 7.2
        positions.extend(ico_vertices * r8)
    
    # Trim or pad to exact n
    if len(positions) >= n:
        return np.array(positions[:n])
    else:
        # Pad with random points near the surface
        rng = np.random.default_rng(42)
        while len(positions) < n:
            angle1 = rng.uniform(0, 2 * np.pi)
            angle2 = rng.uniform(0, np.pi)
            r = spacing * n ** (1/3) * 0.5
            p = r * np.array([np.sin(angle2) * np.cos(angle1),
                              np.sin(angle2) * np.sin(angle1),
                              np.cos(angle2)])
            positions.append(p)
        return np.array(positions)


def _fcc_positions(n, spacing=0.72):
    """Generate positions based on FCC lattice."""
    a = spacing * 1.2  # lattice parameter
    positions = []
    # Generate FCC points in a cubic region
    n_side = int(np.ceil((n / 4) ** (1/3))) + 2
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if len(positions) < n:
                        positions.append(p)
    
    if len(positions) >= n:
        return np.array(positions[:n])
    else:
        # Pad
        rng = np.random.default_rng(42)
        while len(positions) < n:
            p = rng.uniform(-n_side*a, n_side*a, 3)
            positions.append(p)
        return np.array(positions)


def _hcp_positions(n, spacing=0.72):
    """Generate positions based on HCP lattice."""
    a = spacing * 1.2
    c = a * np.sqrt(8/3)
    positions = []
    n_side = int(np.ceil((n / 2) ** (1/3))) + 2
    
    for layer in range(-n_side, n_side + 1):
        z = layer * c
        # AB stacking
        if layer % 2 == 0:
            offset = (0, 0)
        else:
            offset = (a/2, a*np.sqrt(3)/6)
        
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                x = i * a + offset[0]
                y = j * a * np.sqrt(3) + offset[1]
                if len(positions) < n:
                    positions.append(np.array([x, y, z]))
    
    if len(positions) >= n:
        return np.array(positions[:n])
    else:
        rng = np.random.default_rng(42)
        while len(positions) < n:
            p = rng.uniform(-n_side*a, n_side*a, 3)
            p[2] = rng.uniform(-n_side*c, n_side*c)
            positions.append(p)
        return np.array(positions)


def _random_positions(n, spacing=0.72, seed=0):
    """Generate random compact positions."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1/3) * 0.8
    positions = []
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in positions):
                positions.append(p)
        if len(positions) >= n:
            break
        # If we're stuck, just add a random point
        if len(positions) < n and rng.random() < 0.01:
            positions.append(p)
    return np.array(positions[:n])


def _basin_hopping_step(positions, kind, rng, step_size=0.3):
    """Perform a single basin hopping step: perturb then relax."""
    perturbed = positions + rng.normal(0, step_size, positions.shape)
    energy, relaxed = relax(perturbed, 'sutton_chen', kind)
    return energy, relaxed


def _simulated_annealing(positions, kind, rng, t_start=1.0, t_end=0.01, n_steps=50, step_size=0.2):
    """Simulated annealing with Metropolis acceptance."""
    current_energy, current_pos = sutton_chen_energy_gradient(positions, kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(n_steps):
        # Linear cooling
        t = t_start * (t_end / t_start) ** (step / max(n_steps - 1, 1))
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size * (1 + step / n_steps), current_pos.shape)
        new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        
        # Metropolis acceptance
        if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / max(t, 1e-10)):
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Global minima search for Sutton-Chen clusters using multiple strategies."""
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    spacing = 0.72
    
    # Deadline guard
    deadline = time.time() + 280  # 280s out of 330s budget
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting configurations
    start_generators = [
        lambda: _icosahedron_positions(n, spacing),
        lambda: _fcc_positions(n, spacing),
        lambda: _hcp_positions(n, spacing),
        lambda: _random_positions(n, spacing, seed=0),
        lambda: _random_positions(n, spacing, seed=1),
        lambda: _random_positions(n, spacing, seed=2),
    ]
    
    # Phase 1: Relax all starting configurations
    for gen in start_generators:
        if time.time() > deadline - 30:
            break
        try:
            positions = gen()
            energy, relaxed = relax(positions, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        except Exception:
            continue
    
    # Phase 2: Basin hopping from best configuration
    if best_positions is not None and time.time() < deadline - 60:
        rng = np.random.default_rng(42)
        current_energy = best_energy
        current_pos = best_positions.copy()
        
        n_bh_steps = min(30, int((deadline - 60 - time.time()) / 5))
        for step in range(n_bh_steps):
            if time.time() > deadline - 30:
                break
            try:
                new_energy, new_pos = _basin_hopping_step(current_pos, kind, rng, step_size=0.25)
                if new_energy < current_energy:
                    current_energy = new_energy
                    current_pos = new_pos
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_positions = new_pos.copy()
            except Exception:
                continue
    
    # Phase 3: Simulated annealing from best configuration
    if best_positions is not None and time.time() < deadline - 45:
        rng = np.random.default_rng(123)
        try:
            sa_energy, sa_pos = _simulated_annealing(
                best_positions, kind, rng,
                t_start=0.5, t_end=0.01,
                n_steps=40, step_size=0.2
            )
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_positions = sa_pos
        except Exception:
            pass
    
    # Phase 4: Final polish with more basin hopping iterations
    if best_positions is not None and time.time() < deadline - 15:
        rng = np.random.default_rng(999)
        current_energy = best_energy
        current_pos = best_positions.copy()
        
        n_final = min(20, int((deadline - 15 - time.time()) / 5))
        for step in range(n_final):
            if time.time() > deadline - 5:
                break
            try:
                new_energy, new_pos = _basin_hopping_step(current_pos, kind, rng, step_size=0.15)
                if new_energy < current_energy:
                    current_energy = new_energy
                    current_pos = new_pos
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_positions = new_pos.copy()
            except Exception:
                continue
    
    # Validate result
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        # Fallback: return a simple valid configuration
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-2, 2, (n, 3))
        # Ensure minimum distance
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    best_positions[j] = best_positions[i] + 0.4 * rng.normal(0, 1, 3)
    
    return best_positions
# EVOLVE-BLOCK-END
