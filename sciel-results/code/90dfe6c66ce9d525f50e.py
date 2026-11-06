import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
POTENTIAL = 'dzugutov'
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None


def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin, unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _get_13_atom_icosahedron(radius=1.1):
    """13-atom icosahedron: center + 12 vertices at given radius."""
    verts = _get_icosahedron_vertices() * radius
    center = np.zeros((1, 3))
    return np.vstack([center, verts])


def _get_mackay_55():
    """
    Generate a 55-atom Mackay icosahedron.
    Structure: 1 center + 12 (shell 1) + 42 (shell 2).
    Shell 2 atoms are placed at:
      - 20 face centers of the icosahedron (scaled outward)
      - 30 edge midpoints (scaled outward)
      - but that gives 50, not 42.
    
    Actual Mackay 55: The 42 outer atoms form a truncated icosahedron-like shell.
    A practical approach: place 42 atoms on a sphere of radius ~1.8*r1 using
    icosahedral symmetry points (20 face centers + 22 additional from subdividing).
    
    Simpler robust method: use 13-atom icosahedron as core, then add 42 atoms
    via rejection sampling in a shell, then relax.
    """
    core = _get_13_atom_icosahedron(radius=1.1)
    rng = np.random.default_rng(42)
    
    atoms = core.copy()
    min_dist = 0.52
    r_inner = 1.1
    r_outer = 2.0
    
    max_attempts = 20000
    attempts = 0
    while len(atoms) < 55 and attempts < max_attempts:
        r = rng.uniform(r_inner, r_outer)
        theta = rng.uniform(0, 2 * np.pi)
        phi = np.arccos(rng.uniform(-1, 1))
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        dists = np.linalg.norm(atoms - p, axis=1)
        if np.min(dists) >= min_dist:
            atoms = np.vstack([atoms, p])
        attempts += 1
    
    # Relax to get proper local order
    try:
        e, pos = relax(atoms, POTENTIAL)
    except:
        pos = atoms
    return pos


def _fragment_merge_initialization(n_target, rng, n_fragments=3):
    """
    Fragment-merger strategy:
    1. Create n_fragments 13-atom icosahedra (or subsets for smaller n)
    2. Place them at vertices of a larger polyhedron
    3. Coarse-grained annealing: minimize inter-fragment repulsion
    4. Full relaxation
    """
    if n_target <= 13:
        ico = _get_13_atom_icosahedron(radius=1.1)
        if n_target == 13:
            try:
                e, pos = relax(ico, POTENTIAL)
                return pos
            except:
                return ico
        # For n < 13, just use random compact
        return _random_compact(n_target, rng)
    
    # Determine fragment sizes
    # For n=38: 3 fragments of ~13 (39 total, remove 1)
    # For n=45: 3 fragments of 15 (45)
    # For n=50: 4 fragments of ~13 (52, remove 2)
    # For n=55: 4 fragments of ~14 (56, remove 1) or 13+13+13+16
    # For n=57: 4 fragments of ~14 (56, add 1)
    # For n=60: 5 fragments of 12 (60)
    
    if n_target <= 40:
        n_frag = 3
        frag_sizes = [13, 13, 13]
        if n_target < 39:
            # Reduce sizes
            total = 39
            excess = total - n_target
            frag_sizes[0] -= min(excess, 5)
            if excess > 5:
                frag_sizes[1] -= min(excess - 5, 5)
                if excess > 10:
                    frag_sizes[2] -= excess - 10
    elif n_target <= 52:
        n_frag = 4
        frag_sizes = [13, 13, 13, 13]
        total = 52
        excess = total - n_target
        if excess > 0:
            for i in range(n_frag):
                if excess <= 0:
                    break
                reduce = min(excess, 3)
                frag_sizes[i] -= reduce
                excess -= reduce
    else:
        n_frag = 5
        frag_sizes = [12, 12, 12, 12, 12]
        total = 60
        deficit = n_target - total
        if deficit > 0:
            for i in range(n_frag):
                if deficit <= 0:
                    break
                add = min(deficit, 3)
                frag_sizes[i] += add
                deficit -= add
    
    # Generate fragments
    fragments = []
    for fs in frag_sizes:
        if fs <= 13:
            frag = _get_13_atom_icosahedron(radius=1.0)
            if fs < 13:
                # Remove outermost atoms
                dists = np.linalg.norm(frag, axis=1)
                sort_idx = np.argsort(dists)[::-1]
                remove_idx = sort_idx[:13 - fs]
                keep_idx = np.setdiff1d(np.arange(13), remove_idx)
                frag = frag[keep_idx]
        else:
            # Generate larger fragment via random compact + relax
            frag = _random_compact(fs, rng, min_radius=0.8, max_radius=1.8)
            try:
                e, frag = relax(frag, POTENTIAL)
            except:
                pass
        fragments.append(frag)
    
    # Place fragments at vertices of a polyhedron
    if n_frag == 3:
        # Triangle
        R = 2.5
        angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
        centers = np.column_stack([R * np.cos(angles), R * np.sin(angles), np.zeros(3)])
    elif n_frag == 4:
        # Tetrahedron
        R = 2.8
        phi = (1 + np.sqrt(5)) / 2
        centers = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ], dtype=float)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * R
    else:  # n_frag == 5
        # Truncated octahedron or just 5 points
        R = 3.0
        phi = (1 + np.sqrt(5)) / 2
        centers = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [0, 0, 0]
        ], dtype=float)
        # Normalize and scale
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        norms[4] = 1  # center stays at origin
        centers = centers / norms * R
    
    # Combine fragments with their centers
    all_atoms = []
    for i, frag in enumerate(fragments):
        rotated = frag + centers[i]
        all_atoms.append(rotated)
    
    combined = np.vstack(all_atoms)
    
    # Coarse-grained annealing: perturb inter-fragment positions
    for step in range(50):
        # Randomly rotate/translate fragments slightly
        for i in range(n_frag):
            start_idx = sum(frag_sizes[:i])
            end_idx = start_idx + frag_sizes[i]
            frag_atoms = combined[start_idx:end_idx]
            
            # Small random rotation
            angle = rng.uniform(-0.1, 0.1)
            axis = rng.normal(size=3)
            axis = axis / np.linalg.norm(axis)
            
            # Rodrigues rotation
            c, s = np.cos(angle), np.sin(angle)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R_mat = np.eye(3) + s * K + (1 - c) * (K @ K)
            
            combined[start_idx:end_idx] = frag_atoms @ R_mat.T + centers[i]
        
        # Relax
        try:
            e, combined = relax(combined, POTENTIAL)
        except:
            pass
    
    # Adjust to exact n_target
    if len(combined) != n_target:
        if len(combined) > n_target:
            dists = np.linalg.norm(combined, axis=1)
            sort_idx = np.argsort(dists)[::-1]
            remove_idx = sort_idx[:len(combined) - n_target]
            keep_idx = np.setdiff1d(np.arange(len(combined)), remove_idx)
            combined = combined[keep_idx]
        else:
            # Add atoms near surface
            while len(combined) < n_target:
                ref = combined[rng.integers(len(combined))]
                offset = rng.normal(0, 0.3, 3)
                p = ref + offset
                if np.min(np.linalg.norm(combined - p, axis=1)) >= 0.5:
                    combined = np.vstack([combined, p])
    
    try:
        e, combined = relax(combined, POTENTIAL)
    except:
        pass
    
    return combined


def _random_compact(n, rng, min_radius=0.5, max_radius=3.0):
    """Generate a random compact cluster with minimum distance constraint."""
    points = []
    radius = 0.8 * n ** (1/3) + 1.5
    for _ in range(n):
        for _ in range(2000):
            r = rng.uniform(min_radius, max_radius)
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            p = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            if len(points) == 0 or np.min(np.linalg.norm(np.array(points) - p, axis=1)) >= 0.52:
                points.append(p)
                break
        else:
            # Fallback: place at random position
            p = rng.uniform(-max_radius, max_radius, 3)
            points.append(p)
    return np.array(points)


def _basin_hopping(positions, rng, deadline, n_iterations_max=100):
    """
    Basin hopping with adaptive step size and temperature.
    Returns best energy and positions found.
    """
    size = len(positions)
    best_energy = float('inf')
    best_positions = positions.copy()
    current_positions = positions.copy()
    
    temperature = 0.15
    step_size = 0.35
    min_step = 0.05
    max_step = 0.6
    iterations = 0
    
    while time.time() < deadline and iterations < n_iterations_max:
        iterations += 1
        perturbation = rng.normal(0, step_size, size=(size, 3))
        new_positions = current_positions + perturbation
        
        try:
            new_energy, new_relaxed = relax(new_positions, POTENTIAL)
        except:
            continue
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_positions = new_relaxed.copy()
            current_positions = new_relaxed.copy()
            # Slightly increase step size on improvement
            step_size = min(max_step, step_size * 1.1)
        else:
            delta_e = new_energy - best_energy
            acceptance_prob = np.exp(-delta_e / temperature)
            if rng.random() < acceptance_prob:
                current_positions = new_relaxed.copy()
                step_size = max(min_step, step_size * 0.95)
        
        # Cool down
        temperature *= 0.98
        if temperature < 0.01:
            temperature = 0.01
    
    return best_energy, best_positions


def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 285.0  # Safety margin
    
    size = instance["n"]
    start_time = time.time()
    
    # Allocate time: ~45s per instance, but use remaining if less
    remaining = _GLOBAL_DEADLINE - time.time()
    instance_deadline = min(start_time + 45.0, _GLOBAL_DEADLINE - 3.0)
    
    rng = np.random.default_rng(size * 7919 + 1)
    
    # Generate diverse seeds using fragment merger
    seeds = []
    
    # Seed 1: Fragment merger
    try:
        seed1 = _fragment_merge_initialization(size, rng, n_fragments=3)
        seeds.append(seed1)
    except:
        pass
    
    # Seed 2: Fragment merger with different fragment count
    try:
        seed2 = _fragment_merge_initialization(size, rng, n_fragments=4 if size <= 52 else 5)
        seeds.append(seed2)
    except:
        pass
    
    # Seed 3: Mackay 55 based
    try:
        mackay55 = _get_mackay_55()
        if size <= 55:
            dists = np.linalg.norm(mackay55, axis=1)
            sort_idx = np.argsort(dists)[::-1]
            remove_idx = sort_idx[:55 - size]
            keep_idx = np.setdiff1d(np.arange(55), remove_idx)
            seed3 = mackay55[keep_idx]
        else:
            seed3 = mackay55.copy()
            while len(seed3) < size:
                ref = seed3[rng.integers(len(seed3))]
                offset = rng.normal(0, 0.3, 3)
                p = ref + offset
                if np.min(np.linalg.norm(seed3 - p, axis=1)) >= 0.5:
                    seed3 = np.vstack([seed3, p])
        seeds.append(seed3)
    except:
        pass
    
    # Seed 4: Random compact
    try:
        seed4 = _random_compact(size, rng)
        seeds.append(seed4)
    except:
        pass
    
    # Relax all seeds, keep best
    best_energy = float('inf')
    best_positions = None
    
    for seed in seeds:
        if len(seed) != size:
            continue
        try:
            e, pos = relax(seed, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            continue
    
    if best_positions is None:
        # Fallback: just return a random compact cluster
        return _random_compact(size, rng)
    
    # Basin hopping on the best seed
    remaining_deadline = _GLOBAL_DEADLINE - 5.0  # Leave 5s safety
    best_energy, best_positions = _basin_hopping(
        best_positions, rng, remaining_deadline, n_iterations_max=150
    )
    
    return best_positions
