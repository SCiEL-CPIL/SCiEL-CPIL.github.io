"""Dzugutov cluster global minima search using fragment-merger with icosahedral building blocks."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

WORKING_LIMIT = 305.0
START_TIME = time.time()

def _remaining_time():
    return WORKING_LIMIT - (time.time() - START_TIME)

def _icosahedron_vertices():
    """12 vertices of a regular icosahedron centered at origin."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return v

def _random_rotation_matrix(rng):
    """Generate a random rotation matrix."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])
    return R

def _build_13_icosahedron(spacing=1.0, rng=None):
    """Build a 13-atom icosahedron (1 center + 12 vertices)."""
    if rng is None:
        rng = np.random.default_rng()
    verts = _icosahedron_vertices()
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * spacing
    center = np.array([0.0, 0.0, 0.0])
    return np.vstack([center.reshape(1,3), verts])

def _build_14_decahedron(spacing=1.0, rng=None):
    """Build a 14-atom decahedron (1 center + 12 vertices + 1 extra)."""
    if rng is None:
        rng = np.random.default_rng()
    verts = _icosahedron_vertices()
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * spacing
    center = np.array([0.0, 0.0, 0.0])
    extra = rng.normal(size=3) * spacing * 0.3
    return np.vstack([center.reshape(1,3), verts, extra.reshape(1,3)])

def _build_20_icosahedron(spacing=1.0, rng=None):
    """Build a 20-atom structure: 13-atom icosahedron + 7 more on outer shell."""
    if rng is None:
        rng = np.random.default_rng()
    base = _build_13_icosahedron(spacing, rng)
    n_add = 7
    r_outer = spacing * 1.8
    golden_angle = np.pi * (3 - np.sqrt(5))
    extra = np.zeros((n_add, 3))
    for i in range(n_add):
        z = 1 - 2 * i / max(1, n_add - 1)
        r = np.sqrt(1 - z*z)
        theta = golden_angle * i
        extra[i] = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r_outer
    return np.vstack([base, extra])

def _relax_fragment(positions, potential, parameter, max_time=10.0):
    """Relax a fragment with time limit."""
    start = time.time()
    try:
        energy, pos = relax(positions, potential, parameter)
        return energy, pos
    except Exception:
        return np.inf, positions

def _merge_fragments(fragments, positions_list, potential, parameter, rng, max_iter=50):
    """
    Merge fragments by coarse-grained annealing.
    fragments: list of atom arrays
    positions_list: list of center positions for each fragment
    """
    n_frags = len(fragments)
    total_atoms = sum(len(f) for f in fragments)
    
    # Initial assembly
    all_atoms = []
    for i, (frag, center) in enumerate(zip(fragments, positions_list)):
        rotated_frag = frag + center
        all_atoms.append(rotated_frag)
    assembled = np.vstack(all_atoms)
    
    # Coarse-grained annealing: move fragment centers
    best_energy, best_pos = _relax_fragment(assembled, potential, parameter)
    if best_energy == np.inf:
        best_energy = 1e10
        best_pos = assembled.copy()
    
    current_pos = assembled.copy()
    current_energy = best_energy
    
    for it in range(max_iter):
        if _remaining_time() < 5.0:
            break
        
        # Pick a random fragment and perturb its center
        fi = rng.integers(n_frags)
        n_atoms_fi = len(fragments[fi])
        
        # Compute current center of fragment fi
        start_idx = sum(len(fragments[j]) for j in range(fi))
        end_idx = start_idx + n_atoms_fi
        frag_center = np.mean(current_pos[start_idx:end_idx], axis=0)
        
        # Perturb center
        delta = rng.normal(0, 0.5, 3)
        new_center = frag_center + delta
        
        # Move fragment
        new_pos = current_pos.copy()
        new_pos[start_idx:end_idx] += (new_center - frag_center)
        
        # Relax
        new_energy, relaxed_pos = _relax_fragment(new_pos, potential, parameter)
        
        # Metropolis accept
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.3):
            current_energy = new_energy
            current_pos = relaxed_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = relaxed_pos.copy()
    
    return best_energy, best_pos

def _generate_fragment_seed(n_target, potential, parameter, rng):
    """Generate a fragment-merger seed for n_target atoms."""
    # Choose fragment sizes
    frag_sizes = []
    remaining = n_target
    
    # Prefer 13-atom icosahedra and 14-atom decahedra
    while remaining > 0:
        if remaining >= 26 and rng.random() < 0.7:
            frag_sizes.append(13)
            remaining -= 13
        elif remaining >= 28 and rng.random() < 0.5:
            frag_sizes.append(14)
            remaining -= 14
        elif remaining >= 20:
            frag_sizes.append(20)
            remaining -= 20
        elif remaining >= 13:
            frag_sizes.append(13)
            remaining -= 13
        elif remaining >= 14:
            frag_sizes.append(14)
            remaining -= 14
        else:
            frag_sizes.append(remaining)
            remaining = 0
    
    # Build fragments
    fragments = []
    for size in frag_sizes:
        if size == 13:
            frag = _build_13_icosahedron(spacing=1.0 + rng.uniform(-0.05, 0.05), rng=rng)
        elif size == 14:
            frag = _build_14_decahedron(spacing=1.0 + rng.uniform(-0.05, 0.05), rng=rng)
        elif size == 20:
            frag = _build_20_icosahedron(spacing=1.0 + rng.uniform(-0.05, 0.05), rng=rng)
        else:
            # Fallback: random compact
            frag = rng.normal(0, 0.8, (size, 3))
        
        # Relax fragment
        _, frag_relaxed = _relax_fragment(frag, potential, parameter)
        fragments.append(frag_relaxed)
    
    # Place fragments with random rotations and positions
    n_frags = len(fragments)
    positions_list = []
    for i in range(n_frags):
        # Random position in a sphere of radius ~ n_frags
        r = rng.uniform(1.0, 3.0)
        theta = rng.uniform(0, 2*np.pi)
        phi = rng.uniform(0, np.pi)
        center = np.array([r*np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi)])
        positions_list.append(center)
    
    return fragments, positions_list

def _ensure_min_distance(positions, min_dist=0.5, max_iter=30):
    """Ensure all pairs are at least min_dist apart."""
    positions = positions.copy()
    n = len(positions)
    
    for _ in range(max_iter):
        if _remaining_time() < 5.0:
            break
        min_d = np.inf
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
        
        if min_d >= min_dist:
            break
        
        i, j = min_pair
        direction = (positions[i] - positions[j]) / (min_d + 1e-10)
        push = (min_dist - min_d) / 2 + 0.01
        positions[i] += direction * push
        positions[j] -= direction * push
    
    return positions

def _basin_hopping(positions, potential, parameter, max_iterations=30, rng=None):
    """Basin hopping with relaxation."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = relax(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    amplitude = 0.25
    beta = 1.0 / 0.25
    
    for it in range(max_iterations):
        if _remaining_time() < 10.0:
            break
        
        perturbed = current_pos + rng.normal(0, amplitude, positions.shape)
        new_energy, new_pos = relax(perturbed, potential, parameter)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-beta * delta_e):
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        amplitude *= 1.04
    
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Dzugutov clusters using fragment-merger strategy."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.default_rng(n * 7 + 13)
    
    best_energy = np.inf
    best_positions = None
    
    # Multiple restarts with fragment-merger
    n_starts = 3
    
    for start_idx in range(n_starts):
        if _remaining_time() < 40.0:
            break
        
        # Generate fragment-merger seed
        fragments, positions_list = _generate_fragment_seed(n, potential, parameter, rng)
        
        # Merge fragments
        energy, merged_pos = _merge_fragments(fragments, positions_list, potential, parameter, rng, max_iter=30)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = merged_pos.copy()
        
        # Basin hopping from merged structure
        if _remaining_time() > 20.0:
            bh_energy, bh_pos = _basin_hopping(merged_pos, potential, parameter, max_iterations=25, rng=rng)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos.copy()
    
    # Final relaxation
    if best_positions is not None:
        final_energy, final_pos = relax(best_positions, potential, parameter)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    # Ensure minimum distance
    if best_positions is not None:
        best_positions = _ensure_min_distance(best_positions)
    
    return best_positions
# EVOLVE-BLOCK-END
