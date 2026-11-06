"""Improved program: Defect-seeded Mackay icosahedron initialization with multi-stage relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import itertools
from scipy.optimize import minimize
import time

def _get_icosahedron_data():
    """Return vertices, face centers, and edge midpoints of a regular icosahedron (unit sphere)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices
    v = np.array([
        [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
        [1,  phi, 0], [-1,  phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [ phi, 0,  1], [ phi, 0, -1], [-phi, 0,  1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v / norms
    
    # Compute edge length
    edge_len = np.linalg.norm(v_unit[0] - v_unit[1])
    
    # Find all 20 triangular faces
    face_centers = []
    faces = []
    for i, j, k in itertools.combinations(range(12), 3):
        d1 = np.linalg.norm(v_unit[i] - v_unit[j])
        d2 = np.linalg.norm(v_unit[j] - v_unit[k])
        d3 = np.linalg.norm(v_unit[k] - v_unit[i])
        if abs(d1 - edge_len) < 1e-9 and abs(d2 - edge_len) < 1e-9 and abs(d3 - edge_len) < 1e-9:
            fc = (v_unit[i] + v_unit[j] + v_unit[k]) / 3.0
            fc = fc / np.linalg.norm(fc)
            face_centers.append(fc)
            faces.append((i, j, k))
    
    face_centers = np.array(face_centers)
    
    # Find all 30 edges
    edge_mids = []
    edges = []
    for i, j in itertools.combinations(range(12), 2):
        d = np.linalg.norm(v_unit[i] - v_unit[j])
        if abs(d - edge_len) < 1e-9:
            em = (v_unit[i] + v_unit[j]) / 2.0
            em = em / np.linalg.norm(em)
            edge_mids.append(em)
            edges.append((i, j))
    
    edge_mids = np.array(edge_mids)
    
    return v_unit, face_centers, edge_mids, faces, edges

def _build_mackay_55(r_inner=1.0, r_outer=1.9):
    """Build a 55-atom Mackay icosahedron: 1 + 12 + 42.
    Outer shell: 12 vertices + 20 face centers + 10 edge midpoints = 42.
    We select 10 edge midpoints that form a symmetric subset.
    """
    v_unit, face_centers, edge_mids, faces, edges = _get_icosahedron_data()
    
    # For the 10 edge midpoints, select those corresponding to edges that are
    # part of a symmetric pattern. One approach: select edges that are not adjacent
    # to any other selected edge (independent set), or use a known pattern.
    # Actually, for the Mackay icosahedron, the outer shell has 42 atoms at the
    # positions of the icosahedron vertices (12), face centers (20), and a subset
    # of edge midpoints (10). The 10 edge midpoints correspond to the 10 "golden"
    # edges or can be selected as those forming two parallel pentagons.
    
    # Let's select edge midpoints more carefully. In the Mackay icosahedron,
    # the 42 outer atoms are at: 12 vertices, 20 face centers, and 10 points
    # that are midpoints of specific edges. These 10 edge midpoints form two
    # interlocking pentagons.
    
    # For simplicity, let's just pick 10 edge midpoints that are well-distributed.
    # We'll use the first 10 from our list, but let's try to make them symmetric.
    
    # Actually, a better approach: the 10 additional atoms in the Mackay icosahedron
    # outer shell are at the midpoints of the 10 edges that connect vertices
    # of the same "type" in some decomposition. Let's just use all 30 edge mids
    # and pick 10 that maximize minimum distance between them.
    
    # Greedy selection of 10 well-separated edge midpoints
    selected_indices = []
    remaining = list(range(len(edge_mids)))
    for _ in range(10):
        if not remaining:
            break
        best_idx = None
        best_min_dist = -1
        for idx in remaining:
            if not selected_indices:
                best_min_dist = 1e9
                best_idx = idx
            else:
                min_d = min(np.linalg.norm(edge_mids[idx] - edge_mids[si]) for si in selected_indices)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
    
    selected_edge_mids = edge_mids[selected_indices]
    
    # Build the cluster
    atoms = [np.zeros(3)]  # center
    atoms.extend(v_unit * r_inner)  # 12 inner shell
    atoms.extend(v_unit * r_outer)  # 12 outer vertices
    atoms.extend(face_centers * r_outer)  # 20 face centers
    atoms.extend(selected_edge_mids * r_outer)  # 10 edge midpoints
    
    return np.array(atoms)

def _build_mackay_13(r=1.0):
    """Build a 13-atom Mackay icosahedron: 1 + 12."""
    v_unit, _, _, _, _ = _get_icosahedron_data()
    atoms = [np.zeros(3)]
    atoms.extend(v_unit * r)
    return np.array(atoms)

def _build_mackay_55_with_labels(r_inner=1.0, r_outer=1.9):
    """Build 55-atom Mackay with labels: 'C', 'I0'-'I11', 'V0'-'V11', 'F0'-'F19', 'E0'-'E9'."""
    v_unit, face_centers, edge_mids, faces, edges = _get_icosahedron_data()
    
    # Select 10 edge midpoints same way
    selected_indices = []
    remaining = list(range(len(edge_mids)))
    for _ in range(10):
        if not remaining:
            break
        best_idx = None
        best_min_dist = -1
        for idx in remaining:
            if not selected_indices:
                best_min_dist = 1e9
                best_idx = idx
            else:
                min_d = min(np.linalg.norm(edge_mids[idx] - edge_mids[si]) for si in selected_indices)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
    
    selected_edge_mids = edge_mids[selected_indices]
    
    atoms = [np.zeros(3)]
    labels = ['C']
    for i in range(12):
        atoms.append(v_unit[i] * r_inner)
        labels.append(f'I{i}')
    for i in range(12):
        atoms.append(v_unit[i] * r_outer)
        labels.append(f'V{i}')
    for i in range(20):
        atoms.append(face_centers[i] * r_outer)
        labels.append(f'F{i}')
    for i in range(10):
        atoms.append(selected_edge_mids[i] * r_outer)
        labels.append(f'E{i}')
    
    return np.array(atoms), labels

def _remove_atoms_by_priority(positions, labels, n_remove, strategy='outer_first', rng=None):
    """Remove n_remove atoms based on priority.
    Strategies:
    - 'outer_first': remove from outer shell first (V, F, E before I, C)
    - 'symmetric': try to remove in symmetric groups
    - 'random': random removal from outer atoms
    """
    n = len(positions)
    if n_remove <= 0:
        return positions.copy()
    
    # Define priority: C=0, I=1, V=2, F=3, E=4 (higher = more removable)
    def get_priority(label):
        if label == 'C':
            return 0
        elif label.startswith('I'):
            return 1
        elif label.startswith('V'):
            return 2
        elif label.startswith('F'):
            return 3
        elif label.startswith('E'):
            return 4
        return 0
    
    if strategy == 'outer_first':
        # Sort by priority (descending), then by distance from center
        center = np.mean(positions, axis=0)
        dists = np.linalg.norm(positions - center, axis=1)
        priorities = [get_priority(l) for l in labels]
        # Sort: higher priority first, then farther from center
        order = np.argsort([-p - 0.01 * d for p, d in zip(priorities, dists)])
        to_remove = set(order[:n_remove].tolist())
    elif strategy == 'random':
        if rng is None:
            rng = np.random.default_rng(42)
        # Remove random atoms, preferring outer shell
        priorities = [get_priority(l) for l in labels]
        # Weight by priority
        weights = [p + 1 for p in priorities]
        idx = rng.choice(n, size=n_remove, replace=False, p=np.array(weights)/sum(weights))
        to_remove = set(idx.tolist())
    else:  # symmetric
        # Remove in groups that preserve some symmetry
        # For simplicity, just use outer_first
        center = np.mean(positions, axis=0)
        dists = np.linalg.norm(positions - center, axis=1)
        priorities = [get_priority(l) for l in labels]
        order = np.argsort([-p - 0.01 * d for p, d in zip(priorities, dists)])
        to_remove = set(order[:n_remove].tolist())
    
    remaining = [positions[i] for i in range(n) if i not in to_remove]
    return np.array(remaining)

def _add_atoms_to_mackay(positions, n_add, rng, base_r_outer=1.9):
    """Add n_add atoms to a Mackay icosahedron by placing them at symmetry-allowed positions."""
    new_atoms = []
    attempts = 0
    
    # Try to add atoms at positions that maintain icosahedral-like symmetry
    # Place atoms near the surface, at positions that are "natural" extensions
    
    # Define candidate directions: vertices, face centers, edge midpoints, and some random
    v_unit, face_centers, edge_mids, _, _ = _get_icosahedron_data()
    
    # All candidate directions
    candidate_dirs = list(v_unit) + list(face_centers) + list(edge_mids)
    # Add some random directions
    for _ in range(20):
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        candidate_dirs.append(d)
    
    rng.shuffle(candidate_dirs)
    
    for _ in range(n_add):
        placed = False
        for d in candidate_dirs:
            attempts += 1
            if attempts > 500:
                break
            # Try different radii
            for r_factor in [1.0, 1.05, 1.1, 1.15, 1.2]:
                r = base_r_outer * r_factor
                candidate = d * r
                # Check minimum distance
                if all(np.linalg.norm(candidate - p) >= 0.5 for p in positions) and \
                   all(np.linalg.norm(candidate - p) >= 0.5 for p in new_atoms):
                    new_atoms.append(candidate)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            # Fallback: random placement near surface
            for _ in range(50):
                d = rng.normal(size=3)
                d = d / np.linalg.norm(d)
                r = base_r_outer * (1.0 + rng.uniform(0, 0.3))
                candidate = d * r
                if all(np.linalg.norm(candidate - p) >= 0.5 for p in positions) and \
                   all(np.linalg.norm(candidate - p) >= 0.5 for p in new_atoms):
                    new_atoms.append(candidate)
                    break
    
    return np.vstack([positions, np.array(new_atoms)]) if new_atoms else positions

def _relax_with_gradient(positions, potential, parameter, max_iter=1000):
    """Relax using gradient-based optimization as a secondary method."""
    def energy_func(pos):
        try:
            e, _ = relax(pos.reshape(-1, 3), potential, parameter)
            return e
        except:
            return 1e10
    
    # Use L-BFGS-B
    try:
        result = minimize(energy_func, positions.flatten(), method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success or result.fun < 1e9:
            return result.fun, result.x.reshape(-1, 3)
    except:
        pass
    return np.inf, positions

def _generate_random_seed(n, rng, min_dist=0.6):
    """Generate a random seed with minimum distance constraint."""
    points = []
    radius = 1.0 * n ** (1.0/3.0)
    attempts = 0
    while len(points) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        attempts += 1
    if len(points) == n:
        return np.array(points)
    return None

def _generate_icosahedral_seed(n, rng):
    """Generate a seed based on icosahedral symmetry."""
    # Start with 13-atom icosahedron
    mackay_13 = _build_mackay_13(r=1.0)
    if n == 13:
        return mackay_13
    if n < 13:
        # Remove random atoms
        idx = rng.choice(13, size=n, replace=False)
        return mackay_13[idx]
    
    # For n > 13, add atoms randomly around the icosahedron
    current = mackay_13.copy()
    for _ in range(n - 13):
        # Try to add an atom at a random position near the surface
        placed = False
        for _ in range(100):
            # Random direction
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            # Radius between 1.5 and 2.5
            r = rng.uniform(1.5, 2.5)
            p = d * r
            if all(np.linalg.norm(p - q) >= 0.5 for q in current):
                current = np.vstack([current, p])
                placed = True
                break
        if not placed:
            # If we can't place, just add a random point far away (will relax)
            p = rng.normal(size=3) * 3.0
            current = np.vstack([current, p])
    return current

def construct(instance):
    """Construct low-energy cluster using defect-seeded Mackay icosahedron initialization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Time limit per instance: ~50s
    # We will try a set of seeds, relax them, and keep the best.
    # Then we will do a few basin hopping steps on the best.
    
    seeds = []
    
    # 1. Random seeds
    for _ in range(5):
        seed = _generate_random_seed(size, rng, min_dist=0.6)
        if seed is not None:
            seeds.append(seed)
    
    # 2. Icosahedral seeds
    for _ in range(5):
        seed = _generate_icosahedral_seed(size, rng)
        if seed is not None and len(seed) == size:
            seeds.append(seed)
    
    # 3. Perturbed 55-atom Mackay seeds (if size is close to 55)
    if 38 <= size <= 60:
        mackay_55_pos, mackay_55_labels = _build_mackay_55_with_labels(r_inner=1.0, r_outer=1.9)
        if size == 55:
            for _ in range(5):
                perturbed = mackay_55_pos + rng.normal(0, 0.1, size=mackay_55_pos.shape)
                seeds.append(perturbed)
        elif size < 55:
            n_remove = 55 - size
            for _ in range(5):
                candidate = _remove_atoms_by_priority(mackay_55_pos, mackay_55_labels, n_remove, 'random', rng)
                if len(candidate) == size:
                    perturbed = candidate + rng.normal(0, 0.1, size=candidate.shape)
                    seeds.append(perturbed)
        else: # size > 55
            n_add = size - 55
            for _ in range(5):
                candidate = _add_atoms_to_mackay(mackay_55_pos, n_add, rng, base_r_outer=1.9)
                if len(candidate) == size:
                    perturbed = candidate + rng.normal(0, 0.1, size=candidate.shape)
                    seeds.append(perturbed)
    
    # Relax all seeds
    for i, candidate in enumerate(seeds):
        if len(candidate) != size:
            continue
        try:
            energy, positions = relax(candidate, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # Basin Hopping / Local Search on the best result
    if best_positions is not None:
        # Try a few perturbations and relaxations
        for step_size in [0.2, 0.3, 0.4, 0.5]:
            for _ in range(5):
                perturbed = best_positions + rng.normal(0, step_size, size=best_positions.shape)
                try:
                    e3, pos3 = relax(perturbed, potential, parameter)
                    if e3 < best_energy:
                        best_energy = e3
                        best_positions = pos3
                except:
                    continue
    
    if best_positions is None:
        # Fallback: simple grid
        points = []
        for i in range(size):
            p = np.array([i * 1.0, (i // 10) * 1.0, (i // 100) * 1.0])
            points.append(p)
        best_positions = np.array(points)
    
    return best_positions
# EVOLVE-BLOCK-END
