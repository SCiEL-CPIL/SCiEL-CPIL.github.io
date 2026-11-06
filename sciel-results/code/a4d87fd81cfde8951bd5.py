import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

POTENTIAL = 'dzugutov'
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None


def _get_icosahedron_vertices():
    """Return 12 vertices of regular icosahedron, unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _get_icosahedron_edges_and_faces():
    """Return edges (pairs) and faces (triples) of icosahedron."""
    verts = _get_icosahedron_vertices()
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if abs(d1 - min_dist) < 1e-6 and abs(d2 - min_dist) < 1e-6 and abs(d3 - min_dist) < 1e-6:
                    faces.append((i, j, k))
    return edges, faces


def _generate_mackay_55():
    """Generate the 55-atom Mackay icosahedron: 1 + 12 + 42."""
    verts = _get_icosahedron_vertices()
    edges, faces = _get_icosahedron_edges_and_faces()
    phi = (1 + np.sqrt(5)) / 2

    center = np.zeros(3)
    shell1 = verts * 1.0  # 12 atoms at radius 1

    # 42 atoms in shell 2: 30 edge midpoints + 12 face centers
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    edge_mid_norms = np.linalg.norm(edge_mids, axis=1, keepdims=True)
    edge_mids = edge_mids / edge_mid_norms * 1.0 * np.sqrt(1 + 1 / phi ** 2)

    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    fc_norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
    face_centers = face_centers / fc_norms * 1.0 * np.sqrt(1 + 1 / phi ** 2)

    # Select 12 face centers (dodecahedron vertices)
    rng = np.random.default_rng(42)
    face_idx = rng.choice(20, size=12, replace=False)
    selected_face_centers = face_centers[face_idx]

    shell2 = np.vstack([edge_mids, selected_face_centers])

    return np.vstack([center, shell1, shell2])


def _generate_defect_seeded(size, base_55, rng):
    """
    Generate a seed by removing atoms from the 55-atom Mackay icosahedron
    or adding atoms for sizes > 55.
    """
    if size == 55:
        return base_55.copy()

    if size < 55:
        n_remove = 55 - size
        base = base_55.copy()

        if n_remove <= 42:
            remove_idx = list(range(13, 55))
            rng.shuffle(remove_idx)
            remove_idx = remove_idx[:n_remove]
            keep_mask = np.ones(55, dtype=bool)
            keep_mask[remove_idx] = False
            return base[keep_mask]
        else:
            n_from_shell2 = 42
            n_from_shell1 = n_remove - 42
            remove_idx = list(range(13, 55))
            rng.shuffle(remove_idx)
            remove_idx = remove_idx[:n_from_shell2]
            shell1_idx = list(range(1, 13))
            rng.shuffle(shell1_idx)
            remove_idx += shell1_idx[:n_from_shell1]
            keep_mask = np.ones(55, dtype=bool)
            keep_mask[remove_idx] = False
            return base[keep_mask]

    else:
        base = base_55.copy()
        n_add = size - 55
        norms = np.linalg.norm(base, axis=1)
        outer_idx = np.argsort(norms)[-10:]

        new_atoms = []
        for _ in range(n_add):
            ref_idx = rng.choice(outer_idx)
            ref_pos = base[ref_idx]
            offset = rng.normal(0, 0.3, 3)
            new_atom = ref_pos + offset
            new_atoms.append(new_atom)

        return np.vstack([base, np.array(new_atoms)])


def _generate_13_icosahedron():
    """13-atom icosahedron: center + 12 vertices."""
    verts = _get_icosahedron_vertices()
    center = np.zeros(3)
    return np.vstack([center, verts])


def _generate_defect_seeded_from_13(size, rng):
    """
    For sizes 38-54, build from 13-atom icosahedron by adding shells.
    """
    base13 = _generate_13_icosahedron()
    if size == 13:
        return base13

    if size > 13:
        base = base13.copy()
        n_add = size - 13
        phi = (1 + np.sqrt(5)) / 2
        r_layer = 1.0
        verts = _get_icosahedron_vertices()

        edges, faces = _get_icosahedron_edges_and_faces()

        candidates = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2
            norm = np.linalg.norm(mid)
            mid_scaled = mid / norm * r_layer * np.sqrt(1 + 1 / phi ** 2)
            candidates.append(mid_scaled)

        for i, j, k in faces:
            fc = (verts[i] + verts[j] + verts[k]) / 3
            norm = np.linalg.norm(fc)
            fc_scaled = fc / norm * r_layer * np.sqrt(1 + 1 / phi ** 2)
            candidates.append(fc_scaled)

        for _ in range(30):
            u = rng.uniform(0, 2 * np.pi)
            v = rng.uniform(-1, 1)
            r = r_layer * np.sqrt(1 + 1 / phi ** 2) * rng.uniform(0.9, 1.1)
            x = r * np.sqrt(1 - v ** 2) * np.cos(u)
            y = r * np.sqrt(1 - v ** 2) * np.sin(u)
            z = r * v
            candidates.append(np.array([x, y, z]))

        candidates = np.array(candidates)

        selected = []
        rng.shuffle(candidates)
        for cand in candidates:
            if len(selected) >= n_add:
                break
            if all(np.linalg.norm(cand - s) >= 0.5 for s in selected) and \
               all(np.linalg.norm(cand - b) >= 0.5 for b in base):
                selected.append(cand)

        while len(selected) < n_add:
            u = rng.uniform(0, 2 * np.pi)
            v = rng.uniform(-1, 1)
            r = r_layer * np.sqrt(1 + 1 / phi ** 2) * rng.uniform(1.0, 1.3)
            x = r * np.sqrt(1 - v ** 2) * np.cos(u)
            y = r * np.sqrt(1 - v ** 2) * np.sin(u)
            z = r * v
            selected.append(np.array([x, y, z]))

        return np.vstack([base, np.array(selected)])


def _basin_hopping(positions, rng, time_limit, initial_temp=0.2, initial_step=0.3):
    """Adaptive basin hopping."""
    current_pos = positions.copy()
    current_energy, current_relaxed = relax(current_pos, POTENTIAL)
    best_energy = current_energy
    best_pos = current_relaxed.copy()

    temperature = initial_temp
    step_size = initial_step
    consecutive_rejects = 0
    n_atoms = len(positions)

    while time.time() < time_limit:
        scale = step_size * (1 + 0.5 * rng.random())
        perturbation = rng.normal(0, scale, size=(n_atoms, 3))
        new_pos = current_pos + perturbation

        try:
            new_energy, new_relaxed = relax(new_pos, POTENTIAL)
        except:
            continue

        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_relaxed.copy()
            current_pos = new_relaxed.copy()
            consecutive_rejects = 0
            step_size *= 0.97
        else:
            delta_e = new_energy - current_energy
            acceptance_prob = np.exp(-delta_e / temperature)
            if rng.random() < acceptance_prob:
                current_pos = new_relaxed.copy()
                current_energy = new_energy
                consecutive_rejects += 1
            else:
                consecutive_rejects += 1

        if consecutive_rejects > 15:
            temperature *= 1.3
            step_size *= 1.15
            consecutive_rejects = 0
        else:
            temperature *= 0.98
            step_size *= 0.997

        temperature = max(temperature, 0.02)
        step_size = min(step_size, 0.6)

    return best_pos, best_energy


def _parallel_basin_hopping(positions, rng, time_limit, n_restarts=3):
    """Run multiple basin hopping restarts and return the best."""
    best_energy = float('inf')
    best_pos = positions.copy()

    time_per_restart = (time_limit - time.time()) / n_restarts

    for i in range(n_restarts):
        if time.time() > time_limit:
            break
        rng_sub = np.random.default_rng(rng.integers(0, 2 ** 31))

        if i == 0:
            start_pos = positions.copy()
        else:
            start_pos = positions + rng_sub.normal(0, 0.2, size=(len(positions), 3))

        try:
            bh_pos, bh_energy = _basin_hopping(start_pos, rng_sub, time.time() + time_per_restart - 1)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except:
            continue

    return best_pos, best_energy


def _lbfgs_relax(positions, maxiter=200, tol=1e-10):
    """
    Use L-BFGS-B with the Dzugutov gradient to relax positions.
    Returns (energy, relaxed_positions).
    """
    n = len(positions)

    def energy_and_grad(x):
        pos = x.reshape(n, 3)
        try:
            e, grad = dzugutov_energy_gradient(pos)
            return e, grad.reshape(-1)
        except:
            return 1e6, np.zeros_like(x)

    x0 = positions.reshape(-1)
    result = minimize(energy_and_grad, x0, method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'gtol': tol})
    relaxed_pos = result.x.reshape(n, 3)
    energy = result.fun
    return energy, relaxed_pos


def _hybrid_optimize(positions, rng, time_limit):
    """
    Hybrid optimization: alternate between L-BFGS-B gradient descent
    and stochastic basin hopping perturbations.
    """
    current_pos = positions.copy()
    best_energy = float('inf')
    best_pos = positions.copy()

    # Initial gradient-based relaxation
    try:
        e, pos = _lbfgs_relax(current_pos, maxiter=100)
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()
            current_pos = pos.copy()
    except:
        pass

    step_size = 0.15
    temperature = 0.1

    while time.time() < time_limit:
        # Perturb
        scale = step_size * (1 + 0.3 * rng.random())
        perturbation = rng.normal(0, scale, size=(len(positions), 3))
        new_pos = current_pos + perturbation

        # Gradient-based relaxation
        try:
            new_energy, new_relaxed = _lbfgs_relax(new_pos, maxiter=50)
        except:
            continue

        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_relaxed.copy()
            current_pos = new_relaxed.copy()
            step_size *= 0.95
        else:
            delta_e = new_energy - best_energy
            acceptance_prob = np.exp(-delta_e / temperature)
            if rng.random() < acceptance_prob:
                current_pos = new_relaxed.copy()
                step_size *= 0.98
            else:
                step_size *= 1.1

        temperature *= 0.97
        temperature = max(temperature, 0.01)
        step_size = min(step_size, 0.5)

    return best_pos, best_energy


def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE

    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 290.0

    size = instance["n"]
    start_time = time.time()

    instance_deadline = min(start_time + 46.0, _GLOBAL_DEADLINE - 5.0)

    rng = np.random.default_rng(42 + size * 7)

    best_energy = float('inf')
    best_positions = None

    all_seeds = []

    try:
        mackay55 = _generate_mackay_55()
    except:
        mackay55 = None

    if mackay55 is not None and 40 <= size <= 60:
        try:
            seed = _generate_defect_seeded(size, mackay55, rng)
            if seed.shape[0] == size:
                all_seeds.append(seed)
        except:
            pass

    if size <= 54:
        try:
            seed = _generate_defect_seeded_from_13(size, rng)
            if seed.shape[0] == size:
                all_seeds.append(seed)
        except:
            pass

    radius = 0.8 * size ** (1 / 3) + 1.0
    points = []
    for _ in range(size):
        for _ in range(500):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
                break
        else:
            p = rng.uniform(-radius * 1.2, radius * 1.2, 3)
            points.append(p)
    all_seeds.append(np.array(points))

    if mackay55 is not None:
        for i in range(2):
            try:
                perturbed = _generate_defect_seeded(size, mackay55, rng)
                perturbed += rng.normal(0, 0.1, size=(size, 3))
                if perturbed.shape[0] == size:
                    all_seeds.append(perturbed)
            except:
                pass

    # Relax each seed using hybrid optimization
    for seed in all_seeds:
        if time.time() > instance_deadline - 15:
            break
        try:
            e, pos = _hybrid_optimize(seed, rng, min(time.time() + 8, instance_deadline - 12))
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            try:
                e, pos = relax(seed, POTENTIAL)
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except:
                continue

    if best_positions is None:
        best_positions = np.random.randn(size, 3) * 0.3
        try:
            best_energy, best_positions = relax(best_positions, POTENTIAL)
        except:
            pass

    remaining_time = instance_deadline - time.time()
    if remaining_time > 10:
        try:
            bh_pos, bh_energy = _parallel_basin_hopping(best_positions, rng, time.time() + remaining_time - 3, n_restarts=2)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except:
            pass

    return best_positions
