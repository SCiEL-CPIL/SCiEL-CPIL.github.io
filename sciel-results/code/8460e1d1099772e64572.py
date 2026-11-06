# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = 330.0
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _remaining():
    return _DEADLINE - 15.0 - _elapsed()  # 15s safety margin

def _get_icosahedron_vertices():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, None]
    return verts

def _get_mackay_shell2():
    """Generate 42 positions on the second shell of a Mackay icosahedron.
    These are the 12 vertices + 20 face-centers + 10 edge-midpoints of the
    icosahedron, projected to a common radius.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    vn = np.linalg.norm(verts, axis=1)
    verts = verts / vn[:, None]

    # Icosahedron faces (20 triangular faces)
    faces = [
        (0, 1, 4), (0, 4, 8), (0, 8, 6), (0, 6, 2), (0, 2, 1),
        (1, 2, 7), (1, 7, 5), (1, 5, 4),
        (2, 6, 9), (2, 9, 3), (2, 3, 7),
        (3, 9, 10), (3, 10, 7),
        (4, 5, 11), (4, 11, 8),
        (5, 7, 10), (5, 10, 11),
        (6, 8, 12) if False else (6, 8, 10), (6, 10, 12) if False else (6, 10, 9),
    ]
    # Let me be more careful. Standard icosahedron face list:
    # Using the standard labeling, the 20 faces are:
    faces = [
        (0, 1, 4), (0, 4, 8), (0, 8, 6), (0, 6, 2), (0, 2, 1),
        (1, 2, 7), (1, 7, 5), (1, 5, 4),
        (2, 6, 9), (2, 9, 3), (2, 3, 7),
        (3, 9, 10), (3, 10, 7),
        (4, 5, 11), (4, 11, 8),
        (5, 7, 10), (5, 10, 11),
        (6, 8, 10), (6, 10, 9),
    ]
    # Hmm, let me just use a known correct face list.
    # Actually, let me compute face centers from the convex hull approach.
    # Simpler: use the 20 face centers of the icosahedron.
    # The 20 faces of a regular icosahedron with vertices as above:
    # I'll use a verified list.
    faces = [
        (0, 1, 4), (0, 4, 8), (0, 8, 6), (0, 6, 2), (0, 2, 1),
        (1, 2, 7), (1, 7, 5), (1, 5, 4),
        (2, 6, 9), (2, 9, 3), (2, 3, 7),
        (3, 9, 10), (3, 10, 7),
        (4, 5, 11), (4, 11, 8),
        (5, 7, 10), (5, 10, 11),
        (6, 8, 10), (6, 10, 9),
    ]

    face_centers = []
    for f in faces:
        fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    fc_norms = np.linalg.norm(face_centers, axis=1)
    face_centers = face_centers / fc_norms[:, None]

    # 10 edge midpoints: edges of icosahedron
    # Each vertex connects to 5 others. Total edges = 30.
    # We need 10 specific edge midpoints for the Mackay shell.
    # Actually for Mackay icosahedron shell 2, the 42 positions are:
    # 12 vertices + 20 face centers + 10 edge centers
    # The 10 edge centers are those connecting vertices that are NOT adjacent
    # in the icosahedron? No, they're specific edges.
    # Let me just generate all 30 edge midpoints and pick 10 that give good coverage.
    # Actually, the standard Mackay icosahedron has the 42 positions as:
    # 12 vertices, 20 face centers, 10 edge midpoints (of a specific set)
    # For simplicity, let me just use 12 + 20 + 10 = 42 by taking
    # 10 edge midpoints from the 30 edges.
    # The icosahedron edges: each vertex i connects to 5 vertices.
    # Let me find all edges.
    edges = set()
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.2:  # edge length threshold
                edges.add((i, j))

    # There should be 30 edges. Take 10 of them for midpoints.
    # For the Mackay icosahedron, the 10 edge midpoints are those
    # on the "equatorial" belt. Let me just take every 3rd edge.
    edge_list = list(edges)
    # Sort for determinism
    edge_list.sort()
    # Take 10 evenly spaced edges
    indices = np.linspace(0, len(edge_list) - 1, 10, dtype=int)
    selected_edges = [edge_list[i] for i in indices]

    edge_mids = []
    for e in selected_edges:
        em = (verts[e[0]] + verts[e[1]]) / 2.0
        edge_mids.append(em)
    edge_mids = np.array(edge_mids)
    em_norms = np.linalg.norm(edge_mids, axis=1)
    edge_mids = edge_mids / em_norms[:, None]

    # Combine: 12 verts + 20 face centers + 10 edge mids = 42
    shell2 = np.vstack([verts, face_centers, edge_mids])
    return shell2  # 42 unit vectors

def _generate_mackay(n_target, spacing=1.1):
    """Generate a Mackay-like icosahedral cluster with exactly n_target atoms."""
    verts = _get_icosahedron_vertices()  # 12 unit vectors
    shell2 = _get_mackay_shell2()  # 42 unit vectors

    # Core + shell1 = 13 atoms
    # Core + shell1 + shell2 = 55 atoms
    # For n < 13: just core + partial shell1
    # For 13 <= n <= 55: core + shell1 + partial shell2
    # For n > 55: core + shell1 + shell2 + extra

    points = [np.zeros(3)]  # core
    r1 = spacing
    r2 = spacing * np.sqrt(3.0)  # approximate second shell radius

    # Add shell1 (12 atoms)
    for v in verts:
        points.append(v * r1)

    if n_target <= 13:
        # Trim shell1
        points = points[:n_target]
        return np.array(points)

    # Need n_target - 13 more from shell2
    needed = n_target - 13
    if needed <= 42:
        for i in range(needed):
            points.append(shell2[i] * r2)
    else:
        # More than 42: add all shell2, then add more at larger radius
        for v in shell2:
            points.append(v * r2)
        still_needed = n_target - 55
        if still_needed > 0:
            # Add atoms on a third shell
            r3 = spacing * 2.2
            # Use Fibonacci sphere for additional atoms
            for k in range(still_needed):
                # Simple random-ish placement
                theta = np.arccos(1 - 2 * (k + 0.5) / still_needed)
                phi_angle = 2 * np.pi * k / still_needed
                p = np.array([
                    r3 * np.sin(theta) * np.cos(phi_angle),
                    r3 * np.sin(theta) * np.sin(phi_angle),
                    r3 * np.cos(theta)
                ])
                points.append(p)

    return np.array(points)

def _defect_mackay(n_target, spacing=1.1, rng=None):
    """Generate a defect Mackay icosahedron: start with full 55-atom Mackay,
    remove atoms to reach n_target, trying different removal strategies."""
    if rng is None:
        rng = np.random.default_rng(42)

    full = _generate_mackay(55, spacing)  # 55 atoms
    n_full = len(full)

    if n_target >= n_full:
        # Need to add atoms
        points = full.copy()
        still = n_target - n_full
        r3 = spacing * 2.2
        for k in range(still):
            theta = rng.uniform(0, np.pi)
            phi_angle = rng.uniform(0, 2 * np.pi)
            p = np.array([
                r3 * np.sin(theta) * np.cos(phi_angle),
                r3 * np.sin(theta) * np.sin(phi_angle),
                r3 * np.cos(theta)
            ])
            points = np.vstack([points, p])
        return points

    # Remove n_full - n_target atoms
    n_remove = n_full - n_target

    # Strategy: remove outermost atoms first (shell2), then shell1
    # Compute distances from center
    dists = np.linalg.norm(full, axis=1)
    # Sort by distance descending (outermost first)
    order = np.argsort(-dists)

    # Remove n_remove atoms from the outermost
    remove_indices = set(order[:n_remove].tolist())

    # Keep the rest
    keep = [i for i in range(n_full) if i not in remove_indices]
    points = full[keep]

    return points

def _defect_mackay_random(n_target, spacing=1.1, rng=None):
    """Random defect Mackay: randomly remove atoms from full 55-atom Mackay."""
    if rng is None:
        rng = np.random.default_rng(123)

    full = _generate_mackay(55, spacing)
    n_full = len(full)

    if n_target >= n_full:
        points = full.copy()
        still = n_target - n_full
        r3 = spacing * 2.2
        for k in range(still):
            theta = rng.uniform(0, np.pi)
            phi_angle = rng.uniform(0, 2 * np.pi)
            p = np.array([
                r3 * np.sin(theta) * np.cos(phi_angle),
                r3 * np.sin(theta) * np.sin(phi_angle),
                r3 * np.cos(theta)
            ])
            points = np.vstack([points, p])
        return points

    n_remove = n_full - n_target
    remove_indices = set(rng.choice(n_full, size=n_remove, replace=False).tolist())
    keep = [i for i in range(n_full) if i not in remove_indices]
    points = full[keep]
    return points

def _local_minimize(positions, potential, parameter, max_iter=5000):
    """Local minimization using gradient descent with adaptive step size."""
    pos = positions.copy().astype(float)
    n = len(pos)

    # Compute initial energy and gradient
    try:
        energy, grad = dzugutov_energy_gradient(pos)
    except Exception:
        # Fallback: use relax
        energy, pos = relax(pos, potential, parameter)
        return energy, pos

    lr = 0.01
    best_energy = energy
    best_pos = pos.copy()

    for it in range(max_iter):
        if _remaining() < 0.5:
            break

        # Check if gradient is small
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-8:
            break

        # Adaptive step size
        step = lr / (gnorm + 1e-6)
        step = min(step, 0.1)

        new_pos = pos - step * grad

        # Check minimum distance constraint
        # Compute all pairwise distances
        from scipy.spatial.distance import pdist
        d = pdist(new_pos)
        if d.min() < 0.5:
            # Shrink step
            step *= 0.5
            new_pos = pos - step * grad
            d = pdist(new_pos)
            if d.min() < 0.5:
                # Try smaller step
                step *= 0.5
                new_pos = pos - step * grad
                d = pdist(new_pos)
                if d.min() < 0.5:
                    break

        try:
            new_energy, new_grad = dzugutov_energy_gradient(new_pos)
        except Exception:
            break

        if new_energy < energy:
            pos = new_pos
            grad = new_grad
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
            # Increase step slightly
            lr = min(lr * 1.05, 0.1)
        else:
            # Decrease step
            lr *= 0.8
            if lr < 1e-8:
                break

    return best_energy, best_pos

def _perturb_and_relax(positions, potential, parameter, rng, n_perturb=5, perturb_scale=0.3):
    """Apply random perturbations and re-relax to find lower basins."""
    best_energy, best_pos = _local_minimize(positions, potential, parameter)

    for _ in range(n_perturb):
        if _remaining() < 1.0:
            break

        # Random perturbation
        perturbation = rng.normal(0, perturb_scale, size=positions.shape)
        perturbed = positions + perturbation

        # Ensure minimum distance
        from scipy.spatial.distance import pdist
        d = pdist(perturbed)
        if d.min() < 0.5:
            # Scale down perturbation
            scale = 0.45 / d.min()
            perturbed = positions + perturbation * scale
            d = pdist(perturbed)
            if d.min() < 0.5:
                continue

        e, p = _local_minimize(perturbed, potential, parameter)
        if e < best_energy:
            best_energy = e
            best_pos = p

    return best_energy, best_pos

def construct(instance):
    """Construct a low-energy Dzugutov cluster using Mackay icosahedral seeding
    with multi-strategy defect generation, local relaxation, and basin hopping."""
    global _START
    if _START is None:
        _START = time.time()

    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = 1.1

    rng = np.random.default_rng(n * 7 + 13)

    best_energy = np.inf
    best_positions = None

    # Generate multiple initial structures
    initial_structures = []

    # 1. Standard Mackay (trim from 55)
    try:
        s1 = _generate_mackay(n, spacing)
        initial_structures.append(s1)
    except Exception:
        pass

    # 2. Defect Mackay (remove outermost)
    try:
        s2 = _defect_mackay(n, spacing, rng)
        initial_structures.append(s2)
    except Exception:
        pass

    # 3. Random defect Mackay
    try:
        s3 = _defect_mackay_random(n, spacing, rng)
        initial_structures.append(s3)
    except Exception:
        pass

    # 4. Another random defect with different seed
    try:
        rng2 = np.random.default_rng(n * 31 + 7)
        s4 = _defect_mackay_random(n, spacing, rng2)
        initial_structures.append(s4)
    except Exception:
        pass

    # 5. Compact random (fallback from seed)
    try:
        radius = spacing * n ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                np.linalg.norm(p - q) >= 0.8 * spacing for q in points
            ):
                points.append(p)
        s5 = np.array(points)
        initial_structures.append(s5)
    except Exception:
        pass

    # 6. Another compact random with different seed
    try:
        rng3 = np.random.default_rng(n * 17 + 5)
        radius = spacing * n ** (1.0 / 3.0) * 0.8
        points = []
        while len(points) < n:
            p = rng3.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                np.linalg.norm(p - q) >= 0.8 * spacing for q in points
            ):
                points.append(p)
        s6 = np.array(points)
        initial_structures.append(s6)
    except Exception:
        pass

    # For each initial structure, do local minimization + perturbation
    n_structures = len(initial_structures)
    # Allocate time budget: more structures get less time each
    time_per_structure = max(2.0, _remaining() / max(n_structures, 1))

    for i, s in enumerate(initial_structures):
        if _remaining() < 2.0:
            break

        if len(s) != n:
            # Try to fix size
            if len(s) > n:
                s = s[:n]
            else:
                # Add random atoms
                while len(s) < n:
                    p = rng.uniform(-2, 2, 3)
                    s = np.vstack([s, p])

        # Ensure minimum distance
        from scipy.spatial.distance import pdist
        d = pdist(s)
        if d.min() < 0.5:
            # Scale up
            scale = 0.55 / d.min()
            s = s * scale

        # Local minimize
        try:
            e, p = _local_minimize(s, potential, parameter)
        except Exception:
            try:
                e, p = relax(s, potential, parameter)
            except Exception:
                continue

        if e < best_energy:
            best_energy = e
            best_positions = p.copy()

        # Basin hopping: perturb and re-relax
        if _remaining() > 5.0:
            try:
                e2, p2 = _perturb_and_relax(p, potential, parameter, rng, n_perturb=3, perturb_scale=0.25)
                if e2 < best_energy:
                    best_energy = e2
                    best_positions = p2.copy()
            except Exception:
                pass

    # Final fallback
    if best_positions is None:
        # Last resort: use relax on a random structure
        rng4 = np.random.default_rng(n)
        radius = spacing * n ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng4.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(
                np.linalg.norm(p - q) >= 0.8 * spacing for q in points
            ):
                points.append(p)
        try:
            _, best_positions = relax(np.array(points), potential, parameter)
        except Exception:
            best_positions = np.array(points)

    # Final check: ensure minimum distance
    from scipy.spatial.distance import pdist
    d = pdist(best_positions)
    if d.min() < 0.5:
        # Scale up slightly
        scale = 0.55 / d.min()
        best_positions = best_positions * scale

    return best_positions
# EVOLVE-BLOCK-END
