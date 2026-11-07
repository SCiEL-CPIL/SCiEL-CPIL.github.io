"""Structure prediction via hierarchical cluster growth: embed known minima as cores, grow outward."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def _icosahedron_shell(n_shell, radius):
    """Generate n_shell points on a sphere with icosahedral-like symmetry."""
    if n_shell <= 0:
        return np.empty((0, 3))
    if n_shell == 1:
        return np.array([[0, 0, radius]])
    if n_shell == 12:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        verts = verts / norms[:, None] * radius
        return verts
    # For other counts, use Fibonacci sphere
    z = 1 - 2 * (np.arange(n_shell) + 0.5) / n_shell
    r = np.sqrt(np.maximum(0, 1 - z * z))
    theta = 2 * np.pi * np.arange(n_shell) / n_shell
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z]) * radius
    return pts

def _build_core(n_core, kind, rng):
    """Build a core structure of n_core atoms using known motifs."""
    if n_core <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    if n_core == 2:
        return np.array([[0.3, 0, 0], [-0.3, 0, 0]])
    if n_core == 3:
        return np.array([[0.4, 0, 0], [-0.2, 0.35, 0], [-0.2, -0.35, 0]])
    if n_core == 4:
        s = 0.35
        return np.array([[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]])
    if n_core <= 13:
        # Icosahedral motif: 1 center + up to 12 shell
        center = np.array([[0.0, 0.0, 0.0]])
        n_shell = n_core - 1
        if n_shell <= 12:
            shell = _icosahedron_shell(n_shell, 0.7)
            return np.vstack([center, shell])
        return np.vstack([center, _icosahedron_shell(12, 0.7)])
    # For larger cores, use nested shells
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = _icosahedron_shell(12, 0.7)
    remaining = n_core - 13
    if remaining <= 0:
        return np.vstack([center, shell1])
    shell2 = _icosahedron_shell(min(remaining, 30), 1.2)
    if len(shell2) < remaining:
        extra = _icosahedron_shell(remaining - len(shell2), 1.7)
        shell2 = np.vstack([shell2, extra])
    return np.vstack([center, shell1, shell2])

def _find_surface_sites(core, n_add, kind, rng):
    """Find good positions to add n_add atoms around the core."""
    centroid = np.mean(core, axis=0)
    core_radius = np.max(np.linalg.norm(core - centroid, axis=1))
    
    n_candidates = max(n_add * 5, 50)
    z = 1 - 2 * (np.arange(n_candidates) + 0.5) / n_candidates
    r_cand = np.sqrt(np.maximum(0, 1 - z * z))
    theta = 2 * np.pi * np.arange(n_candidates) / n_candidates
    candidates = np.column_stack([r_cand * np.cos(theta), r_cand * np.sin(theta), z])
    
    target_radius = core_radius + 0.65
    candidates = candidates * target_radius + centroid
    
    dists = squareform(pdist(core))
    np.fill_diagonal(dists, np.inf)
    pair_idx = np.argsort(dists, axis=1)[:, :2]
    surface_midpoints = []
    for i in range(len(core)):
        for j in pair_idx[i]:
            if j > i:
                mid = (core[i] + core[j]) / 2
                direction = mid - centroid
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm
                    surface_midpoints.append(mid + direction * 0.3)
    if surface_midpoints:
        surface_midpoints = np.array(surface_midpoints)
        candidates = np.vstack([candidates, surface_midpoints])
    
    selected = []
    for _ in range(n_add):
        if len(candidates) == 0:
            break
        scores = []
        for c in candidates:
            d_core = np.min(np.linalg.norm(core - c, axis=1))
            if selected:
                d_sel = np.min(np.linalg.norm(np.array(selected) - c, axis=1))
            else:
                d_sel = 10.0
            score = d_core + 0.5 * d_sel
            scores.append(score)
        scores = np.array(scores)
        best_idx = np.argmax(scores)
        selected.append(candidates[best_idx])
        candidates = np.delete(candidates, best_idx, axis=0)
    
    if not selected:
        selected = []
        for _ in range(n_add):
            p = centroid + rng.normal(0, 0.5, 3)
            selected.append(p)
    
    return np.array(selected)

def _relax_and_validate(positions, potential, kind, n):
    """Relax positions and check validity."""
    if positions is None or len(positions) != n:
        return None, None
    if not np.all(np.isfinite(positions)):
        return None, None
    if n > 1:
        min_dist = np.min(pdist(positions))
        if min_dist < 0.35:
            return None, None
    try:
        energy, new_positions = relax(positions, potential, kind)
        if np.isfinite(energy) and np.all(np.isfinite(new_positions)):
            if n > 1:
                min_d = np.min(pdist(new_positions))
                if min_d < 0.35:
                    return None, None
            return energy, new_positions
    except:
        pass
    return None, None

def _grow_cluster(n_total, kind, rng, max_iterations=3):
    """Grow a cluster of n_total atoms by starting from a core and adding atoms iteratively."""
    core_size = min(n_total, max(13, int(0.5 * n_total)))
    core = _build_core(core_size, kind, rng)
    
    current = core.copy()
    for iteration in range(max_iterations):
        if len(current) >= n_total:
            break
        n_add = min(n_total - len(current), max(3, (n_total - len(current)) // 2))
        new_atoms = _find_surface_sites(current, n_add, kind, rng)
        current = np.vstack([current, new_atoms])
        
        energy, relaxed = _relax_and_validate(current, 'sutton_chen', kind, len(current))
        if relaxed is not None:
            current = relaxed
    
    if len(current) > n_total:
        current = current[:n_total]
    elif len(current) < n_total:
        n_remaining = n_total - len(current)
        centroid = np.mean(current, axis=0)
        extra = []
        for _ in range(n_remaining):
            p = centroid + rng.normal(0, 0.5, 3)
            extra.append(p)
        current = np.vstack([current, np.array(extra)])
    
    return current

def _random_compact(n, rng, min_dist=0.4):
    """Generate random compact configuration."""
    radius = 0.75 * n ** (1.0 / 3.0) * 1.3
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) == n:
        return np.array(points)
    return None

def construct(instance):
    """Hierarchical cluster growth: embed cores, grow at surface sites, relax."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    rng = np.random.default_rng(size * 7 + 123)
    
    best_energy = np.inf
    best_positions = None
    
    for core_frac in [0.3, 0.5, 0.7]:
        core_size = max(4, int(size * core_frac))
        try:
            grown = _grow_cluster(size, kind, rng, max_iterations=4)
            if grown is not None and len(grown) == size:
                energy, positions = _relax_and_validate(grown, potential, kind, size)
                if positions is not None and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except:
            pass
    
    try:
        center = np.array([[0.0, 0.0, 0.0]])
        shell1 = _icosahedron_shell(12, 0.7)
        base = np.vstack([center, shell1])
        if size <= len(base):
            base = base[:size]
            energy, positions = _relax_and_validate(base, potential, kind, size)
            if positions is not None and energy < best_energy:
                best_energy = energy
                best_positions = positions
        else:
            n_extra = size - len(base)
            shell2 = _icosahedron_shell(min(n_extra, 30), 1.2)
            if n_extra > 30:
                shell3 = _icosahedron_shell(n_extra - 30, 1.7)
                grown = np.vstack([base, shell2, shell3])
            else:
                grown = np.vstack([base, shell2])
            if len(grown) > size:
                grown = grown[:size]
            energy, positions = _relax_and_validate(grown, potential, kind, size)
            if positions is not None and energy < best_energy:
                best_energy = energy
                best_positions = positions
    except:
        pass
    
    for trial in range(4):
        min_d = 0.38 + 0.03 * trial
        rc = _random_compact(size, rng, min_dist=min_d)
        if rc is not None:
            energy, positions = _relax_and_validate(rc, potential, kind, size)
            if positions is not None and energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    try:
        a = 0.75
        max_coord = int(np.ceil((size ** (1/3)) * a * 0.8)) + 2
        points = []
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                for k in range(-max_coord, max_coord + 1):
                    for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                        p = np.array([i * a + dx * a, j * a + dy * a, k * a + dz * a])
                        if np.linalg.norm(p) <= max_coord * a * 0.9:
                            points.append(p)
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        fcc = points[idx[:size]]
        energy, positions = _relax_and_validate(fcc, potential, kind, size)
        if positions is not None and energy < best_energy:
            best_energy = energy
            best_positions = positions
    except:
        pass
    
    if best_positions is None:
        rc = _random_compact(size, rng, min_dist=0.4)
        if rc is not None:
            try:
                energy, positions = relax(rc, potential, kind)
                if np.all(np.isfinite(positions)):
                    best_positions = positions
            except:
                pass
    
    if best_positions is None:
        positions = np.zeros((size, 3))
        for i in range(size):
            positions[i] = [i * 0.5, 0, 0]
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
