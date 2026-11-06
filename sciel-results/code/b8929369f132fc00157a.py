"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _dzugutov_energy(pos):
    """Compute Dzugutov energy for positions."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 0.5:
                return 1e10
            if r < a:
                E += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                E += B * np.exp(d / (r - b))
    return E


def _generate_icosahedral_shell(n, shell_radii, twists):
    """
    Generate a cluster with icosahedral symmetry.
    shell_radii: list of shell radii
    twists: list of twist angles per shell
    """
    if n == 0:
        return np.empty((0, 3))
    
    # Icosahedron vertices (unit)
    phi = (1 + np.sqrt(5)) / 2
    base_ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    base_ico /= np.linalg.norm(base_ico, axis=1, keepdims=True)
    
    # Cuboctahedron vertices (12)
    cubocta = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ], dtype=float)
    cubocta /= np.linalg.norm(cubocta, axis=1, keepdims=True)
    
    # Great circle (20 points for larger clusters)
    # Use points on a great circle at different longitudes
    def make_great_circle(n_pts, radius, angle_offset=0.0):
        pts = []
        for k in range(n_pts):
            theta = 2 * np.pi * k / n_pts + angle_offset
            pts.append([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        return np.array(pts)
    
    # Shell definitions for polytetrahedral clusters
    # Typical Dzugutov clusters: 13 (1+12), 55 (1+12+42), etc.
    # For 38-60 atoms, use shells: 1 center + 12 ico + 12 cubocta + remainder on next shell
    
    positions = []
    remaining = n
    
    # Center atom
    if remaining > 0:
        positions.append(np.array([0.0, 0.0, 0.0]))
        remaining -= 1
    
    # Shell 1: 12 icosahedral vertices
    n1 = min(12, remaining)
    if n1 > 0 and len(shell_radii) > 0:
        r1 = shell_radii[0]
        twist1 = twists[0] if len(twists) > 0 else 0.0
        # Rotate
        for i in range(n1):
            p = base_ico[i % 12].copy()
            p[0] *= r1
            p[1] *= r1
            p[2] *= r1
            # Apply twist around z-axis
            ct, st = np.cos(twist1), np.sin(twist1)
            p = np.array([p[0]*ct - p[1]*st, p[0]*st + p[1]*ct, p[2]])
            positions.append(p)
        remaining -= n1
    
    # Shell 2: 12 cuboctahedral vertices
    n2 = min(12, remaining)
    if n2 > 0 and len(shell_radii) > 1:
        r2 = shell_radii[1]
        twist2 = twists[1] if len(twists) > 1 else 0.0
        for i in range(n2):
            p = cubocta[i % 12].copy()
            p[0] *= r2
            p[1] *= r2
            p[2] *= r2
            ct, st = np.cos(twist2), np.sin(twist2)
            p = np.array([p[0]*ct - p[1]*st, p[0]*st + p[1]*ct, p[2]])
            positions.append(p)
        remaining -= n2
    
    # Shell 3: remaining atoms on a spherical shell
    if remaining > 0 and len(shell_radii) > 2:
        r3 = shell_radii[2]
        # Distribute remaining atoms roughly evenly on a sphere
        # Use Fibonacci sphere
        n3 = remaining
        golden = (1 + np.sqrt(5)) / 2
        for i in range(n3):
            z = 1 - 2 * (i + 0.5) / n3
            r_ring = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            p = np.array([r_ring * np.cos(theta), r_ring * np.sin(theta), z])
            p *= r3
            positions.append(p)
        remaining = 0
    
    # If still remaining, add more on a larger shell
    if remaining > 0:
        r4 = shell_radii[2] if len(shell_radii) > 2 else 3.0
        r4 *= 1.2
        n4 = remaining
        golden = (1 + np.sqrt(5)) / 2
        for i in range(n4):
            z = 1 - 2 * (i + 0.5) / n4
            r_ring = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            p = np.array([r_ring * np.cos(theta), r_ring * np.sin(theta), z])
            p *= r4
            positions.append(p)
    
    return np.array(positions)


def construct(instance):
    """Construct a low-energy Dzugutov cluster using icosahedral symmetry parameterization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Typical shell radii for Dzugutov clusters (in reduced units)
    # First shell ~1.1, second ~1.8, third ~2.5
    base_radii = [1.1, 1.8, 2.5]
    
    # Multiple candidate configurations with different shell radii and twists
    candidates = []
    
    # Configuration 1: Standard icosahedral
    for r_scale in [0.95, 1.0, 1.05]:
        radii = [r * r_scale for r in base_radii]
        for twist in [0.0, 0.1, -0.1]:
            for twist2 in [0.0, 0.15, -0.15]:
                candidates.append((radii, [twist, twist2, 0.0]))
    
    # Configuration 2: More compressed
    for r_scale in [0.9, 0.95]:
        radii = [r * r_scale for r in base_radii]
        candidates.append((radii, [0.0, 0.0, 0.0]))
    
    # Configuration 3: More expanded
    for r_scale in [1.1, 1.15]:
        radii = [r * r_scale for r in base_radii]
        candidates.append((radii, [0.0, 0.0, 0.0]))
    
    best_energy = np.inf
    best_positions = None
    
    for radii, twists in candidates:
        try:
            pos = _generate_icosahedral_shell(size, radii, twists)
            if len(pos) != size:
                continue
            # Check minimum distance
            if np.any(np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=2) < 0.5):
                # Fix by expanding
                pos *= 1.1
            if np.any(np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=2) < 0.5):
                continue
            energy, relaxed_pos = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            continue
    
    # If no valid candidate found, use a fallback
    if best_positions is None:
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * 1.1 for q in points):
                points.append(p)
        energy, best_positions = relax(np.array(points), potential, parameter)
    
    return best_positions
# EVOLVE-BLOCK-END
