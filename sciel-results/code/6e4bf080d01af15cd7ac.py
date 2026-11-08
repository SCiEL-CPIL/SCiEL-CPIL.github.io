"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Track start time for deadline management
_START_TIME = time.time()
_DEADLINE = 315.0  # safety margin before 330s kill


def _time_left():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)


def _fcc_cluster(n, spacing=1.4):
    """Generate an FCC-like cluster with n atoms."""
    pts = []
    # FCC lattice: generate points in a cubic region
    max_idx = 12
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    if np.linalg.norm(p) <= spacing * n ** (1.0 / 3.0) * 1.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        # Pad with random points
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-spacing * n ** (1.0 / 3.0), spacing * n ** (1.0 / 3.0), 3)
            pts = np.vstack([pts, p])
    return pts[:n]


def _hcp_cluster(n, spacing=1.4):
    """Generate an HCP-like cluster with n atoms."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)  # ideal HCP c/a ratio
    # Generate HCP points
    max_layer = 8
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c / 2
        if layer % 2 == 0:
            # A layer
            for i in range(-max_layer, max_layer + 1):
                for j in range(-max_layer, max_layer + 1):
                    x = i * a + (j % 2) * a / 2
                    y = j * a * np.sqrt(3) / 2
                    p = np.array([x, y, z])
                    if np.linalg.norm(p) <= a * n ** (1.0 / 3.0) * 1.5:
                        pts.append(p)
        else:
            # B layer
            for i in range(-max_layer, max_layer + 1):
                for j in range(-max_layer, max_layer + 1):
                    x = i * a + (j % 2) * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                    p = np.array([x, y, z])
                    if np.linalg.norm(p) <= a * n ** (1.0 / 3.0) * 1.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        rng = np.random.default_rng(43)
        while len(pts) < n:
            p = rng.uniform(-a * n ** (1.0 / 3.0), a * n ** (1.0 / 3.0), 3)
            pts = np.vstack([pts, p])
    return pts[:n]


def _icosa_cluster(n, spacing=1.4):
    """Generate an icosahedral-like cluster with n atoms."""
    pts = []
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    a = spacing * 0.8
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0]) * a
    
    # Add layers
    for shell in range(4):
        scale = shell * 0.8
        if shell == 0:
            pts.append([0, 0, 0])
        else:
            for v in ico_verts:
                p = v * (1 + scale)
                pts.append(p)
    
    pts = np.array(pts)
    if len(pts) < n:
        rng = np.random.default_rng(44)
        while len(pts) < n:
            p = rng.uniform(-a * 3, a * 3, 3)
            pts = np.vstack([pts, p])
    return pts[:n]


def _random_compact(n, spacing=1.4, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in pts):
                pts.append(p)
        attempts += 1
    # Pad if needed
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:n])


def _simulated_annealing(positions, kind, n_exp, m_exp, c, t_max=500.0, t_min=0.01, cooling=0.995, seed=None):
    """Simple simulated annealing to escape local minima."""
    rng = np.random.default_rng(seed)
    energy, _ = sutton_chen_energy_gradient(positions, kind)
    best_energy = energy
    best_positions = positions.copy()
    n = len(positions)
    
    for t in np.arange(t_max, t_min, -cooling * t_max * 0.01):
        if _time_left() < 5:
            break
        # Propose a move
        i = rng.integers(n)
        displacement = rng.normal(0, 0.1, 3)
        new_positions = positions.copy()
        new_positions[i] += displacement
        
        new_energy, _ = sutton_chen_energy_gradient(new_positions, kind)
        
        # Acceptance criterion
        if new_energy < energy or rng.random() < np.exp(-(new_energy - energy) / max(t, 1e-10)):
            positions = new_positions
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
    
    return best_energy, best_positions


def construct(instance):
    """Best of multiple strategies with simulated annealing and multi-start relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # Extract parameters
    if kind == "12-6":
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == "9-6":
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == "10-8":
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        n_exp, m_exp, c = 12, 6, 144.41
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: FCC-based
    if _time_left() > 30:
        try:
            fcc_pts = _fcc_cluster(size)
            energy, positions = relax(fcc_pts, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    # Strategy 2: HCP-based
    if _time_left() > 30:
        try:
            hcp_pts = _hcp_cluster(size)
            energy, positions = relax(hcp_pts, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    # Strategy 3: Icosahedral-based
    if _time_left() > 30:
        try:
            ico_pts = _icosa_cluster(size)
            energy, positions = relax(ico_pts, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    # Strategy 4-10: Random compact with simulated annealing
    for seed in range(10):
        if _time_left() < 15:
            break
        try:
            rand_pts = _random_compact(size, seed=seed + 100)
            # First relax locally
            energy, positions = relax(rand_pts, potential, kind)
            
            # Then apply simulated annealing to escape local minima
            sa_energy, sa_positions = _simulated_annealing(
                positions, kind, n_exp, m_exp, c,
                t_max=300.0, t_min=0.05, cooling=0.99, seed=seed
            )
            
            # Final polish
            final_energy, final_positions = relax(sa_positions, potential, kind)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_positions.copy()
        except Exception:
            pass
    
    # Strategy 11-15: More random starts with direct SA
    for seed in range(5):
        if _time_left() < 10:
            break
        try:
            rand_pts = _random_compact(size, seed=seed + 200)
            sa_energy, sa_positions = _simulated_annealing(
                rand_pts, kind, n_exp, m_exp, c,
                t_max=400.0, t_min=0.02, cooling=0.99, seed=seed + 50
            )
            final_energy, final_positions = relax(sa_positions, potential, kind)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_positions.copy()
        except Exception:
            pass
    
    # Final polish of best result
    if best_positions is not None and _time_left() > 5:
        try:
            final_energy, final_positions = relax(best_positions, potential, kind)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_positions.copy()
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
