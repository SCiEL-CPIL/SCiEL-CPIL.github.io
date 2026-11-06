"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Leave 10s safety margin for 300s limit


def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT


def _generate_icosahedron(n, scale=1.0):
    """
    Generate n points with icosahedral symmetry.
    Starts with the 12 vertices of a regular icosahedron, then fills with
    a second shell and interior points to reach n.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    points = [verts * scale]
    
    # If we need more than 12 points, add a second shell (20 points of dodecahedron)
    if n > 12:
        # Dodecahedron vertices (20 points)
        dodec = [
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
            [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
            [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
        ]
        dodec = np.array(dodec, dtype=float)
        dodec_norms = np.linalg.norm(dodec, axis=1, keepdims=True)
        dodec = dodec / dodec_norms * (scale * 1.6)
        points.append(dodec)
    
    # If we still need more, add interior points in a BCC-like pattern
    if n > 32:
        # Add center
        points.append(np.array([[0, 0, 0]]))
        
        # Add more interior points
        remaining = n - sum(len(p) for p in points)
        if remaining > 0:
            # Simple cubic interior points
            interior = []
            s = scale * 0.5
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        if len(interior) >= remaining:
                            break
                        p = np.array([i, j, k]) * s
                        if np.linalg.norm(p) < scale * 1.2:
                            interior.append(p)
            if interior:
                points.append(np.array(interior))
    
    # Concatenate and truncate to n
    all_points = np.vstack(points)
    if len(all_points) > n:
        all_points = all_points[:n]
    elif len(all_points) < n:
        # Pad with random points if needed
        rng = np.random.default_rng(42)
        extra = rng.uniform(-scale, scale, (n - len(all_points), 3))
        all_points = np.vstack([all_points, extra])
    
    return all_points


def construct(instance):
    """
    Find low-energy Dzugutov cluster using icosahedral starting points.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Try different scales for the icosahedron
    for scale in [1.0, 1.1, 0.9, 1.05]:
        if not _check_time():
            break
            
        try:
            positions = _generate_icosahedron(size, scale)
            energy, relaxed = relax(positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            continue
    
    # If time remains, try random restarts
    if _check_time():
        rng = np.random.default_rng(size)
        spacing = 1.1
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        
        for _ in range(10):
            if not _check_time():
                break
                
            try:
                points = []
                while len(points) < size:
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                        points.append(p)
                energy, positions = relax(np.array(points), potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    # Fallback if no valid result
    if best_positions is None:
        rng = np.random.default_rng(size)
        best_positions = rng.uniform(-1, 1, (size, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
