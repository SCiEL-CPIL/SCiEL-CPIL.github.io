"""Improved Lennard-Jones cluster global minimum, 55 atoms."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

# Deadline guard: 150s wall-clock limit, use 145s for safety
_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _lj_energy(pos):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    d = pdist(pos)
    # Avoid division by zero (shouldn't happen with valid inputs)
    d = np.maximum(d, 1e-10)
    r6 = d ** -6
    return 4.0 * np.sum(r6 ** 2 - r6)


def _lj_grad(pos):
    """Compute gradient of LJ energy."""
    n = len(pos)
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                continue
            r6 = r2 ** -3
            r12 = r6 ** 2
            # d/dr of 4*(r^-12 - r^-6) * dr/dri
            # f(r) = 4*(r^-12 - r^-6), f'(r) = 4*(-12*r^-13 + 6*r^-7)
            # grad_i = f'(r) * (pos[i]-pos[j])/r = 4*(-12*r^-14 + 6*r^-8) * rij
            # = 4*(-12*r12 + 6*r6)/r2 * rij
            coeff = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2
            grad[i] += coeff * rij
            grad[j] -= coeff * rij
    return grad


def _lj_energy_and_grad(pos):
    """Combined energy and gradient for L-BFGS-B."""
    return _lj_energy(pos), _lj_grad(pos)


def _mackay_icosahedron_55():
    """Construct the two-shell Mackay icosahedron for 55 atoms.
    
    Shell 1: 12 vertices of icosahedron (radius r1)
    Shell 2: 42 vertices (12 face centers + 30 edge midpoints scaled)
    
    Standard Mackay icosahedron:
    - 1 center atom
    - 12 atoms at icosahedron vertices (radius a)
    - 42 atoms at... actually for 55: 1 + 12 + 42 = 55
    - Inner shell: 12 at radius r1
    - Outer shell: 42 at radius r2
    
    For LJ, optimal radii: r1 ≈ 1.12 (min of LJ), r2 ≈ 2.14 (next minimum)
    Actually the Mackay icosahedron has specific geometry.
    
    Icosahedron vertices (radius 1):
    (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) where φ = (1+√5)/2
    Normalized to unit radius.
    """
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # For LJ, the first shell radius that minimizes interaction with center
    # is at r = 1.12 (2^(1/6)). But in Mackay, we need to optimize.
    # The known best: inner shell at ~1.12, outer shell at ~2.24
    # Let's use the standard Mackay geometry and then relax.
    
    r1 = 1.12  # First shell radius
    r2 = 2.24  # Second shell radius (approximately 2*r1)
    
    # Shell 1: 12 icosahedron vertices at radius r1
    shell1 = verts * r1
    
    # Shell 2: 42 points
    # In Mackay icosahedron, the second shell consists of:
    # - 12 points at face centers (scaled)
    # - 30 points at edge midpoints (scaled)
    
    # Face centers of icosahedron: 20 faces
    # Each face is a triangle. The face center direction is the normalized sum of 3 vertices.
    # But for Mackay, the 42 outer points are at specific positions.
    
    # Actually, let me construct the 42 outer shell points properly.
    # The icosahedron has 12 vertices, 30 edges, 20 faces.
    # Mackay outer shell: 12 (corresponding to vertices, pushed out) + 30 (edge midpoints)
    # = 42 points.
    
    # Edge midpoints: for each pair of connected vertices, take midpoint direction
    # Icosahedron edges: connect vertices that are closest neighbors.
    # In unit icosahedron, edge length = 2 (for the standard coords above, 
    # the distance between (0,1,phi) and (1,phi,0) is sqrt(1 + (phi-1)^2 + phi^2))
    
    # Let me just find all pairs with minimum distance
    from scipy.spatial.distance import cdist
    D = cdist(verts, verts)
    np.fill_diagonal(D, np.inf)
    min_dist = D.min()
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    # Edge midpoint directions (30 edges)
    edge_mids = []
    for i, j in edges:
        mid_dir = (verts[i] + verts[j]) / np.linalg.norm(verts[i] + verts[j])
        edge_mids.append(mid_dir)
    edge_mids = np.array(edge_mids)  # shape (30, 3)
    
    # Vertex directions for outer shell (12 points, same directions as inner but at r2)
    vertex_outer = verts * r2
    
    # Combine: 12 vertex-outer + 30 edge-midpoint-outer = 42
    shell2 = np.vstack([vertex_outer, edge_mids * r2])
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    positions = np.vstack([center, shell1, shell2])
    return positions


def _random_cluster(n=55, seed=None):
    """Generate a random cluster with minimum distance 0.5."""
    rng = np.random.default_rng(seed)
    # Place atoms in a sphere of radius ~6
    pos = []
    for _ in range(n):
        for attempt in range(1000):
            p = rng.uniform(-6, 6, size=3)
            if len(pos) == 0:
                pos.append(p)
                break
            existing = np.array(pos)
            dists = np.linalg.norm(existing - p, axis=1)
            if dists.min() > 0.5:
                pos.append(p)
                break
    if len(pos) < n:
        # Fallback: cubic grid
        side = int(np.ceil(n ** (1/3)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
                    if len(pts) >= n:
                        break
        pos = pts[:n]
    return np.array(pos, dtype=float) * 1.12


def _relax(positions, maxiter=500):
    """L-BFGS-B local relaxation."""
    if _remaining() < 5.0:
        return positions
    
    def fun(x):
        x = x.reshape(-1, 3)
        e = _lj_energy(x)
        g = _lj_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
    return result.x.reshape(-1, 3)


def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9


def construct():
    global _start_time
    _start_time = time.time()
    
    best_pos = None
    best_energy = np.inf
    
    # Strategy 1: Start from Mackay icosahedron and relax
    if _remaining() > 10.0:
        try:
            mackay = _mackay_icosahedron_55()
            relaxed = _relax(mackay, maxiter=300)
            e = _lj_energy(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Strategy 2: Multiple random starts with relaxation
    rng = np.random.default_rng(42)
    n_starts = 8
    for i in range(n_starts):
        if _remaining() < 15.0:
            break
        try:
            # Various random initializations
            if i == 0:
                # Spherical shell random
                pos = _random_cluster(55, seed=42 + i)
            elif i == 1:
                # Two concentric shells random
                pos = np.zeros((55, 3))
                # 1 center
                pos[0] = [0, 0, 0]
                # 12 inner shell
                for k in range(12):
                    theta = rng.uniform(0, 2 * np.pi)
                    phi = np.arccos(rng.uniform(-1, 1))
                    r = 1.12
                    pos[k+1] = [r*np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi)]
                # 42 outer shell
                for k in range(42):
                    theta = rng.uniform(0, 2 * np.pi)
                    phi = np.arccos(rng.uniform(-1, 1))
                    r = 2.24
                    pos[k+13] = [r*np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi)]
            elif i == 2:
                # Cubic lattice with small perturbation
                side = 4
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-1.5)**2 + (p[1]-1.5)**2 + (p[2]-1.5)**2)
                pos = np.array(pts[:55], dtype=float) * 1.12
                pos += rng.normal(0, 0.05, pos.shape)
            elif i == 3:
                # FCC-like packing
                pos = []
                a = 1.12
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            if len(pos) < 55:
                                p = np.array([i, j, k]) * a
                                # FCC offset
                                if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                                    p += np.array([0.5, 0.5, 0.5]) * a
                                pos.append(p)
                pos = np.array(pos[:55], dtype=float)
                pos += rng.normal(0, 0.03, pos.shape)
            else:
                # Different random seeds
                pos = _random_cluster(55, seed=100 + i)
            
            relaxed = _relax(pos, maxiter=200)
            e = _lj_energy(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    # Final polish on best result
    if best_pos is not None and _remaining() > 10.0:
        try:
            polished = _relax(best_pos, maxiter=1000)
            e = _lj_energy(polished)
            if _check_min_dist(polished) and e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: original cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
# EVOLVE-BLOCK-END
