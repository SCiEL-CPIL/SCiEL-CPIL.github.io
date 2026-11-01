"""Lennard-Jones cluster global minimum, 55 atoms: Mackay icosahedron construction + relaxation."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    inv_r6 = 1.0 / (r2 * r2 * r2)
    inv_r12 = inv_r6 * inv_r6
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    # Subtract self terms (diagonal was set to 1, so we added 4*(1-1)=0, fine)
    return energy

def _mckay_icosahedron():
    """
    Construct the Mackay icosahedron (two-shell) for 55 atoms.
    
    The Mackay icosahedron has:
    - 1 atom at center
    - 12 atoms on inner shell (vertices of icosahedron)
    - 30 atoms on middle shell (edge midpoints, scaled)
    - 12 atoms on outer shell (vertices of larger icosahedron)
    
    Total: 1 + 12 + 30 + 12 = 55
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of a regular icosahedron (circumscribed radius ~1)
    # Standard icosahedron vertices: (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = []
    for sign1 in [1, -1]:
        for sign2 in [1, -1]:
            verts.append([0.0, sign1 * 1.0, sign2 * phi])
            verts.append([sign1 * 1.0, sign2 * phi, 0.0])
            verts.append([sign2 * phi, 0.0, sign1 * 1.0])
    verts = np.array(verts)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    # The inner shell radius: for LJ minimum, nearest-neighbor distance ~ 2^(1/6) ≈ 1.12246
    # In the Mackay icosahedron, the inner shell atoms are at distance r1 from center.
    # The distance between adjacent inner-shell atoms (icosahedron edge) should be ~1.12246.
    # For an icosahedron with circumradius R, edge length = 2*R*sin(pi/5) ≈ 2*R*0.5878 = 1.1756*R
    # So R = 1.12246 / 1.1756 ≈ 0.9548
    # But actually, let's just use a reasonable starting radius and let relaxation fix it.
    
    # Use radii based on known Mackay icosahedron geometry
    # Inner shell: 12 vertices
    r_inner = 0.955  # approximate
    inner_atoms = verts_unit * r_inner
    
    # Middle shell: 30 edge midpoints
    # Each edge of the icosahedron connects two vertices. The midpoint of each edge,
    # projected to a sphere of radius r_mid.
    # Find all edges: pairs of vertices with the minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[j, i] = dists[i, j]
    
    # The minimum non-zero distance is the edge length
    min_dist = np.min(dists[dists > 1e-6])
    
    # Find all edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-8:
                edges.append((i, j))
    
    # 30 edges expected
    assert len(edges) == 30, f"Expected 30 edges, got {len(edges)}"
    
    # Edge midpoints, normalized to unit sphere, then scaled
    midpoints = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid_unit = mid / mid_norm
        midpoints.append(mid_unit)
    midpoints = np.array(midpoints)
    
    # Middle shell radius: between inner and outer
    r_mid = 1.35  # approximate
    mid_atoms = midpoints * r_mid
    
    # Outer shell: 12 vertices (same directions as inner, but larger radius)
    r_outer = 1.95  # approximate
    outer_atoms = verts_unit * r_outer
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Combine all atoms
    all_atoms = np.vstack([center, inner_atoms, mid_atoms, outer_atoms])
    
    # Ensure exactly 55 atoms
    assert all_atoms.shape[0] == 55, f"Expected 55 atoms, got {all_atoms.shape[0]}"
    
    return all_atoms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Step 1: Construct Mackay icosahedron
    pos = _mckay_icosahedron()
    
    # Step 2: L-BFGS relaxation
    try:
        result = minimize(
            _lj_energy,
            pos,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if time.time() > deadline:
            return result.x
        pos = result.x
    except Exception:
        pass
    
    # Step 3: If time remains, do a tighter polish
    if time.time() < deadline - 5.0:
        try:
            result2 = minimize(
                _lj_energy,
                pos,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            pos = result2.x
        except Exception:
            pass
    
    return pos
# EVOLVE-BLOCK-END
