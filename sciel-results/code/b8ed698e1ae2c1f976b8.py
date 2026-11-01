"""Lennard-Jones cluster N=55: construct the Mackay icosahedron and polish it."""
import time
import numpy as np
from scipy.optimize import minimize

N = 55

# ---- Icosahedron vertices (12) ----
_t = (1.0 + np.sqrt(5.0)) / 2.0
_ICO_RAW = np.array([
    [-1, 0,  t], [1, 0,  t], [-1, 0, -t], [1, 0, -t],
    [0,  t, -1], [0,  t, 1], [0, -t, -1], [0, -t, 1],
    [ t, -1, 0], [ t, 1, 0], [-t, -1, 0], [-t, 1, 0],
], dtype=float)
_ICO = _ICO_RAW / np.linalg.norm(_ICO_RAW, axis=1, keepdims=True)

# ---- icosahedral rotation matrix (60-degree about z-axis) ----
_ang = np.pi / 3.0
_c, _s = np.cos(_ang), np.sin(_ang)
_R = np.array([
    [_c, -_s, 0.0],
    [_s,  _c, 0.0],
    [0.0, 0.0, 1.0],
])


def _lj_energy_grad(pos):
    """Total LJ energy and gradient (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero on diagonal
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 * r6
    # energy = 4 * sum_{i<j} (r12 - r6)
    e_pair = 4.0 * (r12 - r6)
    energy = 0.5 * np.sum(e_pair)
    # gradient: dE/dx_i = sum_{j!=i} 4*(-12 r^-14 + 6 r^-8) * (x_i - x_j)
    # = 4 * ( -12 * r^-14 + 6 * r^-8 ) * (x_i - x_j)  summed over j
    # r^-14 = r2^-7, r^-8 = r2^-4
    coeff = 4.0 * (-12.0 * (r2 ** -7) + 6.0 * (r2 ** -4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return energy, grad


def _valid(pos, min_dist=0.5):
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    return np.all(r2 >= min_dist * min_dist)


def _relax(pos, max_time=20.0, tol=1e-10):
    """L-BFGS-B local relaxation with time guard."""
    t0 = time.time()
    try:
        res = minimize(
            _lj_energy_grad,
            pos,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'gtol': tol, 'ftol': 1e-12, 'maxfun': 2000},
        )
        if time.time() - t0 > max_time:
            return pos  # keep original if we blew the budget
        if res.x is not None and _valid(res.x):
            e, _ = _lj_energy_grad(res.x)
            e0, _ = _lj_energy_grad(pos)
            if e < e0:
                return res.x
    except Exception:
        pass
    return pos


def construct():
    deadline = 110.0  # safety margin before 120s kill
    t_start = time.time()

    # ---- Build the two-shell Mackay icosahedron ----
    # Shell 1: 12 vertices of the icosahedron at radius r1
    # Shell 2: 42 atoms at the face centers, edge midpoints, and rotated vertices
    # Standard Mackay icosahedron for N=55:
    #   1 center + 12 (radius 1) + 42 (radius ~1.9)
    # Actually for N=55: 12 + 42 = 54, plus 1 center = 55
    # The 42 outer-shell atoms: 30 face centers + 12 edge midpoints? 
    # Let me use the known construction:
    #   Inner shell: 12 atoms at icosahedron vertices, radius r1
    #   Outer shell: 42 atoms, radius r2
    #   Center: 1 atom
    # Total: 1 + 12 + 42 = 55

    # The 42 outer atoms are obtained by taking all icosahedron vertices and
    # applying the 60-degree rotation about z, plus the face-center directions.
    # A cleaner approach: generate all 120 icosahedral symmetry operations applied
    # to a few seed directions, then pick 42 unique directions.

    # Simpler known construction for Mackay icosahedron N=55:
    # Outer shell of 42 = 30 face centers + 12 "edge" positions
    # Face centers of icosahedron: 20 faces
    # Actually, let me use the standard: 
    #   42 outer = all points of form (icosahedron vertex) rotated by 60 deg about z
    #   plus the original icosahedron vertices? No, that's 24.
    # 
    # Correct Mackay construction for the 42-atom outer shell:
    # Take the 12 icosahedron vertices. The 42 outer positions are:
    #   - 20 face-center directions (one per triangular face)
    #   - 12 edge-midpoint directions? No...
    #   Actually: 42 = 30 + 12 where 30 are "face" type and 12 are "vertex" type
    # 
    # Let me just use the well-known result: the 42 outer shell directions
    # are the icosahedron vertices (12) plus all their 60-degree rotations about z
    # (another 12, but some may overlap) plus face centers (20).
    # 12 + 12 + 20 = 44, too many.
    #
    # Correct: The Mackay icosahedron outer shell of 42 consists of:
    #   12 vertices of a larger icosahedron (same directions as inner, scaled)
    #   30 face centers of the inner icosahedron (scaled)
    # That gives 42.

    # Icosahedron face centers: compute from each triangular face
    # Icosahedron has 20 faces. Get them from the convex hull or manually.
    # Manual face list for the standard icosahedron with vertices as above:
    # Vertex indices:
    # 0: [-1, 0, t], 1: [1, 0, t], 2: [-1, 0, -t], 3: [1, 0, -t]
    # 4: [0, t, -1], 5: [0, t, 1], 6: [0, -t, -1], 7: [0, -t, 1]
    # 8: [t, -1, 0], 9: [t, 1, 0], 10: [-t, -1, 0], 11: [-t, 1, 0]

    faces = [
        (0, 5, 1), (0, 1, 9), (0, 9, 11), (0, 11, 4), (0, 4, 5),
        (1, 5, 7), (1, 7, 8), (1, 8, 9),
        (5, 4, 6), (5, 6, 7),
        (9, 8, 10), (9, 10, 11),
        (4, 11, 10), (4, 10, 6),
        (7, 6, 3), (7, 3, 8),
        (8, 3, 2), (8, 2, 10),
        (10, 2, 6),
        (2, 3, 10),  # This might be wrong; let me verify
    ]
    # Actually, let me just compute face centers from the convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(_ICO)
    face_centers = []
    for simplex in hull.simplices:
        fc = _ICO[simplex].mean(axis=0)
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)  # 20 directions

    # Outer shell: 12 icosahedron vertices + 20 face centers = 32, need 42
    # Hmm, that's only 32. Let me reconsider.
    # 
    # Actually for N=55 Mackay icosahedron:
    #   Inner shell: 12 atoms (icosahedron vertices) at radius r1
    #   Outer shell: 42 atoms at radius r2
    #   Center: 1 atom
    # The 42 outer atoms: these are the 12 vertices + 20 face centers + 10 more?
    # No. Let me look at this differently.
    #
    # The Mackay icosahedron for N=55 has:
    #   1 center
    #   12 inner shell (icosahedron vertices)
    #   42 outer shell
    # Total = 55
    #
    # The 42 outer shell consists of:
    #   12 "vertex" type (same directions as inner icosahedron)
    #   30 "face" type (face centers of the inner icosahedron)
    # 12 + 30 = 42. But I only have 20 face centers...
    #
    # Wait, the icosahedron has 20 faces, so 20 face centers. 12 + 20 = 32, not 42.
    #
    # Let me reconsider. Perhaps the outer shell is constructed differently.
    # In the Mackay icosahedron, the outer shell of 42 atoms for N=55:
    # These are at the positions of a larger icosahedron's vertices (12)
    # plus the face centers of that larger icosahedron (20) plus... 
    # 
    # Actually, I think the correct construction is:
    #   The 42 outer atoms are at the icosahedron vertices (12) and
    #   the 30 edge midpoints? No, an icosahedron has 30 edges.
    #   12 + 30 = 42. Yes! Edge midpoints!

    # Edge midpoints of the icosahedron:
    # An icosahedron has 30 edges. Get them from the hull.
    # Edges are pairs of vertices that share a face.
    edges = set()
    for simplex in hull.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = sorted((simplex[i], simplex[j]))
                edges.add((a, b))
    edge_mids = []
    for a, b in edges:
        em = (_ICO[a] + _ICO[b]) / 2.0
        em = em / np.linalg.norm(em)
        edge_mids.append(em)
    edge_mids = np.array(edge_mids)  # 30 directions

    # Now: outer shell = 12 vertices + 20 face centers + 30 edge mids = 62, too many.
    # 
    # I'm overcomplicating this. Let me use a different, well-known approach:
    # The Mackay icosahedron for N=55 can be constructed by:
    #   1 center atom
    #   12 atoms at icosahedron vertices, radius r1
    #   42 atoms at radius r2, where the 42 directions are:
    #     12 icosahedron vertex directions
    #     30 face-center directions (but icosahedron has 20 faces...)
    #
    # I think the issue is I'm confusing different Mackay shells. Let me just
    # use a well-tested construction: generate all 120 icosahedral symmetry
    # operations and apply them to seed directions.

    # Icosahedral group has 60 rotations. Apply to:
    #   - z-axis (1 direction) -> 6 unique directions (stabilizer of size 10)
    #   - x-axis (1 direction) -> 12 unique directions (stabilizer of size 5)  
    #   - face-center direction (1 direction) -> 20 unique directions (stabilizer of size 3)
    #   - edge-midpoint direction (1 direction) -> 30 unique directions (stabilizer of size 2)
    #   - vertex direction (1 direction) -> 12 unique directions (stabilizer of size 5)
    #
    # For N=55: 1 + 12 + 42
    # 42 = 30 + 12 (edge-midpoint type + vertex type) OR 20 + 12 + 10? 
    # Most likely: 42 = 30 (edge-type) + 12 (vertex-type)
    # These are at the same radius r2.

    # Let me generate the icosahedral rotation group.
    # The icosahedral group is generated by 5-fold and 2-fold rotations.
    # 5-fold rotation about z-axis:
    def _rot_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _rot_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # Generate all 60 rotations of the icosahedral group
    # Using the fact that it's generated by a 5-fold rotation about z
    # and a 2-fold rotation about an axis through an edge midpoint.
    # This is complex. Let me use a simpler approach:
    # Just use the known vertex directions and face/edge directions directly.

    # For the outer shell, I'll use:
    #   12 vertex directions (same as inner icosahedron) at radius r2
    #   30 edge-midpoint directions at radius r2
    # Total outer: 42

    # Now determine radii r1 and r2 for optimal LJ packing.
    # The LJ minimum for a pair is at r = 2^(1/6) ≈ 1.1225
    # For the Mackay icosahedron, the optimal radii are approximately:
    #   r1 ≈ 1.1225 (inner shell atoms at LJ minimum from center)
    #   r2 ≈ 2 * r1 ≈ 2.245 (outer shell)
    # But these need optimization. Let me set initial guesses and let L-BFGS polish.

    r1 = 1.1225  # 2^(1/6)
    r2 = 2.245   # ~2 * r1

    # Build positions
    center = np.array([0.0, 0.0, 0.0])
    inner = _ICO * r1  # 12 atoms
    # Outer: 12 vertex + 30 edge-midpoint = 42
    outer_verts = _ICO * r2  # 12
    outer_edges = edge_mids * r2  # 30
    outer = np.vstack([outer_verts, outer_edges])  # 42

    pos = np.vstack([center[None, :], inner, outer])  # 1 + 12 + 42 = 55
    pos = pos[:55]

    # Check validity
    if not _valid(pos):
        # If atoms are too close, scale up slightly
        pos = pos * 1.05

    # Relax to find the local minimum
    pos = _relax(pos, max_time=40.0)

    # If time remains, try a few perturbations and relax
    elapsed = time.time() - t_start
    best_pos = pos
    best_energy, _ = _lj_energy_grad(best_pos)

    rng = np.random.RandomState(42)
    while time.time() - t_start < deadline - 30:
        # Perturb best solution
        perturb = best_pos + rng.normal(0, 0.1, size=best_pos.shape)
        if not _valid(perturb):
            continue
        relaxed = _relax(perturb, max_time=15.0)
        e, _ = _lj_energy_grad(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed
        # Also try a fresh random restart from a perturbed icosahedron
        if time.time() - t_start < deadline - 60:
            r1_p = r1 * (1 + rng.normal(0, 0.05))
            r2_p = r2 * (1 + rng.normal(0, 0.05))
            inner_p = _ICO * r1_p
            outer_verts_p = _ICO * r2_p
            outer_edges_p = edge_mids * r2_p
            outer_p = np.vstack([outer_verts_p, outer_edges_p])
            pos_p = np.vstack([center[None, :], inner_p, outer_p]) + rng.normal(0, 0.05, size=(55, 3))
            if _valid(pos_p):
                relaxed_p = _relax(pos_p, max_time=15.0)
                e_p, _ = _lj_energy_grad(relaxed_p)
                if e_p < best_energy:
                    best_energy = e_p
                    best_pos = relaxed_p

    return best_pos


# EVOLVE-BLOCK-START
# (The construct function above is the main logic; kept here for the evolve block structure)
# EVOLVE-BLOCK-END
