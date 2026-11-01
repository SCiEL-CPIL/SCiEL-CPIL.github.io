"""
Lennard-Jones 55-atom cluster optimisation.

Strategy:
  1. Generate Mackay icosahedron geometry with optimized shell radii.
  2. Optimize radii (r1, r2) via Nelder-Mead on a reduced parameterization.
  3. Use L-BFGS-B with analytical gradient to relax to a local minimum.
  4. Multi-start with small perturbations and keep the best.
  5. Final high-precision polish pass.
  6. Respect the 150 s wall-clock deadline with a safety margin.

The 55-atom Mackay icosahedron has:
  - 1 central atom
  - 12 atoms on the first shell (icosahedral vertices) at distance r1
  - 42 atoms on the second shell at distance r2
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 55
DEADLINE = 145.0  # safety margin before 150 s kill

# ---------------------------------------------------------------------------
# Lennard-Jones energy and gradient
# ---------------------------------------------------------------------------

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms.  pos shape (N,3)."""
    N = pos.shape[0]
    # Pairwise differences: (N, N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # diff[i,j] = pos[i]-pos[j]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid divide by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 ** 2
    # Energy per pair (count each pair once): 4*(r^-12 - r^-6)
    # Sum over i<j  =>  0.5 * (sum over all i≠j)
    energy = 4.0 * 0.5 * np.sum(r12 - r6 - np.diag(r12 - r6))
    # Actually simpler: just sum over all pairs i<j
    # Let me redo: energy = sum_{i<j} 4*(r_ij^-12 - r_ij^-6)
    # Using the matrix: energy = 0.5 * sum_{i≠j} 4*(r12 - r6)
    energy = 2.0 * np.sum(np.triu(r12 - r6, k=1))
    # Gradient: dE/dpos[i] = sum_{j≠i} dE/dpos[i]
    # d/dx_i [4*(r^-12 - r^-6)] = 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j) / r
    # = 4*(6*r^-8 - 12*r^-14) * (x_i - x_j) / r
    # = 24*(r^-8 - 2*r^-14) * (x_i - x_j) / r
    # Note: d(r)/d(x_i) = (x_i - x_j)/r
    # So dE/dx_i = sum_j 4*(12*r^-13 - 6*r^-7) * (x_j - x_i)/r
    #            = sum_j (48*r^-13 - 24*r^-7) * (x_j - x_i) / r
    # Let's use: grad_i = sum_{j≠i} 4 * (12/r^13 - 6/r^7) * (pos_i - pos_j) / r
    # Actually: d/dr [4(r^-12 - r^-6)] = 4(-12 r^-13 + 6 r^-7) = -48 r^-13 + 24 r^-7
    # dE/dpos_i = sum_j dE/dr_ij * d r_ij / dpos_i = sum_j (-48 r^-13 + 24 r^-7) * (pos_i - pos_j)/r
    #           = sum_j (24 r^-7 - 48 r^-13) * (pos_i - pos_j) / r
    #           = sum_j (24 r^-8 - 48 r^-14) * (pos_i - pos_j)
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    coeff = (24.0 * r2 ** -4 - 48.0 * r2 ** -7)  # 24 r^-8 - 48 r^-14
    np.fill_diagonal(coeff, 0.0)
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # = sum_j coeff[i,j]*pos[i] - sum_j coeff[i,j]*pos[j]
    # = (sum_j coeff[i,j]) * pos[i] - sum_j coeff[i,j] * pos[j]
    row_sum = np.sum(coeff, axis=1)
    grad = row_sum[:, np.newaxis] * pos - coeff @ pos
    return energy, grad


# ---------------------------------------------------------------------------
# Icosahedral geometry
# ---------------------------------------------------------------------------

def _icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin,
    with circumscribed radius 1."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts /= norms
    return verts  # (12, 3)


def _mackay_subdivision_42():
    """Generate 42 points on the icosahedral surface corresponding to
    the second shell of the 55-atom Mackay icosahedron.

    Returns 30 edge-midpoint directions and 12 face-center directions,
    all normalized to unit length. The 12 face centers are chosen
    deterministically for maximum symmetry (one per vertex's opposite face).
    """
    ico = _icosahedron_vertices()  # (12, 3), unit radius
    n = 12

    # Find edges: pairs of vertices that are adjacent on icosahedron
    dists = np.linalg.norm(ico[:, np.newaxis, :] - ico[np.newaxis, :, :], axis=2)
    edge_threshold = 1.2
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < edge_threshold:
                edges.append((i, j))
    # Should be 30 edges

    # Edge midpoints
    edge_midpoints = []
    for i, j in edges:
        mid = (ico[i] + ico[j]) / 2.0
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)  # (30, 3)

    # Build adjacency
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    # Find 20 faces (triangles of mutually adjacent vertices)
    faces = set()
    for i in range(n):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and k in adj[i]:
                    face = tuple(sorted((i, j, k)))
                    faces.add(face)
    # Should be 20 faces

    # For each vertex, find the face opposite to it (the face whose centroid
    # is most anti-parallel to the vertex direction). This gives 12 unique
    # face centers (each face is opposite to exactly one vertex in an icosahedron
    # with 20 faces and 12 vertices... actually 20 faces, 12 vertices, so not
    # a perfect 1-to-1. Let's pick 12 faces that maximize symmetry.)
    # 
    # Alternative: In the Mackay icosahedron, the 42 outer atoms are at:
    # 30 edge midpoints + 12 face centers. The 12 face centers correspond to
    # 12 of the 20 faces. A symmetric choice: pick faces that form a
    # "dual" subset. For now, pick the 12 faces with the largest
    # solid angle from the center, or simply use a deterministic selection
    # that respects the icosahedral symmetry as much as possible.
    #
    # Practical approach: pick 12 face centers that are well-distributed.
    # Use a greedy farthest-point selection for determinism and symmetry.
    face_centers_all = []
    for f in sorted(faces):
        center = (ico[f[0]] + ico[f[1]] + ico[f[2]]) / 3.0
        face_centers_all.append(center)
    face_centers_all = np.array(face_centers_all)  # (20, 3)
    fc_norms = np.linalg.norm(face_centers_all, axis=1, keepdims=True)
    face_centers_all /= fc_norms

    # Greedy selection of 12 face centers maximizing minimum pairwise distance
    selected = [0]  # start with face 0
    for _ in range(11):
        # For each unselected face, compute min distance to selected faces
        min_dists = np.full(20, np.inf)
        for idx in selected:
            d = np.linalg.norm(face_centers_all - face_centers_all[idx], axis=1)
            min_dists = np.minimum(min_dists, d)
        # Mask already selected
        for idx in selected:
            min_dists[idx] = -1
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)

    selected_face_centers = face_centers_all[selected]  # (12, 3)

    outer_dirs = np.vstack([edge_midpoints, selected_face_centers])  # (42, 3)
    outer_norms = np.linalg.norm(outer_dirs, axis=1, keepdims=True)
    outer_dirs /= outer_norms
    return ico, outer_dirs  # ico: (12,3) unit, outer_dirs: (42,3) unit


# ---------------------------------------------------------------------------
# Construct initial positions
# ---------------------------------------------------------------------------

def _make_initial_guess(r1=1.1225, r2=2.051):
    """Return (N,3) initial positions close to Mackay icosahedron."""
    ico, outer_dirs = _mackay_subdivision_42()
    center = np.zeros((1, 3))
    shell1 = ico * r1
    shell2 = outer_dirs * r2
    pos = np.vstack([center, shell1, shell2])  # (55, 3)
    return pos


def _optimize_radii(ico, outer_dirs, t0=None, time_limit=20.0):
    """Optimize shell radii r1, r2 by minimizing LJ energy with fixed directions.
    
    Parameters:
        ico: (12, 3) unit icosahedral vertex directions
        outer_dirs: (42, 3) unit outer shell directions
        t0: start time for deadline tracking
        time_limit: max seconds for this optimization
    
    Returns:
        (r1_opt, r2_opt, best_energy)
    """
    if t0 is None:
        t0 = time.time()

    def _energy_from_radii(params):
        r1, r2 = params
        if r1 < 0.5 or r2 < 0.5:
            return 1e10
        center = np.zeros((1, 3))
        shell1 = ico * r1
        shell2 = outer_dirs * r2
        pos = np.vstack([center, shell1, shell2])
        e, _ = _lj_energy_and_grad(pos)
        return e

    # Start near literature values
    x0 = np.array([1.1225, 2.051])
    res = minimize(
        _energy_from_radii,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
    )
    r1_opt, r2_opt = res.x
    best_e = res.fun

    # Check deadline
    if t0 is not None and time.time() - t0 > time_limit:
        pass  # just return what we have

    return r1_opt, r2_opt, best_e


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

def _relax(pos, time_limit=10.0, maxiter=5000, ftol=1e-14, gtol=1e-10):
    """Relax positions using L-BFGS-B.  Returns (positions, energy, elapsed)."""
    start_time = time.time()

    def obj_and_grad(p):
        pos = p.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.ravel()

    res = minimize(
        obj_and_grad,
        pos.ravel(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxiter * 4},
    )
    elapsed = time.time() - start_time
    if res.success or hasattr(res, 'x'):
        new_pos = res.x.reshape(-1, 3)
        e, _ = _lj_energy_and_grad(new_pos)
        return new_pos, e, elapsed
    return pos, _lj_energy_and_grad(pos)[0], elapsed


# ---------------------------------------------------------------------------
# Multi-start search
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()

    # --- Step 1: Get geometry and optimize radii ---
    ico, outer_dirs = _mackay_subdivision_42()
    r1_opt, r2_opt, _ = _optimize_radii(ico, outer_dirs, t0=t0, time_limit=15.0)

    elapsed = time.time() - t0
    if elapsed > DEADLINE - 60.0:
        # Not enough time, use literature values
        r1_opt, r2_opt = 1.1225, 2.051

    # --- Step 2: Build initial guess with optimized radii ---
    pos0 = _make_initial_guess(r1=r1_opt, r2=r2_opt)
    pos_best, e_best, _ = _relax(pos0, time_limit=15.0)

    elapsed = time.time() - t0
    if elapsed > DEADLINE - 40.0:
        return pos_best

    # --- Step 3: Multi-start with small perturbations around best ---
    rng = np.random.RandomState(123)
    n_starts = 4
    for s in range(n_starts):
        if time.time() - t0 > DEADLINE - 25.0:
            break
        noise = rng.normal(0, 0.1, size=(N, 3))
        pos_pert = pos_best + noise
        pos_new, e_new, _ = _relax(pos_pert, time_limit=12.0)
        if e_new < e_best:
            pos_best, e_best = pos_new, e_new

    # --- Step 4: Try perturbations from the original guess ---
    if time.time() - t0 < DEADLINE - 20.0:
        for s in range(2):
            if time.time() - t0 > DEADLINE - 12.0:
                break
            noise = rng.normal(0, 0.2, size=(N, 3))
            pos_pert = pos0 + noise
            pos_new, e_new, _ = _relax(pos_pert, time_limit=10.0)
            if e_new < e_best:
                pos_best, e_best = pos_new, e_new

    # --- Step 5: Final high-precision polish ---
    if time.time() - t0 < DEADLINE - 5.0:
        pos_final, e_final, _ = _relax(
            pos_best, time_limit=8.0, maxiter=10000, ftol=1e-16, gtol=1e-12
        )
        if e_final < e_best:
            pos_best, e_best = pos_final, e_final

    return pos_best
