import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill

    # Build the 11-hexagon cluster on a triangular lattice
    # Unit hexagons with side 1 have circumradius 1, inradius sqrt(3)/2
    # For hexagons of side 1 on a triangular lattice, center-to-center distance is 2*inradius = sqrt(3)
    # Lattice vectors: u = (1.5, sqrt(3)/2), v = (0, sqrt(3))
    u = np.array([1.5, SQRT3 / 2.0])
    v = np.array([0.0, SQRT3])

    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]

    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])

    # Base hexagon vertices (unit hexagon, flat-top orientation)
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [-0.5, SQRT3 / 2.0],
        [-1.0, 0.0],
        [-0.5, -SQRT3 / 2.0],
        [0.5, -SQRT3 / 2.0]
    ])

    # All vertices of all hexagons (66 vertices total)
    all_base_verts = np.tile(base_verts, (N, 1, 1))
    cluster_verts = all_base_verts + cluster_centers[:, np.newaxis, :]
    flat_verts = cluster_verts.reshape(-1, 2)

    # The key insight: for a regular hexagon enclosing a set of points,
    # the minimal side length is determined by the maximum projection of
    # any vertex onto any of the 6 normal directions of the hexagon.
    #
    # For an outer hexagon with side length L and rotation phi,
    # the 6 outward normals are at angles phi + k*60 degrees.
    # A point (x,y) is inside the hexagon iff for all k:
    #   dot(point, normal_k) <= L * inradius = L * sqrt(3)/2
    # So: L >= max_{i,k} dot(v_i, n_k) / (sqrt(3)/2) = 2 * max_{i,k} dot(v_i, n_k) / sqrt(3)
    #
    # The cluster rotates by angle theta. So the rotated vertices are:
    #   v_i(theta) = R(theta) * v_i
    # And we want: min_{theta, phi} 2 * max_{i,k} dot(R(theta)*v_i, n_k(phi)) / sqrt(3)
    #
    # Note: dot(R(theta)*v_i, n_k(phi)) = dot(v_i, R(-theta)*n_k(phi))
    # So the relative angle between cluster and hexagon matters.
    # Let alpha = phi - theta. Then:
    #   dot(R(theta)*v_i, n_k(phi)) = dot(v_i, n_k(alpha))
    # where n_k(alpha) = (cos(alpha + k*60°), sin(alpha + k*60°))
    #
    # So the problem reduces to: min_alpha 2 * max_{i,k} dot(v_i, n_k(alpha)) / sqrt(3)
    # where alpha ranges over [0, 60°) due to hexagonal symmetry.
    #
    # This is a 1D optimization! The function f(alpha) = max_{i,k} dot(v_i, n_k(alpha))
    # is a piecewise sinusoidal function (max of 66*6 = 396 sinusoids).

    # Precompute: for each vertex i and each normal direction k (k=0..5),
    # dot(v_i, n_k(alpha)) = x_i*cos(alpha + k*60°) + y_i*sin(alpha + k*60°)
    # = x_i*(cos(alpha)*cos(k*60°) - sin(alpha)*sin(k*60°)) + y_i*(sin(alpha)*cos(k*60°) + cos(alpha)*sin(k*60°))
    # = cos(alpha)*(x_i*cos_k + y_i*sin_k) + sin(alpha)*(y_i*cos_k - x_i*sin_k)
    # = A_i_k * cos(alpha) + B_i_k * sin(alpha)
    # where A_i_k = x_i*cos_k + y_i*sin_k, B_i_k = y_i*cos_k - x_i*sin_k

    x_flat = flat_verts[:, 0]  # (66,)
    y_flat = flat_verts[:, 1]  # (66,)

    # cos_k and sin_k for k=0..5 (multiples of 60 degrees)
    k_angles = np.arange(6) * np.pi / 3.0
    cos_k = np.cos(k_angles)  # (6,)
    sin_k = np.sin(k_angles)  # (6,)

    # A[i,k] = x_i * cos_k + y_i * sin_k
    # B[i,k] = y_i * cos_k - x_i * sin_k
    A = np.outer(x_flat, cos_k) + np.outer(y_flat, sin_k)  # (66, 6)
    B = np.outer(y_flat, cos_k) - np.outer(x_flat, sin_k)  # (66, 6)

    def compute_side(alpha_deg):
        """Compute the minimal enclosing hexagon side length for relative angle alpha."""
        alpha = np.deg2rad(alpha_deg)
        ca, sa = np.cos(alpha), np.sin(alpha)
        # proj[i,k] = A[i,k]*ca + B[i,k]*sa
        proj = A * ca + B * sa  # (66, 6)
        max_proj = np.max(proj)
        return 2.0 * max_proj / SQRT3

    # Phase 1: Coarse grid search over alpha in [0, 60)
    n_coarse = 6001
    alphas_coarse = np.linspace(0, 60, n_coarse, endpoint=False)
    ca_arr = np.cos(np.deg2rad(alphas_coarse))
    sa_arr = np.sin(np.deg2rad(alphas_coarse))

    # Vectorized: proj[k, i, j] = A[i,j]*ca[k] + B[i,j]*sa[k]
    # max over i and j for each k
    # proj shape: (n_coarse, 66, 6)
    # = A[None, :, :] * ca_arr[:, None, None] + B[None, :, :] * sa_arr[:, None, None]
    # This is 6001 * 66 * 6 = 2.38M floats ~ 19 MB. Fine.
    proj_all = A[None, :, :] * ca_arr[:, None, None] + B[None, :, :] * sa_arr[:, None, None]
    sides_coarse = 2.0 * np.max(proj_all, axis=(1, 2)) / SQRT3

    best_idx = np.argmin(sides_coarse)
    best_alpha = alphas_coarse[best_idx]
    best_side = sides_coarse[best_idx]

    # Phase 2: Fine grid around best
    fine_range = 30.0 / n_coarse * 2  # ~2 grid spacings
    n_fine = 20001
    alphas_fine = np.linspace(best_alpha - fine_range, best_alpha + fine_range, n_fine)
    ca_f = np.cos(np.deg2rad(alphas_fine))
    sa_f = np.sin(np.deg2rad(alphas_fine))
    proj_fine = A[None, :, :] * ca_f[:, None, None] + B[None, :, :] * sa_f[:, None, None]
    sides_fine = 2.0 * np.max(proj_fine, axis=(1, 2)) / SQRT3

    best_idx_f = np.argmin(sides_fine)
    best_alpha = alphas_fine[best_idx_f]
    best_side = sides_fine[best_idx_f]

    # Phase 3: Ternary search for high-precision refinement
    # f(alpha) is piecewise smooth (max of sinusoids). Near the minimum,
    # it should be unimodal in a small neighborhood.
    def ternary_search(f, a, b, tol=1e-12, max_iter=200):
        """Ternary search for minimum of unimodal function on [a, b]."""
        for _ in range(max_iter):
            if b - a < tol:
                break
            m1 = a + (b - a) / 3.0
            m2 = b - (b - a) / 3.0
            f1 = f(m1)
            f2 = f(m2)
            if f1 < f2:
                b = m2
            else:
                a = m1
        return (a + b) / 2.0

    # Use a small window around the grid best
    search_width = fine_range / n_fine * 3  # ~3 fine grid spacings
    a_lo = best_alpha - search_width
    a_hi = best_alpha + search_width

    # Ensure we stay in [0, 60)
    a_lo = max(0.0, a_lo)
    a_hi = min(60.0, a_hi)

    best_alpha_ternary = ternary_search(compute_side, a_lo, a_hi, tol=1e-13, max_iter=300)
    best_side_ternary = compute_side(best_alpha_ternary)

    if best_side_ternary < best_side:
        best_alpha = best_alpha_ternary
        best_side = best_side_ternary

    # Phase 4: Even tighter ternary search
    search_width2 = 1e-6
    a_lo2 = max(0.0, best_alpha - search_width2)
    a_hi2 = min(60.0, best_alpha + search_width2)
    best_alpha_ternary2 = ternary_search(compute_side, a_lo2, a_hi2, tol=1e-14, max_iter=200)
    best_side_ternary2 = compute_side(best_alpha_ternary2)

    if best_side_ternary2 < best_side:
        best_alpha = best_alpha_ternary2
        best_side = best_side_ternary2

    # Phase 5: Additional local refinement with golden section
    def golden_search(f, a, b, tol=1e-14, max_iter=200):
        """Golden section search for minimum."""
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        for _ in range(max_iter):
            if b - a < tol:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        return (a + b) / 2.0

    search_width3 = 1e-7
    a_lo3 = max(0.0, best_alpha - search_width3)
    a_hi3 = min(60.0, best_alpha + search_width3)
    best_alpha_golden = golden_search(compute_side, a_lo3, a_hi3, tol=1e-15, max_iter=200)
    best_side_golden = compute_side(best_alpha_golden)

    if best_side_golden < best_side:
        best_alpha = best_alpha_golden
        best_side = best_side_golden

    # Final verification and tiny safety margin
    final_alpha = best_alpha % 60.0
    final_side = compute_side(final_alpha) + 1e-9

    # Build the inner hexagon positions
    # The cluster rotation is theta, the outer hexagon rotation is phi.
    # We found the optimal relative angle alpha = phi - theta.
    # We can set theta = 0 (cluster unrotated) and phi = alpha.
    # Or any (theta, phi) with phi - theta = alpha.
    # Let's set theta = 0, phi = alpha.
    theta = 0.0
    phi = final_alpha

    # Inner hexagons: centers are cluster_centers (unrotated since theta=0), angle = theta = 0
    inner = []
    for i in range(N):
        x = float(cluster_centers[i, 0])
        y = float(cluster_centers[i, 1])
        ang = float(theta)
        inner.append([x, y, ang])

    return inner, (0.0, 0.0), float(final_side), float(phi)
