import numpy as np
import time
from scipy.spatial.distance import cdist

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(10):
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.90 + 0.05 * attempt)
        k = int(1.0 / spacing) + 4
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        if len(rows) < n_points:
            continue
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.03, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
        d = np.linalg.norm(selected[:, np.newaxis, :] - selected[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        k = int(1.0 / spacing) + 4
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
    return best_pts


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    sq_norms = np.sum(pts * pts, axis=1)
    cross = pts @ pts.T
    sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
    np.maximum(sq_dists, 0.0, out=sq_dists)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair = np.sqrt(sq_dists.min()) / 2.0
    return min(min_pair, min_bound)


def _active_set_newton(pts, n_iter=500, tol=1e-10, time_limit=None, start_time=None):
    """
    Active-set Newton solver for circle packing.
    Maximizes r subject to:
      ||c_i - c_j|| >= 2r  for all i < j
      ||c_i|| + r <= 1     for all i
    
    Uses KKT conditions with active constraint identification.
    """
    if start_time is None:
        start_time = time.time()
    
    pts = pts.copy()
    n = len(pts)
    
    # Get initial radius
    r = _compute_radius(pts)
    if r <= 0:
        return pts, r
    
    # Precompute upper triangle indices
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    
    for it in range(n_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Compute current distances
        norms = np.linalg.norm(pts, axis=1)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 2.0 * r)  # Set diagonal to 2r for consistency
        
        # Identify active constraints (within tolerance)
        tol_active = 1e-8 + 1e-6 * r
        active_pair_mask = dists[iu] <= 2.0 * r + tol_active
        active_bound_mask = norms + r >= 1.0 - tol_active
        
        # Build active constraint set
        active_pair_idx = np.where(active_pair_mask)[0]
        active_bound_idx = np.where(active_bound_mask)[0]
        
        n_active_pairs = len(active_pair_idx)
        n_active_bounds = len(active_bound_idx)
        n_active = n_active_pairs + n_active_bounds
        
        if n_active == 0:
            break
        
        # KKT system: maximize r
        # Variables: r (1) + c_i (2n) = 1 + 2n
        # Constraints (active):
        #   Pair (i,j): ||c_i - c_j|| - 2r >= 0  -> gradient w.r.t. (r, c_i, c_j)
        #   Bound (i): 1 - ||c_i|| - r >= 0      -> gradient w.r.t. (r, c_i)
        
        # For active pair (i,j) with distance d_ij:
        #   d/dr: -2
        #   d/dc_i: (c_i - c_j) / d_ij
        #   d/dc_j: (c_j - c_i) / d_ij
        
        # For active bound (i) with norm ||c_i||:
        #   d/dr: -1
        #   d/dc_i: -c_i / ||c_i||
        
        # The KKT conditions for maximizing r with Lagrange multipliers lambda >= 0:
        # 1 - sum(lambda_pair * (-2) + lambda_bound * (-1)) = 0  (for r)
        # sum(lambda_pair * (c_i-c_j)/d_ij + lambda_bound * (-c_i/||c_i||)) = 0  (for c_i)
        
        # This is a linear system in lambda. Let's set up the system:
        # Let L be the matrix of constraint gradients (n_active x (1+2n))
        # KKT: [0  I] [lambda] = 0  (stationarity for variables)
        #      [I  0] [lambda] = e   (complementary slackness, active constraints)
        
        # Actually, for the active set method, we solve:
        # Minimize 0 subject to:
        #   A * x = b  (active constraints)
        #   x in null space of A
        # Then move in the direction that increases r.
        
        # Simpler approach: use the fact that at optimum, the gradient of r
        # (which is just [1, 0, ..., 0]) must be in the cone of active constraint gradients.
        
        # Let's use a direct approach: solve for the direction of steepest ascent of r
        # subject to the active constraints remaining feasible.
        
        # Build the constraint Jacobian for active constraints
        # Each row corresponds to an active constraint
        # Columns: [r, x_1, y_1, x_2, y_2, ..., x_n, y_n]
        
        J = np.zeros((n_active, 1 + 2 * n))
        
        # Pair constraints
        for k, idx in enumerate(active_pair_idx):
            i, j = iu[0][idx], iu[1][idx]
            d_ij = dists[i, j]
            if d_ij < 1e-12:
                d_ij = 1e-12
            J[k, 0] = -2.0  # d/dr
            J[k, 2*i] = (pts[i, 0] - pts[j, 0]) / d_ij
            J[k, 2*i+1] = (pts[i, 1] - pts[j, 1]) / d_ij
            J[k, 2*j] = (pts[j, 0] - pts[i, 0]) / d_ij
            J[k, 2*j+1] = (pts[j, 1] - pts[i, 1]) / d_ij
        
        # Boundary constraints
        for k, idx in enumerate(active_bound_idx):
            i = active_bound_idx[k]
            nrm = norms[i]
            if nrm < 1e-12:
                nrm = 1e-12
            J[n_active_pairs + k, 0] = -1.0  # d/dr
            J[n_active_pairs + k, 2*i] = -pts[i, 0] / nrm
            J[n_active_pairs + k, 2*i+1] = -pts[i, 1] / nrm
        
        # We want to find a direction d = (dr, dc) such that:
        #   J * d >= 0  (constraints remain feasible)
        #   dr > 0  (we want to increase r)
        #   ||dc|| is bounded
        
        # The optimal direction is the one that maximizes dr subject to J*d >= 0.
        # This is a linear program. For simplicity, we can use the null space approach:
        # Find d in the null space of J (for equality constraints) that maximizes dr.
        
        # Use SVD to find the null space
        try:
            U, S, Vt = np.linalg.svd(J, full_matrices=True)
        except np.linalg.LinAlgError:
            break
        
        # Null space of J is the last (1+2n - rank) rows of Vt
        rank = np.sum(S > 1e-10)
        if rank >= 1 + 2 * n:
            # Full rank, no null space direction
            break
        
        # Null space basis
        N_basis = Vt[rank:, :].T  # (1+2n) x (1+2n-rank)
        
        # We want to maximize dr = d[0] subject to d = N_basis * alpha
        # So dr = N_basis[0, :] @ alpha
        # Maximize this: alpha in direction of N_basis[0, :]
        
        dr_coeff = N_basis[0, :]
        norm_dr = np.linalg.norm(dr_coeff)
        if norm_dr < 1e-12:
            # No improvement direction
            break
        
        # Direction in null space
        alpha = dr_coeff / norm_dr
        d = N_basis @ alpha
        
        # Normalize so that dr = 1 (or a small step)
        dr = d[0]
        if abs(dr) < 1e-12:
            break
        
        # Scale step
        step_size = 1e-4
        d = d * step_size / abs(dr)
        
        # Apply step
        new_r = r + d[0]
        new_pts = pts.copy()
        new_pts[:, 0] += d[1::2]
        new_pts[:, 1] += d[2::2]
        
        # Check validity
        new_norms = np.linalg.norm(new_pts, axis=1)
        if np.any(new_norms + new_r >= 1.0) or new_r <= 0:
            # Step too large, reduce
            step_size *= 0.5
            d = d * 0.5
            new_r = r + d[0]
            new_pts = pts.copy()
            new_pts[:, 0] += d[1::2]
            new_pts[:, 1] += d[2::2]
            if np.any(new_norms + new_r >= 1.0) or new_r <= 0:
                break
        
        # Verify all constraints
        new_dists = np.sqrt(np.sum((new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]) ** 2, axis=2))
        np.fill_diagonal(new_dists, 2.0 * new_r)
        if np.any(new_dists[iu] < 2.0 * new_r - 1e-9):
            # Constraint violated, reduce step
            step_size *= 0.5
            d = d * 0.5
            new_r = r + d[0]
            new_pts = pts.copy()
            new_pts[:, 0] += d[1::2]
            new_pts[:, 1] += d[2::2]
            new_norms = np.linalg.norm(new_pts, axis=1)
            if np.any(new_norms + new_r >= 1.0) or new_r <= 0:
                break
        
        # Accept step
        pts = new_pts
        r = new_r
        
        # Check convergence
        if abs(dr * step_size) < tol:
            break
    
    return pts, r


def _local_improve(pts, n_restarts=3, time_limit=None, start_time=None):
    """
    Local improvement using multiple active-set Newton runs with perturbations.
    """
    if start_time is None:
        start_time = time.time()
    
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    rng = np.random.default_rng(12345)
    
    for restart in range(n_restarts):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Add small perturbation
        if restart > 0:
            perturbation = rng.normal(0, 1e-4 * best_r, pts.shape)
            perturbed = pts + perturbation
            # Ensure validity
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms < 1.0
            if np.sum(mask) < N:
                continue
            perturbed = perturbed[mask][:N]
        else:
            perturbed = pts.copy()
        
        # Run active-set Newton
        improved_pts, improved_r = _active_set_newton(
            perturbed, n_iter=300, tol=1e-10,
            time_limit=time_limit, start_time=start_time
        )
        
        if improved_r > best_r:
            best_r = improved_r
            best_pts = improved_pts.copy()
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial hexagonal packing
    pts = _hex_init(N, seed=42)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Phase 1: Active-set Newton refinement
    if time.time() < deadline - 30.0:
        refined_pts, refined_r = _local_improve(
            pts, n_restarts=5,
            time_limit=deadline - 30.0 - start_time,
            start_time=start_time
        )
        if refined_r > best_r:
            best_r = refined_r
            best_pts = refined_pts.copy()
    
    # Phase 2: Try different initializations
    if time.time() < deadline - 20.0:
        for seed in [0, 1, 2, 3, 4]:
            if time.time() > deadline - 10.0:
                break
            try:
                init_pts = _hex_init(N, seed=seed)
                init_r = _compute_radius(init_pts)
                if init_r < best_r * 0.95:
                    continue
                
                refined_pts, refined_r = _local_improve(
                    init_pts, n_restarts=3,
                    time_limit=deadline - 10.0 - start_time,
                    start_time=start_time
                )
                if refined_r > best_r:
                    best_r = refined_r
                    best_pts = refined_pts.copy()
            except Exception:
                continue
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            final_pts, final_r = _active_set_newton(
                best_pts, n_iter=1000, tol=1e-12,
                time_limit=deadline - 5.0 - start_time,
                start_time=start_time
            )
            if final_r > best_r:
                best_r = final_r
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
    
    return best_pts
