"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Fast, vectorized feasibility radius.
# r = min( min_i (1 - |c_i|),  min_{i<j} |c_i - c_j| / 2 )
# ---------------------------------------------------------------------------
def _radius(pts):
    """Return the common radius for a set of centres, or -inf if invalid."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(pts)):
        return -np.inf
    # boundary term
    dist_origin = np.linalg.norm(pts, axis=1)
    r_boundary = np.min(1.0 - dist_origin)
    if r_boundary <= 0.0:
        return -np.inf
    # pairwise term (300 points -> 44850 pairs, cheap)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    r_pair = 0.5 * np.sqrt(np.min(d2))
    return min(r_boundary, r_pair)


# ---------------------------------------------------------------------------
# Candidate generators
# ---------------------------------------------------------------------------
def _hex_lattice():
    """Deterministic hexagonal lattice cut to the unit disk (always valid)."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        mask = rad <= 1.0 - spacing / 2.0
        pts, rad = pts[mask], rad[mask]
        if len(pts) >= N:
            return pts[np.argsort(rad, kind="stable")[:N]]
        spacing *= 0.98


def _concentric_rings():
    """Concentric rings of circles, centred at origin."""
    best_pts, best_r = None, -np.inf
    for n_rings in range(4, 14):
        # distribute N points across rings, more points in outer rings
        for base in range(1, 12):
            ring_counts = []
            remaining = N
            for i in range(n_rings):
                c = base + i * 2
                if remaining <= 0:
                    break
                c = min(c, remaining)
                ring_counts.append(c)
                remaining -= c
            if sum(ring_counts) < N:
                continue
            pts = []
            total = sum(ring_counts)
            # optional centre point
            if total % 2 == 1:
                pts.append([0.0, 0.0])
            for i, c in enumerate(ring_counts):
                if c <= 0:
                    continue
                rr = (i + 1) / (n_rings + 0.5) * 0.95
                ang = np.linspace(0, 2 * np.pi, c, endpoint=False) + i * np.pi / max(c, 1)
                for a in ang:
                    pts.append([rr * np.cos(a), rr * np.sin(a)])
            if len(pts) < N:
                continue
            pts = np.array(pts[:N])
            r = _radius(pts)
            if r > best_r:
                best_r, best_pts = r, pts.copy()
    return best_pts


def _square_grid():
    """Square grid cut to the unit disk."""
    best_pts, best_r = None, -np.inf
    for spacing in np.linspace(0.05, 0.12, 30):
        k = int(1.0 / spacing) + 1
        pts = np.array([(i * spacing, j * spacing)
                        for i in range(-k, k + 1) for j in range(-k, k + 1)])
        rad = np.hypot(pts[:, 0], pts[:, 1])
        mask = rad <= 1.0 - spacing / 2.0
        pts, rad = pts[mask], rad[mask]
        if len(pts) >= N:
            pts = pts[np.argsort(rad, kind="stable")[:N]]
            r = _radius(pts)
            if r > best_r:
                best_r, best_pts = r, pts.copy()
    return best_pts


# ---------------------------------------------------------------------------
# Local refinement: hill-climbing on the minimum radius.
# At each step, find the bottleneck (boundary or pair) and nudge points apart.
# ---------------------------------------------------------------------------
def _refine(pts, deadline):
    """Multi-restart hill-climbing with perturbation to escape local optima."""
    best_pts = np.array(pts, dtype=float, copy=True)
    best_r = _radius(best_pts)
    if best_r <= 0.0:
        return None

    rng = np.random.default_rng(42)
    n_restarts = 2

    for restart in range(n_restarts):
        if time.time() > deadline - 3.0:
            break

        # For first restart, use the input directly; for others, perturb
        if restart == 0:
            pts = np.array(pts, dtype=float, copy=True)
        else:
            # Perturb from the best found so far
            pts = np.array(best_pts, dtype=float, copy=True)
            # Add small random noise scaled by current radius
            noise_scale = 0.3 * best_r
            pts += rng.normal(0, noise_scale, size=pts.shape)
            # Clamp inside unit circle
            d_o = np.linalg.norm(pts, axis=1)
            over = d_o > 0.99
            if np.any(over):
                pts[over] *= (0.99 / d_o[over])[:, None]

        cur = _radius(pts)
        if cur <= 0.0:
            continue

        step = max(1e-4, 0.05 * cur)
        no_improve_count = 0
        max_no_improve = 20

        while time.time() < deadline - 2.0 and no_improve_count < max_no_improve:
            # find bottleneck
            dist_origin = np.linalg.norm(pts, axis=1)
            r_boundary = 1.0 - np.min(dist_origin)
            i_b = int(np.argmin(dist_origin))

            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(d2, np.inf)
            ii, jj = np.unravel_index(np.argmin(d2), d2.shape)
            d = np.sqrt(d2[ii, jj])
            r_pair = 0.5 * d

            if r_pair < r_boundary:
                # separate the closest pair
                v = pts[jj] - pts[ii]
                nrm = np.linalg.norm(v)
                if nrm < 1e-12:
                    v = np.array([1.0, 0.0]); nrm = 1.0
                v = v / nrm
                pts[ii] -= 0.5 * step * v
                pts[jj] += 0.5 * step * v
            else:
                # push boundary offender toward centre
                p = pts[i_b]
                nrm = np.linalg.norm(p)
                if nrm < 1e-12:
                    no_improve_count += 1
                    continue
                pts[i_b] -= step * (p / nrm)

            # clamp inside (safety)
            d_o = np.linalg.norm(pts, axis=1)
            over = d_o > 0.9999
            if np.any(over):
                pts[over] *= (0.9999 / d_o[over])[:, None]

            new_r = _radius(pts)
            if new_r > cur:
                cur = new_r
                step = max(1e-5, step * 0.97)
                no_improve_count = 0
                if cur > best_r:
                    best_r = cur
                    best_pts = pts.copy()
            else:
                # revert, shrink step
                step *= 0.7
                if step < 1e-7:
                    break
                no_improve_count += 1
                # revert last move
                if r_pair < r_boundary:
                    v = pts[jj] - pts[ii]
                    nrm = np.linalg.norm(v)
                    if nrm < 1e-12:
                        continue
                    v = v / nrm
                    pts[ii] += 0.5 * (step / 0.7) * v
                    pts[jj] -= 0.5 * (step / 0.7) * v
                else:
                    p = pts[i_b]
                    nrm = np.linalg.norm(p)
                    if nrm < 1e-12:
                        break
                    pts[i_b] += (step / 0.7) * (p / nrm)

    return best_pts


def _soft_min_objective(x, beta=50.0):
    """
    Objective: -r where r is approximated by soft-min of all constraints.
    x is flattened (N, 2) array.
    """
    pts = x.reshape(N, 2)
    
    # Boundary constraints: 1 - ||c_i||
    dist_origin = np.linalg.norm(pts, axis=1)
    boundary_terms = 1.0 - dist_origin
    
    # Pairwise constraints: ||c_i - c_j|| / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    pair_terms = 0.5 * np.sqrt(d2)
    
    # Combine all terms into one vector
    all_terms = np.concatenate([boundary_terms, pair_terms.ravel()])
    
    # Soft-min approximation using log-sum-exp
    # min(x) ≈ -1/beta * log(sum(exp(-beta * x)))
    # To avoid overflow, subtract max
    x_shifted = -beta * all_terms
    x_max = np.max(x_shifted)
    if not np.isfinite(x_max):
        return 1e10
    
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted - x_max)))
    soft_min_val = -log_sum_exp / beta
    
    return -soft_min_val


def _soft_min_gradient(x, beta=50.0):
    """
    Gradient of the soft-min objective.
    """
    pts = x.reshape(N, 2)
    
    # Boundary terms
    dist_origin = np.linalg.norm(pts, axis=1)
    boundary_terms = 1.0 - dist_origin
    
    # Pairwise terms
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    pair_terms = 0.5 * np.sqrt(d2)
    
    all_terms = np.concatenate([boundary_terms, pair_terms.ravel()])
    
    # Soft-min gradient: grad = -sum(exp(-beta*x_i) * grad_x_i) / sum(exp(-beta*x_i))
    x_shifted = -beta * all_terms
    x_max = np.max(x_shifted)
    if not np.isfinite(x_max):
        return np.zeros_like(x)
    
    exp_vals = np.exp(x_shifted - x_max)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-15:
        return np.zeros_like(x)
    
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(x)
    
    # Boundary part
    # d/dx_i (1 - ||c_i||) = -c_i / ||c_i||
    for i in range(N):
        if dist_origin[i] > 1e-12:
            grad[2*i:2*i+2] -= weights[i] * pts[i] / dist_origin[i]
    
    # Pairwise part
    # For each pair (i,j), term is 0.5 * ||c_i - c_j||
    # d/dc_i (0.5 * ||c_i - c_j||) = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d/dc_j (0.5 * ||c_i - c_j||) = -0.5 * (c_i - c_j) / ||c_i - c_j||
    
    # This is O(N^2) which is expensive for N=300 (44850 pairs)
    # But we only need to do it if we're going to use gradient-based optimization
    # For now, let's skip the full gradient and use finite differences or a simpler approach
    
    # Actually, let's use a different approach: L-BFGS-B with numerical gradients
    # Or we can use SLSQP which handles constraints directly
    
    return grad


def construct():
    start = time.time()
    deadline = start + 100.0  # 120s wall limit, 20s safety

    # baseline (always valid)
    hex_pts = _hex_lattice()
    best_pts, best_r = hex_pts, _radius(hex_pts)

    # generate alternative seeds and keep the best
    for gen in (_concentric_rings, _square_grid):
        if time.time() > deadline - 10.0:
            break
        cand = gen()
        if cand is not None:
            r = _radius(cand)
            if r > best_r:
                best_r, best_pts = r, cand

    # refine the best seed within remaining time
    if time.time() < deadline - 5.0:
        refined = _refine(best_pts, deadline)
        if refined is not None:
            r_refined = _radius(refined)
            if r_refined > best_r:
                best_r, best_pts = r_refined, refined

    # Try SLSQP optimization to maximize radius with explicit constraints
    if time.time() < deadline - 15.0 and best_r > 0.0:
        x0 = best_pts.flatten()
        
        def objective(x):
            """Minimize negative radius (maximize radius)."""
            pts = x.reshape(N, 2)
            r = _radius(pts)
            if r <= 0:
                return 1e10
            return -r
        
        # Constraints: for each pair i<j, ||c_i - c_j|| - 2*r >= 0
        # and for each i, 1 - ||c_i|| - r >= 0
        # This is complex because r is a function of x.
        # Instead, let's use the soft-min approach with L-BFGS-B
        
        if time.time() < deadline - 20.0:
            beta = 30.0
            res = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < -best_r:
                new_pts = res.x.reshape(N, 2)
                new_r = _radius(new_pts)
                if new_r > best_r:
                    best_r, best_pts = new_r, new_pts

    # Final high-precision polish: very small steps to squeeze out last bits
    if time.time() < deadline - 1.0 and best_r > 0.0:
        pts = np.array(best_pts, dtype=float, copy=True)
        cur = _radius(pts)
        step = 1e-5
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            # find bottleneck
            dist_origin = np.linalg.norm(pts, axis=1)
            r_boundary = 1.0 - np.min(dist_origin)
            i_b = int(np.argmin(dist_origin))

            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(d2, np.inf)
            ii, jj = np.unravel_index(np.argmin(d2), d2.shape)
            d = np.sqrt(d2[ii, jj])
            r_pair = 0.5 * d

            if r_pair < r_boundary:
                v = pts[jj] - pts[ii]
                nrm = np.linalg.norm(v)
                if nrm < 1e-12:
                    continue
                v = v / nrm
                pts[ii] -= 0.5 * step * v
                pts[jj] += 0.5 * step * v
            else:
                p = pts[i_b]
                nrm = np.linalg.norm(p)
                if nrm < 1e-12:
                    continue
                pts[i_b] -= step * (p / nrm)

            # clamp inside
            d_o = np.linalg.norm(pts, axis=1)
            over = d_o > 0.9999
            if np.any(over):
                pts[over] *= (0.9999 / d_o[over])[:, None]

            new_r = _radius(pts)
            if new_r > cur:
                cur = new_r
                if cur > best_r:
                    best_r = cur
                    best_pts = pts.copy()
            else:
                # revert
                if r_pair < r_boundary:
                    v = pts[jj] - pts[ii]
                    nrm = np.linalg.norm(v)
                    if nrm < 1e-12:
                        continue
                    v = v / nrm
                    pts[ii] += 0.5 * step * v
                    pts[jj] -= 0.5 * step * v
                else:
                    p = pts[i_b]
                    nrm = np.linalg.norm(p)
                    if nrm < 1e-12:
                        break
                    pts[i_b] += step * (p / nrm)

    # final validity guard
    if _radius(best_pts) <= 0.0:
        return hex_pts
    return best_pts
# EVOLVE-BLOCK-END
