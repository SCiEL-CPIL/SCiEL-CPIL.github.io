"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def _score(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _energy_and_grad(x, beta=200.0):
    """
    Soft-min energy: sum exp(-beta * (dist_ij - target)) for all pairs,
    plus boundary soft penalty. We want to maximize the minimum, so we minimize
    a function that is large when min distance is small.
    
    Actually, let's use: E = sum_ij max(0, 2*r_target - d_ij)^2 + boundary penalty
    But for L-BFGS we need smooth. Let's use a soft-min proxy:
    E = -log(sum exp(-beta * d_ij / r0)) approximates -beta * min(d_ij) / r0
    This is non-smooth in practice for large beta.
    
    Better approach: minimize a sum of squared violations assuming we know r.
    But we don't know r.
    
    Let's use the approach from inspiration 1: direct bottleneck elimination.
    But since we need a different structural change, let's use a different smooth proxy.
    
    New proxy: 
    For each pair, define f_ij(r) = max(0, 2r - d_ij)^2
    For boundary, f_i(r) = max(0, 2r - margin_i)^2
    We want the largest r such that sum of all f = 0.
    
    Since this is non-smooth, let's use a smooth approximation:
    f_ij(r) = (2r - d_ij)^2 * sigmoid(k*(2r - d_ij)) ... complicated.
    
    Let's use a simpler structural change: use the "repulsive field" approach
    from inspiration 2 but with a different kernel: use 1/d^2 potential (Coulomb-like)
    instead of LJ. This gives strong long-range repulsion that helps global optimization.
    """
    centers = x.reshape(N, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)  # Avoid division by zero
    
    # Coulomb-like repulsion: E_pair = sum_{i<j} 1/d_ij^2
    # This is strong for close pairs, weak for far pairs.
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    
    E_pair = np.sum(1.0 / (dists[mask]**2))
    
    # Boundary penalty: soft wall
    # E_bound = sum_i (1/(0.5 - |c_ik|)^2) for each coordinate
    # But this blows up at boundary. Let's use:
    # E_bound = sum_i sum_k 1/(max(eps, 0.5 - |c_ik|))^2
    
    eps = 1e-6
    margins_x = 0.5 - np.abs(centers[:, 0])
    margins_y = 0.5 - np.abs(centers[:, 1])
    margins_x = np.maximum(margins_x, eps)
    margins_y = np.maximum(margins_y, eps)
    
    E_bound = np.sum(1.0 / margins_x**2) + np.sum(1.0 / margins_y**2)
    
    E = E_pair + 1000.0 * E_bound
    
    # Gradient
    grad = np.zeros_like(x)
    centers_flat = centers
    
    # Gradient of E_pair: d/dx_i (1/d_ij^2) = -2 * (x_i - x_j) / d_ij^4
    for i in range(N):
        for j in range(i + 1, N):
            d = dists[i, j]
            if d < 1e-10:
                continue
            diff_ij = centers_flat[i] - centers_flat[j]
            factor = -2.0 / (d**4)
            grad[2*i] += factor * diff_ij[0]
            grad[2*i + 1] += factor * diff_ij[1]
            grad[2*j] -= factor * diff_ij[0]
            grad[2*j + 1] -= factor * diff_ij[1]
    
    # Gradient of E_bound
    for i in range(N):
        mx = margins_x[i]
        my = margins_y[i]
        # d/dx_i (1/mx^2) = -2/mx^3 * d(mx)/dx_i = -2/mx^3 * (-sign(x_i)) = 2*sign(x_i)/mx^3
        if abs(centers_flat[i, 0]) > eps:
            sign_x = np.sign(centers_flat[i, 0])
            grad[2*i] += 1000.0 * 2.0 * sign_x / (mx**3)
        if abs(centers_flat[i, 1]) > eps:
            sign_y = np.sign(centers_flat[i, 1])
            grad[2*i + 1] += 1000.0 * 2.0 * sign_y / (my**3)
    
    return E, grad

def _generate_hex_initial(n):
    """Generate hexagonal grid initialization."""
    r_est = 0.037
    dx = 2 * r_est
    dy = dx * np.sqrt(3) / 2
    
    best_points = None
    best_score = 0
    
    for n_rows in range(12, 18):
        n_cols = int(np.ceil(n / n_rows))
        if n_rows * n_cols < n:
            continue
            
        points = []
        for row in range(n_rows):
            y = row * dy
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(n_cols):
                x = col * dx + offset
                points.append((x, y))
        
        points = np.array(points[:n])
        center = points.mean(axis=0)
        points -= center
        
        max_extent = np.max(np.abs(points))
        if max_extent > 0:
            scale = 0.48 / max_extent
            points *= scale
            
        if np.all(np.abs(points) <= 0.5 - 1e-10):
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    if best_points is not None:
        return best_points
        
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _bottleneck_local_search(centers, max_iters=5000, deadline=None):
    """
    Direct bottleneck elimination: at each step, find the current limiting
    constraint (closest pair or boundary violation) and move those points
    to increase the minimum distance.
    """
    centers = centers.copy()
    best_score = _score(centers)
    best_centers = centers.copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Find current radius
        current_r = _score(centers)
        
        # Find closest pair
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
        min_pair_idx = np.argmin(dists[:, 1])
        
        # Find boundary limit
        margins = 0.5 - np.max(np.abs(centers), axis=1)
        min_margin = np.min(margins)
        min_margin_idx = np.argmin(margins)
        
        # Determine which is limiting
        if min_pair_dist / 2.0 < min_margin:
            # Pair is limiting
            i = min_pair_idx
            j = idxs[i, 1]
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                # Coincident, random kick
                centers[i] += np.random.normal(0, 0.001, 2)
                centers[j] -= np.random.normal(0, 0.001, 2)
            else:
                direction = diff / d
                # We want to increase distance. Move i and j apart.
                # But we also need to consider if this makes other constraints worse.
                # Simple approach: move each by a small step along the separating direction.
                step = 0.0005
                centers[i] += direction * step
                centers[j] -= direction * step
        else:
            # Boundary is limiting
            idx = min_margin_idx
            # Find which coordinate is closest to boundary
            x, y = centers[idx]
            if abs(x) >= abs(y):
                # Move in x direction
                centers[idx, 0] -= np.sign(x) * 0.0005
            else:
                # Move in y direction
                centers[idx, 1] -= np.sign(y) * 0.0005
        
        # Clamp to square
        centers = np.clip(centers, -0.5 + 1e-12, 0.5 - 1e-12)
        
        new_score = _score(centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = centers.copy()
        elif it > 1000 and new_score < best_score * 0.999:
            # Stagnation: try a random perturbation
            if np.random.random() < 0.1:
                # Perturb the bottleneck points
                if min_pair_dist / 2.0 < min_margin:
                    i = min_pair_idx
                    j = idxs[i, 1]
                    centers[i] += np.random.normal(0, 0.003, 2)
                    centers[j] += np.random.normal(0, 0.003, 2)
                else:
                    centers[min_margin_idx] += np.random.normal(0, 0.003, 2)
                centers = np.clip(centers, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    np.random.seed(42)
    
    # 1. Initialize with hex grid
    best_centers = _generate_hex_initial(N)
    best_score = _score(best_centers)
    
    # 2. Coulomb repulsion optimization with L-BFGS-B
    if time.time() < deadline - 60:
        x0 = best_centers.flatten()
        try:
            res = minimize(_energy_and_grad, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40, 'maxcor': 20})
            if res.x is not None and np.all(np.isfinite(res.x)):
                opt_centers = res.x.reshape(N, 2)
                opt_centers = np.clip(opt_centers, -0.5 + 1e-12, 0.5 - 1e-12)
                opt_score = _score(opt_centers)
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = opt_centers
        except Exception:
            pass
    
    # 3. Bottleneck local search
    if time.time() < deadline - 20:
        centers, score = _bottleneck_local_search(best_centers, max_iters=10000, deadline=deadline - 15)
        if score > best_score:
            best_score = score
            best_centers = centers
    
    # 4. Second L-BFGS-B pass from the improved starting point
    if time.time() < deadline - 15:
        x0 = best_centers.flatten()
        try:
            res = minimize(_energy_and_grad, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40, 'maxcor': 20})
            if res.x is not None and np.all(np.isfinite(res.x)):
                opt_centers = res.x.reshape(N, 2)
                opt_centers = np.clip(opt_centers, -0.5 + 1e-12, 0.5 - 1e-12)
                opt_score = _score(opt_centers)
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = opt_centers
        except Exception:
            pass
    
    # 5. Final bottleneck refinement
    if time.time() < deadline - 5:
        centers, score = _bottleneck_local_search(best_centers, max_iters=5000, deadline=deadline - 2)
        if score > best_score:
            best_score = score
            best_centers = centers
    
    return best_centers
# EVOLVE-BLOCK-END
