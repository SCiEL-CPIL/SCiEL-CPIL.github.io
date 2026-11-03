import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import minimize
import sys

N = 50
DEADLINE = 145.0  # Safety margin before 150s limit

def _true_score(centers):
    """Compute the true packing radius for a given configuration."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return min(boundary, min_dist / 2.0)

def _init_hexagonal():
    """
    Generate a high-quality hexagonal packing initialization.
    We generate a dense hexagonal grid and select the 50 points that
    maximize the minimum distance (greedy farthest-point sampling) 
    within the square, or simply crop the best centered patch.
    """
    best_pts = None
    best_r = -1
    
    # Optimal hex packing density is sqrt(3)/2 * (1/(2r))^2 ... 
    # For N=50 in unit square, r is approx 0.071.
    # Hex spacing s = 2r.
    # Let's try a range of spacings.
    
    for s in np.arange(0.13, 0.16, 0.005):
        pts = []
        row_height = s * np.sqrt(3) / 2
        # Generate enough points to cover the square [-0.5, 0.5]^2
        # y from -0.5 to 0.5
        for i in range(-10, 11):
            y = i * row_height
            if abs(y) > 0.5:
                continue
            # x offset for alternating rows
            offset = (i % 2) * s / 2
            for j in range(-10, 11):
                x = j * s + offset
                if abs(x) <= 0.5:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) < N:
            continue
            
        # We need to select N points.
        # Strategy: Greedy Farthest Point Sampling starting from a random seed 
        # or the center. To be deterministic and robust, we use the center 
        # as the first point, then pick the point that maximizes the min distance 
        # to the existing set. This tends to fill the space uniformly.
        
        selected = []
        # Start with the point closest to (0,0) to ensure symmetry around center
        center_dists = np.linalg.norm(pts, axis=1)
        idx = np.argmin(center_dists)
        selected.append(pts[idx])
        remaining_indices = np.setdiff1d(np.arange(len(pts)), [idx])
        
        for _ in range(N - 1):
            if len(remaining_indices) == 0:
                break
            sel_pts = np.array(selected)
            # Min distance from each remaining point to the selected set
            # dist_sq[i, j] = ||pts[remaining_indices[i]] - sel_pts[j]||^2
            diffs = pts[remaining_indices][:, np.newaxis, :] - sel_pts[np.newaxis, :, :]
            dists_sq = (diffs ** 2).sum(axis=2)
            min_dists_sq = dists_sq.min(axis=1)
            # Pick the one with max min distance
            idx_rel = np.argmax(min_dists_sq)
            idx_abs = remaining_indices[idx_rel]
            selected.append(pts[idx_abs])
            remaining_indices = np.setdiff1d(remaining_indices, [idx_abs])
            
        if len(selected) == N:
            pts_sel = np.array(selected)
            r = _true_score(pts_sel)
            if r > best_r:
                best_r = r
                best_pts = pts_sel
                
    if best_pts is None:
        # Fallback: Simple grid
        rows, cols = 7, 8
        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
        
    return best_pts

def _laplacian_relax(centers, r_target, max_iters=100):
    """
    Relax the configuration using Laplacian energy minimization.
    We fix the Delaunay topology and solve (L + B) x = 0 where B is a boundary force.
    Actually, standard Laplacian smoothing with fixed boundary points (if any) 
    or free boundary points but constrained to stay in the box.
    
    For packing, we want to maximize the minimum distance.
    If we fix the topology and increase r, the springs will push points apart.
    We can simulate this by solving a linear system for a "force-directed" layout 
    where the rest length is 2*r_target.
    
    However, a simpler and more robust approach for "Laplacian" in this context 
    is to minimize the sum of squared deviations from the ideal distance 2r 
    for all edges in the Delaunay triangulation.
    
    Let E = sum_{(i,j) in E} (||x_i - x_j|| - 2r)^2
    This is non-convex. 
    Instead, let's use the standard Laplacian smoothing which minimizes 
    sum_{(i,j) in E} ||x_i - x_j||^2. This contracts the mesh.
    To expand it, we can scale the points.
    
    Better Strategy: 
    1. Compute Delaunay.
    2. Build Laplacian matrix L.
    3. We want to solve for a configuration where the edges are as long as possible 
       but the topology is preserved.
       Actually, if we just run Laplacian smoothing, it shrinks the graph.
       We want to *expand* it.
       
    Let's use a different formulation:
    Minimize the "potential" of the graph where edges act as springs with 
    natural length 2r.
    F_i = sum_{j in N(i)} k * (||x_i - x_j|| - 2r) * (x_i - x_j) / ||x_i - x_j||
    We can use gradient descent on this energy.
    """
    x = centers.copy()
    k = 1.0 # Spring constant
    
    for _ in range(max_iters):
        # Update Delaunay occasionally? No, fix topology for this step to avoid jumps.
        # But if we fix topology, we might get stuck.
        # Let's compute forces based on current Delaunay.
        try:
            tri = Delaunay(x)
        except:
            break
            
        edges = tri.simplices
        # Unique edges
        edge_set = set()
        for simplex in edges:
            for i in range(3):
                j = (i + 1) % 3
                u, v = sorted((simplex[i], simplex[j]))
                edge_set.add((u, v))
        edges_list = list(edge_set)
        
        forces = np.zeros_like(x)
        for u, v in edges_list:
            d_vec = x[u] - x[v]
            d = np.linalg.norm(d_vec)
            if d < 1e-10:
                continue
            # Spring force: pulls if d < 2r, pushes if d > 2r?
            # We want to maximize min dist. 
            # If we set rest length to 2r, and d < 2r, we want to push apart.
            # Force on u is in direction d_vec (away from v) if d < 2r.
            # Force magnitude proportional to (2r - d).
            # This minimizes (d - 2r)^2.
            disp = 2*r - d
            f_vec = disp * d_vec / d
            forces[u] += f_vec
            forces[v] -= f_vec
            
        # Add boundary repulsion to keep points inside
        # If a point is too close to boundary, push it in.
        for i in range(N):
            for k_coord in range(2):
                val = x[i, k_coord]
                # Boundary at 0.5 and -0.5
                # Distance to right wall: 0.5 - val
                # Distance to left wall: val - (-0.5) = val + 0.5
                # We want to keep x[i] in [-0.5+r, 0.5-r] ideally.
                # Simple penalty: if val > 0.5 - r, push left. If val < -0.5 + r, push right.
                # Actually, just clip and let the optimizer handle it?
                # For gradient descent, add a force.
                if val > 0.5 - r:
                    forces[i, k_coord] -= 10.0 * (val - (0.5 - r))
                elif val < -0.5 + r:
                    forces[i, k_coord] += 10.0 * ((-0.5 + r) - val)
                    
        # Update positions
        x += 0.1 * forces
        
        # Clip to square (hard constraint)
        x = np.clip(x, -0.5, 0.5)
        
    return x

def _slsqp_polish(centers, time_budget=10.0):
    """
    Standard SLSQP polish to fix any minor violations and maximize r locally.
    """
    start_time = time.time()
    
    def objective(x):
        return -x[-1]
        
    def gradient(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g
        
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    def constraints(x):
        c = x[:2*N].reshape(N, 2)
        t = x[-1]
        cons_b = 0.5 - np.abs(c) - t
        diffs = c[iu[0], :] - c[iu[1], :]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cons_p = dists - 2 * t
        return np.concatenate([cons_b.ravel(), cons_p])
        
    def jacobian(x):
        c = x[:2*N].reshape(N, 2)
        t = x[-1]
        J = np.zeros((2*N + n_pairs, 2*N + 1))
        
        for i in range(N):
            for k in range(2):
                idx = 2*i + k
                s = np.sign(c[i, k])
                if s == 0: s = 1.0
                J[idx, idx] = -s
                J[idx, -1] = -1.0
                
        diffs = c[iu[0], :] - c[iu[1], :]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists_safe = np.where(dists > 1e-10, dists, 1e-10)
        inv_dists = 1.0 / dists_safe
        
        for p in range(n_pairs):
            i = iu[0][p]
            j = iu[1][p]
            d = diffs[p]
            inv_d = inv_dists[p]
            row = 2*N + p
            J[row, 2*i] = d[0] * inv_d
            J[row, 2*i+1] = d[1] * inv_d
            J[row, 2*j] = -d[0] * inv_d
            J[row, 2*j+1] = -d[1] * inv_d
            J[row, -1] = -2.0
        return J
        
    t0 = _true_score(centers)
    x0 = np.concatenate([centers.ravel(), [t0]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraints, 'jac': jacobian}],
            options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
        )
        if result.success:
            new_centers = result.x[:2*N].reshape(N, 2)
            new_t = result.x[-1]
            true_t = _true_score(new_centers)
            if true_t >= t0 - 1e-9:
                return new_centers, true_t
    except Exception:
        pass
        
    return centers, t0

def construct():
    """
    Main entry point.
    Strategy:
    1. High-quality Hexagonal Init.
    2. Iterative Radius Expansion with Laplacian Relaxation.
       - Start with r_init.
       - Increase r slightly.
       - Relax the configuration using force-directed method (Laplacian-like)
         to accommodate the larger radius while preserving topology as much as possible.
       - If relaxation fails (points overlap or exit square), rollback and reduce step.
    3. Final SLSQP Polish.
    """
    start_time = time.time()
    np.random.seed(42)
    
    # 1. Initialization
    centers = _init_hexagonal()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)
    
    # 2. Iterative Expansion
    # Start from current score
    current_r = best_score
    # We will try to increase r in small steps
    step_size = 0.001
    max_steps = 500 # Limit steps to ensure time control
    
    for step in range(max_steps):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20: # Leave 20s for final polish
            break
            
        # Try to increase r
        target_r = current_r + step_size
        
        # Relax configuration to fit target_r
        # We use the current centers as the starting point for relaxation
        relaxed_centers = _laplacian_relax(centers, target_r, max_iters=20)
        
        # Check if the relaxed configuration is valid for target_r
        # i.e., min dist >= 2*target_r and boundary >= target_r
        score_relaxed = _true_score(relaxed_centers)
        
        if score_relaxed >= target_r - 1e-9:
            # Success: we can accept the new radius
            current_r = target_r
            centers = relaxed_centers
            best_score = max(best_score, current_r)
            best_centers = centers.copy()
            
            # Increase step size slightly if we've been stable?
            # Or keep it constant for robustness.
            # Let's keep it constant but maybe increase if we have time.
            if step % 50 == 0 and elapsed < DEADLINE * 0.5:
                step_size *= 1.1
        else:
            # Failure: the configuration couldn't relax to target_r
            # Try a smaller step? Or break?
            # If we fail, we might be at a local optimum of the topology.
            # We can try to perturb the configuration to break symmetry/topology lock.
            # Let's add small noise and retry with same r or slightly smaller.
            
            # Perturb
            noise = np.random.randn(N, 2) * 0.001
            perturbed = centers + noise
            perturbed = np.clip(perturbed, -0.5, 0.5)
            
            # Try relaxing the perturbed version to target_r
            relaxed_perturbed = _laplacian_relax(perturbed, target_r, max_iters=20)
            score_perturbed = _true_score(relaxed_perturbed)
            
            if score_perturbed >= target_r - 1e-9:
                current_r = target_r
                centers = relaxed_perturbed
                best_score = max(best_score, current_r)
                best_centers = centers.copy()
            else:
                # Failed to increase. 
                # We might be stuck. Let's try to polish the current best with SLSQP 
                # to see if we can squeeze more out of the current topology.
                # If SLSQP doesn't improve, we break the loop and move to final polish.
                # For now, just break the expansion loop and go to final polish.
                # This is a conservative approach to avoid wasting time on failed expansions.
                break
                
    # 3. Final Polish with SLSQP
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5:
        centers_final, score_final = _slsqp_polish(best_centers, time_budget=min(remaining*0.9, 15.0))
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()
            
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

if __name__ == "__main__":
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")

