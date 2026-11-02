"""
Circle packing in perimeter-4 rectangle using Contact Graph and LP-based radius optimization.

Strategy:
1. Represent the packing by circle centers (x, y), a common scale factor s (radius multiplier), 
   and the rectangle aspect ratio (width w, height h = 2-w).
2. In each iteration:
   a. Compute a contact graph based on current positions (edges where distance < threshold * (r_i + r_j)).
   b. Solve an LP to find optimal radii r_i (scaled by s) that maximize sum(r_i) subject to:
      - r_i >= 0
      - r_i + r_j <= dist_ij for all (i,j) in contact graph (plus some safety pairs)
      - r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i (wall constraints)
   c. Use the dual variables (shadow prices) of the LP constraints to compute forces on the centers.
      - For a pair (i,j) with tight constraint r_i + r_j = dist_ij, the force pushes them apart.
      - For wall constraints, the force pushes the circle away from the wall.
   d. Update centers using these forces (gradient ascent on the objective, which is sum of radii).
   e. Update aspect ratio w similarly based on wall forces.
"""
import numpy as np
import time
from scipy.optimize import linprog
import scipy.spatial as spatial

N = 21
DEADLINE = 135.0  # Safety margin

def construct():
    start_time = time.time()
    
    # State variables
    # w: width, h = 2 - w
    # centers: (N, 2) array of [x, y]
    # scale: global scale factor for radii (we optimize relative radii via LP, then scale)
    # Actually, it's better to optimize absolute radii directly if we fix w, h.
    # Let's optimize: w, centers (N, 2). Radii are determined by LP given w, centers.
    
    best_score = 0.0
    best_state = None

    # Initial guess: Grid
    w = 1.0
    h = 1.0
    side = int(np.ceil(np.sqrt(N)))
    centers = np.zeros((N, 2))
    for idx in range(N):
        i = idx % side
        j = idx // side
        centers[idx, 0] = (i + 0.5) * w / side
        centers[idx, 1] = (j + 0.5) * h / side

    # Precompute pairs for contact graph (all pairs, but we will filter dynamically)
    # To make LP faster, we only include constraints for pairs that are "close"
    # and all wall constraints.
    
    def solve_lp(w, h, centers):
        """
        Solve LP to maximize sum(r_i) subject to non-overlap and wall constraints.
        Returns radii, duals for pair constraints, duals for wall constraints, success flag.
        """
        # Variables: r_0, ..., r_{N-1}
        # Maximize sum(r) -> Minimize -sum(r)
        c = -np.ones(N)
        
        A_ub = []
        b_ub = []
        
        # Pair constraints: r_i + r_j <= dist_ij
        # We only add constraints for pairs that are currently close or likely to be close.
        # To be safe and simple, we can add all pairs if N is small (21 -> 210 pairs).
        # 210 constraints is small for linprog.
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # Constraint: r_i + r_j <= dist
                row = np.zeros(N)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
        
        # Wall constraints:
        # r_i <= x_i
        # r_i <= w - x_i
        # r_i <= y_i
        # r_i <= h - y_i
        for i in range(N):
            # r_i - 0 <= x_i
            row = np.zeros(N)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(centers[i, 0])
            
            # r_i <= w - x_i
            A_ub.append(row)
            b_ub.append(w - centers[i, 0])
            
            # r_i <= y_i
            A_ub.append(row)
            b_ub.append(centers[i, 1])
            
            # r_i <= h - y_i
            A_ub.append(row)
            b_ub.append(h - centers[i, 1])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: r_i >= 0
        bounds = [(0, None)] * N
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                return None, None, None, False
            
            radii = res.x
            
            # Extract duals
            # res.ineqlin.marginals are the duals for A_ub rows.
            # Note: scipy linprog minimizes, so duals correspond to constraints A_ub x <= b_ub.
            # The objective is minimize c^T x.
            # The dual variable y_k >= 0 for constraint k.
            # Sensitivity: d(obj)/d(b_k) = y_k.
            # Since we are maximizing sum(r), the "force" on the constraint b_k is related to y_k.
            
            duals = res.ineqlin.marginals
            
            # Parse duals
            # First N*(N-1)/2 duals are for pairs.
            # Next 4*N duals are for walls.
            
            pair_duals = duals[:N*(N-1)//2]
            wall_duals = duals[N*(N-1)//2:]
            
            return radii, pair_duals, wall_duals, True
            
        except Exception:
            return None, None, None, False

    def compute_forces(w, h, centers, radii, pair_duals, wall_duals):
        """
        Compute forces on centers and w based on duals.
        Force on center i:
        - From pair (i,j): If constraint is tight, dual y_ij > 0.
          The constraint is r_i + r_j <= dist_ij.
          d(dist_ij)/d(x_i) = (x_i - x_j) / dist_ij.
          The Lagrangian term is y_ij * (r_i + r_j - dist_ij).
          Wait, standard LP dual interpretation:
          The gradient of the objective w.r.t. centers is determined by how the optimal value changes.
          The optimal value V = sum(r_i).
          dV/d(centers) = sum_k y_k * d(b_k)/d(centers).
          Since we want to MAXIMIZE V, we move centers in the direction of +dV/d(centers).
          
          For pair (i,j): b_k = dist_ij.
          d(dist_ij)/d(x_i) = (x_i - x_j) / dist_ij.
          So force on i from j is y_ij * (x_i - x_j) / dist_ij.
          This pushes i away from j. Correct.
          
          For walls:
          b_k = x_i. d(b_k)/d(x_i) = 1. Force = y_k * 1. Pushes x_i to increase (away from left wall).
          b_k = w - x_i. d(b_k)/d(x_i) = -1. Force = y_k * (-1). Pushes x_i to decrease (away from right wall).
          Note: w is also a variable.
          d(b_k)/d(w) = 1 for right wall. d(b_k)/d(w) = 0 for left.
          
        """
        forces = np.zeros_like(centers)
        force_w = 0.0
        
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                y = pair_duals[idx]
                if y > 1e-8:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 1e-8:
                        fx = y * dx / dist
                        fy = y * dy / dist
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                idx += 1
        
        idx = 0
        for i in range(N):
            # Left wall: r_i <= x_i. b = x_i. dual y.
            # dV/dx_i += y * 1
            y_left = wall_duals[idx]
            if y_left > 1e-8:
                forces[i, 0] += y_left
            idx += 1
            
            # Right wall: r_i <= w - x_i. b = w - x_i. dual y.
            # dV/dx_i += y * (-1)
            # dV/dw += y * (1)
            y_right = wall_duals[idx]
            if y_right > 1e-8:
                forces[i, 0] -= y_right
                force_w += y_right
            idx += 1
            
            # Bottom wall: r_i <= y_i. b = y_i. dual y.
            y_bottom = wall_duals[idx]
            if y_bottom > 1e-8:
                forces[i, 1] += y_bottom
            idx += 1
            
            # Top wall: r_i <= h - y_i. b = h - y_i = (2-w) - y_i. dual y.
            # dV/dy_i += y * (-1)
            # dV/dw += y * (-1)  (since h = 2-w, dh/dw = -1)
            y_top = wall_duals[idx]
            if y_top > 1e-8:
                forces[i, 1] -= y_top
                force_w -= y_top
            idx += 1
            
        return forces, force_w

    # Optimization Loop
    # We use a simple gradient ascent with adaptive step size.
    # Variables: centers (N*2), w (1).
    
    step_size = 0.01
    learning_rate_decay = 0.99
    min_step = 0.0001
    
    # Add some randomness to break symmetry
    rng = np.random.RandomState(42)
    centers += rng.randn(N, 2) * 0.01
    w = 1.0 + rng.randn() * 0.1
    w = np.clip(w, 0.5, 1.5)
    h = 2.0 - w
    
    # Clamp centers to be inside
    centers[:, 0] = np.clip(centers[:, 0], 0.01, w - 0.01)
    centers[:, 1] = np.clip(centers[:, 1], 0.01, h - 0.01)

    for iter in range(1000):
        if time.time() > DEADLINE:
            break
            
        radii, pair_duals, wall_duals, success = solve_lp(w, h, centers)
        
        if not success or radii is None:
            # If LP fails, reduce step size or restart
            step_size *= 0.5
            continue
            
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_state = (w, centers.copy())
            
        forces, force_w = compute_forces(w, h, centers, radii, pair_duals, wall_duals)
        
        # Normalize forces to prevent explosion
        norm_forces = np.linalg.norm(forces)
        if norm_forces > 1.0:
            forces = forces / norm_forces
        norm_fw = abs(force_w)
        if norm_fw > 1.0:
            force_w = force_w / norm_fw
            
        # Update centers
        centers += step_size * forces
        
        # Update w
        w += step_size * force_w
        w = np.clip(w, 0.5, 1.5)
        h = 2.0 - w
        
        # Clamp centers to valid region (with small margin to avoid singularities)
        centers[:, 0] = np.clip(centers[:, 0], 0.001, w - 0.001)
        centers[:, 1] = np.clip(centers[:, 1], 0.001, h - 0.001)
        
        # Decay step size
        step_size *= learning_rate_decay
        if step_size < min_step:
            step_size = min_step
            # If step size is very small, we might be converged. 
            # Try a small random perturbation to escape local minima?
            # For now, just continue.
            
        # Every 50 iterations, try a random perturbation if stuck
        if iter % 50 == 0 and iter > 0:
            # Check if score improved recently? 
            # Simple heuristic: add small noise
            centers += rng.randn(N, 2) * 0.001
            w += rng.randn() * 0.001
            w = np.clip(w, 0.5, 1.5)
            h = 2.0 - w
            centers[:, 0] = np.clip(centers[:, 0], 0.001, w - 0.001)
            centers[:, 1] = np.clip(centers[:, 1], 0.001, h - 0.001)

    # Final verification and cleanup
    if best_state is not None:
        w_best, centers_best = best_state
        h_best = 2.0 - w_best
        radii_best, _, _, _ = solve_lp(w_best, h_best, centers_best)
        
        if radii_best is not None:
            result_array = np.zeros((N, 3))
            for i in range(N):
                x = centers_best[i, 0]
                y = centers_best[i, 1]
                r = radii_best[i]
                
                # Ensure validity: clip radii and centers
                # r <= min(x, w-x, y, h-y)
                max_r = min(x, w_best - x, y, h_best - y)
                r = min(r, max_r)
                r = max(r, 0.0)
                
                # Re-center if clipping changed r? 
                # Actually, if r was clipped, the center might be outside the "safe" zone for that r.
                # But since r <= x and r <= w-x, x is in [r, w-r], so x-r >= 0 and x+r <= w.
                # So center is valid.
                
                result_array[i] = [x, y, r]
                
            # Final check for overlaps (just in case LP had numerical issues)
            # If overlaps exist, we might need to shrink radii slightly.
            # The evaluator checks for disjoint interiors. 
            # A tiny overlap might fail. 
            # Let's do a final "relax" pass on the result to ensure no overlaps.
            
            # Simple overlap resolution:
            for _ in range(10):
                overlaps = False
                for i in range(N):
                    for j in range(i+1, N):
                        dx = result_array[i,0] - result_array[j,0]
                        dy = result_array[i,1] - result_array[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = result_array[i,2] + result_array[j,2]
                        if dist < min_dist - 1e-8:
                            overlaps = True
                            # Push apart or shrink? 
                            # Shrink is safer to maintain boundary constraints.
                            # Reduce both radii slightly
                            reduction = (min_dist - dist) / 2.0 + 1e-9
                            result_array[i,2] -= reduction
                            result_array[j,2] -= reduction
                if not overlaps:
                    break
            
            # Ensure radii are positive
            result_array[:, 2] = np.maximum(result_array[:, 2], 0.0)
            
            return result_array
            
    # Fallback if optimization failed completely
    side = int(np.ceil(np.sqrt(N)))
    w = 1.0
    h = 1.0
    cells = []
    for idx in range(N):
        i = idx % side
        j = idx // side
        r = 0.5 / side - 0.01
        x = (i + 0.5) * w / side
        y = (j + 0.5) * h / side
        cells.append([x, y, r])
    return np.array(cells)

if __name__ == "__main__":
    res = construct()
    print(res)
