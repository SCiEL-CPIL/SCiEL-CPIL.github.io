"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin before 150s kill
    
    best_result = None
    best_score = -1.0
    
    # Precompute indices for pairwise operations (vectorization)
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def is_valid(positions, radii, w, h, tol=1e-9):
        # Vectorized boundary check
        if np.any(positions[:, 0] - radii < -tol) or np.any(positions[:, 0] + radii > w + tol):
            return False
        if np.any(positions[:, 1] - radii < -tol) or np.any(positions[:, 1] + radii > h + tol):
            return False
        
        # Vectorized overlap check
        dx = positions[j_idx, 0] - positions[i_idx, 0]
        dy = positions[j_idx, 1] - positions[i_idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = radii[i_idx] + radii[j_idx]
        
        if np.any(dist < min_dist - tol):
            return False
        return True
    
    def compute_score(positions, radii, w, h):
        if not is_valid(positions, radii, w, h):
            return -1.0
        return np.sum(radii)
    
    def grid_positions(w, h, cols, rows):
        positions = []
        dx = w / cols
        dy = h / rows
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                positions.append([x, y])
        return np.array(positions[:N])
    
    def hexagonal_positions(w, h, cols, rows):
        positions = []
        dx = w / cols
        dy = h / rows
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                if row % 2 == 1:
                    x += dx * 0.25
                positions.append([x, y])
        return np.array(positions[:N])
    
    def initial_radii(positions, w, h):
        radii = np.zeros(N)
        for i in range(N):
            r_max = min(positions[i, 0], w - positions[i, 0],
                        positions[i, 1], h - positions[i, 1])
            for j in range(N):
                if i != j:
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 1e-10:
                        r_max = min(r_max, dist * 0.45)
            radii[i] = max(r_max * 0.9, 0.005)
        return radii

    def relax_vectorized(positions, radii, w, h, max_iter=1000):
        """Fast vectorized force-based relaxation."""
        lr = 0.05
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute pairwise distances
            dx = positions[j_idx, 0] - positions[i_idx, 0]
            dy = positions[j_idx, 1] - positions[i_idx, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq)
            dist[dist < 1e-8] = 1e-8
            
            min_dist = radii[i_idx] + radii[j_idx]
            overlap = min_dist - dist
            
            # Forces
            forces = np.zeros((N, 2))
            # Only apply repulsion if overlapping
            mask = overlap > 0
            if np.any(mask):
                dir_x = dx[mask] / dist[mask]
                dir_y = dy[mask] / dist[mask]
                force_mag = overlap[mask] * 2.0
                
                # Accumulate forces
                np.add.at(forces, i_idx[mask], -force_mag[:, np.newaxis] * np.stack([dir_x, dir_y], axis=-1))
                np.add.at(forces, j_idx[mask], force_mag[:, np.newaxis] * np.stack([dir_x, dir_y], axis=-1))
            
            # Boundary forces
            left_viol = np.maximum(0, radii - positions[:, 0])
            right_viol = np.maximum(0, positions[:, 0] + radii - w)
            bottom_viol = np.maximum(0, radii - positions[:, 1])
            top_viol = np.maximum(0, positions[:, 1] + radii - h)
            
            forces[:, 0] += (left_viol - right_viol) * 5.0
            forces[:, 1] += (bottom_viol - top_viol) * 5.0
            
            # Radius update: grow towards max allowed
            r_boundary = np.minimum(np.minimum(positions[:, 0], w - positions[:, 0]), 
                                    np.minimum(positions[:, 1], h - positions[:, 1]))
            
            r_neighbor_max = np.full(N, np.inf)
            space_for_i = dist - radii[j_idx]
            space_for_j = dist - radii[i_idx]
            np.minimum.at(r_neighbor_max, i_idx, space_for_i)
            np.minimum.at(r_neighbor_max, j_idx, space_for_j)
            
            r_max = np.minimum(r_boundary, r_neighbor_max)
            r_max = np.maximum(r_max, 0.001)
            
            dr = (r_max - radii) * 0.1
            
            # Apply updates
            positions += forces * lr
            radii += dr
            
            # Clamp
            radii = np.maximum(radii, 0.001)
            positions[:, 0] = np.clip(positions[:, 0], radii, w - radii)
            positions[:, 1] = np.clip(positions[:, 1], radii, h - radii)
            
        return positions, radii

    def newton_polish(positions, radii, w, h, max_iter=50):
        """
        Newton-Raphson method to solve for exact jammed configuration.
        We assume the topology (contacts) is fixed from the previous step.
        Variables: positions (2N) and radii (N). Total 3N variables.
        Constraints: 
          1. For each contact pair (i,j): ||p_i - p_j||^2 - (r_i + r_j)^2 = 0
          2. For each wall contact: p_i * n - r_i - c = 0
        We want to maximize sum(r). This is a constrained optimization.
        However, a simpler approach for "polish" is to treat the active set as fixed
        and solve the system of equations where all active constraints are exactly 0.
        To maximize radii, we can use a simple gradient step on radii after solving positions,
        or use a penalty method.
        
        Here, we implement a simplified version:
        1. Identify active contacts (dist < r_i + r_j + tol) and active walls.
        2. Solve the linearized system to adjust positions and radii such that constraints are satisfied.
        3. Since we have more constraints than degrees of freedom usually, we use least squares.
        """
        if time.time() > deadline - 5:
            return positions, radii
            
        # Identify active constraints
        # This is heuristic: use a small tolerance
        tol_active = 1e-4
        
        dx = positions[j_idx, 0] - positions[i_idx, 0]
        dy = positions[j_idx, 1] - positions[i_idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        
        active_pairs = dist < (radii[i_idx] + radii[j_idx]) + tol_active
        active_i = i_idx[active_pairs]
        active_j = j_idx[active_pairs]
        
        active_walls = []
        for i in range(N):
            if positions[i, 0] - radii[i] < tol_active:
                active_walls.append((i, 'left'))
            if w - positions[i, 0] - radii[i] < tol_active:
                active_walls.append((i, 'right'))
            if positions[i, 1] - radii[i] < tol_active:
                active_walls.append((i, 'bottom'))
            if h - positions[i, 1] - radii[i] < tol_active:
                active_walls.append((i, 'top'))
                
        if len(active_pairs) == 0 and len(active_walls) == 0:
            return positions, radii
            
        # Build Jacobian J and residual R
        # Variables x = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
        # Size 3N
        n_vars = 3 * N
        J = np.zeros((len(active_pairs) + len(active_walls), n_vars))
        R = np.zeros(len(active_pairs) + len(active_walls))
        
        row = 0
        # Pair constraints: dist^2 - (r_i + r_j)^2 = 0
        # f = (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2
        # df/dx_i = 2(x_i - x_j), df/dy_i = 2(y_i - y_j), df/dr_i = -2(r_i + r_j)
        # df/dx_j = -2(x_i - x_j), df/dy_j = -2(y_i - y_j), df/dr_j = -2(r_i + r_j)
        
        for k in range(len(active_i)):
            i = active_i[k]
            j = active_j[k]
            xi, yi = positions[i]
            xj, yj = positions[j]
            ri, rj = radii[i], radii[j]
            
            dx_ij = xi - xj
            dy_ij = yi - yj
            r_sum = ri + rj
            
            R[row] = dx_ij**2 + dy_ij**2 - r_sum**2
            
            J[row, 3*i] = 2 * dx_ij
            J[row, 3*i+1] = 2 * dy_ij
            J[row, 3*i+2] = -2 * r_sum
            
            J[row, 3*j] = -2 * dx_ij
            J[row, 3*j+1] = -2 * dy_ij
            J[row, 3*j+2] = -2 * r_sum
            row += 1
            
        # Wall constraints
        # left: x_i - r_i - 0 = 0
        # right: -x_i - r_i + w = 0
        # bottom: y_i - r_i - 0 = 0
        # top: -y_i - r_i + h = 0
        
        for i, wall in active_walls:
            if wall == 'left':
                R[row] = positions[i, 0] - radii[i]
                J[row, 3*i] = 1.0
                J[row, 3*i+2] = -1.0
            elif wall == 'right':
                R[row] = -positions[i, 0] - radii[i] + w
                J[row, 3*i] = -1.0
                J[row, 3*i+2] = -1.0
            elif wall == 'bottom':
                R[row] = positions[i, 1] - radii[i]
                J[row, 3*i+1] = 1.0
                J[row, 3*i+2] = -1.0
            elif wall == 'top':
                R[row] = -positions[i, 1] - radii[i] + h
                J[row, 3*i+1] = -1.0
                J[row, 3*i+2] = -1.0
            row += 1
            
        # Solve least squares: J dx = -R
        try:
            dx, residuals, rank, s = np.linalg.lstsq(J, -R, rcond=None)
        except:
            return positions, radii
            
        # Apply update with damping
        alpha = 0.5
        x_old = np.zeros(n_vars)
        for i in range(N):
            x_old[3*i] = positions[i, 0]
            x_old[3*i+1] = positions[i, 1]
            x_old[3*i+2] = radii[i]
            
        x_new = x_old + alpha * dx
        
        # Extract new positions and radii
        new_pos = np.zeros((N, 2))
        new_rad = np.zeros(N)
        for i in range(N):
            new_pos[i, 0] = x_new[3*i]
            new_pos[i, 1] = x_new[3*i+1]
            new_rad[i] = x_new[3*i+2]
            
        # Clamp to valid range to ensure progress
        new_rad = np.maximum(new_rad, 0.001)
        new_pos[:, 0] = np.clip(new_pos[:, 0], new_rad, w - new_rad)
        new_pos[:, 1] = np.clip(new_pos[:, 1], new_rad, h - new_rad)
        
        # Check if valid
        if is_valid(new_pos, new_rad, w, h):
            return new_pos, new_rad
        else:
            # Try smaller step
            alpha = 0.1
            x_new = x_old + alpha * dx
            new_pos = np.zeros((N, 2))
            new_rad = np.zeros(N)
            for i in range(N):
                new_pos[i, 0] = x_new[3*i]
                new_pos[i, 1] = x_new[3*i+1]
                new_rad[i] = x_new[3*i+2]
            new_rad = np.maximum(new_rad, 0.001)
            new_pos[:, 0] = np.clip(new_pos[:, 0], new_rad, w - new_rad)
            new_pos[:, 1] = np.clip(new_pos[:, 1], new_rad, h - new_rad)
            
            if is_valid(new_pos, new_rad, w, h):
                return new_pos, new_rad
            return positions, radii

    def optimize_with_scipy(positions, radii, w, h):
        n = N
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]
        
        def objective(x):
            pos = x.reshape((n, 3))[:, :2]
            rad = x.reshape((n, 3))[:, 2]
            
            penalty = 0.0
            # Boundary penalties
            for i in range(n):
                bl = max(0.0, rad[i] - pos[i, 0])
                br = max(0.0, pos[i, 0] + rad[i] - w)
                bb = max(0.0, rad[i] - pos[i, 1])
                bt = max(0.0, pos[i, 1] + rad[i] - h)
                penalty += 1000.0 * (bl**2 + br**2 + bb**2 + bt**2)
            
            # Overlap penalties
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = rad[i] + rad[j]
                    overlap = max(0.0, min_dist - dist)
                    penalty += 1000.0 * overlap**2
            
            return -np.sum(rad) + penalty
        
        bounds = []
        for i in range(n):
            bounds.append((0.001, w - 0.001))
            bounds.append((0.001, h - 0.001))
            bounds.append((0.001, min(w, h)/2))
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.success or result.fun < 0:
                pos = result.x.reshape((n, 3))[:, :2]
                rad = result.x.reshape((n, 3))[:, 2]
                
                for i in range(n):
                    pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
                    pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
                    rad[i] = max(rad[i], 0.001)
                
                if is_valid(pos, rad, w, h):
                    return pos, rad
        except:
            pass
        
        return None, None

    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.4, 0.7, 1.3, 1.1, 0.9, 1.6, 0.63]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = ar * (2.0 / (1.0 + ar))
        h = 2.0 - w
        
        if ar >= 1.0:
            cols = int(np.round(np.sqrt(N * ar)))
            rows = int(np.ceil(N / cols))
        else:
            rows = int(np.round(np.sqrt(N / ar)))
            cols = int(np.ceil(N / rows))
        
        while cols * rows < N:
            if ar >= 1.0:
                cols += 1
            else:
                rows += 1
        
        for use_hex in [False, True]:
            if time.time() > deadline:
                break
            
            if use_hex:
                positions = hexagonal_positions(w, h, cols, rows)
            else:
                positions = grid_positions(w, h, cols, rows)
            
            if len(positions) < N:
                continue
            
            positions = positions[:N].copy()
            radii = initial_radii(positions, w, h)
            
            # 1. Fast vectorized relaxation
            positions, radii = relax_vectorized(positions, radii, w, h, max_iter=1000)
            
            score = compute_score(positions, radii, w, h)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([positions, radii]).copy()
            
            # 2. Newton polish (exact solve for active constraints)
            if time.time() < deadline - 10:
                pos_new, rad_new = newton_polish(positions, radii, w, h)
                score_new = compute_score(pos_new, rad_new, w, h)
                if score_new > best_score:
                    best_score = score_new
                    best_result = np.column_stack([pos_new, rad_new]).copy()
            
            # 3. Scipy optimization if time permits
            if time.time() < deadline - 20:
                opt_pos, opt_rad = optimize_with_scipy(positions, radii, w, h)
                if opt_pos is not None:
                    opt_score = compute_score(opt_pos, opt_rad, w, h)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_result = np.column_stack([opt_pos, opt_rad]).copy()
                        
                        # Final Newton polish on the scipy result
                        if time.time() < deadline - 5:
                            pos_final, rad_final = newton_polish(opt_pos, opt_rad, w, h)
                            final_score = compute_score(pos_final, rad_final, w, h)
                            if final_score > best_score:
                                best_score = final_score
                                best_result = np.column_stack([pos_final, rad_final]).copy()
    
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
