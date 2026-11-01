"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def construct():
    """Construct a good packing of 200 equal circles in the unit square [-1/2, 1/2]^2."""
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def compute_radius_and_grad(centers):
        """
        Compute the maximum common radius and the subgradient of the radius function.
        Returns:
            r: the maximum common radius
            grad: subgradient of r w.r.t. centers (shape (N, 2))
        """
        n = len(centers)
        if n < 2:
            return 0.5, np.zeros_like(centers)
        
        # Wall distances
        # d_wall_i = min(0.5 - x_i, x_i + 0.5, 0.5 - y_i, y_i + 0.5)
        # Gradient contribution if wall is active: 
        #   min is 0.5 - x_i -> grad = (-1, 0)
        #   min is x_i + 0.5 -> grad = (1, 0)
        #   min is 0.5 - y_i -> grad = (0, -1)
        #   min is y_i + 0.5 -> grad = (0, 1)
        
        wall_dists = np.stack([
            0.5 - centers[:, 0],
            centers[:, 0] + 0.5,
            0.5 - centers[:, 1],
            centers[:, 1] + 0.5
        ]) # shape (4, N)
        
        min_wall_idx = np.argmin(wall_dists, axis=0) # shape (N,)
        min_wall_val = np.min(wall_dists, axis=0) # shape (N,)
        global_wall_min = np.min(min_wall_val)
        
        # Pairwise distances
        # d_pair_ij = ||c_i - c_j||
        # Gradient w.r.t c_i: (c_i - c_j) / ||c_i - c_j||
        # Gradient w.r.t c_j: (c_j - c_i) / ||c_i - c_j||
        
        dists = pdist(centers) # shape (N*(N-1)/2,)
        min_pair_val = np.min(dists)
        min_pair_idx = np.argmin(dists)
        
        # Convert linear index to (i, j)
        # pdist index k corresponds to i, j where i < j
        # k = i * (2*N - i - 1) / 2 + j - i - 1
        # Inverse:
        # i = floor((2*N - 1 - sqrt(1 + 8*k)) / 2)
        # j = i + 1 + k - i*(2*N - i - 1)/2
        
        m = n
        k = min_pair_idx
        i = int(np.floor((2 * m - 1 - np.sqrt(1 + 8 * k)) / 2))
        j = i + 1 + int(k - i * (2 * m - i - 1) / 2)
        
        # Verify indices (safety check)
        if i >= m or j >= m or i < 0 or j < 0:
            return min(global_wall_min, min_pair_val / 2), np.zeros_like(centers)
            
        d_vec = centers[i] - centers[j]
        dist_val = np.linalg.norm(d_vec)
        
        r_wall = global_wall_min
        r_pair = min_pair_val / 2.0
        
        grad = np.zeros_like(centers)
        
        if r_wall <= r_pair:
            r = r_wall
            # Subgradient from active wall constraints
            # Only walls that achieve the global min contribute? 
            # Actually, the function is min( min_i wall_i, min_ij pair_ij/2 ).
            # The subgradient of min is the convex hull of subgradients of active terms.
            # For a descent step, we can use the subgradient of the most active term (the minimum).
            # If multiple walls are tied for the global min, we sum their subgradients (or average).
            
            active_walls = np.where(min_wall_val == global_wall_min)[0]
            for idx in active_walls:
                wall_type = min_wall_idx[idx]
                if wall_type == 0: # 0.5 - x
                    grad[idx, 0] -= 1.0
                elif wall_type == 1: # x + 0.5
                    grad[idx, 0] += 1.0
                elif wall_type == 2: # 0.5 - y
                    grad[idx, 1] -= 1.0
                elif wall_type == 3: # y + 0.5
                    grad[idx, 1] += 1.0
            # Normalize? No, subgradient of min(f,g) is in conv(df, dg). 
            # Using the specific active one is a valid subgradient.
            # If multiple, any convex combination is valid. Summing is not a convex combination, 
            # but often works as a heuristic. Let's use the average of active subgradients for stability.
            if len(active_walls) > 1:
                grad /= len(active_walls)
        else:
            r = r_pair
            # Subgradient from active pair
            if dist_val > 1e-12:
                g_vec = d_vec / dist_val
                grad[i, :] += g_vec
                grad[j, :] -= g_vec
            else:
                # Random direction if coincident
                grad[i, :] += np.random.normal(0, 1, 2)
                grad[j, :] -= np.random.normal(0, 1, 2)
        
        return r, grad

    def generate_hex_initial(n):
        """Generate a good hexagonal initial configuration for n circles."""
        best_centers = None
        best_r = 0
        
        for rows in range(10, 25):
            base_cols = n // rows
            extra = n % rows
            
            centers = []
            for i in range(rows):
                cols = base_cols + (1 if i < extra else 0)
                if cols == 0:
                    continue
                for j in range(cols):
                    # Adjust spacing to fit better
                    x = (j - (cols - 1) / 2) * 0.1
                    if i % 2 == 1:
                        x += 0.05
                    y = (i - (rows - 1) / 2) * 0.15
                    centers.append([x, y])
            
            centers = np.array(centers)
            if len(centers) != n:
                continue
            
            r, _ = compute_radius_and_grad(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        
        return best_centers if best_centers is not None else None

    def bundle_method(centers, max_iter=500, deadline=None):
        """
        Bundle method for maximizing the non-smooth radius function.
        Maintains a bundle of subgradients and function values to build a PL lower bound.
        """
        current_centers = centers.copy()
        current_r, current_grad = compute_radius_and_grad(current_centers)
        
        # Bundle storage: list of (x_k, r_k, g_k)
        bundle_x = [current_centers.copy()]
        bundle_r = [current_r]
        bundle_g = [current_grad.copy()]
        
        # Parameters
        alpha = 1e-3 # Step size parameter
        beta = 0.9   # Proximity parameter
        max_bundle_size = 100
        
        for k in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
                
            # 1. Solve the bundle subproblem (QP)
            # Minimize: 0.5 * ||x - current_centers||^2 - alpha * (r_k + g_k^T (x - x_k))
            # Subject to: r_j + g_j^T (x - x_j) <= r_k + g_k^T (x - x_k) for all j in bundle
            # 
            # This is a QP. We can solve it using scipy.optimize.minimize with SLSQP or similar,
            # but for speed and robustness, we can implement a simple active set or use a library.
            # Given the constraints are linear in x (after rearranging), and objective is quadratic,
            # we can use scipy.optimize.linprog? No, objective is quadratic.
            # Let's use scipy.optimize.minimize with method='SLSQP' or 'trust-constr'.
            # However, calling a full QP solver every iteration might be slow.
            
            # Alternative: Use the "proximal bundle" update rule directly if we assume
            # the model minimum is at the current point or close to it.
            # A simpler, faster heuristic often used in practice:
            # Compute the "normal" and "steepest" directions.
            
            # Let's try a simpler approach: Subgradient method with averaging (Polyak-Ruppert)
            # or just a robust subgradient descent with adaptive step size.
            # The bundle method is complex to implement from scratch efficiently in Python without C extensions.
            # Let's implement a "Bundle-like" heuristic:
            # 1. Compute subgradient g at current x.
            # 2. Check if g is small (convergence).
            # 3. Move in direction -g (for minimization) or +g (for maximization).
            #    Here we maximize r, so move in +g direction.
            # 4. Use a step size that decreases.
            
            # Actually, let's implement a proper Bundle QP using scipy.optimize.minimize for the subproblem,
            # but limit the bundle size and frequency.
            
            # To keep it fast, let's use a simplified "Bundle" update:
            # We maintain the best point x_best.
            # We take a step in the direction of the subgradient of the current point,
            # but we project it onto the "bundle" polytope (approximated by checking against recent points).
            
            # For simplicity and speed within 150s, let's use a **Subgradient Method with Adaptive Step Size**.
            # This is less robust than a full bundle but much faster to implement and run in Python.
            
            # Adaptive step size:
            # alpha_k = (r_best - r_k) / ||g_k||^2
            
            if len(bundle_r) > 1:
                r_best = max(bundle_r)
            else:
                r_best = current_r
            
            g_norm_sq = np.sum(current_grad ** 2)
            if g_norm_sq < 1e-10:
                break
            
            # Step size
            if r_best > current_r:
                step_size = (r_best - current_r) / g_norm_sq
            else:
                step_size = 1e-4 # Small default
            
            # Limit step size to avoid huge jumps
            step_size = min(step_size, 0.01)
            
            # New point
            new_centers = current_centers + step_size * current_grad
            
            # Clip to box (approximate feasibility)
            new_centers[:, 0] = np.clip(new_centers[:, 0], -0.499, 0.499)
            new_centers[:, 1] = np.clip(new_centers[:, 1], -0.499, 0.499)
            
            new_r, new_grad = compute_radius_and_grad(new_centers)
            
            # Update bundle
            bundle_x.append(new_centers)
            bundle_r.append(new_r)
            bundle_g.append(new_grad)
            
            # Trim bundle
            if len(bundle_x) > max_bundle_size:
                # Keep the best 50 and most recent 20
                indices = list(range(len(bundle_x)))
                # Sort by r descending
                sorted_indices = sorted(indices, key=lambda i: bundle_r[i], reverse=True)
                keep_indices = sorted_indices[:50] + sorted_indices[-20:]
                # Remove duplicates
                keep_indices = list(set(keep_indices))
                
                bundle_x = [bundle_x[i] for i in keep_indices]
                bundle_r = [bundle_r[i] for i in keep_indices]
                bundle_g = [bundle_g[i] for i in keep_indices]
            
            # Update current
            if new_r > current_r:
                current_centers = new_centers
                current_r = new_r
                current_grad = new_grad
            else:
                # If no improvement, maybe take a smaller step or accept?
                # In bundle method, we might not move if the model predicts no improvement.
                # Here, we just continue.
                pass
            
            # Convergence check
            if g_norm_sq < 1e-12:
                break

        return current_centers, current_r

    def active_set_polish(centers, max_iter=200, deadline=None):
        """
        High-precision polish using active set resolution.
        Identifies the active constraints (closest pairs and walls) and solves the linear system
        to maximize the radius subject to these constraints being tight.
        """
        c = centers.copy()
        r = compute_radius_and_grad(c)[0]
        
        for _ in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            r, grad = compute_radius_and_grad(c)
            
            # Identify active constraints
            # 1. Walls
            wall_dists = np.stack([
                0.5 - c[:, 0],
                c[:, 0] + 0.5,
                0.5 - c[:, 1],
                c[:, 1] + 0.5
            ])
            min_wall_val = np.min(wall_dists, axis=0)
            active_walls = np.where(min_wall_val <= r + 1e-9)[0]
            
            # 2. Pairs
            dists = pdist(c)
            min_pair_val = np.min(dists)
            # Find all pairs within tolerance
            tol = 1e-9
            active_pairs = np.where(dists <= 2 * r + tol)[0]
            
            # If no active constraints, we can increase r? 
            # Actually, if no constraints are tight, the local minimum is determined by the "next" constraint.
            # For polishing, we assume the current configuration is near a local optimum where constraints are tight.
            
            # Construct the system:
            # We want to find a direction d such that the radius increases.
            # Or, solve for the optimal local configuration given the active set.
            
            # This is a QP: Maximize r
            # Subject to:
            #   c_i + t * d_i inside square
            #   ||(c_i + t*d_i) - (c_j + t*d_j)|| >= 2*(r + t)
            
            # For small t, linearize:
            #   grad_wall_i^T d_i >= 1
            #   grad_pair_ij^T (d_i - d_j) >= 2
            
            # This is getting complex. Let's use a simple "push" method for the final polish,
            # which is robust and fast.
            
            # Push closest pairs apart
            if len(active_pairs) > 0:
                # Take the closest few
                closest_pairs = active_pairs[:10]
                for idx in closest_pairs:
                    m = len(c)
                    k = idx
                    i = int(np.floor((2 * m - 1 - np.sqrt(1 + 8 * k)) / 2))
                    j = i + 1 + int(k - i * (2 * m - i - 1) / 2)
                    d_vec = c[i] - c[j]
                    dist = np.linalg.norm(d_vec)
                    if dist > 1e-10:
                        g_vec = d_vec / dist
                        # Push apart
                        c[i] += 1e-5 * g_vec
                        c[j] -= 1e-5 * g_vec
            
            # Push against walls
            if len(active_walls) > 0:
                for idx in active_walls:
                    wall_type = np.argmin(wall_dists[:, idx])
                    if wall_type == 0: # 0.5 - x
                        c[idx, 0] -= 1e-5
                    elif wall_type == 1: # x + 0.5
                        c[idx, 0] += 1e-5
                    elif wall_type == 2: # 0.5 - y
                        c[idx, 1] -= 1e-5
                    elif wall_type == 3: # y + 0.5
                        c[idx, 1] += 1e-5
            
            # Clip
            c[:, 0] = np.clip(c[:, 0], -0.499, 0.499)
            c[:, 1] = np.clip(c[:, 1], -0.499, 0.499)
            
            new_r = compute_radius_and_grad(c)[0]
            if new_r > r:
                r = new_r
            else:
                # If no improvement, break or reduce step?
                # Let's break to avoid oscillation.
                break

        return c, r

    # Main algorithm
    # 1. Generate initial hexagonal configuration
    initial = generate_hex_initial(N)
    if initial is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        initial = grid[:N]
    
    best_centers = initial.copy()
    best_r = compute_radius_and_grad(best_centers)[0]
    
    # 2. Bundle Method (Subgradient heuristic)
    if time.time() < deadline - 20:
        centers, r = bundle_method(initial, max_iter=300, deadline=deadline - 20)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # 3. Active Set Polish
    if time.time() < deadline - 5:
        centers, r = active_set_polish(best_centers, max_iter=100, deadline=deadline - 5)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            
    return best_centers
# EVOLVE-BLOCK-END
