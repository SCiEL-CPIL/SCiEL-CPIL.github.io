import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        # Efficient pair distance calculation
        # Using broadcasting might be memory heavy for 200x200x2, but 200 is small.
        # 200*200*8 bytes = 320KB per array. 200*200*2*8 = 640KB. Totally fine.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def lj_potential_and_grad(x, sigma, epsilon=1.0):
        """
        Compute Lennard-Jones potential energy and gradient.
        U = sum_{i<j} 4*epsilon*((sigma/d)^12 - (sigma/d)^6) + wall_potential
        Wall potential: 4*epsilon*((sigma/w)^12 - (sigma/w)^6) where w is distance to wall.
        Actually, for walls, a simple repulsive potential like 1/w^12 or similar is often used.
        Let's use a unified LJ-like form for walls too, treating walls as fixed "atoms" at distance 0.5+sigma?
        No, simpler: U_wall = sum_i sum_k 4*epsilon*((sigma/w_ik)^12 - (sigma/w_ik)^6)
        This pushes points away from walls.
        """
        x = x.reshape(N, 2)
        
        # Pair interactions
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        # Avoid division by zero
        dists_sq_safe = np.where(dists_sq < 1e-12, 1e-12, dists_sq)
        dists = np.sqrt(dists_sq_safe)
        
        # LJ terms: 4*eps*(sigma^12/d^12 - sigma^6/d^6)
        # Let s6 = (sigma^2 / d^2)^3
        s2 = sigma ** 2
        s2_over_d2 = s2 / dists_sq_safe
        s6 = s2_over_d2 ** 3
        s12 = s6 ** 2
        
        # U_pair = 4 * (s12 - s6)
        # Gradient w.r.t x_i:
        # dU/dx_i = sum_j 4 * ( -12*s12/d * (x_i-x_j) + 6*s6/d * (x_i-x_j) ) ?
        # Let's derive:
        # U = 4 (s^12 r^-12 - s^6 r^-6)
        # dU/dr = 4 (-12 s^12 r^-13 + 6 s^6 r^-7)
        # dU/dx_i = (dU/dr) * (dr/dx_i) = (dU/dr) * (x_i - x_j)/r
        # = 4 (-12 s^12 r^-12 + 6 s^6 r^-6) * (x_i - x_j)
        # = 4 (-12 s12 + 6 s6) * (x_i - x_j) / r^2 ? No.
        # r^-13 * (x_i-x_j)/r = (x_i-x_j) r^-14 = (x_i-x_j) / r^2 * r^-12.
        # So term is 4 * ( -12 s12 + 6 s6 ) * (x_i - x_j) / dists_sq
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Force on i from j is -grad. We want grad of U.
        # grad_U_i = sum_{j != i} 4 * ( -12 s12_ij + 6 s6_ij ) * (x_i - x_j) / dists_sq_ij
        
        coeff = 4.0 * (-12.0 * s12 + 6.0 * s6)
        coeff = np.where(mask, coeff, 0.0)
        
        # grad_pair = sum_j coeff_ij * (x_i - x_j) / dists_sq_ij
        # diff is (x_i - x_j)
        # term = coeff / dists_sq_safe
        term = coeff / dists_sq_safe
        term = np.where(mask, term, 0.0)
        
        grad_pair = np.einsum('ij,ijk->ik', term, diff)
        
        U_pair = np.sum(4.0 * (s12 - s6) * mask)
        
        # Wall interactions
        # w_x = 0.5 - |x|
        # We want to repel from walls.
        # Use same LJ form with "wall atoms" effectively at 0.5+sigma?
        # Or just define U_wall = sum 4*( (sigma/w)^12 - (sigma/w)^6 )
        # This goes to infinity as w->0.
        
        w_x = 0.5 - np.abs(x[:, 0])
        w_y = 0.5 - np.abs(x[:, 1])
        
        # Clip to avoid zero
        w_x_safe = np.clip(w_x, 1e-9, None)
        w_y_safe = np.clip(w_y, 1e-9, None)
        
        s2_wx = s2 / (w_x_safe ** 2)
        s6_wx = s2_wx ** 3
        s12_wx = s6_wx ** 2
        
        s2_wy = s2 / (w_y_safe ** 2)
        s6_wy = s2_wy ** 3
        s12_wy = s6_wy ** 2
        
        U_wall = np.sum(4.0 * (s12_wx - s6_wx) + 4.0 * (s12_wy - s6_wy))
        
        # Gradient w.r.t x
        # dU/dx = 4 * ( -12 s12/w * (1/w) * d(1/w)/dx ? )
        # Let's use chain rule.
        # U(w) = 4(s^12 w^-12 - s^6 w^-6)
        # dU/dw = 4(-12 s^12 w^-13 + 6 s^6 w^-7)
        # dw/dx = -sign(x)
        # dU/dx = dU/dw * dw/dx = 4(-12 s^12 w^-13 + 6 s^6 w^-7) * (-sign(x))
        # = 4(12 s^12 w^-13 - 6 s^6 w^-7) * sign(x)
        
        # Note: s^12 w^-13 = (s^2 w^-2)^3 * w^-1 = s6 * w^-1
        # s^6 w^-7 = (s^2 w^-2)^3 * w^-1 ? No. s^6 w^-6 * w^-1 = s6 * w^-1.
        # So dU/dx = 4 * (12 * s6_wx / w_x - 6 * s6_wx / w_x) * sign(x) ?
        # Wait: s^12 w^-13 = s^12 w^-12 * w^-1 = s12 * w^-1.
        # s^6 w^-7 = s^6 w^-6 * w^-1 = s6 * w^-1.
        
        grad_wall_x = 4.0 * (12.0 * s12_wx / w_x_safe - 6.0 * s6_wx / w_x_safe) * np.sign(x[:, 0])
        grad_wall_y = 4.0 * (12.0 * s12_wy / w_y_safe - 6.0 * s6_wy / w_y_safe) * np.sign(x[:, 1])
        
        # Handle sign(0) = 0, which means no force. This is fine, but if x=0, w=0.5, force is non-zero but sign is 0?
        # If x=0, it's stable point for wall potential? No, LJ has minimum at r=sigma.
        # If x=0, w=0.5. If sigma < 0.5, it's repulsive.
        # We need to ensure points don't get stuck at 0 if that's not optimal.
        # Actually, if x=0, the potential is symmetric. The gradient should be 0 at x=0 due to symmetry.
        # But we want to push them to a lattice.
        # The pair interactions will break the symmetry.
        
        grad = np.column_stack((grad_pair[:, 0] + grad_wall_x, grad_pair[:, 1] + grad_wall_y))
        
        U_total = U_pair + U_wall
        
        return U_total, grad.flatten()

    def hexagonal_init(n_rows, n_cols, d_scale=1.0):
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        return pts

    def square_grid_init(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def optimize_lj(init, time_limit):
        x0 = init.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        # Estimate sigma.
        # Approx area per circle = 1/N.
        # d ~ sqrt(1/N).
        # sigma is equilibrium distance.
        # For hex packing, d = sqrt(2/sqrt(3) * 1/N) ?
        # Area of hex cell = d^2 * sqrt(3)/2.
        # d^2 = 2 / (sqrt(3) N).
        # d = sqrt(2 / (sqrt(3) N)).
        sigma = math.sqrt(2.0 / (math.sqrt(3.0) * N))
        
        # Optimize LJ
        start = time.time()
        
        res = minimize(
            lambda x: lj_potential_and_grad(x, sigma)[0],
            x0,
            jac=lambda x: lj_potential_and_grad(x, sigma)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'gtol': 1e-6}
        )
        
        if time.time() > deadline - 5:
            return x0.reshape(N, 2)
            
        x_opt = res.x if res.x is not None else x0
        
        # Inflation Phase
        # Now we have a configuration. We want to maximize the minimum distance.
        # Use the soft-min objective again, but starting from a good LJ solution.
        # Or, simpler: just return the LJ solution?
        # LJ minimization minimizes energy, which correlates with maximizing min distance,
        # but it's not exact.
        # Let's do a quick refinement with the soft-min objective from the previous code,
        # but only for a short time.
        
        def soft_min_obj(x, power=10):
            x = x.reshape(N, 2)
            wall_dists = 0.5 - np.abs(x)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            dists = np.where(np.eye(N, dtype=bool), np.inf, dists)
            
            wall_flat = wall_dists.flatten()
            pair_idx = np.triu_indices(N, k=1)
            pair_flat = dists[pair_idx]
            
            all_dists = np.concatenate([wall_flat, pair_flat])
            v = np.clip(all_dists, 1e-12, None)
            S = np.sum(v ** (-power)) / len(v)
            soft_min_val = S ** (-1.0 / power)
            return -soft_min_val

        def soft_min_grad(x, power=10):
            x = x.reshape(N, 2)
            wall_dists = 0.5 - np.abs(x)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            dists = np.where(np.eye(N, dtype=bool), np.inf, dists)
            
            wall_flat = wall_dists.flatten()
            pair_idx = np.triu_indices(N, k=1)
            pair_flat = dists[pair_idx]
            
            all_dists = np.concatenate([wall_flat, pair_flat])
            v = np.clip(all_dists, 1e-12, None)
            S = np.sum(v ** (-power)) / len(v)
            
            grad = np.zeros_like(x)
            coeff_base = (1.0 / len(v)) * S ** (-1.0 / power - 1.0)
            
            # Wall
            wall_v = np.clip(wall_dists, 1e-12, None)
            wall_coeff = coeff_base * wall_v ** (-power - 1)
            signs = np.sign(x)
            signs[signs == 0] = 1.0
            grad += wall_coeff * signs
            
            # Pair
            pair_v = np.clip(pair_flat, 1e-12, None)
            pair_coeff = coeff_base * pair_v ** (-power - 1)
            
            coeff_matrix = np.zeros((N, N))
            coeff_matrix[pair_idx] = pair_coeff
            coeff_matrix = (coeff_matrix + coeff_matrix.T) / 2.0
            
            dist_safe = np.where(dists < 1e-9, 1e-9, dists)
            term = coeff_matrix / dist_safe
            grad_pairs = np.einsum('ij,ijk->ik', term, diff)
            
            grad = grad - grad_pairs
            return grad.flatten()

        # Run soft-min refinement
        remaining = deadline - time.time()
        if remaining > 5:
             res2 = minimize(
                 lambda x: soft_min_obj(x, 10),
                 x_opt,
                 jac=lambda x: soft_min_grad(x, 10),
                 method='L-BFGS-B',
                 bounds=bounds,
                 options={'maxiter': 200, 'gtol': 1e-7}
             )
             if res2.x is not None:
                 x_opt = res2.x
                 
        return x_opt.reshape(N, 2)

    # Main logic
    best_centers = None
    best_radius = 0.0
    
    inits = []
    
    # Hexagonal
    for n_rows in range(13, 17):
        if time.time() > deadline - 30:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.95, 1.0, 1.05]:
            pts = hexagonal_init(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)
                
    # Square Grid
    m = int(np.ceil(np.sqrt(N)))
    grid = square_grid_init(m)
    inits.append(grid)
    
    # Random
    np.random.seed(42)
    for _ in range(3):
        if time.time() > deadline - 30:
            break
        rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)
        
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        remaining = deadline - time.time()
        t_per_init = remaining / (len(inits) - i)
        t_limit = min(t_per_init, 20.0)
        
        if t_limit < 2.0:
            break
            
        centers = optimize_lj(init, t_limit)
        
        r = compute_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
