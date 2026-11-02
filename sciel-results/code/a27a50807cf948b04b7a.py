import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def compute_score(pts):
        """Compute the packing score (minimum radius)."""
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_bound = np.min(1.0 - radii)
        
        return min(min_pair / 2.0, min_bound)

    def generate_hex_init(spacing, rotation=0.0, noise=0.01):
        """Generate hexagonal lattice points inside unit circle."""
        points = []
        max_j = int(2.0 / (spacing * np.sqrt(3.0))) + 3
        max_i = int(2.0 / spacing) + 3
        
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if rotation != 0.0:
                    xr = x * np.cos(rotation) - y * np.sin(rotation)
                    yr = x * np.sin(rotation) + y * np.cos(rotation)
                else:
                    xr, yr = x, y
                if xr * xr + yr * yr <= 1.0:
                    points.append([xr, yr])
        
        pts = np.array(points)
        if len(pts) < N:
            # Pad with random points if needed
            extra = np.random.randn(N - len(pts), 2) * 0.5
            pts = np.vstack([pts, extra])
        else:
            # Select N closest to center
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
        
        # Add noise
        pts += np.random.randn(N, 2) * noise
        return pts

    def simulated_annealing(init_pts, time_budget):
        """Run Simulated Annealing to maximize the minimum radius."""
        pts = init_pts.copy()
        current_score = compute_score(pts)
        best_pts = pts.copy()
        best_score = current_score
        
        T = 0.05
        step_size = 0.05
        alpha = 0.9995 # Cooling rate
        step_decay = 0.9999 # Step size decay
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            # Pick a random point
            i = np.random.randint(N)
            
            # Generate move
            move = np.random.randn(2) * step_size
            new_pts = pts.copy()
            new_pts[i] += move
            
            # Check bounds immediately to reject obvious bad moves
            r_new_i = np.linalg.norm(new_pts[i])
            if r_new_i > 1.0:
                # Reject if clearly out of bounds
                T *= alpha
                step_size *= step_decay
                continue
            
            new_score = compute_score(new_pts)
            
            if new_score > current_score:
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            else:
                # Accept with probability
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        pts = new_pts
                        current_score = new_score
            
            T *= alpha
            step_size *= step_decay
            
            # Prevent step size from becoming too small
            if step_size < 1e-6:
                step_size = 1e-6
                
        return best_pts, best_score

    def soft_min_and_grad(x, beta):
        """Compute soft-min and its gradient."""
        x_max = np.max(x)
        exp_neg_beta_x = np.exp(-beta * (x - x_max))
        sum_exp = np.sum(exp_neg_beta_x)
        s_val = -1.0 / beta * (np.log(sum_exp) - beta * x_max)
        grad = exp_neg_beta_x / sum_exp
        return s_val, grad

    def objective_and_grad(flat_pts, beta):
        """Objective function and gradient for L-BFGS-B."""
        pts = flat_pts.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_matrix = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(d_matrix, np.inf)
        
        # Upper triangle mask
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_flat = d_matrix[mask]
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        d_bound = 1.0 - radii
        
        # Effective radii
        eff_rad = np.concatenate([d_flat / 2.0, d_bound])
        
        # Soft min
        s_val, grad_eff = soft_min_and_grad(eff_rad, beta)
        obj = -s_val
        
        grad_pair = grad_eff[:len(d_flat)]
        grad_bound = grad_eff[len(d_flat):]
        
        # Gradient via np.add.at for efficiency
        safe_d = np.where(d_matrix < 1e-9, 1e-9, d_matrix)
        
        # Build symmetric gradient matrix
        G = np.zeros((N, N))
        G[mask] = grad_pair
        G.T[mask] = grad_pair
        
        # F_i = sum_j G[i,j] * (p_i - p_j) / d_ij
        weight = G / safe_d
        F_pair = weight @ diff
        
        # Boundary gradient
        safe_radii = np.where(radii < 1e-9, 1e-9, radii)
        F_bound = -grad_bound[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
        
        total_grad = F_pair + F_bound
        
        return obj, total_grad.flatten()

    # 1. Generate diverse initializations
    inits = []
    spacings = [0.105, 0.108, 0.112]
    rotations = [0.0, np.pi / 12, np.pi / 6]
    
    for sp in spacings:
        for rot in rotations:
            pts = generate_hex_init(sp, rot, noise=0.02)
            inits.append(pts)
            
    # 2. Run SA for each initialization
    best_overall_pts = None
    best_overall_score = -np.inf
    
    # Allocate time: 30s per SA run, max 3 runs if time permits
    time_per_sa = 30.0
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 20.0: # Leave 20s for polish
            break
            
        sa_pts, sa_score = simulated_annealing(init_pts, time_per_sa)
        
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall_pts = sa_pts.copy()

    if best_overall_pts is None:
        # Fallback to random if SA failed
        best_overall_pts = np.random.randn(N, 2) * 0.5
        best_overall_score = compute_score(best_overall_pts)

    # 3. Polish with L-BFGS-B
    # Use the remaining time for gradient-based refinement
    if time.time() < deadline - 5.0:
        current_pts = best_overall_pts.copy()
        
        # Anneal beta to sharpen the minimum
        beta_start = 50.0
        beta_end = 50000.0
        num_rounds = 5
        time_per_round = (deadline - time.time() - 5.0) / num_rounds
        
        for round_idx in range(num_rounds):
            if time.time() > deadline - 5.0:
                break
                
            t = round_idx / max(1, num_rounds - 1)
            beta = beta_start * (beta_end / beta_start) ** t
            
            flat_pts = current_pts.flatten()
            
            try:
                result = minimize(
                    objective_and_grad,
                    flat_pts,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 500,
                        'maxfun': 1000,
                        'ftol': 1e-12,
                        'gtol': 1e-12,
                        'maxls': 50
                    }
                )
                
                if result.success or result.fun < 0:
                    new_pts = result.x.reshape(N, 2)
                    
                    radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                    out_of_bounds = radii > 1.0
                    if np.any(out_of_bounds):
                        new_pts[out_of_bounds] *= (0.9999 / radii[out_of_bounds])[:, np.newaxis]
                    
                    new_score = compute_score(new_pts)
                    
                    if new_score > best_overall_score:
                        best_overall_score = new_score
                        best_overall_pts = new_pts.copy()
                        
                    current_pts = new_pts.copy()
                        
            except Exception:
                pass

    return best_overall_pts
