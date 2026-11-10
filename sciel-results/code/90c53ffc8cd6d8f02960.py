import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Leave margin before 150s limit

def construct():
    start_time = time.time()
    
    def remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_radius(centers):
        """Compute the minimum radius for a set of centers."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_pair_sq)
        
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        r = min(min_pair / 2.0, min_wall)
        return r

    def smooth_min_surrogate(centers, beta=50.0):
        """
        Compute a smooth approximation of the minimum radius using log-sum-exp.
        This is differentiable and can be used with gradient-based optimization.
        
        r = min(min_pair/2, min_wall)
        We approximate min(a,b) ≈ -log(exp(-beta*a) + exp(-beta*b))/beta
        """
        # Pair distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_pair_sq)
        
        # Wall distances
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        # Smooth min of min_pair/2 and min_wall
        a = min_pair / 2.0
        b = min_wall
        
        # Use log-sum-exp for numerical stability
        max_ab = max(a, b)
        log_sum = max_ab - np.log(np.exp(-beta * (a - max_ab)) + np.exp(-beta * (b - max_ab)))
        smooth_r = log_sum / beta
        
        return smooth_r

    def smooth_min_surrogate_grad(centers, beta=50.0):
        """
        Compute the smooth minimum radius and its gradient.
        """
        # Pair distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_pair_sq)
        
        # Wall distances
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        a = min_pair / 2.0
        b = min_wall
        
        # Smooth min
        max_ab = max(a, b)
        ea = np.exp(-beta * (a - max_ab))
        eb = np.exp(-beta * (b - max_ab))
        log_sum = max_ab - np.log(ea + eb)
        smooth_r = log_sum / beta
        
        # Gradient
        # d(smooth_r)/da = -ea / (ea + eb)
        # d(smooth_r)/db = -eb / (ea + eb)
        w_a = ea / (ea + eb)
        w_b = eb / (ea + eb)
        
        grad = np.zeros_like(centers)
        
        # Gradient of a = min_pair / 2
        # d(min_pair)/d(centers) needs the active pair(s)
        threshold = min_pair_sq * (1.0 + 1e-5)
        mask = dist_sq <= threshold
        np.fill_diagonal(mask, False)
        
        d = np.sqrt(dist_sq)
        d_safe = np.where(d > 1e-10, d, 1.0)
        
        # Gradient of min_pair wrt centers
        # d(min_pair)/d(c_i) = sum_j (c_i - c_j) / d_ij * mask[i,j]
        W = np.zeros_like(dist_sq)
        W[mask] = 1.0 / d_safe[mask]
        
        W_sum = np.sum(W, axis=1)
        Wc = W.dot(centers)
        grad_pair = centers * W_sum[:, np.newaxis] - Wc
        grad_a = grad_pair / 2.0
        
        # Gradient of b = min_wall
        # Active walls
        threshold_wall = min_wall + 1e-5
        mask_wall = wall <= threshold_wall
        
        # d(min_wall)/d(c_i,k) = -sign(c_i,k) if active
        signs = np.sign(centers)
        signs[signs == 0] = 1.0
        grad_b = -signs * mask_wall
        
        # Combine
        grad = w_a * grad_a + w_b * grad_b
        
        return smooth_r, grad

    def optimize_smooth(centers, beta=50.0, max_iter=500):
        """
        Optimize using smooth surrogate with gradient descent.
        """
        centers = centers.copy().astype(float)
        
        for i in range(max_iter):
            if remaining() < 5.0:
                break
            
            # Compute smooth radius and gradient
            smooth_r, grad = smooth_min_surrogate_grad(centers, beta)
            
            # Normalize gradient
            gmag = np.linalg.norm(grad)
            if gmag < 1e-10:
                break
            grad /= gmag
            
            # Adaptive step size
            step = 0.005
            
            # Try step
            trial = centers + step * grad
            trial = np.clip(trial, -0.4999, 0.4999)
            
            new_smooth_r = smooth_min_surrogate(trial, beta)
            
            if new_smooth_r > smooth_r + 1e-12:
                centers = trial
            else:
                # Try smaller step
                step *= 0.5
                trial = centers + step * grad
                trial = np.clip(trial, -0.4999, 0.4999)
                new_smooth_r = smooth_min_surrogate(trial, beta)
                if new_smooth_r > smooth_r:
                    centers = trial
        
        return centers

    def optimize_scipy(centers, beta=50.0):
        """
        Use scipy.optimize.minimize with the smooth surrogate.
        """
        centers = centers.copy().astype(float)
        
        def objective(x):
            c = x.reshape(N, 3)
            # Keep inside cube
            c = np.clip(c, -0.4999, 0.4999)
            return -smooth_min_surrogate(c, beta)
        
        def gradient(x):
            c = x.reshape(N, 3)
            c = np.clip(c, -0.4999, 0.4999)
            _, grad = smooth_min_surrogate_grad(c, beta)
            return -grad.flatten()
        
        try:
            result = minimize(
                objective, 
                centers.flatten(), 
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            return result.x.reshape(N, 3)
        except:
            return centers

    def force_relax(centers, max_steps=2000, init_temp=0.05):
        """Force-directed relaxation."""
        centers = centers.copy().astype(float)
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        temp = init_temp
        step = 0.01
        no_improve = 0
        
        for i in range(max_steps):
            if remaining() < 5.0:
                break
            
            # Compute forces
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            dist_sq_safe = np.where(dist_sq < 1e-10, 1e-10, dist_sq)
            
            W = 2.0 / (dist_sq_safe * dist_sq_safe)
            np.fill_diagonal(W, 0.0)
            
            W_sum = np.sum(W, axis=1)
            Wc = W.dot(centers)
            pair_forces = centers * W_sum[:, np.newaxis] - Wc
            
            # Wall forces
            wall_forces = np.zeros_like(centers)
            for k in range(3):
                x = centers[:, k]
                d_pos = 0.5 - x
                d_neg = 0.5 + x
                
                f_pos = np.zeros(N)
                mask_pos = d_pos < 0.2
                if np.any(mask_pos):
                    d_safe = np.maximum(d_pos[mask_pos], 0.01)
                    f_pos[mask_pos] = -1.0 / (d_safe * d_safe)
                
                f_neg = np.zeros(N)
                mask_neg = d_neg < 0.2
                if np.any(mask_neg):
                    d_safe = np.maximum(d_neg[mask_neg], 0.01)
                    f_neg[mask_neg] = 1.0 / (d_safe * d_safe)
                
                wall_forces[:, k] = f_pos + f_neg
            
            forces = pair_forces + wall_forces
            
            fmag = np.linalg.norm(forces, axis=1)
            max_f = np.max(fmag)
            if max_f > 1e-10:
                forces = forces / max_f
            
            noise = np.random.randn(N, 3) * temp
            displacement = step * (forces + noise)
            
            trial = centers + displacement
            trial = np.clip(trial, -0.499, 0.499)
            
            new_r = compute_radius(trial)
            
            if new_r >= current_r:
                accept = True
            else:
                delta = current_r - new_r
                if temp > 1e-6:
                    accept = np.random.rand() < np.exp(-delta / temp)
                else:
                    accept = False
            
            if accept:
                centers = trial
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            temp *= 0.995
            step *= 0.999
            
            if no_improve > 200:
                n_perturb = max(1, N // 10)
                idx = np.random.choice(N, size=n_perturb, replace=False)
                centers[idx] += np.random.randn(n_perturb, 3) * 0.02
                centers = np.clip(centers, -0.499, 0.499)
                no_improve = 0
                temp = max(temp, 0.01)
                step = 0.01
                
        return best_centers, best_r

    best_r = 0.0
    best_centers = None
    
    np.random.seed(42)
    
    # Initializations
    inits = []
    
    # 1. 5x5x4 Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.44, 0.44, nx)
    ys = np.linspace(-0.44, 0.44, ny)
    zs = np.linspace(-0.44, 0.44, nz)
    grid_pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                grid_pts.append([x, y, z])
    inits.append(np.array(grid_pts[:N]))
    
    # 2. Random
    inits.append(np.random.uniform(-0.44, 0.44, (N, 3)))
    
    # 3. Jittered Grid
    base = inits[0].copy()
    base += np.random.randn(N, 3) * 0.03
    base = np.clip(base, -0.4999, 0.4999)
    inits.append(base)
    
    # 4. Body-centered cubic-like
    bcc_pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = -0.4 + i * 0.27
                y = -0.4 + j * 0.27
                z = -0.4 + k * 0.27
                bcc_pts.append([x, y, z])
                bcc_pts.append([x + 0.135, y + 0.135, z + 0.135])
    inits.append(np.array(bcc_pts[:N]))
    
    # Process each initialization
    for init_idx, init in enumerate(inits):
        if remaining() < 20.0:
            break
        
        # Phase 1: Force-directed relaxation
        c_force, r_force = force_relax(init, max_steps=1500, init_temp=0.05)
        
        # Phase 2: Smooth surrogate optimization
        c_smooth = optimize_smooth(c_force, beta=50.0, max_iter=200)
        
        # Phase 3: Scipy optimization
        c_scipy = optimize_scipy(c_smooth, beta=50.0)
        
        # Compute final radius
        r_final = compute_radius(c_scipy)
        
        if r_final > best_r:
            best_r = r_final
            best_centers = c_scipy.copy()
    
    # Final polishing
    if remaining() > 15.0 and best_centers is not None:
        for _ in range(3):
            if remaining() < 10.0:
                break
            c = optimize_scipy(best_centers, beta=80.0)
            r = compute_radius(c)
            if r > best_r:
                best_r = r
                best_centers = c
            else:
                break
    
    if best_centers is None:
        return inits[0]
        
    return best_centers
