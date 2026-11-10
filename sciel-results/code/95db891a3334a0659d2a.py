import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 145.0  # Leave margin before 150s limit

def construct():
    start_time = time.time()
    
    def remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_radius(centers):
        """Compute the minimum radius for a set of centers."""
        # Pair distances using cdist for efficiency
        dist = cdist(centers, centers)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        
        # Wall distances
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        r = min(min_pair / 2.0, min_wall)
        return r

    def compute_radius_and_gradient(centers):
        """Compute radius and gradient of the objective (min of pair-dist/2 and wall-dist)."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_pair_sq)
        
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        r = min(min_pair / 2.0, min_wall)
        
        grad = np.zeros_like(centers)
        
        if min_pair / 2.0 <= min_wall + 1e-12:
            # Pair constraint is binding
            threshold = min_pair_sq * (1.0 + 1e-6)
            mask = dist_sq <= threshold
            np.fill_diagonal(mask, False)
            
            d = np.sqrt(dist_sq)
            d_safe = np.where(d > 1e-10, d, 1.0)
            
            W = np.zeros_like(dist_sq)
            W[mask] = 1.0 / (2.0 * d_safe[mask])
            
            W_sum = np.sum(W, axis=1)
            Wc = W.dot(centers)
            
            grad = centers * W_sum[:, np.newaxis] - Wc
            
        else:
            # Wall constraint is binding
            threshold = min_wall + 1e-6
            mask_wall = wall <= threshold
            
            signs = np.sign(centers)
            signs[signs == 0] = 1.0
            
            grad = -signs * mask_wall
        
        gmag = np.linalg.norm(grad)
        if gmag > 1e-10:
            grad /= gmag
        
        return r, grad

    def compute_forces(centers, beta=1.0):
        """Compute repulsion forces between all pairs and wall forces."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq_safe = np.where(dist_sq < 1e-10, 1e-10, dist_sq)
        
        # Inverse square law repulsion
        W = 2.0 / (dist_sq_safe * dist_sq_safe)
        np.fill_diagonal(W, 0.0)
        
        W_sum = np.sum(W, axis=1)
        Wc = W.dot(centers)
        pair_forces = centers * W_sum[:, np.newaxis] - Wc
        
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
        
        return pair_forces + wall_forces

    def optimize_force(centers, max_steps=5000, init_temp=0.05):
        """Simulated annealing / Langevin dynamics with force-directed repulsion."""
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
            
            forces = compute_forces(centers)
            
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

    def optimize_steepest(centers, max_steps=50000, init_step=0.01):
        """Steepest ascent on the radius objective with adaptive step size."""
        centers = centers.copy().astype(float)
        current_r, _ = compute_radius_and_gradient(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        step = init_step
        no_improve = 0
        stall_count = 0
        
        for i in range(max_steps):
            if remaining() < 5.0:
                break
            
            r, grad = compute_radius_and_gradient(centers)
            current_r = r
            
            trial = centers + step * grad
            trial = np.clip(trial, -0.4999, 0.4999)
            
            new_r = compute_radius(trial)
            
            if new_r > current_r + 1e-12:
                centers = trial
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                step = min(step * 1.02, 0.05)
            else:
                step *= 0.5
                no_improve += 1
                stall_count += 1
                
                if stall_count > 30:
                    n_perturb = max(1, N // 10)
                    idx = np.random.choice(N, size=n_perturb, replace=False)
                    centers[idx] += np.random.randn(n_perturb, 3) * 0.02
                    centers = np.clip(centers, -0.4999, 0.4999)
                    step = init_step
                    stall_count = 0
                    no_improve = 0
            
            if no_improve > 500:
                n_perturb = max(2, N // 5)
                idx = np.random.choice(N, size=n_perturb, replace=False)
                centers[idx] += np.random.randn(n_perturb, 3) * 0.04
                centers = np.clip(centers, -0.4999, 0.4999)
                step = init_step
                no_improve = 0
                stall_count = 0
        
        return best_centers, best_r

    best_r = 0.0
    best_centers = None
    
    np.random.seed(42)
    
    inits = []
    
    # Init 1: 5x5x4 grid
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
    
    # Init 2: Random uniform
    inits.append(np.random.uniform(-0.44, 0.44, (N, 3)))
    
    # Init 3: Jittered grid
    base = inits[0].copy()
    base += np.random.randn(N, 3) * 0.03
    base = np.clip(base, -0.4999, 0.4999)
    inits.append(base)
    
    # Init 4: Face-centered cubic inspired
    fc_pts = []
    for layer in range(5):
        z = -0.4 + layer * 0.2
        for i in range(4):
            for j in range(5):
                x = -0.4 + i * 0.2 + (0.05 if layer % 2 else 0)
                y = -0.4 + j * 0.2 + (0.05 if (layer + i) % 2 else 0)
                fc_pts.append([x, y, z])
    inits.append(np.array(fc_pts[:N]))
    
    # Optimize each initialization with both methods
    for init_idx, init in enumerate(inits):
        if remaining() < 20.0:
            break
        
        # Method 1: Force-directed annealing
        c, r = optimize_force(init, max_steps=20000, init_temp=0.05)
        if r > best_r:
            best_r = r
            best_centers = c
        
        # Method 2: Steepest ascent on the result of method 1
        if best_centers is not None and remaining() > 10.0:
            c, r = optimize_steepest(best_centers, max_steps=20000, init_step=0.005)
            if r > best_r:
                best_r = r
                best_centers = c
    
    # Fine-tuning: multiple passes with smaller steps
    if remaining() > 15.0 and best_centers is not None:
        for ft_pass in range(3):
            if remaining() < 10.0:
                break
            c, r = optimize_steepest(best_centers, max_steps=10000, init_step=0.002)
            if r > best_r:
                best_r = r
                best_centers = c
            else:
                break
    
    if best_centers is None:
        return inits[0]
    
    return best_centers
