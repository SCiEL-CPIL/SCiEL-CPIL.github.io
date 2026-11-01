import numpy as np
import scipy.optimize
import time

N = 100

def construct():
        start_time = time.time()
        deadline = start_time + 145.0
        
        # 1. Initial Guess: Jittered Grid
        # 5x5x4 grid = 100 points? 5*5*4 = 100.
        # Or 4x5x5.
        # Let's use a 5x5x5 grid and take the first 100.
        # Or better: random uniform.
        np.random.seed(42)
        c_init = np.random.uniform(-0.5, 0.5, (N, 3))
        
        # Better init: Jittered Grid
        # 100 points. 5x5x4.
        nx, ny, nz = 5, 5, 4
        grid = np.array([(x, y, z) for z in range(nz) for y in range(ny) for x in range(nx)])
        # Scale to [-0.5, 0.5]
        grid[:, 0] = (grid[:, 0] + 0.5) / nx - 0.5
        grid[:, 1] = (grid[:, 1] + 0.5) / ny - 0.5
        grid[:, 2] = (grid[:, 2] + 0.5) / nz - 0.5
        # Add jitter
        jitter = np.random.uniform(-0.05, 0.05, grid.shape)
        c_init = grid + jitter
        # Clip to valid range to be safe
        c_init = np.clip(c_init, -0.45, 0.45)
        
        best_c = c_init
        best_score = -1.0
        
        def calc_score(c):
            # Compute min distance
            # Pairwise
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            
            # Boundary
            # min_k (0.5 - |c_ik|)
            # c is in [-0.5, 0.5]
            # dist to boundary is min(0.5 - c, 0.5 + c) for each dim?
            # No, it's min(0.5 - |c_k|) for each coordinate k.
            # Actually, distance to boundary of cube [-0.5, 0.5] for point c is:
            # min(0.5 - c_x, c_x + 0.5, 0.5 - c_y, c_y + 0.5, 0.5 - c_z, c_z + 0.5)
            # Which is min(0.5 - |c_x|, 0.5 - |c_y|, 0.5 - |c_z|)
            abs_c = np.abs(c)
            min_bound = np.min(0.5 - abs_c)
            
            r = min(min_pair / 2.0, min_bound)
            return r, c

        current_score, _ = calc_score(c_init)
        best_score = current_score
        best_c = c_init.copy()
        
        # Try to improve with Nelder-Mead
        # We minimize the "overlap energy" for a target radius
        # If we can minimize energy to ~0, we have a valid packing.
        
        target_radii = [0.110, 0.111, 0.1113]
        
        for r_target in target_radii:
            if time.time() > deadline - 10.0:
                break
                
            def energy(c_flat):
                c = c_flat.reshape(-1, 3)
                # Clip to ensure validity during optimization? 
                # Nelder-Mead might go out of bounds.
                # We should penalize out-of-bounds heavily.
                
                # Pairwise distances
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                
                # Overlaps
                overlaps = np.maximum(0, 2*r_target - dists)
                e_pairs = np.sum(overlaps**2)
                
                # Boundary violations
                abs_c = np.abs(c)
                dist_bound = 0.5 - abs_c
                # dist_bound can be negative if outside
                # We want to penalize if dist_bound < r_target
                # i.e., if 0.5 - |c_k| < r_target
                # So violation is max(0, r_target - (0.5 - |c_k|))
                bound_viol = np.maximum(0, r_target - dist_bound)
                e_bound = np.sum(bound_viol**2)
                
                return e_pairs + e_bound
            
            try:
                res = scipy.optimize.minimize(
                    energy, 
                    c_init.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True}
                )
                c_opt = res.x.reshape(-1, 3)
                
                # Check if valid
                score, _ = calc_score(c_opt)
                if score > best_score:
                    best_score = score
                    best_c = c_opt.copy()
                    
                # Update c_init to the best found so far for next target?
                # Yes, usually helps.
                c_init = best_c.copy()
                
            except Exception as e:
                pass
        
        return best_c
