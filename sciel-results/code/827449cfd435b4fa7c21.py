import numpy as np
import time

N = 200

def construct():
    # 1. Initialize with a hexagonal grid to provide a good starting structure
    # This is standard practice to avoid starting from pure random which is too slow for SA
    m = 14 # Approx sqrt(200) * sqrt(3)/2 ... 200 points.
    # Let's just use a grid for init
    # 14x14 is 196. 15x14 is 210.
    # Let's use a rectangular grid with jitter
    rows = 14
    cols = 15
    # Remove 10 points to get 200? Or just use 200 in a grid.
    # 200 = 14 * 14 + 4.
    # Let's stick to the hexagonal init from the seed as it's a good baseline, 
    # but the optimization will be SA.
    
    # Re-use the hex grid init logic from seed for a good start
    best_init = None
    best_init_score = -1
    for m in range(10, 22):
        base = N // m
        rem = N % m
        row_counts = [base + 1] * rem + [base] * (m - rem)
        n_max = max(row_counts)
        dx_width = 1.0 / n_max
        dx_height = 1.0 / (1.0 + (m - 1) * np.sqrt(3) / 2.0)
        dx = min(dx_width, dx_height)
        dy = dx * np.sqrt(3) / 2.0
        r_est = dx / 2.0
        if r_est > best_init_score:
            best_init_score = r_est
            best_init = (m, row_counts, dx, dy)
            
    m, row_counts, dx, dy = best_init
    centers = []
    y_start = -((m - 1) * dy) / 2.0
    for j, n in enumerate(row_counts):
        y = y_start + j * dy
        x_start = -((n - 1) * dx) / 2.0
        for i in range(n):
            x = x_start + i * dx
            centers.append([x, y])
            
    P = np.array(centers, dtype=float)
    
    # Add slight jitter to break symmetry and help SA escape local minima
    np.random.seed(42)
    P += np.random.uniform(-0.001, 0.001, P.shape)

    # 2. Simulated Annealing
    # Objective: Minimize Energy = Sum(1/d^2) for pairs + Sum(1/b^2) for bounds
    # This smooths the landscape compared to Min Distance.
    
    def get_energy(P):
        # Pairwise
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        # Avoid division by zero
        np.fill_diagonal(dists_sq, np.inf)
        # Energy from pairs
        E_pairs = np.sum(1.0 / dists_sq)
        
        # Boundary
        # dist to left: 0.5 - P[:, 0]
        # dist to right: 0.5 + P[:, 0]
        # dist to bottom: 0.5 - P[:, 1]
        # dist to top: 0.5 + P[:, 1]
        b_left = 0.5 - P[:, 0]
        b_right = 0.5 + P[:, 0]
        b_bottom = 0.5 - P[:, 1]
        b_top = 0.5 + P[:, 1]
        
        # If any point is outside, energy should be huge
        # But we assume P stays roughly inside. 
        # To be safe, penalize if b < 0.
        # Actually, 1/b^2 blows up near 0. If b < 0, 1/b^2 is positive but we want to push it in.
        # Better: Use max(b, epsilon) for the denominator? 
        # If a point goes outside, b is negative. 1/b^2 is positive. 
        # But we want to minimize energy. 
        # If we use 1/max(b, 1e-6)^2, it penalizes proximity to boundary.
        
        b_vals = np.stack([b_left, b_right, b_bottom, b_top], axis=1)
        # Penalize negative distances heavily?
        # Let's just use 1/b^2. If b is negative, it's still a large positive number? 
        # No, if b is small negative, b^2 is small positive, 1/b^2 is large positive.
        # So it penalizes being close to boundary, whether inside or outside.
        # This encourages staying inside.
        
        # To prevent overflow if b is extremely close to 0
        b_safe = np.maximum(np.abs(b_vals), 1e-6)
        E_bound = np.sum(1.0 / b_safe**2)
        
        return E_pairs + E_bound

    def get_radius(P):
        # The actual score
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        b_left = 0.5 - P[:, 0]
        b_right = 0.5 + P[:, 0]
        b_bottom = 0.5 - P[:, 1]
        b_top = 0.5 + P[:, 1]
        
        # If any point is outside, radius is invalid (or 0)
        # The score definition says: min(min_pair/2, min_bound)
        # If min_bound is negative, the score is negative? 
        # "A centre on or outside the square ... make the candidate invalid."
        # So we must ensure all points are inside.
        
        min_bound = np.min(np.stack([b_left, b_right, b_bottom, b_top], axis=1))
        
        if min_bound < 0:
             return 0.0 # Invalid
             
        return min(min_pair / 2.0, min_bound)

    current_energy = get_energy(P)
    current_radius = get_radius(P)
    best_radius = current_radius
    best_P = P.copy()
    
    start_time = time.time()
    max_time = 110.0
    
    # SA Parameters
    T = 0.01 # Initial Temperature
    T_min = 0.000001
    alpha = 0.9995 # Cooling factor
    max_iters = 50000
    
    for it in range(max_iters):
        if time.time() - start_time > max_time:
            break
            
        # Pick a random point
        idx = np.random.randint(N)
        
        # Generate a random displacement
        # Step size should be related to T
        step_size = T * 10.0
        disp = np.random.randn(2) * step_size
        
        P_new = P.copy()
        P_new[idx] += disp
        
        # Check if it's way outside to reject early? 
        # Energy calculation handles it, but let's be safe.
        if np.any(np.abs(P_new) > 0.6):
             continue
             
        new_energy = get_energy(P_new)
        
        # Acceptance Criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            P = P_new
            current_energy = new_energy
            
            # Update best if radius improved
            new_radius = get_radius(P)
            if new_radius > best_radius:
                best_radius = new_radius
                best_P = P.copy()
                
        # Cool down
        T *= alpha
        
    return best_P
