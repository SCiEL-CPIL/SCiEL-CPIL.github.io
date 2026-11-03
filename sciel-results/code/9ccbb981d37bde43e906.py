import numpy as np
import time

N = 200

def construct():
    # Set random seed for reproducibility in debugging, but let's allow some randomness
    # np.random.seed(42)
    
    # 1. Initialization
    # Start with a random distribution inside the square
    # To improve convergence, we can start with a grid and add noise
    # Or just random. Let's use a grid + noise for a better start.
    
    # Create a grid
    m = 14 # approx sqrt(200) ~ 14.14
    base = N // m
    rem = N % m
    row_counts = [base + 1] * rem + [base] * (m - rem)
    
    centers = []
    y_start = -0.45
    y_end = 0.45
    dy = (y_end - y_start) / max(1, m - 1)
    
    for j, n in enumerate(row_counts):
        y = y_start + j * dy
        x_start = -0.45
        x_end = 0.45
        dx = (x_end - x_start) / max(1, n - 1)
        for i in range(n):
            x = x_start + i * dx
            # Add small noise to break symmetry
            x += np.random.uniform(-0.01, 0.01)
            y += np.random.uniform(-0.01, 0.01)
            centers.append([x, y])
            
    P = np.array(centers, dtype=float)
    
    # Clip to ensure they start inside (though score handles it, valid start helps)
    P[:, 0] = np.clip(P[:, 0], -0.49, 0.49)
    P[:, 1] = np.clip(P[:, 1], -0.49, 0.49)
    
    def calculate_score(P):
        # Calculate minimum distance between any pair and to boundaries
        # Use cdist for efficiency
        # from scipy.spatial.distance import cdist
        # dists = cdist(P, P)
        # np.fill_diagonal(dists, np.inf)
        # min_pair = np.min(dists)
        
        # Manual vectorized calculation for speed if SciPy is slow to import or unavailable (it is available)
        # Let's use the vectorized method which is fast enough for N=200
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        # dist to left: 0.5 - P[:, 0]
        # dist to right: 0.5 + P[:, 0]
        # dist to bottom: 0.5 - P[:, 1]
        # dist to top: 0.5 + P[:, 1]
        bound_dists = np.min([
            0.5 - P[:, 0],
            0.5 + P[:, 0],
            0.5 - P[:, 1],
            0.5 + P[:, 1]
        ], axis=0)
        min_bound = np.min(bound_dists)
        
        return min(min_pair, min_bound)

    current_score = calculate_score(P)
    best_P = P.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    T = 0.01 # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9999
    
    # We will perform moves. 
    # A move consists of jittering a random subset of points or all points.
    # Jittering all points with small steps is good for fine tuning.
    # Jittering a few points is good for structural changes.
    
    # Let's try a mixed strategy: 
    # Mostly jitter all points slightly.
    # Occasionally jitter a small cluster.
    
    while time.time() - start_time < max_time:
        if T < T_min:
            T = T_min
            
        # Generate a new solution
        P_new = P.copy()
        
        # Randomly decide move type
        # 90%: jitter all points
        # 10%: jitter 5-10 random points with larger steps
        
        if np.random.rand() < 0.9:
            # Jitter all points
            step_size = T * 0.1 # Scale step with temperature? Or fixed?
            # Fixed step is often better for packing. Let's use a step that shrinks with T.
            step = 0.005 * (T / 0.01) ** 0.5 
            if step < 1e-5: step = 1e-5
            if step > 0.05: step = 0.05
            
            noise = np.random.randn(N, 2) * step
            P_new += noise
        else:
            # Jitter a random subset
            k = np.random.randint(5, 15)
            idx = np.random.choice(N, k, replace=False)
            step = 0.01
            noise = np.zeros((N, 2))
            noise[idx] = np.random.randn(k, 2) * step
            P_new += noise
            
        # Keep points within bounds to avoid huge penalty, but score handles it.
        # Actually, allowing them to go outside and penalizing via score is standard.
        
        new_score = calculate_score(P_new)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            P = P_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_P = P.copy()
                
        # Cool down
        T *= cooling_rate
        
    # Final local polish: Hill climbing on the best solution found
    P = best_P
    current_score = best_score
    
    # Simple local search to polish
    for it in range(5000):
        if time.time() - start_time > max_time:
            break
            
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        min_bound_vals = np.min([
            0.5 - P[:, 0],
            0.5 + P[:, 0],
            0.5 - P[:, 1],
            0.5 + P[:, 1]
        ], axis=0)
        min_bound_val = np.min(min_bound_vals)
        
        if min_pair_val < min_bound_val:
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            if i == j: continue
            vec = P[i] - P[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                dir_vec = vec / norm
            else:
                dir_vec = np.random.randn(2)
                dir_vec /= np.linalg.norm(dir_vec)
                
            # Try moving i away from j
            for step_mult in [1.0, 0.5, 2.0]:
                step = 0.001 * step_mult
                P_old_i = P[i].copy()
                P[i] = P_old_i + dir_vec * step
                new_score = calculate_score(P)
                if new_score > current_score:
                    current_score = new_score
                    break
                else:
                    P[i] = P_old_i
        else:
            idx = np.argmin(min_bound_vals)
            i = idx
            b_left = 0.5 - P[i, 0]
            b_right = 0.5 + P[i, 0]
            b_bottom = 0.5 - P[i, 1]
            b_top = 0.5 + P[i, 1]
            
            if b_left == min_bound_vals[i]:
                dir_vec = np.array([1, 0])
            elif b_right == min_bound_vals[i]:
                dir_vec = np.array([-1, 0])
            elif b_bottom == min_bound_vals[i]:
                dir_vec = np.array([0, 1])
            else:
                dir_vec = np.array([0, -1])
                
            for step_mult in [1.0, 0.5, 2.0]:
                step = 0.001 * step_mult
                P_old_i = P[i].copy()
                P[i] = P_old_i + dir_vec * step
                new_score = calculate_score(P)
                if new_score > current_score:
                    current_score = new_score
                    break
                else:
                    P[i] = P_old_i
                    
    return P
