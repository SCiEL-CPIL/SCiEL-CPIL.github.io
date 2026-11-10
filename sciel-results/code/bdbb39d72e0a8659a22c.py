# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N
    t0 = time.time()
    DEADLINE = 590.0  # Safety margin for return overhead

    def elapsed():
        return time.time() - t0

    # Pre-compute indices for tubes
    # k is in [0, 64)
    # i = k // 8, j = k % 8
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    # Pre-compute shifts for x and y as functions of c
    # x_k(c) = x_k + c * (i_k * w)
    # y_k(c) = y_k + c * (j_k * w)
    # We want to compute the union area of squares [x_k(c), x_k(c)+w] x [y_k(c), y_k(c)+w]
    
    # To speed up the volume calculation, we use a deterministic integration over c.
    # We sample M slices in c in [0, 1].
    # For each slice, we compute the union area of 64 axis-aligned squares.
    # Union area of axis-aligned squares can be computed by sweeping or by 
    # checking overlaps. Since N=64 is small, we can use a simple grid or 
    # exact interval union.
    
    # Exact interval union for 1D is O(N log N).
    # For 2D union of rectangles, it's more complex.
    # However, since all squares have the same size w, we can use a 
    # "pixel" approach or a more efficient geometric algorithm.
    
    # Given the speed requirements, a vectorized MC with a fixed seed 
    # but high sample count might be faster than complex geometric algorithms 
    # in Python if not carefully optimized. 
    # BUT, the directive asks for a structural change.
    # Let's use a deterministic grid integration for c, and for each c, 
    # compute the 2D union area using a vectorized "rasterization" on a 
    # fine grid, or better, an exact algorithm if possible.
    
    # Actually, let's stick to the MC but make it VERY fast and deterministic 
    # to serve as the loss for a more robust optimizer, OR use the exact 
    # volume calculation if we can make it fast enough.
    
    # Let's try the exact volume calculation using the "sweep line" or 
    # inclusion-exclusion is too hard for 64 sets.
    
    # Alternative: Use a fixed set of points in the BOUNDING BOX of the union?
    # No, the bounding box changes.
    
    # Let's use the property that the volume is the integral of the cross-sectional area.
    # A(c) = Area(Union_k [x_k + c*i_k*w, x_k + c*i_k*w + w] x [y_k + c*j_k*w, y_k + c*j_k*w + w])
    
    # We can compute A(c) exactly for a given c by finding the union of 64 rectangles.
    # Algorithm for union of rectangles:
    # 1. Collect all x-coordinates and y-coordinates of corners.
    # 2. Create a grid of cells defined by these coordinates.
    # 3. Check if each cell is covered by any rectangle.
    # With 64 rectangles, there are at most 128 x-coords and 128 y-coords.
    # Grid size ~ 127x127 = 16k cells.
    # Checking coverage for each cell against 64 rects is 16k * 64 = 1M ops.
    # Doing this for M=100 slices is 100M ops. In Python this is too slow (~10s+).
    
    # We need a faster loss function.
    # Let's go back to MC but optimize the loop.
    # The current program uses 2000 samples. That's very fast.
    # The issue is the optimizer (Nelder-Mead in 128D is bad).
    
    # New Strategy:
    # 1. Use a fast, deterministic, low-noise loss function: 
    #    Stratified MC with a fixed set of points in [0,1]^3 mapped to the current box.
    #    This makes the loss landscape smoother than pure random MC.
    # 2. Use a more suitable optimizer for high dimensions: 
    #    Use `scipy.optimize.dual_annealing` or `differential_evolution` if time permits,
    #    or a custom local search with random restarts.
    #    Given 600s budget, `differential_evolution` might be too slow.
    #    Let's use a custom stochastic local search (Simulated Annealing or Hill Climbing)
    #    which is very fast per iteration.
    
    def fast_volume_loss(params):
        """
        Computes union volume using a fixed set of stratified samples.
        """
        x = params[:64]
        y = params[64:]
        
        # Box bounds
        x_max = x + (i_idx + 1) * w
        y_max = y + (j_idx + 1) * w
        
        X0 = np.min(x)
        X1 = np.max(x_max)
        Y0 = np.min(y)
        Y1 = np.max(y_max)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return 1e10
            
        # Use a pre-computed set of points in [0,1]^3
        # We generate these once outside the function.
        u = fast_volume_loss.rand_u
        v = fast_volume_loss.rand_v
        c = fast_volume_loss.rand_c
        
        a = X0 + u * (X1 - X0)
        b = Y0 + v * (Y1 - Y0)
        
        # Vectorized check
        # shift_x[i, k] = c_i * i_k * w
        # shift_y[i, k] = c_i * j_k * w
        
        # i_idx, j_idx are shape (64,)
        # c is shape (M,)
        
        shift_x = c[:, None] * (i_idx[None, :] * w)
        shift_y = c[:, None] * (j_idx[None, :] * w)
        
        diff_x = a[:, None] - shift_x - x[None, :]
        diff_y = b[:, None] - shift_y - y[None, :]
        
        valid_x = (diff_x >= 0) & (diff_x <= w)
        valid_y = (diff_y >= 0) & (diff_y <= w)
        
        covered = np.any(valid_x & valid_y, axis=1)
        fraction = np.mean(covered)
        
        return area * fraction

    # Initialize the random points for the loss function
    # Use a large number of points for accuracy, but not too large for speed.
    # 5000 points * 64 tubes = 320k operations. Very fast.
    rng_init = np.random.default_rng(12345)
    n_samples = 5000
    fast_volume_loss.rand_u = rng_init.uniform(0, 1, n_samples)
    fast_volume_loss.rand_v = rng_init.uniform(0, 1, n_samples)
    fast_volume_loss.rand_c = rng_init.uniform(0, 1, n_samples)

    # Initial guess: Keich construction (separable)
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(bin(idx)[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    keich = keich - keich.min()
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    x_init = keich[i_flat]
    y_init = keich[j_flat]
    
    # Add a small random perturbation to break symmetry and allow non-separable solutions
    rng_perturb = np.random.default_rng(42)
    x_init += rng_perturb.normal(0, 0.05, 64)
    y_init += rng_perturb.normal(0, 0.05, 64)
    
    x0 = np.concatenate([x_init, y_init])
    
    best_params = x0.copy()
    best_loss = fast_volume_loss(x0)
    
    # Local Search: Stochastic Hill Climbing with Restart
    # We will perform many small random steps, accepting improvements.
    # To escape local minima, we occasionally take larger steps or restart.
    
    rng_search = np.random.default_rng(999)
    
    # Parameters for search
    step_size = 0.1
    max_iter = 50000 # Upper bound, but we stop by time
    
    for _ in range(max_iter):
        if elapsed() > DEADLINE:
            break
            
        # Generate a random perturbation
        # Gaussian noise
        noise = rng_search.normal(0, step_size, 128)
        new_params = best_params + noise
        
        new_loss = fast_volume_loss(new_params)
        
        if new_loss < best_loss:
            best_params = new_params
            best_loss = new_loss
            # Reduce step size slightly on success to refine
            step_size *= 0.99
        else:
            # Increase step size slightly to explore
            step_size *= 1.01
            # Cap step size
            if step_size > 0.5:
                step_size = 0.5
            if step_size < 0.001:
                step_size = 0.001
                
        # Occasional large jump to escape local minima
        if _ % 1000 == 0 and _ > 0:
            noise_large = rng_search.normal(0, 0.5, 128)
            new_params = best_params + noise_large
            new_loss = fast_volume_loss(new_params)
            if new_loss < best_loss:
                best_params = new_params
                best_loss = new_loss
                step_size = 0.1 # Reset step size

    # Final refinement with a smaller step size if time permits
    if elapsed() < DEADLINE - 50:
        step_size = 0.01
        for _ in range(10000):
            if elapsed() > DEADLINE - 50:
                break
            noise = rng_search.normal(0, step_size, 128)
            new_params = best_params + noise
            new_loss = fast_volume_loss(new_params)
            if new_loss < best_loss:
                best_params = new_params
                best_loss = new_loss
                step_size *= 0.95
            else:
                step_size *= 1.05
                if step_size > 0.05:
                    step_size = 0.05

    x_final = best_params[:64]
    y_final = best_params[64:]
    
    positions = np.stack([x_final, y_final], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
