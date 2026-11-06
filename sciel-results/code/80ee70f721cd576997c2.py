import numpy as np
from scipy.optimize import minimize
import time

def design(problem):
    size = problem["pixels"] # 32
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # We model the cell as a 4x4 array of 8x8 sub-cells.
    # Each sub-cell contains a centered square (or circle) of dielectric.
    # Symmetry constraints on the 4x4 grid of sub-cell radii (r_ij):
    # 1. 180 degree rotation: r_ij == r_(3-i)(3-j)
    # 2. Transposition: r_ij == r_ji
    # Independent parameters:
    # r00, r01, r02
    # r11, r12
    # r22
    # Total 6 parameters.
    
    # Map parameters to the 4x4 grid
    def get_grid_from_params(params):
        r00, r01, r02, r11, r12, r22 = params
        # Create 4x4 array of radii
        # Indices 0..3
        radii = np.zeros((4, 4))
        radii[0, 0] = r00
        radii[0, 1] = r01
        radii[0, 2] = r02
        radii[0, 3] = r02 # 3-0=3, 3-2=1 -> r02? Wait.
        # Let's re-verify symmetry: r_ij = r_(3-i)(3-j)
        # r_03 = r_(3-0)(3-3) = r_30.
        # r_30 = r_(3-3)(3-0) = r_03.
        # Also r_ij = r_ji.
        # So r_03 = r_30.
        # From r_03 = r_(3-0)(3-3) is wrong index mapping?
        # Rotation 180: (i,j) -> (3-i, 3-j).
        # r_00 -> r_33. So r00 == r33.
        # r_01 -> r_32. So r01 == r32.
        # r_02 -> r_31. So r02 == r31.
        # r_03 -> r_30. So r03 == r30.
        # r_10 -> r_23. So r10 == r23.
        # r_11 -> r_22. So r11 == r22.
        # r_12 -> r_21. So r12 == r21.
        # r_13 -> r_20. So r13 == r20.
        # r_20 -> r_13.
        # r_21 -> r_12.
        # r_22 -> r_11.
        # r_23 -> r_10.
        # r_30 -> r_03.
        # r_31 -> r_02.
        # r_32 -> r_01.
        # r_33 -> r_00.
        
        # Transposition: r_ij = r_ji
        # r_01 = r_10
        # r_02 = r_20
        # r_03 = r_30
        # r_12 = r_21
        # r_13 = r_31
        # r_23 = r_32
        
        # Combining:
        # r00 = r33
        # r01 = r10 = r32 = r23
        # r02 = r20 = r31 = r13
        # r03 = r30
        # r11 = r22
        # r12 = r21
        
        # Independent:
        # r00
        # r01 (implies r10, r32, r23)
        # r02 (implies r20, r31, r13)
        # r03 (implies r30)
        # r11 (implies r22)
        # r12 (implies r21)
        
        # So we have 6 parameters:
        # p0 = r00
        # p1 = r01
        # p2 = r02
        # p3 = r03
        # p4 = r11
        # p5 = r12
        
        radii[0, 0] = p0
        radii[0, 1] = p1
        radii[0, 2] = p2
        radii[0, 3] = p3
        
        radii[1, 0] = p1
        radii[1, 1] = p4
        radii[1, 2] = p5
        radii[1, 3] = p2
        
        radii[2, 0] = p2
        radii[2, 1] = p5
        radii[2, 2] = p4
        radii[2, 3] = p1
        
        radii[3, 0] = p3
        radii[3, 1] = p2
        radii[3, 2] = p1
        radii[3, 3] = p0
        
        return radii

    def build_cell(params):
        radii = get_grid_from_params(params)
        cell = np.zeros((size, size), dtype=int)
        sub_size = size // 4 # 8
        
        # Precompute local coordinates for an 8x8 sub-cell
        # Center is at 3.5, 3.5
        yy, xx = np.mgrid[0:sub_size, 0:sub_size]
        # Distance from center (3.5, 3.5)
        # We use a square shape for the dielectric in each sub-cell to allow higher harmonics
        # Actually, a circle is smoother. Let's stick to circles for robustness, 
        # but the "checker" hint suggests squares might be better for higher order gaps?
        # Let's use squares (Manhattan distance or Chebyshev distance) to create sharper features.
        # Chebyshev distance: max(|x-cx|, |y-cy|) < radius
        # This creates a square dielectric blob.
        
        for i in range(4):
            for j in range(4):
                r = radii[i, j]
                # r is in pixels, max 4.0 (half of 8)
                # Create mask for this sub-cell
                # Local coords 0..7
                # Center 3.5
                dist = np.maximum(np.abs(xx - 3.5), np.abs(yy - 3.5))
                mask = dist < r
                # Place in global cell
                start_i = i * sub_size
                start_j = j * sub_size
                cell[start_i:start_i+sub_size, start_j:start_j+sub_size] = mask.astype(int)
        
        return cell

    def surrogate_score(params):
        cell = build_cell(params)
        # Compute 1/epsilon
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        
        # Compute 2D DFT
        # We care about components (2,0) and (0,2) and maybe (1,0), (0,1)
        # The grid is 32x32.
        # FFT indices: 0..31.
        # Component (n,m) corresponds to index (n,m) if n,m < 16, or (32-n, 32-m) if > 16.
        # We want low frequency components.
        fft_inv_eps = np.fft.fft2(inv_eps)
        
        # Extract (1,0), (0,1), (2,0), (0,2)
        # Note: DC is (0,0).
        # (1,0) is at [1, 0]
        # (0,1) is at [0, 1]
        # (2,0) is at [2, 0]
        # (0,2) is at [0, 2]
        
        f_10 = np.abs(fft_inv_eps[1, 0])
        f_01 = np.abs(fft_inv_eps[0, 1])
        f_20 = np.abs(fft_inv_eps[2, 0])
        f_02 = np.abs(fft_inv_eps[0, 2])
        
        # Heuristic: The 4-5 gap is often driven by (2,0) and (0,2) interactions.
        # However, if (1,0) is too strong, it might open the 1-2 or 2-3 gap and close 4-5?
        # Let's maximize a combination.
        # A common metric for gap opening is the magnitude of the Fourier component
        # corresponding to the reciprocal vector connecting the degenerate states.
        # For the 4-5 gap in square lattice, the critical points are often M (pi, 0) or X (pi, pi) or G (pi, 0) etc.
        # Let's try maximizing f_20 + f_02.
        score = f_20 + f_02
        return -score # Minimize negative score

    # Optimization
    # Initial guess: uniform radii, maybe slightly varied
    # r in [0.5, 3.5] pixels
    x0 = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    bounds = [(0.5, 3.5) for _ in range(6)]
    
    # Use Nelder-Mead
    # Set a time limit for optimization
    start_time = time.time()
    time_limit = 100.0 # 100 seconds for optimization
    
    best_params = x0
    best_score = surrogate_score(x0)
    
    # Simple loop with Nelder-Mead restarts or just one long run?
    # Nelder-Mead can be slow. Let's try a few random starts and short optimizations.
    
    # Actually, let's just do a grid search on a coarse grid first?
    # 6 dimensions is too high for full grid.
    # Let's try a few random starts and run minimize with a time check.
    
    np.random.seed(42)
    
    for i in range(20):
        if time.time() - start_time > time_limit - 10:
            break
            
        if i == 0:
            current_params = x0
        else:
            current_params = np.random.uniform(0.5, 3.5, 6)
            
        try:
            # Use a simple local search
            # scipy.optimize.minimize with method='Nelder-Mead'
            # We need to pass the time limit via a flag or just trust it's fast enough
            # surrogate_score is fast (FFT of 32x32)
            
            # To avoid getting stuck, we can limit the number of iterations
            res = minimize(surrogate_score, current_params, method='Nelder-Mead', 
                           options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 1000, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Final check on time
    if time.time() - start_time > time_limit - 5:
        # If we are running out of time, just use the best so far
        pass

    # Construct final cell
    final_cell = build_cell(best_params)
    
    # Convert to list of lists
    return final_cell.tolist()

