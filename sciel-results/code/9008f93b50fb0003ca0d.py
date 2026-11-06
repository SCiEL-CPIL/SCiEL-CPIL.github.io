"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 unit cell using a Fourier-target constructive approach.
    We synthesize the dielectric profile to maximize specific low-order 
    Fourier components of 1/epsilon (m=2, n=0; m=0, n=2; m=2, n=2) which 
    are critical for opening the 4-5 TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill

    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)

    # 1. Construct a candidate cell in real space based on Fourier targets.
    # The gap between TM bands 4 and 5 is strongly influenced by the 
    # (2,0), (0,2), and (2,2) Fourier components of 1/eps.
    # We want to maximize the magnitude of these components while maintaining 
    # the required symmetry.
    
    # We will use a parametric family of shapes that can be tuned to 
    # optimize these coefficients. A "rounded cross" or "superellipse" 
    # with adjustable exponents and aspect ratios is a good candidate.
    # Alternatively, we can directly optimize a few parameters of a 
    # superellipse: |x/a|^n + |y/b|^n <= 1.
    
    # Let's try a superellipse centered at (0,0) with parameters a, b, n.
    # Due to symmetry, we only need to define the first quadrant.
    
    best_score = -np.inf
    best_cell = None
    
    # Precompute coordinate grid
    coords = np.arange(size) - (size - 1) / 2.0
    X, Y = np.meshgrid(coords, coords)
    # Normalize to [-0.5, 0.5]
    X = X / size
    Y = Y / size
    
    # We want to search over a, b, and n (superellipse exponent).
    # a, b in [0.2, 0.5], n in [1, 10].
    # Also consider the case of 4 separate rods (which is a limit of 
    # disconnected superellipses, but let's stick to one connected shape 
    # for now, or a 4-rod setup which is effectively 4 superellipses).
    
    # Actually, the 4-rod setup was the baseline. Let's try a single 
    # connected "fat" shape that has strong 2nd order harmonics.
    # A shape with high curvature at the diagonals and flat at the axes 
    # might be good for (2,2) vs (2,0).
    
    # Let's define a function to compute the score (approximate gap proxy) 
    # or just return a well-tuned static shape if time is tight, 
    # but we have 200s, so we can do a small optimization.
    
    # Strategy:
    # 1. Generate a set of candidate superellipse parameters.
    # 2. For each, create the pixel map.
    # 3. Enforce symmetry.
    # 4. (Optional) Do a quick local search on the best few.
    
    # Since we don't have the band solver here, we need a proxy or 
    # rely on the known good parameters. 
    # However, the task says "evaluator solves...". We are in the design 
    # phase. We should produce the best static cell we can reason about 
    # or use a fast heuristic.
    
    # Known good structure for 4-5 TM gap: often involves a shape that 
    # is elongated along the axes or has specific hole patterns.
    # The 4-rod design (score 5.78) is decent. 
    # Let's try to improve it by varying the rod radius and position, 
    # or by using a single connected "plus" shape.
    
    # Let's try a "plus" shape (cross) which has strong (2,0) and (0,2) 
    # components.
    # Width of the arms and length of the arms.
    
    # Candidate 1: 4 rods (baseline)
    # Candidate 2: Plus shape
    # Candidate 3: Diamond shape
    
    # We will implement a quick local search on a parameterized family 
    # using a simplified metric if possible, or just pick the best 
    # from a coarse grid based on physics intuition.
    
    # Physics intuition: To maximize the gap between band 4 and 5, 
    # we need strong coupling between states at X and M points.
    # This is often achieved by shapes that break the rotational symmetry 
    # in a specific way or by optimizing the fill fraction and shape.
    
    # Let's try optimizing the 4-rod design parameters: 
    # center offset (cx, cy) and radius (r).
    # Symmetry requires centers at (±cx, ±cy).
    
    # We'll do a coarse grid search over cx, cy, r.
    # cx, cy in [0.1, 0.4], r in [0.05, 0.25].
    
    # But wait, the evaluator uses a specific band solver. Without it, 
    # we can't score. 
    # The prompt implies we should use the available time to *find* 
    # a better cell. If we can't run the solver, we must rely on 
    # pre-computed knowledge or a very fast approximation.
    
    # However, often in these tasks, if a solver isn't provided in the 
    # sandbox, we assume we can't run it. But the "inspiration" 
    # mentions "Fourier-target constructive design". This suggests 
    # we can compute Fourier coefficients easily and optimize those 
    # as a proxy.
    
    # Let's compute the Fourier coefficients of 1/eps for a given cell.
    # 1/eps is 1 for air, 1/12 for dielectric.
    # We want to maximize the magnitude of G_{2,0}, G_{0,2}, G_{2,2}.
    
    def compute_fourier_proxy(cell):
        """
        Compute a proxy score based on Fourier coefficients of 1/eps.
        """
        cell_arr = np.array(cell, dtype=float)
        # 1/eps map
        inv_eps = np.where(cell_arr == 1, 1.0/eps_high, 1.0)
        
        # Compute DFT (using numpy fft2)
        # The Fourier coefficients are related to the DFT.
        # G_{m,n} = (1/N^2) sum_{x,y} inv_eps(x,y) exp(-i 2pi (mx+ny)/N)
        # numpy.fft.fft2 gives sum exp(-i 2pi (kx x + ky y)/N)
        # So G_{m,n} is proportional to fft2(inv_eps)[m, n]
        
        fft_coeffs = np.fft.fft2(inv_eps)
        
        # We care about (2,0), (0,2), (2,2) and their symmetric counterparts.
        # Note: fft2 indices are 0..N-1. 
        # Index for (2,0) is [2, 0] or [N-2, 0] (for negative freq).
        # Since inv_eps is real and symmetric, the coefficients are real 
        # and even.
        
        # Let's sum the magnitudes of the relevant low-order modes.
        # Modes: (±1,0), (0,±1) are usually small for centered shapes.
        # (±2,0), (0,±2), (±2,±2), (±1,±1) etc.
        
        # For the 4-5 TM gap, (2,0) and (2,2) are critical.
        
        # Get magnitudes
        mag = np.abs(fft_coeffs)
        
        # Extract specific modes
        # (2,0) -> [2,0] and [N-2,0]
        # (0,2) -> [0,2] and [0,N-2]
        # (2,2) -> [2,2], [2,N-2], [N-2,2], [N-2,N-2]
        
        val_20 = mag[2, 0] + mag[N-2, 0]
        val_02 = mag[0, 2] + mag[0, N-2]
        val_22 = mag[2, 2] + mag[2, N-2] + mag[N-2, 2] + mag[N-2, N-2]
        
        # We want to maximize a combination.
        # Let's try maximizing val_22 + val_20 + val_02
        return val_22 + val_20 + val_02

    # Now, let's search for a shape that maximizes this proxy.
    # We'll use a parameterized 4-rod shape.
    
    def make_4_rod_cell(cx, cy, r):
        """
        Create a 32x32 cell with 4 rods at (±cx, ±cy) with radius r.
        """
        cell = [[0]*size for _ in range(size)]
        # Pixel centers
        xs = np.arange(size) - (size - 1)/2.0
        ys = np.arange(size) - (size - 1)/2.0
        X, Y = np.meshgrid(xs, ys)
        
        # Normalized coordinates
        Xn = X / size
        Yn = Y / size
        
        # Rod centers
        centers = [(-cx, -cy), (cx, -cy), (-cx, cy), (cx, cy)]
        
        mask = np.zeros((size, size), dtype=bool)
        for ccx, ccy in centers:
            dist_sq = (Xn - ccx)**2 + (Yn - ccy)**2
            mask |= (dist_sq < r**2)
        
        # Convert to list of lists
        for i in range(size):
            for j in range(size):
                if mask[i, j]:
                    cell[i][j] = 1
                else:
                    cell[i][j] = 0
        
        # Enforce symmetry (should be exact, but just in case)
        # The generation is symmetric.
        return cell

    # Grid search over cx, cy, r
    # cx, cy: 0.1 to 0.4 step 0.02
    # r: 0.05 to 0.25 step 0.01
    
    best_proxy = -np.inf
    best_params = (0.25, 0.25, 0.15) # Default to baseline
    
    # To save time, we'll do a coarse grid first.
    cx_vals = np.arange(0.10, 0.41, 0.02)
    cy_vals = np.arange(0.10, 0.41, 0.02)
    r_vals = np.arange(0.05, 0.26, 0.02)
    
    for cx in cx_vals:
        for cy in cy_vals:
            for r in r_vals:
                # Check time
                if time.time() > deadline - 10:
                    break
                cell = make_4_rod_cell(cx, cy, r)
                score = compute_fourier_proxy(cell)
                if score > best_proxy:
                    best_proxy = score
                    best_params = (cx, cy, r)
        
        if time.time() > deadline - 10:
            break
    if time.time() > deadline - 10:
        break

    # Now refine around the best parameters found
    cx0, cy0, r0 = best_params
    
    # Local search
    for _ in range(20): # 20 iterations
        if time.time() > deadline - 5:
            break
        # Random perturbation
        cx = cx0 + np.random.uniform(-0.02, 0.02)
        cy = cy0 + np.random.uniform(-0.02, 0.02)
        r = r0 + np.random.uniform(-0.02, 0.02)
        
        # Clip to valid range
        cx = np.clip(cx, 0.05, 0.45)
        cy = np.clip(cy, 0.05, 0.45)
        r = np.clip(r, 0.02, 0.30)
        
        cell = make_4_rod_cell(cx, cy, r)
        score = compute_fourier_proxy(cell)
        if score > best_proxy:
            best_proxy = score
            best_params = (cx, cy, r)
            cx0, cy0, r0 = cx, cy, r

    # Finalize cell with best params
    final_cell = make_4_rod_cell(best_params[0], best_params[1], best_params[2])
    
    # Double check symmetry
    # The function make_4_rod_cell ensures symmetry.
    
    return final_cell

# EVOLVE-BLOCK-END
