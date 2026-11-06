"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    
    Strategy:
    1. Use a circular rod representation (like the seed/inspiration 1) which is known to work well.
    2. Use a local search / hill climbing approach with a proxy metric based on the 
       Fourier spectrum of 1/epsilon (since TM bands depend on 1/epsilon).
    3. The proxy metric will be the magnitude of the Fourier components at specific 
       reciprocal lattice vectors that couple bands 4 and 5.
    4. We will start from the seed design (radius 0.2) and try to improve it by 
       varying the radius and potentially adding perturbations, but since we need 
       strict D4 symmetry and a simple representation, we'll stick to varying 
       the radius of the central circular rod and maybe the fill fraction.
       
    Actually, the ring-based approach in the current program is too restrictive 
    (square rings). The circular rod is a better structural family.
    Let's refine the circular rod radius to maximize a proxy for the 4-5 gap.
    
    The 4-5 gap is a higher-order gap. It is sensitive to the Fourier components 
    of the dielectric function at specific wavevectors.
    For a square lattice, the first few reciprocal lattice vectors are G = (m, n) * 2*pi/a.
    The coupling between bands k and k+G is proportional to epsilon_G.
    For TM, it's (1/epsilon)_G.
    
    Let's compute the Fourier coefficients of 1/epsilon for a circular rod and 
    try to maximize the magnitude of the coefficients that are relevant for the 4-5 gap.
    However, identifying the exact G vectors is tricky without knowing the band structure.
    
    Alternative: Use a simple local search on the radius parameter.
    Since we can't run the band solver, we have to rely on a proxy or just 
    return a very good fixed design.
    
    The seed (radius 0.2) scored 1.0.
    Let's try to find a radius that might be slightly better.
    Often, the optimal radius for the first gap is around 0.2-0.3.
    For higher gaps, the structure might need to be more complex, but a simple 
    rod is a good baseline.
    
    Let's implement a local search over the radius r in [0.15, 0.35] with step 0.01.
    We will use a proxy: the ratio of the power in the first few non-zero Fourier 
    components of 1/epsilon to the total power. A higher contrast in the relevant 
    Fourier components might help.
    
    Actually, let's just use the radius that is known to be good for the first gap 
    and assume it's decent for higher gaps too, or try a few specific radii.
    
    Let's try:
    1. r = 0.20 (seed)
    2. r = 0.22
    3. r = 0.25
    4. r = 0.28
    
    And pick the one with the "best" proxy.
    
    Proxy: Sum of |FFT[1/eps][m,n]|^2 for m^2+n^2 in [1, 4, 5, 8, 9, 10, 13, 16, 17, 18] 
    (small G vectors).
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def generate_cell(radius_frac):
        """Generate a 32x32 cell with a central circular rod of given radius fraction."""
        cell = [[0] * size for _ in range(size)]
        r_px = radius_frac * size
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - cx
                y = (i + 0.5) - cy
                if x * x + y * y < r_px * r_px:
                    cell[i][j] = 1
        return cell
    
    def compute_proxy(cell):
        """
        Compute a proxy metric based on Fourier components of 1/epsilon.
        We focus on low-frequency components that are likely to couple the bands.
        """
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        
        # Compute 2D FFT
        fft_inv_eps = np.fft.fft2(inv_eps)
        
        # We are interested in components with small m^2 + n^2.
        # The BZ is [-pi, pi]^2.
        # DFT frequencies: m/size * 2*pi.
        # Let's sum the power for m^2+n^2 <= 10 (excluding DC).
        
        total_power = 0.0
        # Use fftshift to center the DC component
        fft_shifted = np.fft.fftshift(fft_inv_eps)
        
        for m in range(-size//2, size//2 + 1):
            for n in range(-size//2, size//2 + 1):
                if m == 0 and n == 0:
                    continue
                if m*m + n*n > 10:  # Limit to low frequencies
                    continue
                idx_m = m + size//2
                idx_n = n + size//2
                total_power += np.abs(fft_shifted[idx_m, idx_n])**2
        
        return total_power
    
    # Candidate radii
    radii = [0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]
    
    best_proxy = -1
    best_radius = 0.20  # Default to seed
    
    for r in radii:
        if time.time() - start_time > deadline:
            break
        cell = generate_cell(r)
        proxy = compute_proxy(cell)
        if proxy > best_proxy:
            best_proxy = proxy
            best_radius = r
    
    # Generate the final cell
    final_cell = generate_cell(best_radius)
    
    # Enforce D4 symmetry strictly (just in case, though circular rod is already symmetric)
    for i in range(size):
        for j in range(size):
            vals = [
                final_cell[i][j],
                final_cell[size-1-i][j],
                final_cell[i][size-1-j],
                final_cell[size-1-i][size-1-j],
                final_cell[j][i],
                final_cell[size-1-j][size-1-i],
                final_cell[j][size-1-i],
                final_cell[size-1-j][i],
            ]
            avg = sum(vals) / len(vals)
            final_cell[i][j] = 1 if avg >= 0.5 else 0
    
    return final_cell

# EVOLVE-BLOCK-END
