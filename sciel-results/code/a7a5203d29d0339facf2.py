"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    
    Strategy: Use a ring-based representation with D4 symmetry.
    The cell is divided into 16 concentric square rings (from center outward).
    Each ring is either dielectric (1) or air (0).
    We search over binary patterns of these rings.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (or 230s kill)
    
    # The cell has D4 symmetry: invariant under 90-degree rotation and reflection.
    # We can represent the cell by the values in one 8x8 quadrant (including center line).
    # Actually, with 32x32, the center is between pixels. 
    # D4 symmetry about the center of the cell.
    # Center is at (15.5, 15.5) in pixel coordinates (0-indexed).
    # Symmetry operations:
    # 1. Reflection about x=15.5: cell[i][j] == cell[i][31-j]
    # 2. Reflection about y=15.5: cell[i][j] == cell[31-i][j]
    # 3. Rotation 90 deg: cell[i][j] == cell[j][31-i]
    
    # This implies the cell is determined by the top-left 16x16 quadrant?
    # No, let's think about the fundamental domain.
    # The fundamental domain for D4 symmetry on a 32x32 grid (centered) is 8x8 pixels.
    # Specifically, pixels (i, j) for i in 0..15, j in 0..15 determine the rest?
    # Let's verify.
    # If we know cell[i][j] for i in [0,15], j in [0,15]:
    # cell[i][31-j] is determined by reflection x.
    # cell[31-i][j] is determined by reflection y.
    # cell[31-i][31-j] is determined by both.
    # What about rotation?
    # cell[j][31-i] must equal cell[i][j].
    # This imposes constraints on the 16x16 block.
    # For i, j in [0,15]:
    # If we pick i, j such that i <= 15 and j <= 15, the rotation maps (i,j) to (j, 31-i).
    # 31-i is in [16, 31]. So the rotated pixel is in the bottom-right or bottom-left quadrant.
    # This means the top-left 16x16 is NOT independent.
    
    # Let's define the fundamental domain more carefully.
    # The center is (15.5, 15.5).
    # Distance from center determines the "ring".
    # Max distance is sqrt(15.5^2 + 15.5^2) ~ 22.
    # Min distance is 0.5.
    
    # Alternative approach:
    # Just generate random D4 symmetric cells and evaluate a fast proxy.
    # But we don't have a fast proxy for the band gap.
    
    # Let's try the ring idea again.
    # Define rings by max(|i-15.5|, |j-15.5|).
    # i, j are integer pixel indices 0..31.
    # i-15.5 is half-integer: ..., -16.5, -15.5, ..., -0.5, 0.5, ..., 15.5, 16.5
    # |i-15.5| takes values 0.5, 1.5, ..., 15.5, 16.5.
    # So there are 16 distinct values for |i-15.5|.
    # Let k = int(|i-15.5| + 0.5) - 1?
    # If i=15, |15-15.5|=0.5. k=0.
    # If i=14, |14-15.5|=1.5. k=1.
    # ...
    # If i=0, |0-15.5|=15.5. k=15.
    # If i=31, |31-15.5|=15.5. k=15.
    # So k ranges from 0 to 15.
    # The "Chebyshev radius" R = max(|i-15.5|, |j-15.5|).
    # R takes values 0.5, 1.5, ..., 15.5.
    # Let ring index r = int(R - 0.5) = int(max(|i-15.5|, |j-15.5|) - 0.5).
    # r ranges from 0 to 15.
    
    # If we assign a value to each ring r, does that guarantee D4 symmetry?
    # Yes, because the ring definition is symmetric under D4.
    # However, this restricts us to "square rings" which are axis-aligned.
    # This is a very restricted class of structures.
    # But it's a good starting point for a structural change.
    
    # Let's also consider circular rings?
    # Distance d = sqrt((i-15.5)^2 + (j-15.5)^2).
    # This breaks the strict pixel grid alignment for symmetry?
    # No, the distance function is D4 symmetric.
    # But assigning rings by distance might be more complex to implement efficiently.
    # Let's stick to Chebyshev distance (square rings) first.
    
    # We will search over binary patterns of 16 bits.
    # 2^16 = 65536 possibilities.
    # We can't evaluate the band gap for all of them (too slow).
    # We need a heuristic or a small subset.
    
    # Heuristic:
    # For TM bands, dielectric rods in air are common.
    # So we expect the center to be dielectric (1) and outer rings to be air (0).
    # Or vice versa?
    # The seed program (score 1.0) was a single central rod.
    # That corresponds to ring 0=1, rings 1-15=0.
    
    # Let's try to find a pattern that might open a higher gap.
    # Perhaps alternating rings?
    # Or a specific thickness?
    
    # Since we can't compute the gap, let's use a proxy:
    # The band gap is related to the Fourier components of the dielectric function.
    # A stronger contrast and specific spatial frequencies can open gaps.
    
    # Let's try a few specific patterns:
    # 1. Central rod (seed): 1000000000000000
    # 2. Central rod + 1 ring: 1100000000000000
    # 3. Central rod + 2 rings: 1110000000000000
    # 4. Alternating: 1010101010101010
    # 5. Inverted: 0100000000000000 (hole in center)
    
    # Actually, the seed score is 1.0. We need to beat 1.0.
    # The seed is a single rod.
    # Maybe a larger rod? Or a ring?
    
    # Let's implement the ring-based cell generator.
    
    def generate_cell_from_rings(rings):
        """
        rings: list of 16 ints (0 or 1), where rings[r] is the value for ring r.
        Returns 32x32 cell.
        """
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                r = int(max(abs(i - 15.5), abs(j - 15.5)) - 0.5)
                cell[i][j] = rings[r]
        return cell
    
    # We will test a set of candidate patterns.
    candidates = []
    
    # Pattern 1: Seed (single rod)
    candidates.append([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
    # Pattern 2: Slightly larger rod
    candidates.append([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    candidates.append([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
    # Pattern 3: Ring (hole in center)
    candidates.append([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    candidates.append([0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
    # Pattern 4: Alternating
    candidates.append([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
    candidates.append([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
    
    # Pattern 5: Random search
    # We'll do a random search if time permits.
    
    # But wait, the seed score is 1.0.
    # The current program (score 0.26) is worse.
    # We need to be at least as good as the seed, ideally better.
    
    # Let's think about physics.
    # For a square lattice of dielectric rods in air, the first TM gap is around k=0.3-0.4.
    # The 4-5 gap is higher.
    # To maximize the gap, we might need to tune the radius.
    # The seed used radius 0.2 (in lattice units).
    # In pixels, 0.2 * 32 = 6.4 pixels radius.
    # Diameter ~ 12.8 pixels.
    # Center at 15.5.
    # So it covers pixels 15.5-6.4=9.1 to 15.5+6.4=21.9.
    # So pixels 10..21 roughly.
    # That's 12x12 pixels.
    # In our ring representation:
    # Ring 0: 1 pixel (center? No, center is between pixels).
    # Let's check the ring mapping again.
    # i=15, j=15: r=0.
    # i=15, j=16: r=0.
    # i=16, j=15: r=0.
    # i=16, j=16: r=0.
    # So ring 0 is a 2x2 block at the center.
    # Ring 1 is the next layer of pixels.
    # A rod of radius 6.4 would cover rings 0, 1, 2, 3?
    # Max dist for ring 3 is 3.5.
    # Max dist for ring 4 is 4.5.
    # 6.4 is between 6.5 (ring 6) and 5.5 (ring 5).
    # So a rod of radius 6.4 covers rings 0-5 partially?
    # Actually, the ring boundary is at R=0.5, 1.5, etc.
    # A circle of radius 6.4 includes all pixels with R <= 6.4.
    # R=6.5 is ring 6.
    # So it includes rings 0, 1, 2, 3, 4, 5.
    # So the seed pattern is approximately [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0].
    
    # Let's add this to candidates.
    candidates.append([1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
    candidates.append([1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
    candidates.append([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0])
    
    # Let's also try some "checkerboard-like" patterns that might have high symmetry.
    # But checkerboard is not D4 symmetric in this ring sense?
    # Actually, a checkerboard pattern is D4 symmetric.
    # But it's not ring-based.
    
    # Let's implement a fast evaluation of a proxy metric.
    # We can compute the Fourier transform of the dielectric function.
    # The band structure depends on the Fourier coefficients of 1/epsilon (for TM) or epsilon (for TE).
    # For TM, the matrix element involves 1/epsilon.
    # So we should look at the Fourier components of 1/epsilon.
    # epsilon is 1 or 12.
    # 1/epsilon is 1 or 1/12.
    
    # Let's compute the magnitude of the first few Fourier coefficients of 1/epsilon.
    # A strong contrast and specific spatial frequencies might correlate with a large gap.
    
    def compute_proxy(cell):
        """
        Compute a proxy metric based on Fourier components of 1/epsilon.
        """
        import numpy as np
        eps = np.array([[eps_high if cell[i][j] == 1 else eps_low for j in range(size)] for i in range(size)])
        inv_eps = 1.0 / eps
        
        # Compute 2D FFT
        fft_inv_eps = np.fft.fft2(inv_eps)
        
        # We are interested in the low-frequency components (small k).
        # The Brillouin zone boundary is at pi/a.
        # In pixel units, the BZ boundary is at 0.5 cycles/pixel?
        # No, the lattice constant is 32 pixels.
        # The BZ boundary is at 1/(2a) = 1/64 cycles/pixel?
        # No, k_max = pi/a.
        # In discrete Fourier transform, the frequency index m corresponds to k = 2*pi*m/N.
        # m=1 corresponds to k = 2*pi/32 = pi/16.
        # The BZ boundary is at k = pi/a = pi/32? No, a=1 (in lattice units).
        # The unit cell is 1x1.
        # The BZ is [-pi, pi] x [-pi, pi].
        # The pixel grid is 32x32.
        # So the DFT frequencies are m/32 * 2*pi.
        # m=1 -> 2*pi/32 = pi/16.
        # The BZ boundary is pi.
        # So m goes up to 16.
        
        # Let's sum the magnitudes of the first few non-zero Fourier components.
        # Exclude the DC component (m=0, n=0).
        # Focus on components with m^2 + n^2 <= some threshold.
        
        total_power = 0
        for m in range(-16, 17):
            for n in range(-16, 17):
                if m == 0 and n == 0:
                    continue
                if m*m + n*n > 25: # Limit to low frequencies
                    continue
                idx_m = m % 32
                idx_n = n % 32
                total_power += np.abs(fft_inv_eps[idx_m, idx_n])**2
        
        return total_power
    
    # Let's evaluate the candidates with this proxy.
    # But wait, the proxy might not correlate well with the 4-5 gap.
    # The 4-5 gap is a higher-order gap.
    # It might depend on higher Fourier components.
    
    # Let's just pick the candidate that is closest to the seed but slightly modified.
    # Or we can do a random search.
    
    # Let's do a random search over the 16-bit patterns.
    # We'll pick 100 random patterns and evaluate the proxy.
    # Then pick the best one.
    
    best_proxy = -1
    best_pattern = candidates[0]
    
    # Evaluate seed
    proxy_seed = compute_proxy(generate_cell_from_rings(candidates[0]))
    best_proxy = proxy_seed
    
    # Evaluate other fixed candidates
    for pattern in candidates[1:]:
        if time.time() - start_time > deadline:
            break
        cell = generate_cell_from_rings(pattern)
        proxy = compute_proxy(cell)
        if proxy > best_proxy:
            best_proxy = proxy
            best_pattern = pattern
    
    # Random search
    num_random = 50
    for _ in range(num_random):
        if time.time() - start_time > deadline:
            break
        pattern = [random.randint(0, 1) for _ in range(16)]
        cell = generate_cell_from_rings(pattern)
        proxy = compute_proxy(cell)
        if proxy > best_proxy:
            best_proxy = proxy
            best_pattern = pattern
    
    # Generate the final cell from the best pattern
    final_cell = generate_cell_from_rings(best_pattern)
    
    # Enforce D4 symmetry strictly (just in case)
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
