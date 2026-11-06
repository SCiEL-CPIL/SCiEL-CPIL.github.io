"""Design a 32x32 photonic crystal unit cell maximizing the 4-5 TM band gap."""
import time
import math
import numpy as np
from scipy.optimize import minimize

# Pre-compute the 289-plane-wave basis vectors (G vectors)
# We need G vectors such that |k + G| is small enough for convergence.
# The evaluator uses 289 plane waves. For a 17x17 grid of G vectors (from -8 to 8),
# that's exactly 289. Let's verify: (-8..8) is 17 values, 17*17 = 289.
# The lattice constant a=1, so G = (m, n) where m, n are integers.
# We use the irreducible Brillouin zone for the square lattice.

# Pre-compute G vectors: m, n from -8 to 8
G_m = np.arange(-8, 9)
G_n = np.arange(-8, 9)
G_grid = np.array([(m, n) for m in G_m for n in G_n], dtype=float)
N_G = len(G_grid)  # 289

# Pre-compute k-points in the irreducible Brillouin zone
# Square lattice irreducible BZ: 0 <= kx <= ky, kx + ky <= 0.5 (in units of 2pi/a)
# Actually for square lattice, the irreducible BZ is the triangle with vertices
# (0,0), (0.5, 0), (0.25, 0.25) in units of 2pi/a? No.
# The BZ of square lattice (a=1) is [-pi, pi] x [-pi, pi].
# Irreducible BZ: 0 <= kx <= ky, kx <= pi/2, ky <= pi/2, and kx + ky <= pi
# Wait, let me reconsider. The square lattice BZ is [-pi/a, pi/a]^2.
# The irreducible BZ (1/8 of the BZ) is: 0 <= kx <= ky, kx + ky <= pi/a, kx, ky >= 0
# Actually it's the triangle with vertices (0,0), (pi/a, 0), (pi/a, pi/a)? No.
# The irreducible BZ of the square lattice is the triangle with vertices
# (0,0), (pi/a, 0), and (pi/(2a), pi/(2a)).
# Let me use a set of 30 k-points that cover this region.

def generate_k_points(n=30):
    """Generate k-points in the irreducible BZ of the square lattice."""
    kpts = []
    # Include high-symmetry points
    kpts.append((0.0, 0.0))       # Gamma
    kpts.append((math.pi, 0.0))   # X
    kpts.append((math.pi/2, math.pi/2))  # M
    # Add points along Gamma-X, Gamma-M, X-M
    for i in range(1, 9):
        t = i / 9.0
        kpts.append((t * math.pi, 0.0))           # Gamma to X
        kpts.append((t * math.pi/2, t * math.pi/2)) # Gamma to M
        kpts.append((math.pi - t * math.pi/2, t * math.pi/2))  # X to M
    # Trim to 30 if needed
    kpts = kpts[:30]
    return kpts

K_POINTS = generate_k_points(30)
N_K = len(K_POINTS)

# Pre-compute the inverse epsilon matrix for a given pixel map
# For TM polarization, the eigenvalue problem is:
# sum_j (G + k) x (G_j + k) / eps(G - G_j) * c_j = omega^2 * c_i
# where eps is the inverse dielectric function.

def compute_fourier_coeffs(pixel_map, size=32):
    """Compute Fourier coefficients of the pixel map.
    
    The pixel map is a 32x32 array of 0s and 1s.
    We compute the Fourier coefficients for G vectors (m, n) where m, n in [-8, 8].
    
    The dielectric function is eps(r) = eps_air + (eps_high - eps_air) * f(r)
    where f(r) is 1 for dielectric, 0 for air.
    
    For the inverse epsilon (needed for TM), we need 1/eps(r).
    """
    eps_air = 1.0
    eps_high = 12.0
    
    # Convert pixel map to a continuous function
    # Each pixel is a square of size 1/32
    # f(r) = 1 if pixel is dielectric, 0 otherwise
    
    # Compute Fourier coefficients of 1/eps(r)
    # 1/eps(r) = 1/eps_air for air, 1/eps_high for dielectric
    # = (1 - f(r))/eps_air + f(r)/eps_high
    # = 1/eps_air + f(r) * (1/eps_high - 1/eps_air)
    
    inv_eps_air = 1.0 / eps_air
    inv_eps_high = 1.0 / eps_high
    delta_inv_eps = inv_eps_high - inv_eps_air
    
    # Compute Fourier coefficients of f(r)
    # f(r) is a step function on the pixel grid
    # The Fourier coefficient for G = (m, n) is:
    # c_G = integral f(r) exp(-i G.r) d^2r / a^2
    # Since f(r) is piecewise constant on pixels:
    # c_G = sum_{pixels} f_{ij} * integral_{pixel_ij} exp(-i G.r) d^2r / a^2
    
    # For a pixel at position (i, j) with center at ((i+0.5)/size, (j+0.5)/size)
    # and extent [i/size, (i+1)/size] x [j/size, (j+1)/size]:
    # integral = exp(-i G.x * (i+0.5)/size) * exp(-i G.y * (j+0.5)/size) * 
    #           sinc(G.x / (2*pi*size)) * sinc(G.y / (2*pi*size)) * (1/size^2)
    # where sinc(x) = sin(x)/x
    
    coeffs = np.zeros(N_G, dtype=complex)
    pixel_size = 1.0 / size
    
    for idx, (m, n) in enumerate(G_grid):
        Gx = 2 * math.pi * m
        Gy = 2 * math.pi * n
        
        if m == 0 and n == 0:
            # DC component: just the fraction of dielectric
            coeffs[idx] = np.sum(pixel_map) / (size * size)
            continue
        
        # Compute the integral
        # For each pixel, the contribution is:
        # f_{ij} * exp(-i Gx * (i+0.5)/size) * exp(-i Gy * (j+0.5)/size) * 
        #           sin(Gx/(2*pi*size))/(Gx/(2*pi*size)) * sin(Gy/(2*pi*size))/(Gy/(2*pi*size)) * pixel_size^2
        
        # This is separable, so we can compute it as a product of 1D sums
        # Let's compute the 1D Fourier coefficients for x and y separately
        
        # x-direction: sum_i f_col_i * exp(-i Gx * (i+0.5)/size) * sinc(Gx * pixel_size)
        # where f_col_i is the sum over j of pixel_map[i,j]... no, that's not right.
        
        # Actually, f(r) is 2D, so:
        # c_G = (1/size^2) * sum_i sum_j pixel_map[i,j] * 
        #       exp(-i Gx*(i+0.5)/size) * exp(-i Gy*(j+0.5)/size) *
        #       sinc(Gx/(2*pi*size)) * sinc(Gy/(2*pi*size))
        
        # = sinc(Gx/(2*pi*size)) * sinc(Gy/(2*pi*size)) * 
        #   (1/size^2) * sum_i sum_j pixel_map[i,j] * exp(-i Gx*(i+0.5)/size) * exp(-i Gy*(j+0.5)/size)
        
        # The double sum is the 2D DFT of the pixel map (with appropriate scaling)
        
        sx = math.sin(Gx * pixel_size) / (Gx * pixel_size) if Gx != 0 else 1.0
        sy = math.sin(Gy * pixel_size) / (Gy * pixel_size) if Gy != 0 else 1.0
        
        # Compute the 2D sum
        # exp(-i Gx*(i+0.5)/size) = exp(-i Gx/size) * exp(-i Gx*i/size)
        # Let phi_x = Gx/size, phi_y = Gy/size
        # sum_i sum_j pixel_map[i,j] * exp(-i phi_x * (i+0.5)) * exp(-i phi_y * (j+0.5))
        # = exp(-i phi_x/2) * exp(-i phi_y/2) * sum_i sum_j pixel_map[i,j] * exp(-i phi_x * i) * exp(-i phi_y * j)
        
        phi_x = Gx / size
        phi_y = Gy / size
        
        # Use numpy for the 2D sum
        i_idx = np.arange(size)
        j_idx = np.arange(size)
        exp_x = np.exp(-1j * phi_x * i_idx)
        exp_y = np.exp(-1j * phi_y * j_idx)
        
        # 2D sum = sum_i exp_x[i] * sum_j pixel_map[i,j] * exp_y[j]
        # = exp_x @ pixel_map @ exp_y
        val = np.exp(-1j * phi_x / 2) * np.exp(-1j * phi_y / 2) * (exp_x @ pixel_map @ exp_y)
        
        coeffs[idx] = sx * sy * val / (size * size)
    
    # Now compute 1/eps Fourier coefficients
    # 1/eps(r) = inv_eps_air + delta_inv_eps * f(r)
    # So the Fourier coefficients of 1/eps are:
    # inv_eps_G = inv_eps_air * delta_{G,0} + delta_inv_eps * f_G
    
    inv_eps_coeffs = np.zeros(N_G, dtype=complex)
    for idx in range(N_G):
        if G_grid[idx][0] == 0 and G_grid[idx][1] == 0:
            inv_eps_coeffs[idx] = inv_eps_air + delta_inv_eps * coeffs[idx]
        else:
            inv_eps_coeffs[idx] = delta_inv_eps * coeffs[idx]
    
    return inv_eps_coeffs

def compute_bands(inv_eps_coeffs, kx, ky):
    """Compute the TM band structure for a given k-point.
    
    The eigenvalue problem for TM polarization is:
    sum_j (G_i + k) x (G_j + k) * inv_eps_{G_i - G_j} * c_j = omega^2 * c_i
    
    where (G + k) x (G + k) = |G + k|^2 for TM (the transverse component).
    Wait, for TM polarization in 2D, the electric field is perpendicular to the plane.
    The eigenvalue problem is:
    sum_j (k + G_i) * (k + G_j) / eps(G_i - G_j) * c_j = omega^2 c_i
    
    Actually, for TM (out-of-plane E), the equation is:
    sum_j (k + G_i) . (k + G_j) * inv_eps(G_i - G_j) * c_j = (omega/c)^2 * a^2 * c_i
    
    Let me be more careful. In the plane-wave expansion for TM:
    E_z(r) = sum_G c_G exp(i (k + G) . r)
    The wave equation gives:
    sum_G' (k + G) x (k + G') / eps(G - G') c_{G'} = (omega/c)^2 a^2 c_G
    
    where for 2D TM, (k+G) x (k+G') means the cross product in the 2D plane,
    which is (k_x + G_x)(k_x + G'_x) + (k_y + G_y)(k_y + G'_y) = (k+G).(k+G')
    
    So the matrix element is:
    M_{G,G'} = (k+G).(k+G') * inv_eps(G - G')
    
    And we solve M c = (omega/c)^2 a^2 c
    """
    # Build the matrix M
    M = np.zeros((N_G, N_G), dtype=complex)
    
    # Pre-compute k + G for each G
    kG = G_grid + np.array([kx, ky])
    
    # For each pair (i, j), compute M[i,j] = (k+G_i).(k+G_j) * inv_eps(G_i - G_j)
    for i in range(N_G):
        for j in range(N_G):
            # G_i - G_j
            dG = G_grid[i] - G_grid[j]
            dm = int(round(dG[0]))
            dn = int(round(dG[1]))
            
            # Find the index of (dm, dn) in G_grid
            # G_grid has entries from -8 to 8 in both dimensions
            idx = (dm + 8) * 17 + (dn + 8)
            if 0 <= idx < N_G:
                eps_val = inv_eps_coeffs[idx]
            else:
                # If the difference is outside the computed range, assume it's 0
                # (this shouldn't happen with 17x17 G vectors)
                eps_val = 0.0
            
            dot = kG[i, 0] * kG[j, 0] + kG[i, 1] * kG[j, 1]
            M[i, j] = dot * eps_val
    
    # Solve the eigenvalue problem
    # M is Hermitian, so use eigh
    eigvals, eigvecs = np.linalg.eigh(M)
    
    # The eigenvalues are (omega/c)^2 * a^2
    # We want omega in units of 2pi*c/a, so omega * a / (2pi*c) = sqrt(eigval) / (2pi)
    # Actually, let's just return sqrt(eigval) as the frequency in units of c/a
    # The score uses ratios, so the absolute scale doesn't matter.
    
    freqs = np.sqrt(np.real(eigvals))
    return freqs

def compute_gap(pixel_map):
    """Compute the 4-5 TM band gap ratio for a given pixel map."""
    inv_eps_coeffs = compute_fourier_coeffs(pixel_map)
    
    max_omega_4 = 0.0
    min_omega_5 = float('inf')
    
    for kx, ky in K_POINTS:
        freqs = compute_bands(inv_eps_coeffs, kx, ky)
        # freqs is sorted in ascending order
        # Band 4 is index 3, band 5 is index 4
        if len(freqs) >= 5:
            omega_4 = freqs[3]
            omega_5 = freqs[4]
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
    
    if min_omega_5 == float('inf') or max_omega_4 == 0.0:
        return 0.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    
    return 2.0 * gap / midgap

def make_symmetric(cell):
    """Ensure the cell is symmetric under the square symmetry group."""
    size = len(cell)
    # Make it symmetric under 90-degree rotation and reflection
    # The cell must satisfy: cell == cell[::-1] row-wise, cell == cell[::-1] column-wise, cell == cell.T
    
    # Start with the top-left quadrant and mirror it
    result = [[0] * size for _ in range(size)]
    half = size // 2
    
    # Fill the top-left quadrant and mirror
    for i in range(half):
        for j in range(half):
            val = cell[i][j]
            result[i][j] = val
            result[i][size - 1 - j] = val  # horizontal mirror
            result[size - 1 - i][j] = val  # vertical mirror
            result[size - 1 - i][size - 1 - j] = val  # both mirrors
    
    # Handle the middle row and column (if size is even, there's no exact middle)
    # For size=32, half=16, so indices 0..15 and 16..31
    # The symmetry is: result[i][j] == result[i][31-j] == result[31-i][j] == result[31-i][31-j]
    
    return result

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the 4-5 TM band gap."""
    start_time = time.time()
    deadline = start_time + 180.0  # Leave 50s margin
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Initial guess: 4 rods at the corners of the cell
    # Positions: (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
    # Radius: 0.15
    def initial_cell():
        cell = [[0] * size for _ in range(size)]
        centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        radius = 0.15
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size
                y = (j + 0.5) / size
                for cx, cy in centers:
                    if (x - cx)**2 + (y - cy)**2 < radius**2:
                        cell[i][j] = 1
                        break
        return cell
    
    cell = initial_cell()
    cell = make_symmetric(cell)
    pixel_map = np.array(cell, dtype=float)
    
    current_score = compute_gap(pixel_map)
    best_score = current_score
    best_cell = [row[:] for row in cell]
    
    # Local search: flip pixels and check if the score improves
    # Use a greedy approach with random restarts
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Try flipping a random pixel (and its symmetric counterparts)
        i = rng.integers(0, size // 2)
        j = rng.integers(0, size // 2)
        
        # Get the symmetric positions
        positions = [(i, j), (i, size - 1 - j), (size - 1 - i, j), (size - 1 - i, size - 1 - j)]
        
        # Flip all symmetric positions
        old_vals = []
        for pi, pj in positions:
            old_vals.append(cell[pi][pj])
            cell[pi][pj] = 1 - cell[pi][pj]
        
        new_pixel_map = np.array(cell, dtype=float)
        new_score = compute_gap(new_pixel_map)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = [row[:] for row in cell]
        else:
            # Revert
            for (pi, pj), old_val in zip(positions, old_vals):
                cell[pi][pj] = old_val
    
    return best_cell
