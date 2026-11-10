import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # Initialize cell: Dielectric background (1) with a central air hole (0)
    # We will use a diamond shape initially as it often yields better TM gaps
    cell = np.ones((size, size), dtype=int)
    
    # Define a central diamond hole
    # |i - center| + |j - center| < radius
    center = (size - 1) / 2.0
    radius = 6.0 # Initial guess
    for i in range(size):
        for j in range(size):
            if abs(i - center) + abs(j - center) < radius:
                cell[i, j] = 0

    # Identify independent pixels for D4 symmetry
    # The cell is symmetric under:
    # 1. Horizontal flip: (i, j) -> (i, size-1-j)
    # 2. Vertical flip: (i, j) -> (size-1-i, j)
    # 3. Transpose: (i, j) -> (j, i)
    # 
    # We can define the independent set as pixels where i <= size//2 and j <= i
    # Wait, for size=32, center is 15.5.
    # Symmetry axes are between 15 and 16.
    # Independent region: 0 <= i <= 15, 0 <= j <= i
    # Let's verify reflection.
    # If we change cell[i][j], we must change:
    # cell[i][size-1-j], cell[size-1-i][j], cell[size-1-i][size-1-j]
    # And if we change cell[j][i] (if j != i), we must change its reflections too.
    
    # To simplify, we will just iterate over ALL pixels in the top-left 16x16 quadrant?
    # No, that breaks transpose symmetry if we don't handle it.
    
    # Let's define the list of independent coordinates.
    # For i in 0..15:
    #   for j in 0..i:
    #     This covers the triangle.
    #     When we flip (i,j), we must flip:
    #     (i, size-1-j)
    #     (size-1-i, j)
    #     (size-1-i, size-1-j)
    #     AND if i != j, we must also flip the transpose set:
    #     (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
    
    independent_pixels = []
    for i in range(size // 2):
        for j in range(i + 1):
            independent_pixels.append((i, j))
            
    # Precompute G vectors
    # Plane wave count
    N_pw = 9 
    G_max = (N_pw - 1) // 2
    G_vectors = []
    for gx in range(-G_max, G_max + 1):
        for gy in range(-G_max, G_max + 1):
            G_vectors.append((gx, gy))
    G_vectors = np.array(G_vectors)
    N_G = len(G_vectors)
    
    # Precompute pixel positions
    # Pixel centers: (i+0.5)/size - 0.5
    pixel_coords = []
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            pixel_coords.append((x, y))
    pixel_coords = np.array(pixel_coords)
    
    # Precompute phase factors for each pixel and each G
    # Phase = exp(-i * (Gx * x + Gy * y))
    # G is in units of 2*pi/a. Since our coords are in units of a, 
    # the phase is exp(-i * 2*pi * (Gx * x + Gy * y))
    phases = np.zeros((size * size, N_G), dtype=complex)
    for idx in range(size * size):
        x, y = pixel_coords[idx]
        for g_idx in range(N_G):
            gx, gy = G_vectors[g_idx]
            phases[idx, g_idx] = np.exp(-1j * 2 * np.pi * (gx * x + gy * y))
            
    # Function to compute epsilon_G
    def compute_epsilon_G(cell_flat):
        # cell_flat is size*size array of 0/1
        # eps = 1 + 11 * cell
        eps_vals = 1.0 + 11.0 * cell_flat
        # epsilon_G = sum( eps_vals * phases ) / (size*size)
        # Vectorized: eps_vals (N_pix, 1) * phases (N_pix, N_G)
        eps_G = np.sum(eps_vals[:, np.newaxis] * phases, axis=0) / (size * size)
        return eps_G

    # Function to compute bands at a k-point
    def compute_bands(kx, ky, eps_G):
        # k is in units of 2*pi/a
        # Matrix M[G, G'] = (1/eps_{G-G'}) * (k+G) dot (k+G')
        # Note: eps_{G-G'} is indexed by the difference vector.
        # We have eps_G for G in G_vectors.
        # We need to map difference G-G' to the index in eps_G.
        
        # Create a dictionary for fast lookup of G index
        # G_vectors is sorted? We generated it in a loop.
        # Let's create a map (gx, gy) -> index
        G_map = {}
        for idx, (gx, gy) in enumerate(G_vectors):
            G_map[(gx, gy)] = idx
            
        M = np.zeros((N_G, N_G), dtype=complex)
        
        for i_g in range(N_G):
            gx, gy = G_vectors[i_g]
            kx_g = kx + gx
            ky_g = ky + gy
            for j_g in range(N_G):
                gpx, gpy = G_vectors[j_g]
                kx_gp = kx + gpx
                ky_gp = ky + gpy
                
                # Difference G - G'
                diff_gx = gx - gpx
                diff_gy = gy - gpy
                
                # Check if diff is in our G set
                if (diff_gx, diff_gy) in G_map:
                    idx_diff = G_map[(diff_gx, diff_gy)]
                    eps_diff = eps_G[idx_diff]
                    if abs(eps_diff) < 1e-9:
                        M[i_g, j_g] = 0
                    else:
                        # TM: dot product
                        dot = kx_g * kx_gp + ky_g * ky_gp
                        M[i_g, j_g] = dot / eps_diff
                else:
                    M[i_g, j_g] = 0
                    
        # Solve eigenvalue problem
        # M is Hermitian? 
        # (k+G) dot (k+G') / eps_{G-G'}
        # M_{ij} = (k+Gi) dot (k+Gj) / eps_{Gi-Gj}
        # M_{ji} = (k+Gj) dot (k+Gi) / eps_{Gj-Gi}
        # Since eps is real and symmetric (eps_{-G} = eps_G* = eps_G for real eps), 
        # and dot is symmetric, M is Hermitian.
        eigenvalues = np.linalg.eigvalsh(M)
        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)
        # Return the first 5 bands (frequencies squared)
        return eigenvalues[:5]

    # High symmetry points
    # Gamma: (0, 0)
    # X: (0.5, 0)  (Since k is in 2pi/a units, X is at pi/a -> 0.5)
    # M: (0.5, 0.5)
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5)
    ]

    def calculate_surrogate_gap(cell):
        cell_flat = cell.flatten()
        eps_G = compute_epsilon_G(cell_flat)
        
        min_omega_5 = np.inf
        max_omega_4 = -np.inf
        
        for kx, ky in k_points:
            bands_sq = compute_bands(kx, ky, eps_G)
            # bands_sq are omega^2
            # omega = sqrt(bands_sq)
            # We need the 4th and 5th bands (indices 3 and 4)
            if len(bands_sq) >= 5:
                omega_4 = np.sqrt(bands_sq[3])
                omega_5 = np.sqrt(bands_sq[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
                
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0
        return gap / midgap

    # Local Search
    best_gap = calculate_surrogate_gap(cell)
    best_cell = cell.copy()
    
    # Precompute reflection indices for each independent pixel
    # For a pixel (i, j), the reflected pixels are:
    # (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)
    # If i != j, also:
    # (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
    
    reflection_map = {}
    for (i, j) in independent_pixels:
        pixels = [(i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)]
        if i != j:
            pixels.extend([(j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)])
        # Unique indices
        unique_pixels = list(set(pixels))
        reflection_map[(i, j)] = unique_pixels

    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for (i, j) in independent_pixels:
            if time.time() - start_time > time_limit:
                break
                
            # Try flipping
            old_val = cell[i, j]
            new_val = 1 - old_val
            
            # Apply flip to all reflected pixels
            pixels_to_flip = reflection_map[(i, j)]
            for (ri, rj) in pixels_to_flip:
                cell[ri, rj] = new_val
                
            new_gap = calculate_surrogate_gap(cell)
            
            if new_gap > best_gap + 1e-6:
                best_gap = new_gap
                best_cell = cell.copy()
                improved = True
            else:
                # Revert
                for (ri, rj) in pixels_to_flip:
                    cell[ri, rj] = old_val
    
    # Convert to list of lists
    return best_cell.tolist()
