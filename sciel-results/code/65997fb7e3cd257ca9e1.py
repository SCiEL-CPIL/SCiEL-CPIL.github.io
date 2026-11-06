"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    
    Strategy: Use a D4-symmetric binary mask derived from a superposition of
    multi-scale features (approximating wavelet-like structures) that couple
    specific high-frequency reciprocal lattice vectors. We explore square pillars
    (which have richer Fourier spectra than circles) and optimize their size
    and arrangement using a fast surrogate model.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 230s limit
    
    def generate_square_pillars(radius):
        """
        Generate a 32x32 cell with square pillars arranged in a D4-symmetric pattern.
        We'll use a 2x2 supercell arrangement of square pillars.
        """
        cell = np.zeros((size, size), dtype=int)
        
        # Centers of the 4 pillars in a 2x2 arrangement
        centers = [
            (8, 8),
            (8, 24),
            (24, 8),
            (24, 24)
        ]
        
        # Square pillar: |x - cx| < radius and |y - cy| < radius
        for cx, cy in centers:
            x1 = max(0, int(cx - radius))
            x2 = min(size, int(cx + radius + 1))
            y1 = max(0, int(cy - radius))
            y2 = min(size, int(cy + radius + 1))
            cell[x1:x2, y1:y2] = 1
        
        return cell
    
    def generate_circular_rods(radius):
        """Generate a 32x32 cell with circular rods arranged in a D4-symmetric pattern."""
        cell = np.zeros((size, size), dtype=int)
        
        centers = [
            (8, 8),
            (8, 24),
            (24, 8),
            (24, 24)
        ]
        
        for cx, cy in centers:
            for i in range(size):
                for j in range(size):
                    dx = i - cx
                    dy = j - cy
                    if dx*dx + dy*dy < radius*radius:
                        cell[i, j] = 1
        
        return cell
    
    def approximate_gap_ratio(cell, max_g=4, n_kpoints=8):
        """
        Fast surrogate model to estimate the gap ratio.
        We'll use a simplified band structure calculation with a small basis set.
        """
        # For speed, we'll only calculate at a few high-symmetry k-points
        # and use a small number of plane waves.
        
        # We need 1/eps(x,y) for the TM band structure
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        inv_fft = np.fft.fft2(inv_eps) / (size * size)
        
        # Define a few k-points in the irreducible Brillouin zone
        # The BZ is [-pi, pi] x [-pi, pi] (in units of 2*pi/L)
        # Irreducible wedge: 0 <= kx <= ky <= pi
        k_points = [
            (0.0, 0.0),      # Gamma
            (np.pi/2, 0.0),  # X
            (np.pi/2, np.pi/2), # M
            (np.pi/4, 0.0),
            (np.pi/4, np.pi/4),
            (np.pi/6, 0.0),
            (np.pi/3, 0.0),
            (np.pi/3, np.pi/6),
        ]
        
        # Use a small basis: G vectors with |m|, |n| <= max_g
        g_vectors = []
        for m in range(-max_g, max_g + 1):
            for n in range(-max_g, max_g + 1):
                g_vectors.append((m, n))
        
        n_g = len(g_vectors)
        
        # Precompute the inverse epsilon Fourier coefficients
        inv_coeff_dict = {}
        for m in range(-max_g, max_g + 1):
            for n in range(-max_g, max_g + 1):
                m_idx = m % size
                n_idx = n % size
                inv_coeff_dict[(m, n)] = inv_fft[m_idx, n_idx]
        
        # Calculate eigenvalues at each k-point
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build the matrix H where H_{G,G'} = (k+G)^2 * inv_eps_{G-G'}
            H = np.zeros((n_g, n_g), dtype=complex)
            
            for i, (m1, n1) in enumerate(g_vectors):
                kx_g = kx + 2 * np.pi * m1
                ky_g = ky + 2 * np.pi * n1
                k_sq = kx_g**2 + ky_g**2
                
                for j, (m2, n2) in enumerate(g_vectors):
                    dm = m1 - m2
                    dn = n1 - n2
                    if (dm, dn) in inv_coeff_dict:
                        H[i, j] = k_sq * inv_coeff_dict[(dm, dn)]
                    else:
                        H[i, j] = 0.0
            
            # Solve the eigenvalue problem
            # H c = omega^2 c
            try:
                eigvals = np.linalg.eigvalsh(H)
                eigvals = np.sort(np.real(eigvals))
                
                # We want the 4th and 5th bands (indices 3 and 4)
                if len(eigvals) >= 5:
                    omega_4 = math.sqrt(max(0, eigvals[3]))
                    omega_5 = math.sqrt(max(0, eigvals[4]))
                    
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except:
                continue
        
        if min_omega_5 <= 0 or max_omega_4 <= 0 or min_omega_5 == float('inf'):
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap
    
    # Main optimization loop
    best_score = -1.0
    best_cell = None
    best_radius = None
    best_type = 'square'
    
    # Define a search space for pillar/rod radius
    # Radius in pixels, from 2 to 10
    radii = np.arange(2.0, 10.5, 0.5)
    
    # First pass: coarse search over square pillars
    for radius in radii:
        if time.time() > deadline:
            break
        
        cell = generate_square_pillars(radius)
        score = approximate_gap_ratio(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            best_radius = radius
            best_type = 'square'
    
    # Second pass: fine search around the best radius
    if best_cell is not None and best_radius is not None:
        fine_radii = np.arange(max(1.5, best_radius - 1.0), min(12.0, best_radius + 1.0), 0.1)
        for radius in fine_radii:
            if time.time() > deadline:
                break
            
            cell = generate_square_pillars(radius)
            score = approximate_gap_ratio(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                best_radius = radius
                best_type = 'square'
    
    # Third pass: try circular rods if we have time
    if time.time() < deadline - 15:
        circ_radii = np.arange(2.0, 10.5, 0.5)
        for radius in circ_radii:
            if time.time() > deadline:
                break
            
            cell = generate_circular_rods(radius)
            score = approximate_gap_ratio(cell)
            
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
                best_radius = radius
                best_type = 'circle'
        
        # Fine search for circular rods if they were better
        if best_type == 'circle' and best_radius is not None:
            circ_fine_radii = np.arange(max(1.5, best_radius - 1.0), min(12.0, best_radius + 1.0), 0.1)
            for radius in circ_fine_radii:
                if time.time() > deadline:
                    break
                
                cell = generate_circular_rods(radius)
                score = approximate_gap_ratio(cell)
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
                    best_radius = radius
                    best_type = 'circle'
    
    # Convert numpy array to nested list
    if best_cell is not None:
        return best_cell.tolist()
    else:
        # Fallback: return a default cell
        cell = generate_square_pillars(4.0)
        return cell.tolist()

# EVOLVE-BLOCK-END
