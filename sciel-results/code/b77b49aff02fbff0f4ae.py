import numpy as np
import time
import itertools

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses a proxy plane-wave model to search and optimize D4-symmetric designs
    targeting the 4th-5th TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Precompute coordinates for symmetry enforcement and generation
    # We work with a 32x32 grid.
    # D4 symmetry: invariant under 90 deg rotation and reflection.
    # We define orbits of pixels under D4.
    # For a 32x32 grid, center is at (15.5, 15.5) if 0-indexed, but symmetry is about center.
    # Let's use 0..31 indices.
    # Symmetry operations:
    # 1. Identity: (i, j)
    # 2. Rot 90: (j, size-1-i)
    # 3. Rot 180: (size-1-i, size-1-j)
    # 4. Rot 270: (size-1-j, i)
    # 5. Refl X: (i, size-1-j)
    # 6. Refl Y: (size-1-i, j)
    # 7. Refl Diag: (j, i)
    # 8. Refl Anti-Diag: (size-1-j, size-1-i)
    
    # To enforce symmetry, we can just generate a pattern and symmetrize it.
    # Or, define the pattern on the fundamental domain and replicate.
    # Fundamental domain for D4 on NxN grid:
    # i from 0 to N//2 - 1 (if N even)
    # j from 0 to N//2 - 1
    # Actually, for N=32, center is between 15 and 16.
    # The orbits are sets of pixels.
    
    def get_orbits(size):
        orbits = []
        visited = np.zeros((size, size), dtype=bool)
        for i in range(size):
            for j in range(size):
                if not visited[i, j]:
                    orbit = []
                    # Generate all 8 symmetric points
                    points = [
                        (i, j),
                        (j, size-1-i),
                        (size-1-i, size-1-j),
                        (size-1-j, i),
                        (i, size-1-j),
                        (size-1-i, j),
                        (j, i),
                        (size-1-j, size-1-i)
                    ]
                    # Unique points
                    unique_points = []
                    for p in points:
                        if not visited[p[0], p[1]]:
                            unique_points.append(p)
                            visited[p[0], p[1]] = True
                    orbits.append(unique_points)
        return orbits

    orbits = get_orbits(size)
    
    # Fast Proxy Model
    # Use a small basis of G-vectors.
    # G = (m, n) * 2*pi / L. We work in units where L=1.
    # We need coefficients of 1/eps.
    # eps is piecewise constant.
    # 1/eps is 1/12 for dielectric, 1 for air.
    # Let's compute the Fourier coefficients of 1/eps for a given cell.
    # For speed, we can use FFT.
    
    def compute_inv_eps_coeffs(cell):
        # cell is 32x32 array of 0s and 1s.
        # 1 = dielectric (eps=12), 0 = air (eps=1)
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # FFT of inv_eps
        # The Fourier series is: f(x,y) = sum_{m,n} C_{mn} exp(i 2 pi (m x + n y))
        # numpy.fft.fft2 gives F[k,l] = sum_{x,y} f[x,y] exp(-i 2 pi (k x + l y) / N)
        # So C_{mn} = F[m % N, n % N] / N^2
        F = np.fft.fft2(inv_eps)
        return F / (size * size)

    # We need to evaluate the band structure.
    # H = (k+G)^T * diag(1/eps) * (k+G) is not quite right for TM.
    # For TM: H_{G', G} = (k+G') . (k+G) * (1/eps)_{G'-G}
    # Wait, the standard form is:
    # sum_G' (k+G) . (k+G') * (1/eps)_{G-G'} * c_{G'} = omega^2 c_G
    # Let's use a small basis of G-vectors.
    # G-vectors: (m, n) for m, n in [-g_max, g_max]
    
    g_max = 4
    g_vectors = [(m, n) for m in range(-g_max, g_max+1) for n in range(-g_max, g_max+1)]
    N_g = len(g_vectors)
    
    # K-points in irreducible BZ: Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Note: X is (0.5, 0) and M is (0.5, 0.5) in some conventions, or vice versa.
    # Square lattice: Gamma (0,0), M (0.5, 0.5) (corner), X (0.5, 0) (mid-edge).
    k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
    
    def estimate_gap(cell):
        """
        Estimate the gap ratio between band 4 and 5.
        """
        if time.time() > deadline:
            return -1.0
            
        coeffs = compute_inv_eps_coeffs(cell)
        
        # Map G-vector difference to index in coeffs
        # coeffs[m % size, n % size] corresponds to G = (m, n)
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # Build Hamiltonian matrix H
            H = np.zeros((N_g, N_g), dtype=complex)
            for i, Gp in enumerate(g_vectors):
                kx_total = kx + Gp[0]
                ky_total = ky + Gp[1]
                for j, G in enumerate(g_vectors):
                    m_diff = Gp[0] - G[0]
                    n_diff = Gp[1] - G[1]
                    
                    # Get coeff for (m_diff, n_diff)
                    m_idx = m_diff % size
                    n_idx = n_diff % size
                    eps_inv_G = coeffs[m_idx, n_idx]
                    
                    # H_{Gp, G} = (k+Gp) . (k+G) * (1/eps)_{Gp-G}
                    # Wait, the dot product is (k+Gp) dot (k+G)
                    dot = (kx_total) * (kx + G[0]) + (ky_total) * (ky + G[1])
                    H[i, j] = dot * eps_inv_G
            
            # Solve eigenvalue problem
            # H is Hermitian
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            # We want bands 4 and 5 (index 3 and 4)
            if len(eigvals) >= 5:
                omega_4 = eigvals[3]
                omega_5 = eigvals[4]
                
                if omega_5 < min_omega_5:
                    min_omega_5 = omega_5
                if omega_4 > max_omega_4:
                    max_omega_4 = omega_4
        
        if min_omega_5 == float('inf') or max_omega_4 <= 0:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap <= 0:
            return -1.0
            
        return gap / midgap

    # Design Generators
    # We represent the design as a binary array.
    # To ensure symmetry, we define the pattern on the fundamental domain or use orbits.
    # Let's use the orbit approach for optimization.
    # Each orbit is a set of pixels. We assign a binary value to each orbit.
    # The cell is constructed by filling all pixels in an orbit with the same value.
    
    def build_cell_from_orbit_values(orbit_values):
        cell = np.zeros((size, size), dtype=int)
        for orbit, val in zip(orbits, orbit_values):
            for i, j in orbit:
                cell[i, j] = val
        return cell

    # Initial candidates:
    # 1. Single central rod
    # 2. Ring
    # 3. Cross
    # 4. Four rods
    
    def generate_single_rod(radius_frac):
        cell = np.zeros((size, size), dtype=int)
        cx = size / 2.0
        cy = size / 2.0
        r = radius_frac * size
        for i in range(size):
            for j in range(size):
                x = j - cx
                y = i - cy
                if x*x + y*y < r*r:
                    cell[i, j] = 1
        return cell

    def generate_ring(inner_frac, outer_frac):
        cell = np.zeros((size, size), dtype=int)
        cx = size / 2.0
        cy = size / 2.0
        r_in = inner_frac * size
        r_out = outer_frac * size
        for i in range(size):
            for j in range(size):
                x = j - cx
                y = i - cy
                r2 = x*x + y*y
                if r_in*r_in <= r2 < r_out*r_out:
                    cell[i, j] = 1
        return cell

    def generate_cross(width_frac, length_frac):
        cell = np.zeros((size, size), dtype=int)
        cx = size / 2.0
        cy = size / 2.0
        w = width_frac * size / 2.0
        l = length_frac * size / 2.0
        for i in range(size):
            for j in range(size):
                x = j - cx
                y = i - cy
                if (abs(x) < w and abs(y) < l) or (abs(y) < w and abs(x) < l):
                    cell[i, j] = 1
        return cell

    def generate_four_rods(dx, dy, radius_frac):
        cell = np.zeros((size, size), dtype=int)
        r = radius_frac * size
        centers = [
            (dx * size, dy * size),
            (-dx * size, dy * size),
            (dx * size, -dy * size),
            (-dx * size, -dy * size)
        ]
        cx = size / 2.0
        cy = size / 2.0
        for i in range(size):
            for j in range(size):
                x = j - cx
                y = i - cy
                for ccx, ccy in centers:
                    if (x-ccx)**2 + (y-ccy)**2 < r*r:
                        cell[i, j] = 1
                        break
        return cell

    # Evaluate initial candidates
    candidates = []
    
    # Single rods
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        cell = generate_single_rod(r)
        score = estimate_gap(cell)
        candidates.append((score, cell))
        
    # Rings
    for r_in in [0.0, 0.1, 0.2]:
        for r_out in [0.3, 0.4, 0.5, 0.6]:
            if r_in < r_out:
                cell = generate_ring(r_in, r_out)
                score = estimate_gap(cell)
                candidates.append((score, cell))
                
    # Crosses
    for w in [0.1, 0.2, 0.3]:
        for l in [0.4, 0.5, 0.6, 0.8]:
            cell = generate_cross(w, l)
            score = estimate_gap(cell)
            candidates.append((score, cell))
            
    # Four rods
    for dx in [0.2, 0.25, 0.3]:
        for dy in [0.2, 0.25, 0.3]:
            for r in [0.1, 0.15, 0.2]:
                cell = generate_four_rods(dx, dy, r)
                score = estimate_gap(cell)
                candidates.append((score, cell))
                
    # Sort by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 candidates and refine
    best_score = -1.0
    best_cell = candidates[0][1] if candidates else np.zeros((size, size), dtype=int)
    
    if candidates:
        best_score, best_cell = candidates[0]
        
    top_k = 5
    for score, cell in candidates[:top_k]:
        if time.time() > deadline - 10:
            break
            
        # Local optimization: flip orbits
        # Convert cell to orbit values
        orbit_values = []
        for orbit in orbits:
            # Majority vote
            val = sum(cell[i, j] for i, j in orbit)
            orbit_values.append(1 if val >= len(orbit)/2 else 0)
            
        current_score = score
        current_orbit_values = orbit_values.copy()
        
        # Simple hill climbing
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try flipping each orbit
            for idx in range(len(orbits)):
                # Flip
                old_val = current_orbit_values[idx]
                current_orbit_values[idx] = 1 - old_val
                
                new_cell = build_cell_from_orbit_values(current_orbit_values)
                new_score = estimate_gap(new_cell)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_cell = new_cell
                else:
                    # Revert
                    current_orbit_values[idx] = old_val
                    
    # Return best cell
    return best_cell.tolist()

