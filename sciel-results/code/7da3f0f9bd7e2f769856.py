"""Design a 32x32 symmetric photonic-crystal cell to maximize the TM 4-5 band gap."""
import time
import numpy as np
from scipy.linalg import eigh

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")
    
    deadline = time.time() + 220  # safety margin before 230s kill
    
    # --- Fast surrogate: plane-wave method with limited G-vectors ---
    def fast_gap(cell):
        """Compute approximate gap using a small set of G-vectors."""
        N = 5  # half-width of G-vector grid
        G_list = []
        for gx in range(-N, N + 1):
            for gy in range(-N, N + 1):
                G_list.append((gx, gy))
        
        # Compute Fourier coefficients of 1/epsilon for TM
        # epsilon = eps_low + (eps_high - eps_low) * cell
        eps_inv = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                eps_inv[i, j] = 1.0 / (eps_low + (eps_high - eps_low) * cell[i][j])
        
        # Precompute Fourier coefficients
        G_coeff = {}
        for gx, gy in G_list:
            # F[G] = (1/N^2) sum_{i,j} f(i,j) exp(-2pi i (Gx*i + Gy*j)/N)
            # Using DFT
            G_coeff[(gx, gy)] = np.fft.ifft2(
                np.fft.fftshift(
                    np.fft.fft2(eps_inv) * (size ** 2)
                )
            ).astype(complex)
            # Actually, let's do this more carefully
            # f(i,j) = eps_inv[i,j]
            # F(G) = sum_{i,j} f(i,j) exp(-2pi i (Gx*i + Gy*j)/size) / size^2
            # This is just the 2D DFT normalized
            pass
        
        # Recompute properly
        F = np.fft.fft2(eps_inv) / (size ** 2)
        
        # For a given G, F[G] where G is in units of 2pi/size
        # G_x = gx * 2pi/size, so the index in F is gx mod size
        def get_F(gx, gy):
            idx_x = gx % size
            idx_y = gy % size
            return F[idx_x, idx_y]
        
        # Build matrices for a few k-points
        k_points = [
            (0.0, 0.0),      # Gamma
            (0.5, 0.0),      # X
            (0.5, 0.5),      # M
            (0.25, 0.0),
            (0.25, 0.25),
            (0.0, 0.25),
            (0.375, 0.125),
            (0.125, 0.375),
        ]
        
        max_omega_lower = 0.0
        min_omega_upper = float('inf')
        
        for kx, ky in k_points:
            k = (kx, ky)
            # Build the eigenvalue problem
            n = len(G_list)
            A = np.zeros((n, n), dtype=complex)
            
            for i, (Gix, Giy) in enumerate(G_list):
                G_i = (Gix, Giy)
                for j, (Gjx, Gjy) in enumerate(G_list):
                    G_j = (Gjx, Gjy)
                    G_diff = (Gix - Gjx, Giy - Gjy)
                    
                    # (k + G_i) . (k + G_j)
                    kpi = (kx + Gix, ky + Giy)
                    kpj = (kx + Gjx, ky + Gjy)
                    dot = kpi[0] * kpj[0] + kpi[1] * kpj[1]
                    
                    # Get F[G_diff]
                    F_val = get_F(G_diff[0], G_diff[1])
                    
                    A[i, j] = dot * F_val
            
            # Solve eigenvalue problem: A * c = omega^2 * c
            # A is Hermitian
            A = (A + A.conj().T) / 2
            try:
                eigvals = eigh(A, eigvals_only=True)
                eigvals = np.real(eigvals)
                eigvals = np.sort(eigvals)
                
                # Take bands 4 and 5 (0-indexed: 3 and 4)
                if len(eigvals) >= 5:
                    omega4 = np.sqrt(max(eigvals[3], 0))
                    omega5 = np.sqrt(max(eigvals[4], 0))
                    max_omega_lower = max(max_omega_lower, omega4)
                    min_omega_upper = min(min_omega_upper, omega5)
            except:
                return 0.0
        
        if min_omega_upper == float('inf'):
            return 0.0
        
        gap = 2 * (min_omega_upper - max_omega_lower) / (min_omega_upper + max_omega_lower)
        return gap if gap > 0 else 0.0
    
    # --- Symmetric cell generation and optimization ---
    
    def make_symmetric_cell(pixel_map):
        """Enforce 4-fold symmetry (D4) on a 32x32 cell."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                val = pixel_map[i, j]
                # Apply D4 symmetries
                cell[i, j] = val
                cell[j, i] = val  # transpose
                cell[size-1-i, j] = val  # row flip
                cell[i, size-1-j] = val  # col flip
                cell[size-1-i, size-1-j] = val  # 180 rot
                cell[size-1-j, i] = val  # other transposes
                cell[size-1-i, size-1-j] = val
                # The full D4 orbit: (i,j), (j,i), (N-1-i,j), (N-1-j,i),
                # (N-1-i,N-1-j), (N-1-j,N-1-i), (i,N-1-j), (j,N-1-i)
                cell[i, size-1-j] = val
                cell[j, size-1-i] = val
        return cell
    
    def cell_to_list(cell_np):
        return [list(row) for row in cell_np]
    
    # Generate initial candidates
    np.random.seed(42)
    
    def random_symmetric_cell():
        """Generate a random D4-symmetric cell."""
        # Only need to specify the fundamental domain
        # For 32x32 with D4 symmetry, the fundamental domain is roughly 1/8 of the cell
        # Points: (i, j) where 0 <= i <= j <= 15 (upper triangle of first quadrant)
        # Actually, for a 32x32 cell centered at (15.5, 15.5), the symmetry axes are at 15.5
        # Fundamental domain: i in [0, 15], j in [i, 15] (upper-left triangle of top-left quadrant)
        # Plus the diagonal points
        
        # Simpler: just set random values and enforce symmetry
        raw = np.random.randint(0, 2, (size, size))
        return make_symmetric_cell(raw)
    
    def rod_cell(radius_frac):
        """Dielectric rod of given radius fraction."""
        cell = np.zeros((size, size), dtype=int)
        cx = cy = (size - 1) / 2.0
        r = radius_frac * size / 2
        for i in range(size):
            for j in range(size):
                dx = i - cx
                dy = j - cy
                if dx*dx + dy*dy < r*r:
                    cell[i, j] = 1
        return cell
    
    def ring_cell(r_inner, r_outer):
        """Dielectric ring."""
        cell = np.zeros((size, size), dtype=int)
        cx = cy = (size - 1) / 2.0
        ri = r_inner * size / 2
        ro = r_outer * size / 2
        for i in range(size):
            for j in range(size):
                dx = i - cx
                dy = j - cy
                d2 = dx*dx + dy*dy
                if ri*ri <= d2 <= ro*ro:
                    cell[i, j] = 1
        return cell
    
    def cross_cell(arm_width):
        """Cross/plus shape."""
        cell = np.zeros((size, size), dtype=int)
        mid = size // 2
        half = arm_width // 2
        for i in range(size):
            for j in range(size):
                if (abs(i - mid + 0.5) < half or abs(j - mid + 0.5) < half):
                    cell[i, j] = 1
        return make_symmetric_cell(cell)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Rods with various radii
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        candidates.append(rod_cell(r))
    
    # Rings
    for ri, ro in [(0.1, 0.3), (0.2, 0.4), (0.15, 0.35), (0.25, 0.45), (0.1, 0.25)]:
        candidates.append(ring_cell(ri, ro))
    
    # Crosses
    for w in [4, 6, 8, 10, 12]:
        candidates.append(cross_cell(w))
    
    # Random cells
    for _ in range(20):
        candidates.append(random_symmetric_cell())
    
    # Evaluate all candidates with fast surrogate
    best_cell = None
    best_score = -1
    
    for cell in candidates:
        if time.time() > deadline:
            break
        score = fast_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Local search: flip pixels in the symmetric orbit
    if best_cell is not None and time.time() < deadline - 30:
        current = best_cell.copy()
        current_score = best_score
        max_iters = 5000
        iters = 0
        
        # Precompute orbits for efficiency
        # For each pixel, compute its full D4 orbit
        orbits = {}
        for i in range(size):
            for j in range(size):
                orbit = set()
                orbit.add((i, j))
                orbit.add((j, i))
                orbit.add((size-1-i, j))
                orbit.add((i, size-1-j))
                orbit.add((size-1-i, size-1-j))
                orbit.add((size-1-j, i))
                orbit.add((i, size-1-j))
                orbit.add((j, size-1-i))
                orbits[(i, j)] = orbit
        
        # Get unique orbits (each orbit counted once)
        seen_orbits = set()
        unique_orbits = []
        for key, orbit in orbits.items():
            froz = frozenset(orbit)
            if froz not in seen_orbits:
                seen_orbits.add(froz)
                unique_orbits.append(list(orbit))
        
        # Local search
        no_improve_count = 0
        while iters < max_iters and time.time() < deadline - 10:
            iters += 1
            # Pick a random orbit to flip
            orbit = unique_orbits[np.random.randint(len(unique_orbits))]
            
            # Try flipping
            for (i, j) in orbit:
                current[i, j] = 1 - current[i, j]
            
            new_score = fast_gap(current)
            
            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current.copy()
            else:
                # Revert
                for (i, j) in orbit:
                    current[i, j] = 1 - current[i, j]
                no_improve_count += 1
                
                if no_improve_count > 200:
                    # Perturb: flip a few random orbits
                    for _ in range(3):
                        orbit2 = unique_orbits[np.random.randint(len(unique_orbits))]
                        for (i, j) in orbit2:
                            current[i, j] = 1 - current[i, j]
                    no_improve_count = 0
    
    # Also try some targeted designs based on known good photonic crystal structures
    if time.time() < deadline - 20:
        # Honeycomb-like pattern (triangular lattice approximation in square cell)
        # Try a pattern with high contrast
        for trial in range(50):
            if time.time() > deadline:
                break
            # Random restart with local search
            start = random_symmetric_cell()
            s = fast_gap(start)
            if s > best_score:
                best_score = s
                best_cell = start.copy()
            
            # Quick local search
            cur = start.copy()
            cur_s = s
            for _ in range(50):
                if time.time() > deadline:
                    break
                orbit = unique_orbits[np.random.randint(len(unique_orbits))]
                for (i, j) in orbit:
                    cur[i, j] = 1 - cur[i, j]
                ns = fast_gap(cur)
                if ns > cur_s:
                    cur_s = ns
                    if cur_s > best_score:
                        best_score = cur_s
                        best_cell = cur.copy()
                else:
                    for (i, j) in orbit:
                        cur[i, j] = 1 - cur[i, j]
    
    if best_cell is None:
        # Fallback: simple rod
        best_cell = rod_cell(0.25)
    
    return cell_to_list(best_cell)
