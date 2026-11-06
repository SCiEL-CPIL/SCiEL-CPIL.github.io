"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """Design a 4-rod photonic crystal unit cell to maximize the TM 4-5 band gap."""
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 195.0  # Safety margin before 230s kill

    def make_cell(cx, cy, radius):
        """Create a symmetric 4-rod cell."""
        cell = np.zeros((size, size), dtype=np.uint8)
        # The four rods are at (cx, cy), (-cx, cy), (cx, -cy), (-cx, -cy)
        # In pixel coordinates (0 to size-1), center is size/2
        # We want the rods to be at fractional positions cx, cy relative to center
        # Pixel centers are at (i+0.5)/size - 0.5
        
        # Efficient vectorized generation
        x_coords = (np.arange(size) + 0.5) / size - 0.5
        y_coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Distances to the four rod centers
        # Note: cx, cy are fractions of the lattice constant
        d1 = np.sqrt((X - cx)**2 + (Y - cy)**2)
        d2 = np.sqrt((X + cx)**2 + (Y - cy)**2)
        d3 = np.sqrt((X - cx)**2 + (Y + cy)**2)
        d4 = np.sqrt((X + cx)**2 + (Y + cy)**2)
        
        mask = (d1 < radius) | (d2 < radius) | (d3 < radius) | (d4 < radius)
        cell[mask] = 1
        
        # Enforce symmetry explicitly to handle floating point edge cases
        # The construction is theoretically symmetric, but let's ensure strict binary symmetry
        cell = (cell + cell[::-1, ::-1]) // 2
        cell = (cell + cell.T) // 2
        
        return cell.tolist()

    def fast_score(cell_list):
        """
        Fast approximation of the band gap score.
        Uses a reduced plane-wave basis for speed.
        This is a proxy for the full 289-pw calculation.
        """
        cell = np.array(cell_list, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        
        # Compute Fourier coefficients of epsilon
        # We need eps_G for G in the basis
        N = 9
        G_vals = np.fft.fftfreq(N, d=1.0/N)
        # G vectors are 2*pi * G_vals
        G2 = np.zeros((N, N))
        eps_f = np.zeros((N, N), dtype=complex)
        
        # Precompute pixel coordinates
        px = (np.arange(size) + 0.5) / size - 0.5
        py = (np.arange(size) + 0.5) / size - 0.5
        PX, PY = np.meshgrid(px, py, indexing='xy')
        
        for i in range(N):
            for j in range(N):
                kx = 2 * np.pi * G_vals[i]
                ky = 2 * np.pi * G_vals[j]
                G2[i, j] = kx**2 + ky**2
                # Calculate Fourier coefficient: (1/A) * sum_eps * exp(-i G . r)
                phase = np.exp(-1j * (kx * PX + ky * PY))
                eps_f[i, j] = np.sum(eps * phase) / (size * size)
                
        # Find band gap
        # We check a few critical k-points in the irreducible BZ
        # Irreducible BZ of square lattice: 0 <= ky <= kx <= 0.5 (in units of 2pi/a)
        # Key points: Gamma (0,0), M (0.5, 0.5), X (0.5, 0)
        k_points = [
            (0.0, 0.0),    # Gamma
            (0.5, 0.5),    # M
            (0.5, 0.0),    # X
            (0.25, 0.25),  # Midpoint Gamma-M
            (0.5, 0.25),   # Midpoint M-X
            (0.25, 0.0),   # Midpoint Gamma-X
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        # Basis indices for N=9: -4 to 4
        indices = np.arange(-N//2, N//2 + 1)
        
        for kx, ky in k_points:
            # kx, ky are in units of 2pi/a
            # Actual k vector: k = 2*pi*(kx, ky)
            # G vector: G = 2*pi*(Gx, Gy) where Gx, Gy are integers
            # The eigenvalue problem for TM:
            # sum_G' eps_{G-G'} |G'+k|^2 c_{G'} = omega^2 |G+k|^2 c_G
            # This is a generalized eigenvalue problem or can be transformed.
            # Let's use the form:
            # A c = omega^2 c, where A_ij = eps_{G_i - G_j} * |G_j + k|^2 / |G_i + k|^2 ?
            # No, the standard form is:
            # sum_G' eps_{G-G'} (G'+k)^2 c_{G'} = omega^2 (G+k)^2 c_G
            # This is not a standard symmetric eigenvalue problem directly because of the (G+k)^2 on the RHS.
            # However, we can transform it. Let d_G = c_G * |G+k|.
            # Then sum_G' eps_{G-G'} (G'+k)^2 d_{G'}/|G'+k| = omega^2 |G+k| d_G
            # This is getting complicated. Let's use the simpler symmetric form if possible.
            # Actually, for TM, the matrix is:
            # M_{ij} = eps_{G_i - G_j} * |G_j + k|^2
            # And we solve M c = omega^2 * D c, where D is diagonal with D_ii = |G_i + k|^2.
            # This is a generalized eigenvalue problem: M c = lambda D c.
            # We can transform to symmetric standard form:
            # D^{-1/2} M D^{-1/2} y = lambda y, where c = D^{-1/2} y.
            
            N_basis = N
            G_vecs = []
            for i in range(N_basis):
                for j in range(N_basis):
                    G_vecs.append((indices[i], indices[j]))
                    
            # This is too slow for N=9 (81x81) in a Python loop for many k-points and many cells.
            # Let's stick to a very small basis for the proxy score, e.g., N=5 (25 plane waves).
            # Or even better, just check the gap at Gamma and M using a small basis.
            
            # Let's try N=5 (indices -2 to 2)
            N_small = 5
            idx_small = np.arange(-N_small//2, N_small//2 + 1)
            G_small = []
            for i in idx_small:
                for j in idx_small:
                    G_small.append((i, j))
            
            M = np.zeros((len(G_small), len(G_small)), dtype=complex)
            D = np.zeros(len(G_small))
            
            for i, (gx_i, gy_i) in enumerate(G_small):
                # G_i + k
                kgx = gx_i + kx
                kgy = gy_i + ky
                kg2 = kgx**2 + kgy**2
                if kg2 < 1e-10:
                    kg2 = 1e-10
                D[i] = kg2
                
                for j, (gx_j, gy_j) in enumerate(G_small):
                    dgx = gx_i - gx_j
                    dgy = gy_i - gy_j
                    # Get eps_f[ dgx, dgy ]
                    # Map dgx, dgy to index in N=9 grid
                    # Index in N=9: i = (dgx + 4) % 9
                    ix = (int(dgx) + 4) % 9
                    iy = (int(dgy) + 4) % 9
                    M[i, j] = eps_f[ix, iy] * D[j]
            
            # Solve generalized eigenvalue problem: M c = lambda D c
            # Transform to symmetric: A = D^{-1/2} M D^{-1/2}
            D_inv_sqrt = 1.0 / np.sqrt(D)
            A = M * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]
            A = (A + A.conj().T) / 2.0 # Ensure Hermitian
            
            try:
                evals = np.linalg.eigvalsh(A)
                evals = np.real(evals)
                evals = np.sort(evals)
                # We need the 4th and 5th bands (indices 3 and 4)
                if len(evals) >= 5:
                    omega4 = np.sqrt(evals[3])
                    omega5 = np.sqrt(evals[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return -1.0
        
        gap = 2 * (min_omega5 - max_omega4)
        midgap = min_omega5 + max_omega4
        if midgap == 0:
            return -1.0
        return gap / midgap

    # Search Strategy
    best_cell = None
    best_score = -np.inf
    
    # Coarse Grid Search
    # Radius: 0.05 to 0.25
    # Position: 0.15 to 0.30 (cx=cy due to symmetry)
    
    radii = np.arange(0.05, 0.26, 0.05)
    positions = np.arange(0.15, 0.31, 0.05)
    
    for r in radii:
        if time.time() - start_time > time_limit - 20:
            break
        for pos in positions:
            if time.time() - start_time > time_limit - 20:
                break
            cell = make_cell(pos, pos, r)
            score = fast_score(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
                best_r = r
                best_pos = pos
                
    # Fine-tune around best
    if best_cell is not None:
        # Fine grid around best_r, best_pos
        fine_radii = np.arange(best_r - 0.02, best_r + 0.03, 0.01)
        fine_positions = np.arange(best_pos - 0.02, best_pos + 0.03, 0.01)
        
        for r in fine_radii:
            if time.time() - start_time > time_limit - 5:
                break
            for pos in fine_positions:
                if time.time() - start_time > time_limit - 5:
                    break
                cell = make_cell(pos, pos, r)
                score = fast_score(cell)
                if score > best_score:
                    best_score = score
                    best_cell = cell
                    best_r = r
                    best_pos = pos

    if best_cell is None:
        # Fallback: Standard 4-rod
        best_cell = make_cell(0.25, 0.25, 0.15)
        
    return best_cell
# EVOLVE-BLOCK-END
