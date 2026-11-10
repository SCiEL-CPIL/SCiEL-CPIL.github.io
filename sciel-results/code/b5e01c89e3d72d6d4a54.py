"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _compute_fourier_coefficients(cell, size, n_pw):
    """Compute exact Fourier coefficients of the dielectric permittivity."""
    eps = np.array(cell, dtype=np.float64)
    eps = 1.0 + (12.0 - 1.0) * eps  # eps_low=1, eps_high=12
    
    # Compute Fourier coefficients for plane waves with indices in [-n_pw, n_pw]
    k_vals = np.arange(-n_pw, n_pw + 1)
    coeffs = np.zeros((len(k_vals), len(k_vals)), dtype=np.complex128)
    
    for i, kx in enumerate(k_vals):
        for j, ky in enumerate(k_vals):
            if kx == 0 and ky == 0:
                coeffs[i, j] = eps.mean()
            else:
                # Fourier coefficient: sum over pixels of eps(x,y) * exp(-i 2π (kx*x + ky*y)) / N^2
                x = (np.arange(size) + 0.5) / size - 0.5
                y = (np.arange(size) + 0.5) / size - 0.5
                X, Y = np.meshgrid(x, y, indexing='ij')
                phase = np.exp(-1j * 2 * np.pi * (kx * X + ky * Y))
                coeffs[i, j] = np.sum(eps * phase) / (size * size)
    
    return coeffs, k_vals

def _solve_tm_bands(coeffs, k_vals, n_pw, kx, ky, n_bands=5):
    """Solve for TM bands at given wave vector using plane-wave Galerkin method."""
    # For TM polarization: (k - G)^2 * eps^{-1}(G-G') * E(G') = omega^2 E(G)
    # Build the eigenvalue problem
    
    n = len(k_vals)
    # Precompute (k-G)^2 for each plane wave
    kg2 = np.zeros(n)
    for i, kxi in enumerate(k_vals):
        for j, kyi in enumerate(k_vals):
            pass  # We'll handle this in the matrix construction
    
    # Actually, let's build the matrix properly
    # For each pair of plane waves G_i, G_j, the matrix element is:
    # (k - G_i)^2 * sum_{G'} eps^{-1}(G_i - G') * delta_{G', G_j}
    
    # First, build eps_inv in Fourier space
    # eps_inv(G) = 1/eps(-G) (since 1/eps is also periodic)
    # But we need the convolution: A_{ij} = (k-G_i)^2 * (eps^{-1})_{ij}
    
    # Let's compute eps_inv coefficients
    eps_inv_coeffs = np.zeros((n, n), dtype=np.complex128)
    for i, kxi in enumerate(k_vals):
        for j, kyi in enumerate(k_vals):
            # eps_inv(kx, ky) = 1/eps(-kx, -ky)
            # Find index for (-kx, -ky)
            ikx = k_vals.index(-kxi) if -kxi in k_vals else None
            iky = k_vals.index(-kyi) if -kyi in k_vals else None
            if ikx is not None and iky is not None:
                eps_inv_coeffs[i, j] = 1.0 / coeffs[ikx, iky] if coeffs[ikx, iky] != 0 else 0.0
    
    # Build the eigenvalue matrix
    # H_{ij} = (k - G_i)^2 * eps_inv_coeffs[i, j]
    # But this isn't quite right for the Galerkin method.
    
    # For TM: The equation is (k - G)^2 * eps^{-1}(G - G') E(G') = omega^2 E(G)
    # In matrix form: sum_{G'} [(k - G)^2 * eps^{-1}(G - G')] E(G') = omega^2 E(G)
    
    # So H_{i,j} = (k - G_i)^2 * eps^{-1}(G_i - G_j)
    
    H = np.zeros((n, n), dtype=np.complex128)
    for i, kxi in enumerate(k_vals):
        for j, kyi in enumerate(k_vals):
            pass
    
    # Let me redo this more carefully
    # G_i = (k_vals[i], 0) + (0, k_vals[j])? No, each plane wave is 2D.
    # Actually, for a 2D problem with square lattice, plane waves are indexed by (mx, my)
    # where G = (mx * 2π, my * 2π)
    
    # Let me restructure: use 2D indices
    mx_vals = np.arange(-n_pw, n_pw + 1)
    my_vals = np.arange(-n_pw, n_pw + 1)
    n_m = len(mx_vals)
    n_total = n_m * n_m
    
    # Flatten index: idx = mx_idx * n_m + my_idx
    def idx2d(mx_idx, my_idx):
        return mx_idx * n_m + my_idx
    
    # Recompute coeffs with proper 2D indexing
    coeffs2d = np.zeros((n_m, n_m), dtype=np.complex128)
    x = (np.arange(size) + 0.5) / size - 0.5
    y = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for i, mx in enumerate(mx_vals):
        for j, my in enumerate(my_vals):
            if mx == 0 and my == 0:
                coeffs2d[i, j] = eps.mean()
            else:
                phase = np.exp(-1j * 2 * np.pi * (mx * X + my * Y))
                coeffs2d[i, j] = np.sum(eps * phase) / (size * size)
    
    # Build eps_inv in 2D
    eps_inv2d = np.zeros((n_m, n_m), dtype=np.complex128)
    for i, mx in enumerate(mx_vals):
        for j, my in enumerate(my_vals):
            # eps_inv(mx, my) = 1/eps(-mx, -my)
            imx = mx_vals.index(-mx) if -mx in mx_vals else None
            imy = my_vals.index(-my) if -my in my_vals else None
            if imx is not None and imy is not None:
                val = coeffs2d[imx, imy]
                eps_inv2d[i, j] = 1.0 / val if abs(val) > 1e-15 else 0.0
    
    # Now build the Hamiltonian matrix for wave vector (kx, ky)
    # H_{(mx1,my1),(mx2,my2)} = (k - G1)^2 * eps_inv2d[mx1-mx2, my1-my2]
    # where G1 = (2π*mx1, 2π*my1)
    
    H = np.zeros((n_total, n_total), dtype=np.complex128)
    for i in range(n_m):
        for j in range(n_m):
            mx1, my1 = mx_vals[i], my_vals[j]
            G1x, G1y = 2 * np.pi * mx1, 2 * np.pi * my1
            kg1x, kg1y = kx - G1x, ky - G1y
            kg1_sq = kg1x**2 + kg1y**2
            
            for i2 in range(n_m):
                for j2 in range(n_m):
                    mx2, my2 = mx_vals[i2], my_vals[j2]
                    dmx, dmy = mx1 - mx2, my1 - my2
                    # Get eps_inv2d[dmx, dmy]
                    imx = mx_vals.index(dmx) if dmx in mx_vals else None
                    imy = my_vals.index(dmy) if dmy in my_vals else None
                    if imx is not None and imy is not None:
                        H[idx2d(i, j), idx2d(i2, j2)] = kg1_sq * eps_inv2d[imx, imy]
    
    # Solve eigenvalue problem
    eigvals = np.linalg.eigvalsh(H)
    eigvals = np.sort(eigvals.real)
    
    # omega^2 = eigvals, so omega = sqrt(eigvals)
    omega = np.sqrt(np.maximum(eigvals, 0))
    
    return omega[:n_bands]

def _evaluate_band_gap(cell, size, n_pw=6, n_bands=5):
    """Compute the gap-midgap ratio between TM bands 4 and 5."""
    # Irreducible Brillouin zone for square lattice
    # BZ: 0 <= kx <= ky <= pi (for lattice constant 1, reciprocal lattice constant 2π)
    # Key points: Gamma (0,0), M (π,0), X (π/2, π/2), and paths between them
    
    # Sample 30 wave vectors in the irreducible BZ
    kx_vals = []
    ky_vals = []
    
    # Along Gamma to X: (0,0) to (π/2, π/2)
    for t in np.linspace(0, 1, 10):
        kx_vals.append(np.pi / 2 * t)
        ky_vals.append(np.pi / 2 * t)
    
    # Along X to M: (π/2, π/2) to (π, 0)
    for t in np.linspace(0, 1, 10):
        kx_vals.append(np.pi / 2 + np.pi / 2 * t)
        ky_vals.append(np.pi / 2 * (1 - t))
    
    # Along M to Gamma: (π, 0) to (0, 0)
    for t in np.linspace(0, 1, 10):
        kx_vals.append(np.pi * (1 - t))
        ky_vals.append(0)
    
    # Compute bands at each k point
    max_omega4 = -np.inf
    min_omega5 = np.inf
    
    for kx, ky in zip(kx_vals, ky_vals):
        omega = _solve_tm_bands(cell, size, n_pw, kx, ky, n_bands=5)
        if len(omega) >= 5:
            max_omega4 = max(max_omega4, omega[3])  # 4th band (0-indexed: 3)
            min_omega5 = min(min_omega5, omega[4])  # 5th band (0-indexed: 4)
    
    if max_omega4 <= 0 or min_omega5 <= 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    
    if midgap <= 0:
        return -1.0
    
    return gap / midgap

def _make_symmetric(cell, size):
    """Ensure the cell has 4-fold rotational and reflection symmetry."""
    cell = np.array(cell, dtype=np.int32)
    # Apply 90-degree rotation symmetry: cell[i][j] = cell[j][size-1-i] = cell[size-1-i][size-1-j] = cell[size-1-j][i]
    # For each orbit under D4 symmetry, set all pixels to the majority value
    for i in range(size):
        for j in range(size):
            # Get all symmetric equivalents
            equivs = [(i, j), (j, size-1-i), (size-1-i, size-1-j), (size-1-j, i)]
            vals = [cell[e[0], e[1]] for e in equivs]
            majority = 1 if sum(vals) >= 2 else 0
            for e in equivs:
                cell[e[0], e[1]] = majority
    return cell.tolist()

def design(problem):
    """Design a photonic crystal unit cell with large TM band gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Start with the seed design
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            row.append(1 if x*x + y*y < 0.2**2 else 0)
        cell.append(row)
    
    # Ensure symmetry
    cell = _make_symmetric(cell, size)
    
    # Evaluate initial design
    start_time = time.time()
    deadline = start_time + 220  # 220s deadline with 10s margin
    
    best_cell = [row[:] for row in cell]
    best_score = _evaluate_band_gap(cell, size, n_pw=6, n_bands=5)
    
    # Hill climbing: try flipping pixels
    np.random.seed(42)
    
    # Get all unique orbits under D4 symmetry
    orbits = []
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            # Get orbit
            orbit = [(i, j), (j, size-1-i), (size-1-i, size-1-j), (size-1-j, i)]
            orbit = list(set(orbit))
            orbits.append(orbit)
            for pos in orbit:
                visited.add(pos)
    
    # Shuffle orbits for randomization
    np.random.shuffle(orbits)
    
    current_cell = [row[:] for row in cell]
    current_score = best_score
    
    iterations = 0
    while time.time() < deadline - 30:  # Keep 30s margin
        iterations += 1
        
        # Try flipping a random orbit
        if len(orbits) == 0:
            break
        
        orbit_idx = np.random.randint(len(orbits))
        orbit = orbits[orbit_idx]
        
        # Get current value
        current_val = current_cell[orbit[0][0]][orbit[0][1]]
        new_val = 1 - current_val
        
        # Apply flip
        for (i, j) in orbit:
            current_cell[i][j] = new_val
        
        # Evaluate
        new_score = _evaluate_band_gap(current_cell, size, n_pw=6, n_bands=5)
        
        # If better, accept
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = [row[:] for row in current_cell]
        else:
            # Revert
            for (i, j) in orbit:
                current_cell[i][j] = current_val
    
    return best_cell
# EVOLVE-BLOCK-END
