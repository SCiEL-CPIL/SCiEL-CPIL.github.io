"""Design a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5."""
import numpy as np
import time
from itertools import product

def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    start_time = time.time()
    deadline = start_time + 190.0  # Leave safety margin before 200s limit

    # Precompute pixel coordinates centered at origin
    coords = np.array([(i + 0.5) / size - 0.5 for i in range(size)])
    X, Y = np.meshgrid(coords, coords, indexing='ij')
    
    # Define key Fourier modes for the dielectric distribution
    # Modes that couple to bands 4-5 in a square lattice
    # Using modes: (1,0), (0,1), (2,0), (0,2), (1,1), (2,2), (1,-1)
    modes = [(1,0), (0,1), (2,0), (0,2), (1,1), (2,2), (1,-1), (2,-1), (-1,2), (-2,1)]
    
    # Precompute mode functions
    mode_funcs = []
    for (mx, my) in modes:
        mode_funcs.append(np.cos(2*np.pi*mx*X) * np.cos(2*np.pi*my*Y) + 
                          np.sin(2*np.pi*mx*X) * np.sin(2*np.pi*my*Y))
    mode_funcs = np.array(mode_funcs)  # shape: (n_modes, 32, 32)
    
    n_modes = len(modes)
    
    # Initial parameters: amplitudes and bias
    # Start with a reasonable initial guess
    best_params = np.zeros(n_modes + 1)
    best_params[0] = 0.5  # bias
    best_params[1] = 0.3  # (1,0) mode
    best_params[2] = 0.3  # (0,1) mode
    best_params[3] = 0.2  # (2,0) mode
    best_params[4] = 0.2  # (0,2) mode
    best_params[5] = 0.1  # (1,1) mode
    best_params[6] = 0.1  # (2,2) mode
    
    def params_to_cell(params):
        """Convert parameter vector to a 32x32 binary cell."""
        bias = params[0]
        amplitudes = params[1:]
        field = bias + amplitudes @ mode_funcs  # shape: (32, 32)
        cell = (field > 0.5).astype(int)
        # Ensure symmetry: enforce all four symmetries
        cell = np.minimum(cell, cell[::-1, :])
        cell = np.minimum(cell, cell[:, ::-1])
        cell = np.minimum(cell, cell.T)
        return cell.tolist()
    
    def compute_score(cell):
        """Compute the gap-midgap ratio using plane wave method."""
        n_pw = 17  # 17x17 = 289 plane waves
        # Build reciprocal lattice vectors
        kx_vals = np.arange(-n_pw//2, n_pw//2 + 1)
        ky_vals = np.arange(-n_pw//2, n_pw//2 + 1)
        KX, KY = np.meshgrid(kx_vals, ky_vals, indexing='ij')
        KX = KX.ravel()
        KY = KY.ravel()
        n_K = len(KX)
        
        # Fourier coefficients of epsilon
        # epsilon(r) = eps_low + (eps_high - eps_low) * cell(r)
        eps_low = 1.0
        eps_high_val = eps_high
        eps_diff = eps_high_val - eps_low
        
        # Compute Fourier coefficients
        eps_FK = np.zeros(n_K, dtype=complex)
        for idx in range(n_K):
            kx = KX[idx]
            ky = KY[idx]
            if kx == 0 and ky == 0:
                eps_FK[idx] = eps_low + eps_diff * np.mean(cell)
            else:
                # Fourier coefficient: (1/N^2) sum cell(i,j) exp(-2pi i (kx*i/N + ky*j/N))
                phase = np.exp(-2j*np.pi*(kx*X + ky*Y))
                eps_FK[idx] = eps_low * (0 if (kx != 0 or ky != 0) else 1) + eps_diff * np.mean(cell * phase)
        
        # Define wave vectors in irreducible Brillouin zone
        # Square lattice BZ: [0, 1] x [0, 1] in units of 2pi
        # Irreducible zone: 0 <= ky <= kx, 0 <= kx <= 1
        wave_vectors = []
        # Along Gamma-X: (k, 0) for k = 0, 0.1, 0.2, ..., 1.0
        for k in np.linspace(0, 1, 11):
            wave_vectors.append((k, 0))
        # Along X-M: (1, k) for k = 0, 0.1, ..., 0.5
        for k in np.linspace(0, 0.5, 6):
            wave_vectors.append((1, k))
        # Along M-Gamma: (k, k) for k = 0, 0.1, ..., 0.5
        for k in np.linspace(0, 0.5, 6):
            wave_vectors.append((k, k))
        # Along Gamma-Y: (0, k) for k = 0, 0.1, ..., 1.0
        for k in np.linspace(0, 1, 11):
            wave_vectors.append((0, k))
        
        # Remove duplicates
        unique_wv = []
        seen = set()
        for wv in wave_vectors:
            key = (round(wv[0], 4), round(wv[1], 4))
            if key not in seen:
                seen.add(key)
                unique_wv.append(wv)
        
        max_omega_4 = 0
        min_omega_5 = float('inf')
        
        for (kx, ky) in unique_wv:
            # Solve eigenvalue problem for TM mode
            # H = D * E, where D is the dielectric matrix
            # For TM: (k + G)^2 * eps(k-G) * E_G = omega^2 * E_G
            # In matrix form: M * E = omega^2 * E
            # where M_G,G' = (k+G)^2 * eps(k-G-G')
            
            # Build the matrix
            M = np.zeros((n_K, n_K), dtype=complex)
            for i in range(n_K):
                ki_x = KX[i] + kx
                ki_y = KY[i] + ky
                ki_sq = ki_x**2 + ki_y**2
                for j in range(n_K):
                    # G_diff = G_i - G_j
                    Gdx = KX[i] - KX[j]
                    Gdy = KY[i] - KY[j]
                    # Find index of G_diff in our plane wave set
                    # Check if G_diff is in our set
                    found = False
                    for l in range(n_K):
                        if KX[l] == Gdx and KY[l] == Gdy:
                            M[i, j] = ki_sq * eps_FK[l]
                            found = True
                            break
                    if not found:
                        M[i, j] = 0
            
            # Solve eigenvalue problem
            # M is Hermitian, so use eigh
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
                eigvals = np.real(eigvals)
                # Sort eigenvalues
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                
                # Find bands 4 and 5 (0-indexed: 3 and 4)
                if len(eigvals) >= 5:
                    omega_4 = np.sqrt(eigvals[3])
                    omega_5 = np.sqrt(eigvals[4])
                    max_omega_4 = max(max_omega_4, omega_4)
                    min_omega_5 = min(min_omega_5, omega_5)
            except:
                continue
        
        if max_omega_4 == 0 or min_omega_5 == float('inf'):
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2
        if midgap == 0:
            return -1.0
        return 2 * gap / midgap
    
    # Evaluate initial design
    best_cell = params_to_cell(best_params)
    best_score = compute_score(best_cell)
    
    # Local optimization using coordinate descent
    step_sizes = [0.05] * (n_modes + 1)
    n_iterations = 0
    
    while time.time() < deadline:
        improved = False
        for dim in range(n_modes + 1):
            if time.time() > deadline:
                break
            
            # Try positive and negative steps
            for direction in [1, -1]:
                new_params = best_params.copy()
                new_params[dim] += direction * step_sizes[dim]
                
                # Clamp bias to [0, 1] and amplitudes to [-0.5, 0.5]
                new_params[0] = np.clip(new_params[0], 0, 1)
                for d in range(1, n_modes + 1):
                    new_params[d] = np.clip(new_params[d], -0.5, 0.5)
                
                new_cell = params_to_cell(new_params)
                new_score = compute_score(new_cell)
                
                if new_score > best_score:
                    best_params = new_params
                    best_cell = new_cell
                    best_score = new_score
                    improved = True
                    break
        
        if not improved:
            # Reduce step sizes
            step_sizes = [s * 0.5 for s in step_sizes]
            if step_sizes[0] < 0.001:
                break
        
        n_iterations += 1
    
    return best_cell

