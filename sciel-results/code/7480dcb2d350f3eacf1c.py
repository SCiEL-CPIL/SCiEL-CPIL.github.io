"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time
import random

def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    t_start = time.time()
    DEADLINE = 225.0  # safety margin before 230s kill
    
    def time_ok():
        return time.time() - t_start < DEADLINE
    
    # Plane wave setup: 17x17 = 289 plane waves
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289
    
    # K-points in irreducible BZ (30 points as specified)
    pi = np.pi
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)
    
    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2
        
    # Precompute difference indices for Fourier coupling
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            idx_x = int(dGx) + half
            idx_y = int(dGy) + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)
    
    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs
        
    def compute_band_ratio(coeffs):
        """Compute gap-midgap ratio."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            # Build H matrix efficiently
            H_vals = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                if eigvals[3] > max_band4:
                    max_band4 = eigvals[3]
                if eigvals[4] < min_band5:
                    min_band5 = eigvals[4]
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def enforce_symmetry(cell):
        """Enforce 4-fold rotational + reflection symmetry."""
        cell = np.array(cell, dtype=int)
        # 4-fold rotation: 90, 180, 270 degrees
        rot90 = np.rot90(cell, 1)
        rot180 = np.rot90(cell, 2)
        rot270 = np.rot90(cell, 3)
        # Reflections
        hflip = cell[:, ::-1].copy()
        vflip = cell[::-1, :].copy()
        # Transpose
        trans = cell.T.copy()
        # All 8 symmetries
        avg = (cell + rot90 + rot180 + rot270 + hflip + vflip + trans + np.rot90(trans, 1))
        result = (avg >= 4).astype(int)
        return result

    # Fourier series representation with modes |h|+|k| <= 3
    # This gives us a low-dimensional parameter space to optimize
    # Modes: (0,0), (1,0),(0,1),(-1,0),(0,-1), (2,0),(0,2),(-2,0),(0,-2),
    #        (1,1),(1,-1),(-1,1),(-1,-1), (3,0),(0,3),(-3,0),(0,-3),
    #        (2,1),(1,2),(-2,1),(-1,2),(2,-1),(1,-2),(-2,-1),(-1,-2),
    #        (3,1),(1,3),(-3,1),(-1,3),(3,-1),(1,-3),(-3,-1),(-1,-3)
    # But we only need real-valued symmetric modes
    
    # For square symmetry, we can use:
    # DC term, cos(2πx), cos(2πy), cos(2π(x+y)), cos(2π(x-y)),
    # cos(4πx), cos(4πy), cos(4π(x+y)), cos(4π(x-y)),
    # etc. up to the 3rd order harmonics
    
    def generate_fourier_cell(params):
        """
        Generate cell from Fourier series parameters.
        params: array of amplitudes for symmetric Fourier modes
        """
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        # Define symmetric Fourier modes up to |h|+|k| <= 3
        # Using real-valued cosine modes that respect square symmetry
        
        # Order 0: DC
        field = np.zeros((size, size))
        
        # Order 1: (1,0), (0,1) - but these break 4-fold symmetry unless combined
        # Actually for full square symmetry, we need modes that are invariant under 90° rotation
        # cos(2πx) -> cos(2πy) under 90° rotation, so we need equal amplitudes
        # Let's use combinations that are 4-fold symmetric
        
        # 4-fold symmetric modes:
        # f1(x,y) = cos(2πx) + cos(2πy)  [order 1, but this is actually 4-fold symmetric]
        # Wait, cos(2πx) under 90° rotation becomes cos(2πy), so f1 is NOT invariant
        # We need f(x,y) = f(y,x) = f(-x,y) = f(x,-y) etc.
        
        # True 4-fold symmetric modes:
        # cos(2π(x+y)) + cos(2π(x-y)) = 2cos(2πx)cos(2πy)
        # Actually let's think more carefully.
        
        # A function f(x,y) is 4-fold symmetric if f(x,y) = f(y,x) = f(-x,y) = f(x,-y)
        # and f(x,y) = f(y,-x) = f(-y,x) etc.
        
        # Simple 4-fold symmetric basis functions:
        # g1 = cos(2πx)cos(2πy)  [order 2]
        # g2 = cos(4πx) + cos(4πy)  [order 4, but let's stay within |h|+|k|<=3]
        # 
        # For |h|+|k| <= 3, 4-fold symmetric modes:
        # (0,0): 1
        # (1,1), (1,-1), (-1,1), (-1,-1): cos(2π(x±y)) combinations
        #   cos(2π(x+y)) + cos(2π(x-y)) = 2cos(2πx)cos(2πy)
        # (2,0), (0,2), (-2,0), (0,-2): cos(4πx) + cos(4πy) [but |2|+|0|=2 <= 3]
        # (3,0), (0,3), (-3,0), (0,-3): cos(6πx) + cos(6πy) [|3|+|0|=3 <= 3]
        # (2,1), (1,2), etc.: more complex combinations
        
        # Let's define a set of 4-fold symmetric Fourier modes:
        # Mode 0: DC = 1
        # Mode 1: cos(2πx)cos(2πy)  [from (1,1) and (1,-1) pairs]
        # Mode 2: cos(4πx) + cos(4πy)  [from (2,0) and (0,2)]
        # Mode 3: cos(6πx) + cos(6πy)  [from (3,0) and (0,3)]
        # Mode 4: cos(2π(x+y))  [but this is not 4-fold symmetric alone]
        # Actually cos(2π(x+y)) under 90° rotation: cos(2π(y-x)) = cos(2π(x-y))
        # So cos(2π(x+y)) + cos(2π(x-y)) is 4-fold symmetric = 2cos(2πx)cos(2πy)
        
        # Let's use a simpler approach: generate the field as a sum of terms
        # that are explicitly 4-fold symmetric.
        
        # Term 1: DC
        field += params[0] * 1.0
        
        # Term 2: cos(2πx)cos(2πy) - this is 4-fold symmetric
        field += params[1] * np.cos(2*np.pi*X) * np.cos(2*np.pi*Y)
        
        # Term 3: cos(4πx) + cos(4πy) - 4-fold symmetric
        field += params[2] * (np.cos(4*np.pi*X) + np.cos(4*np.pi*Y))
        
        # Term 4: cos(6πx) + cos(6πy) - 4-fold symmetric  
        field += params[3] * (np.cos(6*np.pi*X) + np.cos(6*np.pi*Y))
        
        # Term 5: cos(2π(x+y)) + cos(2π(x-y)) = 2cos(2πx)cos(2πy) - same as term 2
        # Let's use different modes:
        
        # Term 5: cos(4π(x+y)) + cos(4π(x-y)) = 2cos(4πx)cos(4πy)
        field += params[4] * np.cos(4*np.pi*X) * np.cos(4*np.pi*Y)
        
        # Term 6: cos(6π(x+y)) + cos(6π(x-y)) = 2cos(6πx)cos(6πy)
        field += params[5] * np.cos(6*np.pi*X) * np.cos(6*np.pi*Y)
        
        # Term 7: cos(4πx)cos(2πy) + cos(2πx)cos(4πy) - check symmetry
        # Under 90°: cos(4πy)cos(2πx) + cos(2πy)cos(4πx) - same! So 4-fold symmetric
        field += params[6] * (np.cos(4*np.pi*X) * np.cos(2*np.pi*Y) + np.cos(2*np.pi*X) * np.cos(4*np.pi*Y))
        
        # Threshold to binary
        threshold = 0.0
        cell = (field >= threshold).astype(int)
        
        return enforce_symmetry(cell)

    # Phase 1: Initialize with several random Fourier parameter sets
    n_params = 7
    best_ratio = -np.inf
    best_params = None
    best_cell = None
    
    # Try several random initializations
    for trial in range(20):
        if not time_ok():
            break
        
        # Random parameters in reasonable range
        params = np.random.randn(n_params) * 0.3
        params[0] = 0.0  # Keep DC at 0 for balanced dielectric/air
        
        cell = generate_fourier_cell(params)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_params = params.copy()
            best_cell = cell.copy()
    
    if best_cell is None:
        # Fallback: simple disk
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        R = np.sqrt(X**2 + Y**2)
        best_cell = (R <= 0.25).astype(int)
        best_cell = enforce_symmetry(best_cell)
        best_params = np.zeros(n_params)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))

    # Phase 2: Local optimization of Fourier parameters using random perturbation
    current_params = best_params.copy()
    current_ratio = best_ratio
    current_cell = best_cell.copy()
    
    search_start = time.time()
    search_time_limit = 120.0
    
    step_sizes = [0.1, 0.05, 0.02, 0.01]
    current_step_idx = 0
    
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
        # Random perturbation
        step = step_sizes[current_step_idx]
        perturbation = np.random.randn(n_params) * step
        new_params = current_params + perturbation
        
        # Keep DC near 0
        new_params[0] = current_params[0]
        
        new_cell = generate_fourier_cell(new_params)
        coeffs = get_fourier_coeffs(new_cell)
        new_ratio = compute_band_ratio(coeffs)
        
        if new_ratio > current_ratio + 1e-8:
            current_params = new_params
            current_ratio = new_ratio
            current_cell = new_cell
            no_improve_count = 0
            # Occasionally reduce step size
            if no_improve_count == 0 and random.random() < 0.1 and current_step_idx < len(step_sizes) - 1:
                current_step_idx += 1
        else:
            no_improve_count += 1
            # Increase step size if stuck
            if no_improve_count > 15 and current_step_idx > 0:
                current_step_idx -= 1
                no_improve_count = 0

    # Phase 3: Pixel-level refinement
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 40:
        # Identify independent pixels (top-left quadrant)
        indep_pixels = [(i, j) for i in range(16) for j in range(16)]
        random.shuffle(indep_pixels)
        
        refine_start = time.time()
        refine_limit = 30.0
        
        no_improve_refine = 0
        max_refine_no_improve = 80
        
        for (i, j) in indep_pixels:
            if time.time() - refine_start > refine_limit or not time_ok():
                break
            
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                no_improve_refine = 0
            else:
                no_improve_refine += 1
                if no_improve_refine > max_refine_no_improve:
                    break

    # Final verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)
    
    if final_ratio < best_ratio - 0.01:
        current_cell = best_cell
    else:
        best_ratio = final_ratio
        
    return current_cell.tolist()
