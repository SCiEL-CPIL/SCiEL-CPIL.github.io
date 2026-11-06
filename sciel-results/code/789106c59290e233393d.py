import time
import numpy as np
from numpy.linalg import eigh
import math

def _enforce_d4(cell):
    """Enforce D4 symmetry by averaging over all 8 transformations."""
    N = len(cell)
    cell = np.array(cell, dtype=float)
    # All 8 D4 transformations
    transforms = [
        cell,
        cell[:, ::-1],        # mirror x
        cell[::-1, :],        # mirror y
        cell[::-1, ::-1],     # 180
        cell.T,               # transpose
        cell.T[:, ::-1],      # transpose + mirror x
        cell.T[::-1, :],      # transpose + mirror y
        cell.T[::-1, ::-1],   # transpose + 180
    ]
    averaged = np.mean(transforms, axis=0)
    return (averaged > 0.5).astype(int)

def _compute_tm_gap(cell, n_k=12, pw_radius=12):
    """
    Compute the TM gap-midgap ratio using plane-wave expansion.
    """
    N = len(cell)
    eps = np.array(cell, dtype=float)
    eps = 1.0 + 11.0 * eps  # epsilon = 1 + 11 * dielectric
    inv_eps = 1.0 / eps
    
    # Fourier coefficients of 1/eps
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    # Generate plane wave vectors
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    # Precompute the difference matrix indices
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    # Fourier coefficient matrix: element (i,j) = fft_inv[G_i - G_j]
    dqx = Gx_diff.astype(int)
    dqy = Gy_diff.astype(int)
    inv_eps_mat = fft_inv[dqx % N, dqy % N]
    
    # Key points in the irreducible Brillouin zone (square lattice)
    # Gamma (0,0), M (0.5, 0), X (0, 0.5), K (0.5, 0.5)
    k_points = [
        (0.0, 0.0), 
        (0.25, 0.0), 
        (0.5, 0.0), 
        (0.5, 0.25),
        (0.5, 0.5), 
        (0.4, 0.4),
        (0.3, 0.3),
        (0.2, 0.2),
        (0.15, 0.15),
        (0.35, 0.35),
        (0.45, 0.45),
        (0.4, 0.45),
    ][:n_k]
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2 * (min_w5 - max_w4) / (min_w5 + max_w4)

def _make_rod(radius, center=(0.5, 0.5), N=32):
    """Create a circular rod in the unit cell."""
    cell = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - center[0]
            y = (j + 0.5) / N - center[1]
            if x*x + y*y < radius*radius:
                cell[i, j] = 1
    return cell

def _make_square(half_width, center=(0.5, 0.5), N=32):
    """Create a square hole/rod in the unit cell."""
    cell = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - center[0]
            y = (j + 0.5) / N - center[1]
            if abs(x) < half_width and abs(y) < half_width:
                cell[i, j] = 1
    return cell

def _make_ring(inner_radius, outer_radius, N=32):
    """Create a ring (annulus) in the unit cell."""
    cell = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            r = math.sqrt(x*x + y*y)
            if inner_radius <= r <= outer_radius:
                cell[i, j] = 1
    return cell

def _fourier_cell(coeffs, N=32):
    """
    Generate a cell from C4v-symmetric Fourier coefficients.
    coeffs: [bg, c1, c2, c3, c4, c5]
    """
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    x = (x + 0.5) / N - 0.5
    y = (y + 0.5) / N - 0.5
    
    profile = np.zeros((N, N))
    
    if len(coeffs) > 0:
        profile += coeffs[0]
    if len(coeffs) > 1:
        profile += coeffs[1] * (np.cos(2 * np.pi * 2 * x) + np.cos(2 * np.pi * 2 * y))
    if len(coeffs) > 2:
        profile += coeffs[2] * (np.cos(2 * np.pi * (x + y)) + np.cos(2 * np.pi * (x - y)))
    if len(coeffs) > 3:
        profile += coeffs[3] * (np.cos(2 * np.pi * 4 * x) + np.cos(2 * np.pi * 4 * y))
    if len(coeffs) > 4:
        profile += coeffs[4] * (np.cos(2 * np.pi * (2*x + 2*y)) + np.cos(2 * np.pi * (2*x - 2*y)))
    if len(coeffs) > 5:
        profile += coeffs[5] * (np.cos(2 * np.pi * 6 * x) + np.cos(2 * np.pi * 6 * y))
    
    cell = (profile > 0.5).astype(int)
    return cell

def design(problem):
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin for 200s limit
    
    N = 32
    best_cell = None
    best_score = -10.0
    
    # Phase 1: Evaluate geometric seeds
    seeds = []
    
    # Circular rods of various radii
    for r in [0.15, 0.20, 0.25, 0.30, 0.35]:
        seeds.append(_make_rod(r))
    
    # Square rods
    for hw in [0.15, 0.20, 0.25, 0.30]:
        seeds.append(_make_square(hw))
    
    # Rings
    for ir, orr in [(0.15, 0.30), (0.20, 0.35), (0.10, 0.25), (0.25, 0.40)]:
        seeds.append(_make_ring(ir, orr))
    
    # Fourier-generated seeds
    np.random.seed(42)
    for _ in range(15):
        coeffs = [
            np.random.uniform(0.3, 0.7),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1)
        ]
        seeds.append(_fourier_cell(coeffs))
    
    # Evaluate seeds
    for cell in seeds:
        if time.time() > deadline - 60:
            break
        cell = np.array(_enforce_d4(cell.tolist()), dtype=int)
        try:
            score = _compute_tm_gap(cell, n_k=8, pw_radius=10)
            if score > best_score:
                best_score = score
                best_cell = cell
        except:
            pass
    
    # Phase 2: Local search with pixel mutations
    if best_cell is not None:
        current_cell = best_cell.copy()
        current_score = best_score
        no_improve = 0
        max_no_improve = 30
        
        while time.time() < deadline - 20 and no_improve < max_no_improve:
            # Try flipping 1-3 random pixels
            new_cell = current_cell.copy()
            n_flips = np.random.randint(1, 4)
            for _ in range(n_flips):
                ri = np.random.randint(0, N)
                rj = np.random.randint(0, N)
                new_cell[ri, rj] = 1 - new_cell[ri, rj]
            
            new_cell = np.array(_enforce_d4(new_cell.tolist()), dtype=int)
            
            if np.array_equal(new_cell, current_cell):
                continue
            
            try:
                new_score = _compute_tm_gap(new_cell, n_k=6, pw_radius=8)
            except:
                new_score = -10.0
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current_cell.copy()
            else:
                no_improve += 1
    
    # Phase 3: Fourier space optimization
    if time.time() < deadline - 30:
        best_fourier_score = best_score
        best_fourier_cell = best_cell.copy() if best_cell is not None else None
        
        # Try several Fourier starting points
        fourier_starts = [
            [0.5, 0.1, 0.0, -0.1, 0.0, 0.0],
            [0.5, -0.1, 0.0, 0.1, 0.0, 0.0],
            [0.4, 0.2, 0.1, -0.15, 0.05, 0.0],
            [0.6, 0.0, 0.2, 0.0, 0.0, -0.1],
        ]
        
        for init_coeffs in fourier_starts:
            if time.time() > deadline - 10:
                break
            
            current_coeffs = init_coeffs.copy()
            current_cell = np.array(_enforce_d4(_fourier_cell(current_coeffs).tolist()), dtype=int)
            
            try:
                current_score = _compute_tm_gap(current_cell, n_k=6, pw_radius=8)
            except:
                current_score = -10.0
            
            if current_score > best_fourier_score:
                best_fourier_score = current_score
                best_fourier_cell = current_cell.copy()
            
            # Local search in Fourier space
            no_improve = 0
            step_size = 0.08
            
            while time.time() < deadline - 5 and no_improve < 25:
                mutated_coeffs = current_coeffs.copy()
                for i in range(len(mutated_coeffs)):
                    if np.random.random() < 0.8:
                        mutated_coeffs[i] += np.random.normal(0, step_size)
                
                mutated_cell = np.array(_enforce_d4(_fourier_cell(mutated_coeffs).tolist()), dtype=int)
                
                try:
                    new_score = _compute_tm_gap(mutated_cell, n_k=5, pw_radius=7)
                except:
                    new_score = -10.0
                
                if new_score > current_score:
                    current_score = new_score
                    current_coeffs = mutated_coeffs
                    current_cell = mutated_cell
                    no_improve = 0
                    step_size *= 0.9
                    if current_score > best_fourier_score:
                        best_fourier_score = current_score
                        best_fourier_cell = current_cell.copy()
                        if current_score > best_score:
                            best_score = current_score
                            best_cell = current_cell.copy()
                else:
                    no_improve += 1
                    if no_improve > 5:
                        step_size *= 1.15
    
    # Final fallback
    if best_cell is None:
        best_cell = _make_rod(0.25)
    
    return best_cell.tolist()
