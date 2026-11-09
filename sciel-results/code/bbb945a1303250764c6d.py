"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Initialize with a set of parameterized symmetric shapes (superellipses, crosses).
    2. Use a fast 9x9 plane-wave surrogate to evaluate the gap.
    3. Run Simulated Annealing with restarts to refine the pixel map.
    4. Enforce C4v symmetry exactly.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- 1. Fast Plane-Wave Surrogate ---
    # Use 9x9 plane waves (81 modes) for better accuracy than 7x7, still fast
    pw_half = 4
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    # Precompute G differences for FFT indexing
    G_diff_h = np.zeros((N_pw, N_pw), dtype=int)
    G_diff_k = np.zeros((N_pw, N_pw), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diff_h[i, j] = G_vectors[i][0] - G_vectors[j][0]
            G_diff_k[i, j] = G_vectors[i][1] - G_vectors[j][1]
            
    # Modulo size for FFT indexing
    G_diff_h_idx = G_diff_h % size
    G_diff_k_idx = G_diff_k % size
    
    # K-points in irreducible BZ (Triangle: Gamma-M-X)
    k_points = []
    # Gamma (0,0)
    k_points.append((0.0, 0.0))
    # M (0.5, 0)
    k_points.append((0.5, 0.0))
    # X (0.5, 0.5)
    k_points.append((0.5, 0.5))
    
    # Along Gamma-M: (0, t)
    for t in np.linspace(0.1, 0.4, 4):
        k_points.append((0.0, t))
        
    # Along M-X: (0.5, t)
    for t in np.linspace(0.1, 0.4, 4):
        k_points.append((0.5, t))
        
    # Along Gamma-X: (t, t)
    for t in np.linspace(0.1, 0.4, 4):
        k_points.append((t, t))
        
    # Interior points
    k_points.append((0.25, 0.1))
    k_points.append((0.4, 0.2))
    k_points.append((0.3, 0.3))
    k_points.append((0.2, 0.25))
    
    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap ratio between bands 4 and 5.
        cell_grid: 32x32 array of 0/1 (0=air, 1=dielectric)
        """
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # FFT for Fourier coefficients of 1/epsilon
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_h_idx, G_diff_k_idx] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H = (K_i . K_j) * coeffs_{G_i - G_j}
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            # Get eigenvalues
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 2. Symmetry Helpers ---
    def get_orbit(i, j):
        points = [
            (i, j),
            (size - 1 - i, j),
            (i, size - 1 - j),
            (size - 1 - i, size - 1 - j),
            (j, i),
            (size - 1 - j, j),
            (j, size - 1 - i),
            (size - 1 - j, size - 1 - i)
        ]
        return list(set(points))

    def enforce_symmetry(c):
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                orbit = get_orbit(i, j)
                s = sum(c[pi, pj] for pi, pj in orbit)
                sym[i, j] = 1 if s >= len(orbit)/2 else 0
        return sym

    # --- 3. Initial Candidates ---
    def make_superellipse(cx, cy, R, n, is_dielectric=True):
        """
        Creates a superellipse centered at (cx, cy).
        """
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                dx = abs(i - cx)
                dy = abs(j - cy)
                val = (dx/R)**n + (dy/R)**n
                if val <= 1.0:
                    cell[i, j] = 1 if is_dielectric else 0
                else:
                    cell[i, j] = 0 if is_dielectric else 1
        return enforce_symmetry(cell)

    candidates = []
    
    # Generate candidates with varying R and n
    # R: radius, n: exponent (2=circle, 1=diamond, >2=boxy, <1=star)
    for R in [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]:
        for n in [0.5, 1.0, 2.0, 4.0, 8.0]:
            # Central shape (Dielectric)
            c1 = make_superellipse(15.5, 15.5, R, n, is_dielectric=True)
            candidates.append(c1)
            
            # Central shape (Air hole in Dielectric background)
            c2 = 1 - c1
            candidates.append(c2)
            
            # Corner shapes (Dielectric at corners)
            # Place one at (8,8) and let symmetry replicate
            c3 = np.zeros((size, size), dtype=int)
            for i in range(size):
                for j in range(size):
                    dx = abs(i - 8.0)
                    dy = abs(j - 8.0)
                    val = (dx/R)**n + (dy/R)**n
                    if val <= 1.0:
                        c3[i, j] = 1
            c3 = enforce_symmetry(c3)
            candidates.append(c3)
            
            # Corner shapes (Air at corners, Dielectric background)
            c4 = 1 - c3
            candidates.append(c4)

    # Evaluate all candidates
    best_cell = None
    best_score = -1.0
    for c in candidates:
        score = evaluate_gap(c)
        if score > best_score:
            best_score = score
            best_cell = c.copy()

    # --- 4. Simulated Annealing ---
    current_cell = best_cell.copy()
    current_score = best_score
    best_cell_sa = best_cell.copy()
    best_score_sa = best_score
    
    temperature = 0.5
    cooling_rate = 0.998
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Pick random orbit to flip
        i = np.random.randint(size)
        j = np.random.randint(size)
        orbit = get_orbit(i, j)
        
        new_cell = current_cell.copy()
        for pi, pj in orbit:
            new_cell[pi, pj] = 1 - new_cell[pi, pj]
        
        new_score = evaluate_gap(new_cell)
        
        # Metropolis acceptance
        delta = new_score - current_score
        if delta > 0 or np.random.random() < math.exp(delta / max(temperature, 1e-6)):
            current_cell = new_cell
            current_score = new_score
            
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_cell_sa = current_cell.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Reheat if stuck
            if no_improve_count > max_no_improve:
                temperature = 0.5
                no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                temperature = 0.5
                no_improve_count = 0
                
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01

    # --- 5. Finalize ---
    final_cell = enforce_symmetry(best_cell_sa)
    return final_cell.tolist()

# EVOLVE-BLOCK-END
