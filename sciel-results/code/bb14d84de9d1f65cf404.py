"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
import random

# EVOLVE-BLOCK-START
def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Use a higher fidelity evaluator (11x11 plane waves) for better accuracy.
    2. Perform multi-start local search from diverse symmetric initializations.
    3. Enforce C4v symmetry throughout.
    4. Return the best design found within the time budget.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    # Use 195s deadline to be safe within 200s limit (evaluator kills at 230s)
    deadline = start_time + 195.0 

    # --- 1. Symmetry Enforcement ---
    def enforce_symmetry(c):
        """
        Enforces C4v symmetry by averaging the orbit and thresholding.
        Input: 32x32 array of 0/1
        Output: 32x32 array of 0/1 with C4v symmetry
        """
        sym = np.zeros_like(c, dtype=int)
        # Precompute orbits for speed? No, just do it directly, it's fast enough for 32x32
        for i in range(size):
            for j in range(size):
                # Orbit of (i, j) under C4v
                # (i, j), (i, S-1-j), (S-1-i, S-1-j), (S-1-i, j)
                # (j, i), (S-1-j, i), (S-1-j, S-1-i), (j, S-1-i)
                # Note: Some might be duplicates
                orbit = [
                    (i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (size - 1 - i, j),
                    (j, i),
                    (size - 1 - j, i),
                    (size - 1 - j, size - 1 - i),
                    (j, size - 1 - i)
                ]
                # Sum values in orbit
                s = sum(c[pi, pj] for pi, pj in orbit)
                # Threshold: if majority is 1, set to 1
                sym[i, j] = 1 if s >= 4 else 0
        return sym

    # --- 2. High-Fidelity Evaluator ---
    # Use 11x11 plane waves (121 modes) for better accuracy
    pw_half = 5
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    # Precompute G vector arrays
    Gx_arr = np.array([g[0] for g in G_vectors], dtype=float)
    Gy_arr = np.array([g[1] for g in G_vectors], dtype=float)
    
    # Precompute G differences for FFT lookup
    # We need indices into the FFT of 1/epsilon
    G_diffs_h = np.zeros((N_pw, N_pw), dtype=int)
    G_diffs_k = np.zeros((N_pw, N_pw), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs_h[i, j] = (G_vectors[i][0] - G_vectors[j][0]) % size
            G_diffs_k[i, j] = (G_vectors[i][1] - G_vectors[j][1]) % size

    # K-points in irreducible BZ
    # We use a denser set of points for the local search to be more robust
    # But keep it manageable. 30 points is good.
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # Gamma to M (0, 0.5)
    for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        k_points.append((0.0, t))
    # M to X (0.5, 0.5)
    for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        k_points.append((t, 0.5 - t))
    # X is already included as t=0.5 in M-X? No, M-X is (t, 0.5-t). 
    # At t=0.5, we get (0.5, 0). This is X? No, X is (0.5, 0) or (0, 0.5)?
    # In square lattice, Gamma=(0,0), M=(0, 0.5), X=(0.5, 0.5) is usually the corner?
    # Wait, standard notation:
    # Gamma: (0,0)
    # M: (0, 1/2) (midpoint of edge)
    # X: (1/2, 1/2) (corner)
    # Irreducible BZ is 1/8th of the full BZ.
    # Boundaries: kx=0, ky=0, kx=ky.
    # So points:
    # 1. (0, t) for t in [0, 0.5] (Gamma to M)
    # 2. (t, 0.5-t) for t in [0, 0.5] (M to X)
    # This covers the 1/8 triangle.
    # My k_points list above covers:
    # (0, t) for t in [0, 0.5] -> 11 points
    # (t, 0.5-t) for t in [0, 0.5] -> 11 points
    # Total 22 points. This is sufficient and fast.

    def evaluate_gap(cell_grid):
        """
        Computes the approximate band gap between bands 4 and 5.
        cell_grid: 32x32 array of 0/1 (0=air, 1=dielectric)
        """
        # Build epsilon grid
        eps_grid = np.where(cell_grid == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # FFT for Fourier coefficients of 1/epsilon
        # coeffs_{G_i - G_j} = (1/N^2) * FFT(inv_eps)_{G_i - G_j}
        fft_1eps = np.fft.fft2(inv_eps)
        # We need to index fft_1eps with G_diffs
        # Note: fft2 output is ordered 0..N-1. 
        # G_diffs are already mod size.
        coeffs = fft_1eps[G_diffs_h, G_diffs_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            # H = (K_i · K_j) * coeffs_{G_i - G_j}
            # dots[i, j] = Kx[i]*Kx[j] + Ky[i]*Ky[j]
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            # Get eigenvalues
            # We only need the 4th and 5th smallest eigenvalues
            # eigvalsh returns sorted values
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

    # --- 3. Optimization: Multi-start Local Search ---
    
    def create_initial_cell(radius, center_offset, add_center_dot=False):
        """
        Creates a symmetric initial cell with 4 disks.
        """
        cell = np.zeros((size, size), dtype=int)
        
        # Centers at (8,8), (8,24), (24,8), (24,24)
        # With offset, we can shift them slightly to break degeneracy?
        # But we must maintain symmetry.
        # Let's just vary the radius and maybe add a central dot.
        
        centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    if dist < radius:
                        cell[i, j] = 1
        
        if add_center_dot:
            # Add a small dot in the center
            c_cy, c_cx = 15.5, 15.5
            r_c = 2.0
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - c_cy)**2 + (j - c_cx)**2)
                    if dist < r_c:
                        cell[i, j] = 1
        
        return enforce_symmetry(cell)

    def local_search(cell, time_limit=10.0):
        """
        Runs a local search (hill climbing with random restarts) on the given cell.
        Returns the best cell and score found.
        """
        current_cell = cell.copy()
        current_score = evaluate_gap(current_cell)
        best_cell = current_cell.copy()
        best_score = current_score
        
        start_ts = time.time()
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - start_ts < time_limit:
            # Pick a random orbit to flip
            i = np.random.randint(size)
            j = np.random.randint(size)
            
            orbit = [
                (i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (size - 1 - i, j),
                (j, i),
                (size - 1 - j, i),
                (size - 1 - j, size - 1 - i),
                (j, size - 1 - i)
            ]
            orbit = list(set(orbit))
            
            new_cell = current_cell.copy()
            for pi, pj in orbit:
                new_cell[pi, pj] = 1 - new_cell[pi, pj]
            
            new_score = evaluate_gap(new_cell)
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_cell = current_cell.copy()
            else:
                no_improve_count += 1
                # If stuck for a while, do a random jump (restart locally)
                if no_improve_count > max_no_improve:
                    # Flip 5 random orbits
                    for _ in range(5):
                        ri = np.random.randint(size)
                        rj = np.random.randint(size)
                        r_orbit = [
                            (ri, rj),
                            (ri, size - 1 - rj),
                            (size - 1 - ri, size - 1 - rj),
                            (size - 1 - ri, rj),
                            (rj, ri),
                            (size - 1 - rj, ri),
                            (size - 1 - rj, size - 1 - ri),
                            (rj, size - 1 - ri)
                        ]
                        r_orbit = list(set(r_orbit))
                        for pi, pj in r_orbit:
                            current_cell[pi, pj] = 1 - current_cell[pi, pj]
                    current_score = evaluate_gap(current_cell)
                    no_improve_count = 0
        
        return best_cell, best_score

    # --- 4. Main Loop ---
    
    # Define a set of diverse initial configurations
    initial_configs = []
    
    # 1. Standard 4 disks, radius 4
    initial_configs.append(create_initial_cell(radius=4.0))
    
    # 2. Larger disks, radius 5
    initial_configs.append(create_initial_cell(radius=5.0))
    
    # 3. Smaller disks, radius 3
    initial_configs.append(create_initial_cell(radius=3.0))
    
    # 4. Disks with central dot
    initial_configs.append(create_initial_cell(radius=4.0, add_center_dot=True))
    
    # 5. Disks with central dot, larger
    initial_configs.append(create_initial_cell(radius=5.0, add_center_dot=True))
    
    # 6. Random symmetric start
    rand_cell = np.random.randint(0, 2, (size, size))
    rand_cell = enforce_symmetry(rand_cell)
    initial_configs.append(rand_cell)
    
    # 7. Inverted disks (mostly dielectric, air holes)
    # This might be useful if the gap is better in the "hole" configuration
    # But the task says 1=dielectric. So this is just a different topology.
    cell_inv = np.ones((size, size), dtype=int)
    centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
    radius = 4.0
    for cy, cx in centers:
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                if dist < radius:
                    cell_inv[i, j] = 0
    cell_inv = enforce_symmetry(cell_inv)
    initial_configs.append(cell_inv)
    
    global_best_cell = None
    global_best_score = -float('inf')
    
    # Allocate time to each start
    # We have ~195s.
    # Let's do 7 starts. ~27s each.
    # But local search might be fast. We can do more if they finish early.
    
    for idx, init_cell in enumerate(initial_configs):
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time: first few get more, later ones less if time is tight
        # Simple: equal time allocation based on remaining time / remaining starts
        remaining_starts = len(initial_configs) - idx
        time_left = deadline - time.time()
        time_per_start = time_left / remaining_starts
        
        # Cap time per start to avoid one taking too long if others are quick
        # But if time_per_start is large, use it.
        # If time_per_start is small, use it.
        
        best_cell, best_score = local_search(init_cell, time_limit=time_per_start)
        
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_cell = best_cell.copy()
            
        # If we found a very high score, we might want to polish it with more time?
        # The local search already polishes.
        
    # If no global best found (should not happen), fallback
    if global_best_cell is None:
        global_best_cell = enforce_symmetry(np.random.randint(0, 2, (size, size)))
        
    # Final check: enforce symmetry one last time just in case
    final_cell = enforce_symmetry(global_best_cell)
    
    return final_cell.tolist()

# EVOLVE-BLOCK-END
