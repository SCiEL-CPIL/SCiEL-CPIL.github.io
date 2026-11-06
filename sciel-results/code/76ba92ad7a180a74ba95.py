"""
Strategy: Fourier-domain direct optimization.
Instead of parameterizing the shape (circles/polygons) and searching in shape-space,
we directly optimize the Fourier coefficients of the dielectric function.
The TM 4-5 band gap in a square lattice is primarily driven by the (2,2) Fourier component
at the M point (k=0.5, 0.5). We construct a continuous dielectric field using a small
set of symmetry-allowed Fourier modes, optimize their amplitudes to maximize the
effective gap metric, and then threshold the field to get the binary 32x32 cell.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    start_time = time.time()
    time_limit = 190.0  # Safety margin before 230s kill

    # Precompute pixel coordinates
    # Center the cell at (0,0) for symmetry
    x = np.linspace(-0.5, 0.5, size, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')
    
    # Symmetry group D4:
    # Identity: (x, y)
    # Rot 90: (-y, x)
    # Rot 180: (-x, -y)
    # Rot 270: (y, -x)
    # Refl X: (x, -y)
    # Refl Y: (-x, y)
    # Refl Diag1: (y, x)
    # Refl Diag2: (-y, -x)
    
    def get_sym_points(x_val, y_val):
        return [
            (x_val, y_val),
            (-y_val, x_val),
            (-x_val, -y_val),
            (y_val, -x_val),
            (x_val, -y_val),
            (-x_val, y_val),
            (y_val, x_val),
            (-y_val, -x_val)
        ]

    # Define the set of Fourier modes to optimize.
    # We focus on modes that contribute to the (2,2) component or nearby.
    # Due to D4 symmetry, we can group modes.
    # Mode (l, m) with l>=m>=0.
    # We will optimize amplitudes A_lm and phases phi_lm.
    # However, for binary-like structures, phases are often 0 or pi.
    # Let's allow continuous phases but initialize to 0/pi.
    
    # Key modes for 4-5 TM gap:
    # The gap is at M (0.5, 0.5). The coupling is between states separated by G=(2,2) or similar.
    # Actually, for TM, the operator involves epsilon_G.
    # The 4-5 gap is often associated with the (2,2) component.
    # Let's include modes (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3).
    
    modes = [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    
    n_modes = len(modes)
    
    def build_field(params):
        """
        params: [A_00, A_10, phi_10, A_11, phi_11, ...]
        A_00 is the average epsilon (DC component).
        """
        eps_field = np.zeros((size, size), dtype=float)
        
        # DC component: average epsilon
        # We want the average to be around (1+12)/2 = 6.5 for balanced fill?
        # Actually, the DC component is fixed by the average fill fraction.
        # In this parameterization, A_00 represents the average epsilon.
        # Let's fix A_00 to 6.5 (50% fill) or optimize it slightly?
        # Optimizing A_00 changes the average fill.
        # Let's include it as the first parameter.
        eps_field += params[0]
        
        idx = 1
        for l, m in modes:
            A = params[idx]
            phi = params[idx+1]
            idx += 2
            
            # Sum over all 8 symmetric points
            # For (l,m), the symmetric partners are:
            # (l,m), (-m,l), (-l,-m), (m,-l)
            # (l,-m), (-m,-l), (-l,m), (m,l)
            # Note: if l=m, some are duplicates. If l=0 or m=0, some are duplicates.
            # To be safe and efficient, we just sum over the unique generators and their reflections.
            # A better way: generate all 8, remove duplicates, sum.
            
            pts = get_sym_points(l, m)
            # Remove duplicates
            unique_pts = []
            for p in pts:
                if p not in unique_pts:
                    unique_pts.append(p)
            
            val = 0.0
            for ul, um in unique_pts:
                # Term: A * cos(2*pi*(ul*X + um*Y) + phi)
                # Wait, the standard Fourier series for a real function with D4 symmetry:
                # f(x,y) = sum A_lm cos(2*pi*(l*x + m*y) + phi_lm)
                # But we must sum over the orbit.
                # Actually, the basis functions for D4 are:
                # cos(2*pi*(l*x + m*y)) and sin(2*pi*(l*x + m*y))
                # With D4 symmetry, the coefficients are constrained.
                # Let's use the explicit sum of cosines for the orbit.
                # f(x,y) = A * sum_{(l',m') in orbit} cos(2*pi*(l'*x + m'*y) + phi)
                # This is not quite right because the phase might need to vary for sine terms.
                # However, for "flower" shapes, cosine terms with 0 or pi phase are sufficient.
                # Let's stick to A * sum cos(...).
                
                term = np.cos(2 * np.pi * (ul * X + um * Y) + phi)
                val += term
            
            # Normalize by number of unique points to keep amplitude scale consistent
            eps_field += A * val / len(unique_pts)
        
        return eps_field

    def score_cell(cell):
        """
        Fast approximate score for a binary cell.
        We compute the (2,2) Fourier coefficient of the binary map.
        The gap size is roughly proportional to |epsilon_22|.
        """
        cell_arr = np.array(cell, dtype=float)
        # Compute Fourier coefficient (2,2)
        # epsilon_22 = sum_{i,j} eps_ij * exp(-i 2*pi (2*i/32 + 2*j/32))
        # Since eps is real and symmetric, we can use cos/sin.
        # Actually, let's just use the field we built to estimate the score, 
        # but we need to threshold it first.
        
        # Threshold to binary
        # We want 50% fill roughly.
        # Sort the field and pick the median as threshold?
        # Or use a fixed threshold based on the mean?
        # Let's use the median of the field to ensure 50% fill.
        thresh = np.median(cell_arr)
        binary = (cell_arr > thresh).astype(float)
        
        # Compute (2,2) component magnitude
        # G = (2, 2)
        # eps_G = 1/N^2 sum_{i,j} binary[i,j] exp(-i 2*pi (2*i/N + 2*j/N))
        # For real binary, this is complex. Magnitude is what matters.
        
        # We can compute this efficiently with FFT or direct sum.
        # Direct sum is 32*32 = 1024 ops, very fast.
        
        # Phase factors
        # i is row, j is col.
        # exp(-i 2*pi (2*i/32 + 2*j/32)) = exp(-i pi/8 * (2i + 2j)) = exp(-i pi/4 * (i+j))
        # Wait, 2*pi * 2 * i / 32 = pi * i / 8.
        # So phase = -pi/8 * (2*i + 2*j) = -pi/4 * (i+j).
        
        # Let's use numpy for speed
        i_idx = np.arange(size)
        j_idx = np.arange(size)
        I, J = np.meshgrid(i_idx, i_idx, indexing='ij')
        
        phase = np.exp(-1j * (np.pi / 4.0) * (I + J))
        
        eps_22 = np.sum(binary * phase) / (size * size)
        
        # Also check (0,2) and (2,0) just in case, though (2,2) is key for M point.
        # The 4-5 TM gap is at M. The coupling is between (k, k) and (k+G, k+G).
        # At M=(0.5, 0.5), the states are (0.5, 0.5) and (2.5, 0.5) or similar?
        # Actually, the zone boundary is at 0.5.
        # The relevant Fourier component for the gap at M is usually (1,1) or (2,2)?
        # For a square lattice, the M point is (0.5, 0.5).
        # The reciprocal lattice vector connecting M to Gamma is (0.5, 0.5)? No, G is integer.
        # The states at M are degenerate with states at M+G.
        # The gap is opened by epsilon_G where G connects the two.
        # For M=(0.5, 0.5), the degenerate states are k=(0.5, 0.5) and k'=(0.5+1, 0.5) = (1.5, 0.5) ??
        # No, the Brillouin zone is [0,1]x[0,1] in units of 2pi/a?
        # Let's assume standard units where BZ is [-0.5, 0.5].
        # M is (0.5, 0.5).
        # The states coupled are k and k+G.
        # If k=(0.5, 0.5), k+G=(1.5, 0.5) which is equivalent to (-0.5, 0.5) which is M.
        # So G=(1,0) or (0,1) connects M to M?
        # Actually, the 4-5 TM gap is often associated with the (2,2) component in some contexts,
        # but let's check the (1,1) component too.
        
        # Let's compute (1,1) as well.
        # Phase for (1,1): exp(-i 2*pi (1*i/32 + 1*j/32)) = exp(-i pi/16 (i+j))
        phase_11 = np.exp(-1j * (np.pi / 16.0) * (I + J))
        eps_11 = np.sum(binary * phase_11) / (size * size)
        
        # And (2,0)
        # Phase for (2,0): exp(-i 2*pi (2*i/32)) = exp(-i pi/8 i)
        phase_20 = np.exp(-1j * (np.pi / 8.0) * I)
        eps_20 = np.sum(binary * phase_20) / (size * size)
        
        # Combine them. The (2,2) is usually the strongest driver for the high-order gap.
        # Let's return a weighted sum of magnitudes.
        score = 2.0 * np.abs(eps_22) + 1.0 * np.abs(eps_11) + 1.0 * np.abs(eps_20)
        
        return score, binary.tolist()

    # Optimization
    # We use a simple random restart + local search (Nelder-Mead or just coordinate descent)
    # Given the small number of parameters (1 DC + 2*n_modes), we can do a coarse grid or random search.
    
    best_score = -np.inf
    best_cell = None
    
    # Initial parameters
    # A_00 = 6.5
    # Others small random
    
    n_restarts = 20
    steps_per_restart = 50
    
    # Parameters: [A00, A10, p10, A11, p11, ...]
    param_count = 1 + 2 * n_modes
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Initialize params
        params = np.zeros(param_count)
        params[0] = 6.5 + np.random.uniform(-1, 1)
        
        for i in range(n_modes):
            idx_a = 1 + 2*i
            idx_p = 2 + 2*i
            # Random amplitude
            params[idx_a] = np.random.uniform(-1.0, 1.0)
            # Phase: 0 or pi
            params[idx_p] = np.random.choice([0, np.pi])
        
        # Local search: Coordinate descent with random steps
        current_score, _ = score_cell(build_field(params))
        
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Pick a random parameter to perturb
            p_idx = np.random.randint(0, param_count)
            old_val = params[p_idx]
            
            # Perturb
            if p_idx == 0:
                params[p_idx] = old_val + np.random.uniform(-0.5, 0.5)
            else:
                params[p_idx] = old_val + np.random.uniform(-0.2, 0.2)
                
            new_field = build_field(params)
            new_score, _ = score_cell(new_field)
            
            if new_score > current_score:
                current_score = new_score
            else:
                # Revert
                params[p_idx] = old_val
        
        # Final evaluation for this restart
        final_field = build_field(params)
        final_score, final_cell = score_cell(final_field)
        
        if final_score > best_score:
            best_score = final_score
            best_cell = final_cell
            # Print for debugging (optional)
            # print(f"Restart {r}: Score {best_score}")

    if best_cell is None:
        # Fallback: Centered disk
        best_cell = np.zeros((size, size), dtype=int)
        center = size // 2
        radius = 8
        for i in range(size):
            for j in range(size):
                if (i - center)**2 + (j - center)**2 < radius**2:
                    best_cell[i, j] = 1
        best_cell = best_cell.tolist()

    # Ensure symmetry in the final binary cell
    # The field was symmetric, but thresholding might break it if there are ties.
    # Enforce symmetry by averaging with its reflections.
    cell_arr = np.array(best_cell, dtype=float)
    
    # Reflect
    ref1 = cell_arr[::-1, ::-1] # 180 rot
    ref2 = cell_arr.T # Transpose (Refl Diag)
    ref3 = cell_arr[:, ::-1] # Refl Y
    
    # Average and threshold
    # Actually, just enforce the required symmetries:
    # cell == cell[::-1] (row-wise) -> 180 rotation? No, "row-wise" usually means reverse rows.
    # "cell == cell[::-1] row-wise and column-wise"
    # This implies:
    # 1. cell[i][j] == cell[size-1-i][size-1-j] (180 rot)
    # 2. cell[i][j] == cell[i][size-1-j] (Refl Y)?
    # Let's check the contract: "cell == cell[::-1] row-wise and column-wise and cell == its transpose"
    # cell[::-1] reverses rows. If cell == cell[::-1], it's symmetric top-bottom.
    # "row-wise and column-wise" is ambiguous.
    # Usually "symmetric about the cell centre" means D4.
    # Let's enforce D4 by averaging all 8 transformations.
    
    sym_sum = np.zeros_like(cell_arr)
    for i in range(size):
        for j in range(size):
            # Get all 8 symmetric positions
            pts = get_sym_points(i, j)
            # Note: get_sym_points uses continuous coords. Here we have indices.
            # Map index to coord: x = (i + 0.5)/size - 0.5
            # This is complex. Let's just use array flips.
            
            # 1. (i, j)
            sym_sum[i, j] += cell_arr[i, j]
            # 2. (size-1-i, size-1-j) - 180
            sym_sum[size-1-i, size-1-j] += cell_arr[i, j]
            # 3. (i, size-1-j) - Refl Y
            sym_sum[i, size-1-j] += cell_arr[i, j]
            # 4. (size-1-i, j) - Refl X
            sym_sum[size-1-i, j] += cell_arr[i, j]
            # 5. (j, i) - Transpose
            sym_sum[j, i] += cell_arr[i, j]
            # 6. (size-1-j, size-1-i) - 180 of Transpose
            sym_sum[size-1-j, size-1-i] += cell_arr[i, j]
            # 7. (j, size-1-i) - Transpose of Refl X?
            sym_sum[j, size-1-i] += cell_arr[i, j]
            # 8. (size-1-j, i) - Transpose of Refl Y?
            sym_sum[size-1-j, i] += cell_arr[i, j]
            
    # The sum should be 8 * value if symmetric.
    # Threshold at 4.0 (half of 8)
    final_sym_cell = (sym_sum >= 4.0).astype(int)
    
    return final_sym_cell.tolist()
# EVOLVE-BLOCK-END
