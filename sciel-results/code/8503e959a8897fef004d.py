"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    t0 = time.time()
    deadline = t0 + 200.0  # safety margin before 230s kill

    # --- Symmetric parameterization via top-left 16x16 quadrant ---
    # The cell must satisfy:
    #   cell[i][j] == cell[31-i][j]  (vertical flip)
    #   cell[i][j] == cell[i][31-j]  (horizontal flip)
    #   cell[i][j] == cell[j][i]     (transpose)
    # This is C4v symmetry about the cell center.
    # The top-left 16x16 quadrant (i in 0..15, j in 0..15) determines the full cell
    # IF the quadrant itself is symmetric: quadrant[i][j] == quadrant[j][i].
    # We enforce this by only storing the upper triangle (i <= j) and mirroring.
    
    half = size // 2  # 16
    
    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from a symmetric 16x16 quadrant.
        quadrant[i][j] is defined for i <= j (upper triangle).
        We mirror to get the full quadrant, then mirror to get the full cell.
        """
        N = size
        # First, complete the 16x16 quadrant using transpose symmetry
        full_quad = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                val = quadrant[i][j]
                full_quad[i][j] = val
                full_quad[j][i] = val
        
        # Now mirror the 16x16 quadrant to get the full 32x32 cell
        cell = [[0]*N for _ in range(N)]
        for i in range(half):
            for j in range(half):
                val = full_quad[i][j]
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
        return cell

    # --- Fast internal PWE solver ---
    def compute_gap(cell):
        """Compute the TM band-4/5 gap-midgap ratio using a plane-wave method.
        Returns the ratio (higher is better), or -1 on failure.
        """
        N = size
        # Compute 1/epsilon map
        eps_inv = np.ones((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if cell[i][j] == 1:
                    eps_inv[i, j] = 1.0 / eps_high
        
        # Fourier coefficients of 1/epsilon via FFT
        fft_coeff = np.fft.fft2(eps_inv) / (N * N)
        
        def get_coeff(mx, my):
            return fft_coeff[my % N, mx % N]
        
        # Plane-wave basis: mx, my in -8..8 (289 total, matching evaluator)
        max_g = 8
        g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
        num_g = len(g_indices)
        
        # Build eps_inv coupling matrix
        eps_mat = np.zeros((num_g, num_g), dtype=complex)
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                eps_mat[p, q] = get_coeff(mx_p - mx_q, my_p - my_q)
        
        # G vectors
        G_vecs = np.array([(2.0*math.pi*mx, 2.0*math.pi*my) for mx, my in g_indices])
        
        # K-points: 30 points covering the irreducible BZ
        # Irreducible BZ of square lattice under D4: triangle with vertices
        # Gamma=(0,0), X=(pi,0) mapped to (0,pi), M=(pi/2,pi/2)
        # Region: 0 <= kx <= ky, kx + ky <= pi
        k_points = []
        # Along Gamma to X (0,0) to (0,pi): 7 points
        for t in np.linspace(0, math.pi, 7):
            k_points.append((0.0, t))
        # Along Gamma to M (0,0) to (pi/2,pi/2): 7 points
        for t in np.linspace(0, math.pi/2, 7):
            k_points.append((t, t))
        # Along M to X (pi/2,pi/2) to (0,pi): 7 points
        for t in np.linspace(0, math.pi/2, 7):
            kx = math.pi/2 - t
            ky = math.pi/2 + t
            k_points.append((kx, ky))
        # Interior points: 9 points
        interior = [
            (math.pi/6, math.pi/3),
            (math.pi/4, math.pi/2),
            (math.pi/8, 3*math.pi/8),
            (math.pi/5, 2*math.pi/5),
            (math.pi/3, 2*math.pi/3),
            (math.pi/12, math.pi/4),
            (math.pi/10, math.pi/3),
            (math.pi/7, math.pi/2),
            (math.pi/9, 5*math.pi/18),
        ]
        for pt in interior:
            k_points.append(pt)
        
        k_points = k_points[:30]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            # |k + G_q|^2 for each plane wave
            kG_sq = (kx + G_vecs[:, 0])**2 + (ky + G_vecs[:, 1])**2
            
            # Matrix M[p,q] = eps_mat[p,q] * |k + G_q|^2
            M = eps_mat * kG_sq[None, :]
            
            # Ensure Hermitian (should be, but numerical safety)
            M = (M + M.conj().T) / 2.0
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            eigvals = np.sort(np.real(eigvals))
            if len(eigvals) < 5:
                continue
            
            omega4 = math.sqrt(max(eigvals[3], 0.0))
            omega5 = math.sqrt(max(eigvals[4], 0.0))
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf') or max_omega4 <= 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # --- Design generation using Chebyshev polynomial contours ---
    # Strategy: superpose four concentric C4v-symmetric contours defined by
    # Chebyshev polynomials of different orders to create multi-scale structure.
    # The dielectric is where the sum of signed contour functions exceeds a threshold.
    
    def make_chebyshev_cell(params):
        """
        Create a cell from Chebyshev polynomial contours.
        params: list of (order, radius_scale, thickness, sign) for each contour
        The dielectric condition: sum_k sign_k * T_{order_k}(r / radius_scale_k) > threshold
        where T_n is the Chebyshev polynomial of the first kind.
        """
        N = size
        cell = [[0]*N for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                # Distance from center of cell (15.5, 15.5)
                dx = j - 15.5
                dy = i - 15.5
                r = math.sqrt(dx*dx + dy*dy)
                
                # Also compute the C4v-symmetric distance (use max of |dx|, |dy| for square symmetry)
                # Actually, for proper C4v we need the radial distance to be C4v-symmetric.
                # The Euclidean distance r is C4v-symmetric (invariant under 90° rotation and reflection).
                
                total = 0.0
                for (order, radius_scale, thickness, sign) in params:
                    if radius_scale <= 0:
                        continue
                    x = r / radius_scale
                    if abs(x) > 1.0:
                        # Outside the domain of Chebyshev; use clamped value
                        x_clamped = 1.0 if x > 0 else -1.0
                        # T_n(1) = 1, T_n(-1) = (-1)^n
                        val = 1.0 if order % 2 == 0 else -1.0
                    else:
                        # Chebyshev polynomial T_n(x) = cos(n * arccos(x))
                        val = math.cos(order * math.acos(max(-1.0, min(1.0, x))))
                    
                    # Apply thickness: only contribute within a band around the contour
                    # The contour is where T_n(x) = 0 (or some level)
                    # Use a Gaussian-like window around the zero-crossings
                    # Simpler: use the value directly with sign
                    total += sign * val * thickness
                
                if total > 0.0:
                    cell[i][j] = 1
                else:
                    cell[i][j] = 0
        
        return cell

    # --- Simpler parametric family: four concentric rings with varying radii ---
    # This is more robust and directly creates multi-scale structure.
    
    def make_ring_cell(radii, thicknesses):
        """
        Create a cell from concentric rings.
        radii: list of center radii for each ring
        thicknesses: list of half-thicknesses for each ring
        Dielectric where any ring is active.
        """
        N = size
        cell = [[0]*N for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                dx = j - 15.5
                dy = i - 15.5
                r = math.sqrt(dx*dx + dy*dy)
                
                is_diel = False
                for rad, thick in zip(radii, thicknesses):
                    if abs(r - rad) <= thick:
                        is_diel = True
                        break
                
                cell[i][j] = 1 if is_diel else 0
        
        return cell

    # --- Generate and evaluate candidate designs ---
    # Start with several parametric families, then do local refinement.
    
    candidates = []
    
    # Family 1: Single ring (from inspiration 1, which scored well)
    for r_center in [4.0, 5.0, 6.0, 7.0, 8.0]:
        for r_width in [1.5, 2.0, 2.5, 3.0]:
            candidates.append(make_ring_cell([r_center], [r_width]))
    
    # Family 2: Two concentric rings
    for r1 in [3.0, 4.0, 5.0, 6.0]:
        for r2 in [r1+3.0, r1+4.0, r1+5.0]:
            if r2 < 14.0:
                for w1 in [1.5, 2.0]:
                    for w2 in [1.5, 2.0]:
                        candidates.append(make_ring_cell([r1, r2], [w1, w2]))
    
    # Family 3: Four concentric rings (Chebyshev-inspired multi-scale)
    for base_r in [2.0, 3.0, 4.0]:
        for spacing in [2.5, 3.0, 3.5, 4.0]:
            radii = [base_r + k*spacing for k in range(4)]
            if radii[-1] < 14.0:
                for w in [1.0, 1.5, 2.0]:
                    candidates.append(make_ring_cell(radii, [w]*4))
    
    # Family 4: Disk (solid center)
    for r in [5.0, 6.0, 7.0, 8.0, 9.0]:
        candidates.append(make_ring_cell([r/2.0], [r/2.0]))
    
    # Evaluate all candidates
    best_score = -1.0
    best_cell = candidates[0]
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline - 40:
            break
        score = compute_gap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # --- Local refinement: flip individual pixels in the best cell ---
    # Convert best_cell to a quadrant representation for efficient manipulation
    # The cell is C4v-symmetric, so we work with the 16x16 upper-triangle quadrant
    
    def cell_to_quadrant(cell):
        """Extract the upper-triangle 16x16 quadrant from a C4v-symmetric cell."""
        N = size
        half = N // 2
        quad = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                quad[i][j] = cell[i][j]
        return quad
    
    def quadrant_to_cell(quad):
        """Build full cell from upper-triangle quadrant."""
        N = size
        half = N // 2
        # Complete the 16x16 quadrant
        full_quad = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                full_quad[i][j] = quad[i][j]
                full_quad[j][i] = quad[i][j]
        # Mirror to full cell
        cell = [[0]*N for _ in range(N)]
        for i in range(half):
            for j in range(half):
                val = full_quad[i][j]
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
        return cell
    
    current_quad = cell_to_quadrant(best_cell)
    current_score = best_score
    
    # Get list of independent positions
    positions = [(i, j) for i in range(half) for j in range(i, half)]
    
    # Local search: try flipping each position, keep if improved
    import random
    random.seed(42)
    
    no_improve = 0
    max_no_improve = 30
    
    while time.time() < deadline - 15 and no_improve < max_no_improve:
        improved = False
        shuffled_positions = positions[:]
        random.shuffle(shuffled_positions)
        
        for (i, j) in shuffled_positions:
            if time.time() > deadline - 10:
                break
            
            old_val = current_quad[i][j]
            current_quad[i][j] = 1 - old_val
            
            test_cell = quadrant_to_cell(current_quad)
            test_score = compute_gap(test_cell)
            
            if test_score > current_score:
                current_score = test_score
                best_score = test_score
                best_cell = test_cell
                improved = True
                no_improve = 0
            else:
                current_quad[i][j] = old_val
                no_improve += 1
        
        if not improved:
            break
    
    # Final verification of symmetry
    final_cell = best_cell
    N = size
    for i in range(N):
        for j in range(N):
            if final_cell[i][j] != final_cell[N-1-i][j]:
                # Fix: force symmetry
                final_cell[i][j] = final_cell[N-1-i][j]
            if final_cell[i][j] != final_cell[i][N-1-j]:
                final_cell[i][j] = final_cell[i][N-1-j]
            if final_cell[i][j] != final_cell[j][i]:
                final_cell[i][j] = final_cell[j][i]
    
    return final_cell
# EVOLVE-BLOCK-END
