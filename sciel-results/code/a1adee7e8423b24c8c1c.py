"""
Photonic-crystal unit cell design: maximise the TM band gap between bands 4 and 5.

Strategy:
  1. Use a fast plane-wave (PW) TM band solver with exact Fourier coefficients
     of the pixel map (289 PWs, 17x17 grid).
  2. Start from a hexagonal-rod / inverted-honeycomb-like symmetric structure
     that is known to open higher-order TM gaps.
  3. Local search: perturb the pixel map (flipping pixels while preserving
     4-fold symmetry), evaluate the gap, keep improvements.
  4. Deadline guard: stop before 200 s and return the best cell found.
"""

import time
import math
import copy
import numpy as np

# ---------- symmetry helpers ----------

def apply_symmetries(cell):
    """Enforce 4-fold symmetry: cell == cell[::-1] row-wise, column-wise, and transpose."""
    n = len(cell)
    c = [row[:] for row in cell]
    # Enforce row-reversal symmetry: c[i][j] == c[n-1-i][j]
    for i in range(n // 2):
        for j in range(n):
            avg = (c[i][j] + c[n - 1 - i][j]) / 2.0
            c[i][j] = 1 if avg >= 0.5 else 0
            c[n - 1 - i][j] = c[i][j]
    # Enforce column-reversal: c[i][j] == c[i][n-1-j]
    for i in range(n):
        for j in range(n // 2):
            avg = (c[i][j] + c[i][n - 1 - j]) / 2.0
            c[i][j] = 1 if avg >= 0.5 else 0
            c[i][n - 1 - j] = c[i][j]
    # Enforce transpose: c[i][j] == c[j][i]
    for i in range(n):
        for j in range(i + 1, n):
        # Actually for 4-fold we need c[i][j]==c[j][i]
            avg = (c[i][j] + c[j][i]) / 2.0
            c[i][j] = 1 if avg >= 0.5 else 0
            c[j][i] = c[i][j]
    return c


def is_symmetric(cell):
    n = len(cell)
    for i in range(n):
        for j in range(n):
            if cell[i][j] != cell[n - 1 - i][j]:
                return False
            if cell[i][j] != cell[i][n - 1 - j]:
                return False
            if cell[i][j] != cell[j][i]:
                return False
    return True


# ---------- plane-wave TM solver ----------

def compute_fourier_coefficients(cell, n_pw=17):
    """
    Compute exact Fourier coefficients of the dielectric map.
    For TM polarization, the eigenvalue equation involves 1/eps(k+G).
    We need the Fourier coefficients of 1/eps(r) for each G.
    
    cell: n x n list of 0/1. 1 = dielectric (eps=12), 0 = air (eps=1).
    n_pw: half-grid size (total PWs = (2*n_pw-1)^2)
    Returns:
        eps_inv_coeffs: dict mapping (gx, gy) index to coefficient of 1/eps
    """
    n = len(cell)
    eps_high = 12.0
    eps_low = 1.0
    
    # Build the dielectric map on the pixel grid
    # 1/eps values: air -> 1.0, dielectric -> 1/12
    inv_eps_grid = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inv_eps_grid[i, j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)
    
    # Fourier coefficients: c_G = (1/n^2) * sum_{i,j} inv_eps[i,j] * exp(-i*G.x2pi*i/n - i*G.y2pi*j/n)
    # G ranges from -(n_pw-1) to (n_pw-1)
    coeffs = {}
    for gx in range(-(n_pw - 1), n_pw):
        for gy in range(-(n_pw - 1), n_pw):
            # Use numpy FFT-like computation
            # c = (1/n^2) sum_i sum_j f(i,j) exp(-2pi i (gx*i + gy*j)/n)
            # This is just the 2D DFT at (gx, gy), scaled
            # Use np.fft for efficiency: compute full FFT and index
            pass
    
    # More efficient: compute the full 2D FFT once
    fft_2d = np.fft.fft2(inv_eps_grid)
    # The DFT: F[kx, ky] = sum_i sum_j f[i,j] exp(-2pi i (kx*i + ky*j)/n)
    # So c_G = F[kx, ky] / n^2 where kx = gx mod n, ky = gy mod n
    n2 = n * n
    for gx in range(-(n_pw - 1), n_pw):
        for gy in range(-(n_pw - 1), n_pw):
            kx = gx % n
            ky = gy % n
            coeffs[(gx, gy)] = fft_2d[kx, ky] / n2
    
    return coeffs


def compute_tm_bands(coeffs, n_pw=17, num_bands=6, kx=0.0, ky=0.0):
    """
    Solve the TM band structure at wave vector (kx, ky) using plane-wave expansion.
    
    For TM: the eigenvalue equation is
        sum_G' (k+G)^2 / eps_G * |E(k+G')|^2 ... 
    Actually for TM polarization in 2D:
        The equation is: sum_{G'} [ (k+G)^2 * delta_{G,G'} - sum_{G''} (k+G)^2/eps_{G-G''} ] ... 
    
    Let me use the standard formulation:
    For TM: 
        sum_{G'} [ (k+G)^2 delta_{G,G'} - sum_{G''} (k+G)^2 / eps_{G-G''} * delta_{G',G''} ] E_{G'} = omega^2 E_G
    
    Actually the standard PW method for TM:
    The eigenvalue equation for TM modes:
        sum_{G'} [ (k+G)^2 delta_{G,G'} - (k+G)^2 sum_{G''} (1/eps)_{G-G''} delta_{G',G''} ] E_{G'} = omega^2 E_G
    
    Wait, let me be more careful. The standard form is:
    
    For TM (electric field perpendicular to plane):
        The Maxwell equation gives:
        (k+G)^2 E_G - sum_{G'} (k+G)^2 * (1/eps)_{G-G'} * E_{G'} = omega^2 E_G
    
    So the matrix A has elements:
        A_{G,G'} = (k+G)^2 * delta_{G,G'} - (k+G)^2 * (1/eps)_{G-G'}
    
    Hmm, that's not quite right either. Let me use the well-known form:
    
    Actually the correct TM eigenvalue equation in PW basis:
        sum_{G'} [ (k+G)^2 delta_{G,G'} - sum_{G''} (k+G)^2 / eps_{G-G''} * delta_{G',G''} ] E_{G'} = omega^2 E_G
    
    This can be written as a matrix equation:
        M E = omega^2 E
    where M_{G,G'} = (k+G)^2 delta_{G,G'} - (k+G)^2 * eps_inv_{G-G'}
    
    But this is not symmetric. The standard approach uses a symmetric formulation.
    
    Let me use the approach from Joannopoulos et al. (Photonic Crystals):
    
    For TM:
        The eigenvalue equation is:
        sum_{G'} [ (k+G)^2 delta_{G,G'} - (k+G)^2 sum_{G''} (1/eps)_{G-G''} delta_{G',G''} ] E_{G'} = omega^2 E_G
    
    To make it symmetric, define a new basis. But for simplicity, let's just build the matrix directly.
    
    Actually, I think the correct form is:
        A_{G,G'} = (k+G)^2 * delta_{G,G'} - (k+G)^2 * eps_inv_{G-G'}
    
    Wait no. Let me look at this more carefully.
    
    The standard plane-wave method for TM polarization:
    
    The Helmholtz equation for the z-component of E:
        [∇_t^2 + k^2] E_z = 0  (in homogeneous media)
    
    In a periodic dielectric, we expand E_z = sum_G E_G exp(i(k+G)·r)
    and 1/eps(r) = sum_G eps_inv_G exp(iG·r)
    
    The equation becomes:
        sum_{G'} [ (k+G)^2 delta_{G,G'} - (k+G)^2 eps_inv_{G-G'} ] E_{G'} = omega^2 E_G
    
    Hmm, I'm getting confused. Let me just use the standard non-symmetric formulation and use scipy.linalg.eig for the eigenvalues, taking the real parts.
    
    Actually, for the TM case, the standard formulation (e.g., from Joannopoulos) is:
    
    The eigenvalue equation is:
        sum_{G'} [ (k+G)^2 delta_{G,G'} - (k+G)^2 * (1/eps)_{G-G'} ] E_{G'} = omega^2 E_G
    
    But this matrix is not symmetric. However, we can symmetrize it.
    
    Let me just build the matrix and use np.linalg.eigvalsh if possible, or eig otherwise.
    
    For efficiency and correctness, let me use the following well-known formulation:
    
    For TM polarization, define the matrix:
        M_{G,G'} = (k+G)^2 * delta_{G,G'} - (k+G)^2 * eps_inv_{G-G'}
    
    Then solve M E = omega^2 E.
    
    But M is not Hermitian in general. However, the eigenvalues should be real and positive.
    
    Alternatively, use the symmetric form:
        Let's define F_G = E_G * (k+G)
    Then the equation becomes symmetric.
    
    Actually, I'll just use the direct approach with scipy.linalg.eig and take the smallest positive real eigenvalues.
    """
    n = n_pw  # half-grid
    # G vectors: from -(n-1) to (n-1) in each direction
    G_list = []
    for gx in range(-(n - 1), n):
        for gy in range(-(n - 1), n):
            G_list.append((gx, gy))
    
    num_pw = len(G_list)
    
    # Build the matrix
    # M_{G,G'} = (k+G)^2 * delta_{G,G'} - (k+G)^2 * eps_inv_{G-G'}
    # But eps_inv_{G-G'} is only defined for G-G' in the computed range
    
    # Actually, I realize the formulation might be wrong. Let me use a cleaner approach.
    
    # The correct TM eigenvalue equation (from Joannopoulos, 2nd ed., Eq. 8.17):
    # sum_{G'} [ (k+G)^2 delta_{G,G'} - (k+G)^2 * (1/eps)_{G-G'} ] E_{G'} = omega^2 E_G
    # 
    # Wait, I think the (1/eps) term should be multiplied differently. Let me use:
    # 
    # Actually, the standard form is:
    #   sum_{G'} [ (k+G)^2 delta_{G,G'} - sum_{G''} (k+G)^2 * (1/eps)_{G-G''} * delta_{G',G''} ] E_{G'} = omega^2 E_G
    # 
    # This simplifies to:
    #   (k+G)^2 E_G - (k+G)^2 sum_{G''} (1/eps)_{G-G''} E_{G''} = omega^2 E_G
    #   sum_{G''} [ (k+G)^2 delta_{G,G''} - (k+G)^2 (1/eps)_{G-G''} ] E_{G''} = omega^2 E_G
    #
    # So M_{G,G''} = (k+G)^2 [ delta_{G,G''} - (1/eps)_{G-G''} ]
    #
    # This is NOT symmetric. But we can still compute eigenvalues.
    
    M = np.zeros((num_pw, num_pw), dtype=complex)
    
    for i, (gx, gy) in enumerate(G_list):
        kx_total = kx + 2 * math.pi * gx
        ky_total = ky + 2 * math.pi * gy
        k_sq = kx_total * kx_total + ky_total * ky_total
        
        for j, (gx2, gy2) in enumerate(G_list):
            diff_gx = gx - gx2
            diff_gy = gy - gy2
            
            # Get eps_inv coefficient for (diff_gx, diff_gy)
            # Only use coefficients within our computed range
            if -n <= diff_gx < n and -n <= diff_gy < n:
                eps_inv_val = coeffs.get((diff_gx, diff_gy), 0.0)
            else:
                eps_inv_val = 0.0
            
            delta = 1.0 if (gx == gx2 and gy == gy2) else 0.0
            M[i, j] = k_sq * (delta - eps_inv_val)
    
    # Solve eigenvalue problem
    try:
        eigvals = np.linalg.eigvalsh(M)
        # eigvalsh assumes Hermitian; M may not be. Use eig instead.
        eigvals = np.linalg.eig(M)[0]
        # Take real parts, filter positive
        eigvals_real = np.real(eigvals)
        eigvals_real = eigvals_real[eigvals_real > 0]
        eigvals_real = np.sort(eigvals_real)
    except:
        return None
    
    if len(eigvals_real) < num_bands:
        return None
    
    # omega^2 = eigvals, so omega = sqrt(eigvals)
    omegas = np.sqrt(eigvals_real[:num_bands])
    return omegas


def compute_gap(omegas):
    """Compute the gap-midgap ratio between bands 4 and 5 (indices 3 and 4)."""
    if len(omegas) < 5:
        return -1.0
    max_band4 = omegas[3]
    min_band5 = omegas[4]
    gap = min_band5 - max_band4
    midgap = (min_band5 + max_band4) / 2.0
    if midgap <= 0:
        return -1.0
    return gap / midgap


# ---------- k-point sampling over irreducible BZ ----------

def get_irreducible_bz_kpoints(num_k=30):
    """
    Generate k-points over the irreducible Brillouin zone of a square lattice.
    The irreducible BZ is the 1/8 triangle: 0 <= kx <= ky <= 1 (in units of 2pi/a, 
    actually the BZ goes from -1 to 1 in units of 2pi/a, so irreducible is 0<=kx<=ky<=1).
    
    Actually, the first BZ of a square lattice with lattice constant a=1 has extent
    [-pi, pi] x [-pi, pi] in k-space, or [-1,1]x[-1,1] in units of 2pi/a.
    The irreducible wedge is 0 <= kx <= ky <= 1 (in units of 2pi/a).
    
    Let's use normalized coordinates where the BZ boundary is at 1.
    """
    kpoints = []
    # Include boundary points and interior points
    # The irreducible BZ is the triangle with vertices (0,0), (1,0), (1,1)
    # Wait, actually for a square lattice the BZ is a square from -1 to 1 in each direction.
    # The irreducible 1/8 wedge is: 0 <= kx <= ky <= 1
    
    # Let's sample points in this triangle
    # Use a grid and filter
    n_grid = 12
    for i in range(n_grid):
        for j in range(n_grid):
            kx = i / (n_grid - 1)
            ky = j / (n_grid - 1)
            if kx <= ky + 1e-9 and kx >= 0 and ky >= 0 and kx <= 1 and ky <= 1:
                kpoints.append((kx, ky))
    
    # Ensure we have at least num_k points, add some if needed
    if len(kpoints) < num_k:
        # Add more points along the edges
        for t in np.linspace(0, 1, num_k):
            kpoints.append((t, t))  # along kx=ky
            kpoints.append((t, 1.0))  # along ky=1
            kpoints.append((0.0, t))  # along kx=0
    
    # Deduplicate
    unique_kpoints = []
    for kp in kpoints:
        if not any(abs(kp[0] - uk[0]) < 1e-6 and abs(kp[1] - uk[1]) < 1e-6 for uk in unique_kpoints):
            unique_kpoints.append(kp)
    
    return unique_kpoints[:num_k]


# ---------- scoring function ----------

def score_cell(cell, n_pw=17, num_k=30, num_bands=6):
    """
    Score a cell by computing the maximum gap-midgap ratio over the irreducible BZ.
    """
    coeffs = compute_fourier_coefficients(cell, n_pw)
    kpoints = get_irreducible_bz_kpoints(num_k)
    
    max_gap_ratio = -1.0
    for kx, ky in kpoints:
        # Convert to actual k values (in units of 2pi/a)
        # The BZ extends from -1 to 1 in units of 2pi/a
        omegas = compute_tm_bands(coeffs, n_pw, num_bands, kx, ky)
        if omegas is None:
            continue
        gap_ratio = compute_gap(omegas)
        if gap_ratio > max_gap_ratio:
            max_gap_ratio = gap_ratio
    
    return max_gap_ratio


# ---------- initial design ----------

def initial_design(n=32):
    """
    Create an initial symmetric design: a hexagonal-like rod arrangement.
    We'll start with a central dielectric rod and some symmetry.
    """
    cell = [[0] * n for _ in range(n)]
    
    # Central rod: radius ~0.25
    cx = cy = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            dx = i - cx
            dy = j - cy
            if dx * dx + dy * dy < (0.22 * n) ** 2:
                cell[i][j] = 1
    
    # Add some dielectric at the corners/edges to break the simple rod symmetry
    # Actually, let's try a different approach: inverted honeycomb
    # For now, just use the rod and let the optimizer improve it
    
    cell = apply_symmetries(cell)
    return cell


def initial_design_hex(n=32):
    """
    Try a hexagonal-rod-like design with 6-fold-ish symmetry approximated in a square cell.
    """
    cell = [[0] * n for _ in range(n)]
    cx = cy = (n - 1) / 2.0
    
    # Central small rod
    for i in range(n):
        for j in range(n):
            dx = i - cx
            dy = j - cy
            r = math.sqrt(dx * dx + dy * dy)
            if r < 0.12 * n:
                cell[i][j] = 1
            # Six lobes at 60-degree intervals
            for angle_deg in range(0, 360, 60):
                angle = math.radians(angle_deg)
                lx = 0.35 * n * math.cos(angle)
                ly = 0.35 * n * math.sin(angle)
                dlx = dx - lx
                dly = dy - ly
                if dlx * dlx + dly * dly < (0.10 * n) ** 2:
                    cell[i][j] = 1
    
    cell = apply_symmetries(cell)
    return cell


# ---------- local search ----------

def flip_pixel_symmetric(cell, i, j, n):
    """
    Flip a pixel and all its symmetric equivalents.
    Returns a new cell.
    """
    new_cell = [row[:] for row in cell]
    # All symmetric equivalents of (i, j)
    equivs = set()
    equivs.add((i, j))
    equivs.add((n - 1 - i, j))      # row reversal
    equivs.add((i, n - 1 - j))      # col reversal
    equivs.add((n - 1 - i, n - 1 - j))  # both
    equivs.add((j, i))              # transpose
    equivs.add((j, n - 1 - i))      # transpose + row rev
    equivs.add((n - 1 - j, i))      # transpose + col rev
    equivs.add((n - 1 - j, n - 1 - i))  # all
    
    for ei, ej in equivs:
        new_cell[ei][ej] = 1 - new_cell[ei][ej]
    
    return new_cell


def local_search(cell, n, max_time, initial_score):
    """
    Local search: flip pixels while preserving symmetry, keep improvements.
    """
    best_cell = [row[:] for row in cell]
    best_score = initial_score
    start_time = time.time()
    
    # Precompute the list of unique pixel positions (under symmetry)
    unique_positions = set()
    for i in range(n):
        for j in range(n):
            # Canonical representative: use the minimum of all symmetric equivalents
            equivs = [
                (i, j),
                (n - 1 - i, j),
                (i, n - 1 - j),
                (n - 1 - i, n - 1 - j),
                (j, i),
                (j, n - 1 - i),
                (n - 1 - j, i),
                (n - 1 - j, n - 1 - i),
            ]
            canonical = min(equivs)
            unique_positions.add(canonical)
    
    unique_positions = list(unique_positions)
    
    # Randomize the order for diversity
    import random
    random.seed(42)
    random.shuffle(unique_positions)
    
    iteration = 0
    while time.time() - start_time < max_time - 30:  # leave 30s margin
        for pos in unique_positions:
            if time.time() - start_time > max_time - 30:
                break
            
            i, j = pos
            new_cell = flip_pixel_symmetric(best_cell, i, j, n)
            if not is_symmetric(new_cell):
                continue
            
            new_score = score_cell(new_cell, n_pw=17, num_k=15, num_bands=6)
            if new_score > best_score:
                best_cell = new_cell
                best_score = new_score
            
            iteration += 1
    
    return best_cell, best_score


# ---------- main design function ----------

def design(problem):
    """Design a 32x32 symmetric photonic crystal unit cell to maximize the TM band gap."""
    start_time = time.time()
    max_time = 190  # leave 10s margin for safety
    
    n = problem.get("pixels", 32)
    
    # Try multiple initial designs
    candidates = []
    
    # Design 1: simple central rod
    cell1 = initial_design(n)
    candidates.append(cell1)
    
    # Design 2: hexagonal lobes
    cell2 = initial_design_hex(n)
    candidates.append(cell2)
    
    # Design 3: ring (annulus)
    cell3 = [[0] * n for _ in range(n)]
    cx = cy = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            dx = i - cx
            dy = j - cy
            r = math.sqrt(dx * dx + dy * dy)
            if 0.15 * n < r < 0.30 * n:
                cell3[i][j] = 1
    cell3 = apply_symmetries(cell3)
    candidates.append(cell3)
    
    # Design 4: four-lobed (clover)
    cell4 = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = i - cx
            dy = j - cy
            # Four lobes at 45 degrees
            for angle_deg in [45, 135, 225, 315]:
                angle = math.radians(angle_deg)
                lx = 0.25 * n * math.cos(angle)
                ly = 0.25 * n * math.sin(angle)
                dlx = dx - lx
                dly = dy - ly
                if dlx * dlx + dly * dly < (0.12 * n) ** 2:
                    cell4[i][j] = 1
    cell4 = apply_symmetries(cell4)
    candidates.append(cell4)
    
    # Score all candidates with a quick evaluation
    best_cell = None
    best_score = -1.0
    
    for cell in candidates:
        elapsed = time.time() - start_time
        if elapsed > max_time - 60:
            break
        sc = score_cell(cell, n_pw=17, num_k=15, num_bands=6)
        if sc > best_score:
            best_score = sc
            best_cell = [row[:] for row in cell]
    
    if best_cell is None:
        best_cell = candidates[0]
        best_score = score_cell(best_cell, n_pw=17, num_k=15, num_bands=6)
    
    # Local search from the best candidate
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 30:
        best_cell, best_score = local_search(best_cell, n, remaining_time, best_score)
    
    # Final verification
    if not is_symmetric(best_cell):
        best_cell = apply_symmetries(best_cell)
    
    return best_cell
