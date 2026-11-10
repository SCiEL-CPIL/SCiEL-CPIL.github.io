"""Rewritten program: optimize 4-fold symmetric pixel cell for max 4-5 TM gap
using a local plane-wave band solver + simulated annealing + greedy refinement."""
import time
import math
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

# ---------------- Local TM plane-wave band solver ----------------
def _tm_bands(cell, n_pw, kx, ky, eps_high=12.0):
    """Compute TM eigenvalues (omega^2) using plane-wave Galerkin with exact
    Fourier coefficients of the pixel map. Returns sorted eigenvalues."""
    N = len(cell)
    eps = np.asarray(cell, dtype=float)
    # Fourier coefficients: eps(G) = (1/N^2) sum eps(x) exp(-i G.x)
    # Use 2D FFT. eps_grid[i,j] corresponds to position (i+0.5, j+0.5)/N - 0.5
    # G indices: g = -N/2 .. N/2-1
    # FFT of eps gives coefficients for G = 2*pi*g/L with L=1 (lattice constant)
    eps_fft = np.fft.fft2(eps)
    eps_fft = np.fft.fftshift(eps_fft) / (N * N)
    # G vectors
    gvals = np.fft.fftfreq(N, d=1.0 / N)  # gives -N/2..N/2-1 * 2*pi? Let's be careful.
    # Actually we want G = 2*pi * (m / L) with L = 1, m integer.
    # For a pixel grid of N pixels over length 1, the FFT frequencies are m (cycles per unit).
    # So G = 2*pi*m.
    m = np.fft.fftfreq(N, d=1.0 / N)  # this gives m values: 0,1,...,N/2-1,-N/2,...,-1
    m = np.fft.fftshift(m).astype(int)
    # Build G grid
    Gx, Gy = np.meshgrid(m, m, indexing='ij')
    # Plane wave selection: |G| <= n_pw (in units of 2*pi, i.e. |m| <= n_pw)
    # Actually n_pw is a radius in G-space. Let's use |m| <= n_pw.
    # k is in units of 2*pi (reduced coordinates).
    # kx, ky are given in 2*pi units (i.e. k = 2*pi*(kx, ky)).
    # Plane wave k + G has components kx + m_x, ky + m_y (in 2*pi units).
    # TM: eigenvalue problem: (k+G)^2 * A(G') = sum_G' delta(G-G') * eps(G-G') / eps_avg? 
    # Standard TM formulation: 
    # (k+G)^2 * A(G') = (1/eps_avg) sum_{G''} eps(G-G'') * (k+G'')^2 * A(G'')
    # where eps_avg = <eps> = eps(0).
    # Actually the standard form: 
    # sum_{G'} [ (k+G)^2 delta_{G,G'} - (1/eps(G-G')) (k+G')^2 ] A(G') = 0
    # This is not symmetric. The symmetric form uses 1/eps:
    # Let's use the standard symmetric formulation:
    # (k+G)^2 * A(G') = sum_{G''} [1/eps(G-G'')] (k+G'')^2 * A(G'')
    # Wait, the symmetric form is:
    # sum_{G'} [ delta_{G,G'} (k+G)^2 - (k+G)(k+G') / eps(G-G') ] A(G') = 0
    # where eps(G) here is the Fourier coefficient of 1/eps, not eps.
    # For TM: the symmetric eigenproblem is:
    # (k+G)^2 A(G') = sum_{G''} [ (k+G)(k+G'') / eps(G-G'') ] A(G'')
    # where eps(G) is the Fourier coefficient of 1/eps(x).
    # So we need Fourier coefficients of 1/eps, not eps!
    
    # Let me redo: For TM polarization, the field satisfies:
    # grad_perp^2 H_z + (omega^2/c^2) eps(x) H_z = 0
    # Plane wave expansion: H_z = sum_G A(G) exp(i(k+G).r)
    # This leads to: (k+G)^2 A(G) = sum_{G'} [ (k+G)(k+G') / eps(G-G') ] A(G')
    # where 1/eps(x) = sum_G [1/eps](G) exp(iG.r)
    # So we need Fourier coefficients of 1/eps.
    
    inv_eps = np.where(eps > 0, 1.0 / eps, 0.0)
    inv_eps_fft = np.fft.fft2(inv_eps)
    inv_eps_fft = np.fft.fftshift(inv_eps_fft) / (N * N)
    
    # Select plane waves with |m| <= n_pw
    mask = (Gx * Gx + Gy * Gy) <= n_pw * n_pw
    G_indices = np.where(mask)
    M = len(G_indices[0])
    if M == 0:
        return np.array([])
    
    # k in 2*pi units
    kx2 = kx
    ky2 = ky
    
    # Build matrix
    # For each selected plane wave i with G_i = (m_xi, m_yi):
    # (kx2 + m_xi)^2 + (ky2 + m_yi)^2 on diagonal
    # Off-diagonal: ((kx2+m_xi)(kx2+m_xj) + (ky2+m_yi)(ky2+m_yj)) * inv_eps_fft[Gi-Gj]
    
    mx = Gx[G_indices]
    my = Gy[G_indices]
    
    kx_plus = kx2 + mx
    ky_plus = ky2 + my
    diag = kx_plus * kx_plus + ky_plus * ky_plus
    
    # inv_eps_fft is indexed by (m_x - m_x', m_y - m_y') in shifted FFT coordinates
    # We need inv_eps_fft at G_i - G_j for all pairs.
    # G_i - G_j = (m_xi - m_xj, m_yi - m_yj)
    # In the shifted FFT array, the index for frequency (a, b) is at:
    # idx_x = (a + N//2) % N, idx_y = (b + N//2) % N
    # But our m range is -N/2 to N/2-1, so m_xi - m_xj ranges from -(N-1) to N-1.
    # We need to wrap into the FFT grid.
    
    mat = np.zeros((M, M), dtype=complex)
    for i in range(M):
        for j in range(M):
            dx = mx[i] - mx[j]
            dy = my[i] - my[j]
            # Wrap into FFT index
            idx_x = (dx + N // 2) % N
            idx_y = (dy + N // 2) % N
            val = inv_eps_fft[idx_x, idx_y]
            mat[i, j] = (kx_plus[i] * kx_plus[j] + ky_plus[i] * ky_plus[j]) * val
            if i == j:
                mat[i, j] = diag[i]
    
    # Solve eigenvalue problem
    # mat is Hermitian (should be real symmetric if inv_eps is real and even)
    mat = (mat + mat.conj().T) / 2.0
    mat = mat.real  # should be real
    
    try:
        vals = np.linalg.eigvalsh(mat)
        vals = vals[vals > 0]
        return vals
    except Exception:
        return np.array([])


def _score_cell(cell, n_pw=10, n_kpts=12):
    """Score a cell: compute 4-5 TM gap-midgap ratio over BZ."""
    # Irreducible BZ of square lattice: triangle with vertices (0,0), (0.5,0), (0.5,0.5)
    # in 2*pi units. Sample points on boundary and interior.
    # Key points: Gamma (0,0), X (0.5,0), M (0.5,0.5), and lines.
    kpts = []
    # Gamma
    kpts.append((0.0, 0.0))
    # X
    kpts.append((0.5, 0.0))
    # M
    kpts.append((0.5, 0.5))
    # Line Gamma-X: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5)[1:]:
        kpts.append((t, 0.0))
    # Line X-M: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5)[1:]:
        kpts.append((0.5, t))
    # Line Gamma-M: (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 5)[1:]:
        kpts.append((t, t))
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in kpts:
        vals = _tm_bands(cell, n_pw, kx, ky)
        if len(vals) < 5:
            continue
        omega4 = math.sqrt(vals[3])  # 4th band (0-indexed: 3)
        omega5 = math.sqrt(vals[4])  # 5th band
        if omega4 > max_omega4:
            max_omega4 = omega4
        if omega5 < min_omega5:
            min_omega5 = omega5
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return -1.0
    
    gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    return gap_ratio


# ---------------- Symmetry enforcement ----------------
def _apply_4fold_sym(cell):
    """Enforce 4-fold rotational symmetry about cell center.
    For a 32x32 cell, the center is at (15.5, 15.5).
    90-degree rotation about center: (i, j) -> (j, N-1-i) for N=32.
    Actually for even N, rotation about center: 
    new_i = j, new_j = N-1-i
    Let's verify: center is at ((N-1)/2, (N-1)/2) = (15.5, 15.5).
    Point (i, j) relative to center: (i-15.5, j-15.5).
    90 deg CCW: (-(j-15.5), i-15.5) = (15.5-j, i-15.5).
    Absolute: (i', j') = (15.5-j+15.5, i-15.5+15.5) = (31-j, i).
    So (i, j) -> (31-j, i) for N=32.
    """
    N = len(cell)
    cell = np.array(cell)
    # Apply 4 rotations and average (or take majority)
    c0 = cell.copy()
    c1 = np.rot90(c0, 1)
    c2 = np.rot90(c0, 2)
    c3 = np.rot90(c0, 3)
    # Average and threshold
    avg = (c0.astype(float) + c1.astype(float) + c2.astype(float) + c3.astype(float)) / 4.0
    return (avg >= 0.5).astype(int)


def _make_symmetric_from_coarse(coarse, block=4):
    """Create a 32x32 cell from an 8x8 coarse grid, each coarse pixel = 4x4 block.
    Then enforce 4-fold symmetry."""
    N_coarse = len(coarse)
    N = N_coarse * block
    cell = np.zeros((N, N), dtype=int)
    for i in range(N_coarse):
        for j in range(N_coarse):
            cell[i*block:(i+1)*block, j*block:(j+1)*block] = coarse[i, j]
    # Enforce 4-fold symmetry
    cell = _apply_4fold_sym(cell)
    return cell.tolist()


# ---------------- Main design function ----------------
def design(problem):
    """Optimize a 4-fold symmetric 32x32 pixel cell for max 4-5 TM gap."""
    t_start = time.time()
    deadline = t_start + 180.0  # safety margin before 230s kill
    
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12.0)
    
    # Start with a reasonable initial design: dielectric rod
    N_coarse = 8
    block = size // N_coarse  # 4
    coarse = np.zeros((N_coarse, N_coarse), dtype=int)
    # Initial: central 2x2 coarse pixels = dielectric (radius ~0.25)
    coarse[3:5, 3:5] = 1
    
    cell = _make_symmetric_from_coarse(coarse, block)
    
    # Evaluate initial
    best_score = _score_cell(cell, n_pw=8)
    best_cell = [row[:] for row in cell]
    
    # Simulated annealing on coarse grid
    # Free variables: independent coarse pixels under 4-fold symmetry
    # For 8x8 with 4-fold symmetry about center (3.5, 3.5):
    # (i,j) -> (7-j, i) for 90 deg rotation.
    # Fundamental domain: i <= 3, and for i==3, j <= 3 (roughly)
    # Let's just allow all coarse pixels but enforce symmetry after each flip.
    
    T = 1.0
    T_min = 0.01
    cooling = 0.98
    
    np.random.seed(42)
    
    while T > T_min:
        if time.time() > deadline - 30.0:
            break
        
        # Random flip in coarse grid
        i = np.random.randint(N_coarse)
        j = np.random.randint(N_coarse)
        old_val = coarse[i, j]
        coarse[i, j] = 1 - old_val
        
        new_cell = _make_symmetric_from_coarse(coarse, block)
        new_score = _score_cell(new_cell, n_pw=8)
        
        delta = new_score - best_score
        if delta > 0 or np.random.random() < math.exp(delta / T):
            best_score = new_score
            best_cell = [row[:] for row in new_cell]
        else:
            coarse[i, j] = old_val  # revert
        
        T *= cooling
    
    # Greedy refinement on full 32x32 grid
    cell = np.array(best_cell)
    improved = True
    while improved and time.time() < deadline - 20.0:
        improved = False
        for i in range(size):
            for j in range(size):
                if time.time() > deadline - 20.0:
                    break
                old_val = cell[i, j]
                cell[i, j] = 1 - old_val
                cell = _apply_4fold_sym(cell)
                new_score = _score_cell(cell, n_pw=8)
                if new_score > best_score:
                    best_score = new_score
                    best_cell = cell.tolist()
                    improved = True
                else:
                    cell[i, j] = old_val
                    cell = _apply_4fold_sym(cell)
            if time.time() > deadline - 20.0:
                break
    
    # Final: try a few more n_pw evaluations to verify
    # Return best cell
    return best_cell
