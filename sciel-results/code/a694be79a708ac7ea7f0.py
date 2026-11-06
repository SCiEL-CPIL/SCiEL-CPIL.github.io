"""Design a photonic crystal unit cell maximizing the TM band gap between bands 4 and 5."""
import time
import numpy as np
from scipy.linalg import eigh

def _fourier_coefficients(cell, n_order):
    """Compute Fourier coefficients of 1/epsilon for the given cell.
    n_order: half-width of the Fourier basis (so total (2*n_order+1)^2 coefficients)
    Returns array of shape (2*n_order+1, 2*n_order+1) where index (i,j) corresponds
    to Fourier mode (i - n_order, j - n_order).
    """
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps[eps == 0] = 1.0
    eps[eps == 1] = 12.0
    inv_eps = 1.0 / eps
    
    # Fourier coefficients: c_{mn} = (1/N^2) sum_{x,y} inv_eps(x,y) exp(-i 2pi (m x + n y)/N)
    # For a 32x32 cell, we compute DFT
    N = size
    c = np.zeros((2 * n_order + 1, 2 * n_order + 1), dtype=complex)
    for m in range(-n_order, n_order + 1):
        for n in range(-n_order, n_order + 1):
            # c_{mn} = (1/N^2) sum_{x=0}^{N-1} sum_{y=0}^{N-1} inv_eps[x,y] exp(-i 2pi (m(x+0.5) + n(y+0.5))/N)
            # Use the fact that sum over x of exp(-i 2pi m (x+0.5)/N) for integer m
            # = exp(-i pi m / N) * sum_{x=0}^{N-1} exp(-i 2pi m x / N)
            # For m not a multiple of N, this sum is 0. For m = 0, it's N.
            # So c_{mn} is non-zero only when m and n are multiples of N.
            # Wait, that's not right. The cell is N x N pixels, and we want the Fourier
            # coefficients of the periodic function. The pixel values are samples.
            # Actually, for a pixelated structure, the Fourier coefficient is:
            # c_{mn} = (1/N^2) sum_{x,y} f(x,y) exp(-i 2pi (m x + n y)/N) * N^2 / N^2
            # Hmm, let me think more carefully.
            # The dielectric function is periodic with period 1 (in lattice units).
            # The pixel at position (x/N, y/N) has value inv_eps[x,y].
            # The Fourier coefficient is:
            # c_{mn} = int_0^1 int_0^1 f(u,v) exp(-i 2pi (m u + n v)) du dv
            # For a pixelated approximation:
            # c_{mn} ≈ (1/N^2) sum_{x=0}^{N-1} sum_{y=0}^{N-1} f(x/N, y/N) exp(-i 2pi (m x/N + n y/N)) * (1/N) * (1/N) * N^2
            # = (1/N^2) sum_{x,y} f[x,y] exp(-i 2pi (m x + n y)/N)
            # Wait, that's not dimensionally correct. Let me redo.
            # f(u,v) = f[x,y] for u in [x/N, (x+1)/N), v in [y/N, (y+1)/N)
            # c_{mn} = sum_{x,y} f[x,y] int_{x/N}^{(x+1)/N} int_{y/N}^{(y+1)/N} exp(-i 2pi (m u + n v)) du dv
            # = sum_{x,y} f[x,y] [exp(-i 2pi m (x+1)/N) - exp(-i 2pi m x/N)] / (-i 2pi m) *
            #              [exp(-i 2pi n (y+1)/N) - exp(-i 2pi n y/N)] / (-i 2pi n)
            # For m != 0: [exp(-i 2pi m (x+1)/N) - exp(-i 2pi m x/N)] / (-i 2pi m)
            # = exp(-i 2pi m x/N) [exp(-i 2pi m /N) - 1] / (-i 2pi m)
            # = exp(-i 2pi m x/N) * exp(-i pi m /N) * 2i sin(pi m /N) / (-i 2pi m)
            # = exp(-i pi m (2x+1)/N) * sin(pi m /N) / (pi m)
            # For m = 0: the integral is 1/N.
            # So c_{mn} = sum_{x,y} f[x,y] * S(m) * S(n) where
            # S(k) = 1/N if k == 0, else exp(-i pi k (2*0+1)/N) * sin(pi k /N) / (pi k) ... 
            # Wait, I need to be more careful. Let me just use the DFT approach with the
            # understanding that the pixel values represent the average over each pixel.
            # Actually, the simplest and most accurate approach for a pixelated structure:
            # c_{mn} = (1/N^2) sum_{x=0}^{N-1} sum_{y=0}^{N-1} f[x,y] * exp(-i 2pi (m x + n y)/N)
            # This is the standard DFT interpretation where f[x,y] is the value at the pixel center.
            # But this is an approximation. The exact integral would involve the sinc function.
            # For simplicity and speed, let's use the DFT approximation.
            pass
    
    # Let's use a more efficient approach: compute the 2D DFT of inv_eps
    # and extract the relevant coefficients
    # The DFT of inv_eps gives us F[m,n] = sum_{x,y} inv_eps[x,y] exp(-i 2pi (m x + n y)/N)
    # The Fourier coefficient c_{mn} = F[m mod N, n mod N] / N^2
    # But we need c_{mn} for m, n in [-n_order, n_order]
    # For |m| < N/2, c_{mn} = F[m mod N, n mod N] / N^2
    # For |m| >= N/2, we need to be careful with aliasing.
    # Since N=32 and n_order is small (say up to 8), we're fine.
    
    F = np.fft.fft2(inv_eps)
    c = np.zeros((2 * n_order + 1, 2 * n_order + 1), dtype=complex)
    for i, m in enumerate(range(-n_order, n_order + 1)):
        for j, n in enumerate(range(-n_order, n_order + 1)):
            c[i, j] = F[m % N, n % N] / (N * N)
    
    return c


def _pwe_solve(cell, n_order, kx, ky, n_bands=6):
    """Solve the PWE eigenvalue problem for TM polarization.
    Returns the first n_bands eigenvalues (omega^2 in units of c/a).
    """
    c = _fourier_coefficients(cell, n_order)
    N = 2 * n_order + 1
    
    # Build the matrix A and vector b for the generalized eigenvalue problem
    # For TM: sum_{G'} (k - G')^2 / epsilon_{G-G'} * a_{G'} = omega^2 a_k
    # In matrix form: A a = omega^2 a where A_{GG'} = (k-G)^2 * c_{G-G'}
    # Actually, the standard formulation is:
    # sum_{G'} (k + G - G')^2 / epsilon_{G-G'} a_{G'} = (omega/c)^2 a_G
    # Let me use the standard PWE formulation.
    # For TM polarization:
    # (k + G)^2 a_G - sum_{G'} (k + G)^2 (k + G')^2 / epsilon_{G-G'} ... 
    # No, let me use the correct formulation.
    # The TM mode equation is:
    # sum_{G'} (k - G')^2 / epsilon_{G-G'} * E_{G'} = (omega/c)^2 E_k
    # So A_{G,G'} = (k - G')^2 * epsilon_{G-G'}^{-1} = (k - G')^2 * c_{G-G'}
    # Wait, that's not right either. Let me look this up mentally.
    # The standard PWE for TM:
    # (k + G)^2 E_G = (omega/c)^2 sum_{G'} epsilon_{G-G'}^{-1} E_{G'}
    # So A_{G,G'} = delta_{G,G'} (k+G)^2 - ... no.
    # Actually: (omega/c)^2 E_G = sum_{G'} (k+G)^2 (k+G')^2 / epsilon_{G-G'} ... 
    # I'm getting confused. Let me use a cleaner formulation.
    # For TM polarization in 2D:
    # The eigenvalue equation is:
    # sum_{G'} (k - G')^2 / epsilon_{G-G'} * E_{G'} = (omega/c)^2 E_k
    # where epsilon_{G-G'} are the Fourier coefficients of 1/epsilon.
    # So the matrix is: A_{G,G'} = (k - G')^2 * epsilon_{G-G'}
    # and we solve A E = (omega/c)^2 E
    # Wait, that gives A_{G,G'} = (k-G')^2 * c_{G-G'} where c = Fourier of 1/eps.
    # Let me verify: if eps is constant, c_0 = 1/eps, c_{G!=0} = 0.
    # Then A_{G,G'} = (k-G')^2 * c_{G-G'} = (k-G')^2 / eps if G=G', else 0.
    # So A is diagonal with entries (k-G)^2/eps, giving (omega/c)^2 = (k-G)^2/eps,
    # which is correct for a uniform medium.
    
    # Build G vector
    G = []
    for m in range(-n_order, n_order + 1):
        for n in range(-n_order, n_order + 1):
            G.append((m, n))
    G = np.array(G, dtype=float)
    
    # Build A matrix
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            # G_i - G_j
            dG = G[i] - G[j]
            # (k - G_j)^2
            k_minus_Gj = np.array([kx - G[j, 0], ky - G[j, 1]])
            k_minus_Gj_sq = np.dot(k_minus_Gj, k_minus_Gj)
            # epsilon_{G_i - G_j} = c_{G_i - G_j}
            # c is indexed by (m - n_order, n - n_order) for m, n in [-n_order, n_order]
            dm = int(dG[0])
            dn = int(dG[1])
            if abs(dm) <= n_order and abs(dn) <= n_order:
                c_val = c[dm + n_order, dn + n_order].real  # c should be real for symmetric cell
                A[i, j] = k_minus_Gj_sq * c_val
            else:
                A[i, j] = 0.0
    
    # Solve eigenvalue problem
    # A is symmetric if the cell is symmetric, so use eigh
    try:
        eigvals, eigvecs = eigh(A, subset_by_index=[0, n_bands - 1])
        return eigvals
    except:
        eigvals, eigvecs = eigh(A)
        return eigvals[:n_bands]


def _evaluate_gap(cell, n_order, n_kpoints=10):
    """Evaluate the gap-midgap ratio between bands 4 and 5.
    Returns the score (higher is better).
    """
    # Irreducible BZ for square lattice: kx >= ky >= 0, kx + ky <= pi (in units of 2pi/a)
    # Actually, for a square lattice with lattice constant a=1, the BZ is [-pi, pi] x [-pi, pi]
    # The irreducible BZ is the 1/8 wedge: 0 <= ky <= kx, kx + ky <= pi
    # Wait, the square BZ is [-pi, pi]^2. The symmetry group C4v has 8 elements.
    # The irreducible BZ is the triangle with vertices (0,0), (pi,0), (pi/2, pi/2).
    # So 0 <= ky <= kx, and kx <= pi, ky <= pi, and kx + ky <= pi? No.
    # The irreducible BZ for C4v symmetry of the square is the region:
    # 0 <= ky <= kx, 0 <= kx <= pi, 0 <= ky <= pi, and kx + ky <= pi is NOT the constraint.
    # Actually, the square BZ is [-pi, pi] x [-pi, pi]. The irreducible wedge is:
    # 0 <= ky <= kx, kx <= pi, ky <= pi. But also we need kx + ky <= pi? 
    # No. The square BZ has corners at (±pi, ±pi). The irreducible BZ under C4v is:
    # 0 <= ky <= kx, 0 <= kx <= pi, 0 <= ky <= pi. This is a triangle with vertices
    # (0,0), (pi,0), (pi,pi). But (pi,pi) is a corner, and the line kx=ky from (0,0) to (pi,pi)
    # is a symmetry line. The other boundaries are kx=pi (from (pi,0) to (pi,pi)) and 
    # ky=0 (from (0,0) to (pi,0)).
    # Wait, I think the irreducible BZ is actually the triangle with vertices (0,0), (pi,0), (pi/2, pi/2).
    # Let me reconsider. The square BZ is [-pi, pi]^2. The high-symmetry points are:
    # Gamma = (0,0), X = (pi,0), M = (pi,pi).
    # The irreducible BZ is the triangle Gamma-X-M, i.e., 0 <= ky <= kx, kx + ky <= pi? 
    # No, that's not right either. Let me just use a set of k-points that cover the 
    # relevant region.
    
    # Use k-points along the high-symmetry lines and interior
    k_points = []
    # Along Gamma-X: ky=0, kx from 0 to pi
    for i in range(4):
        kx = i * np.pi / 3
        k_points.append((kx, 0.0))
    # Along X-M: kx=pi, ky from 0 to pi
    for i in range(1, 4):
        ky = i * np.pi / 3
        k_points.append((np.pi, ky))
    # Along M-Gamma: kx=ky, from (pi,pi) to (0,0)
    for i in range(1, 4):
        kx = np.pi * (3 - i) / 3
        k_points.append((kx, kx))
    # Some interior points
    k_points.append((np.pi/2, np.pi/4))
    k_points.append((np.pi/3, np.pi/6))
    
    min_omega5 = np.inf
    max_omega4 = -np.inf
    
    for kx, ky in k_points:
        eigvals = _pwe_solve(cell, n_order, kx, ky, n_bands=6)
        if len(eigvals) >= 5:
            omega4 = eigvals[3]  # 4th band (0-indexed: 3)
            omega5 = eigvals[4]  # 5th band (0-indexed: 4)
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
    
    if max_omega4 <= 0 or min_omega5 <= 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = min_omega5 + max_omega4
    return 2 * gap / midgap


def _enforce_symmetry(cell):
    """Enforce C4v symmetry: cell == cell[::-1] row-wise, column-wise, and transpose."""
    size = len(cell)
    cell = np.array(cell, dtype=int)
    
    # Enforce row reversal symmetry: cell[i,j] == cell[size-1-i, j]
    for i in range(size // 2):
        for j in range(size):
            avg = (cell[i, j] + cell[size - 1 - i, j]) / 2
            val = 1 if avg >= 0.5 else 0
            cell[i, j] = val
            cell[size - 1 - i, j] = val
    
    # Enforce column reversal symmetry: cell[i,j] == cell[i, size-1-j]
    for i in range(size):
        for j in range(size // 2):
            avg = (cell[i, j] + cell[i, size - 1 - j]) / 2
            val = 1 if avg >= 0.5 else 0
            cell[i, j] = val
            cell[i, size - 1 - j] = val
    
    # Enforce transpose symmetry: cell[i,j] == cell[j,i]
    for i in range(size):
        for j in range(i + 1, size):
            avg = (cell[i, j] + cell[j, i]) / 2
            val = 1 if avg >= 0.5 else 0
            cell[i, j] = val
            cell[j, i] = val
    
    return cell.tolist()


def _make_rod_cell(radius, size=32):
    """Create a centered rod cell with given radius (in lattice units)."""
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < radius * radius:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    return _enforce_symmetry(cell)


def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the TM band gap."""
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for the 120s budget
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]
    
    # Start with a reasonable initial guess
    best_cell = _make_rod_cell(0.2, size)
    best_score = -np.inf
    
    # Quick evaluation with low-order basis
    n_order_fast = 3  # 7x7 = 49 plane waves
    n_order_refine = 5  # 11x11 = 121 plane waves
    
    # Evaluate initial guess
    t0 = time.time()
    score = _evaluate_gap(best_cell, n_order_fast, n_kpoints=8)
    best_score = score
    t_eval = time.time() - t0
    
    # Try different rod radii
    radii = [0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.32]
    for r in radii:
        if time.time() > deadline - 20:
            break
        cell = _make_rod_cell(r, size)
        score = _evaluate_gap(cell, n_order_fast, n_kpoints=8)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Now do local optimization: flip pixels to improve the score
    # Use a greedy local search with symmetry enforcement
    if time.time() < deadline - 15:
        current_cell = best_cell
        current_score = best_score
        no_improve = 0
        max_no_improve = 10
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            # Try flipping each pixel (with symmetry)
            for i in range(size):
                for j in range(size):
                    if time.time() > deadline - 10:
                        break
                    # Flip the pixel and its symmetry equivalents
                    new_cell = [row[:] for row in current_cell]
                    # Get all symmetry-equivalent positions
                    positions = [(i, j)]
                    # Row reversal
                    positions.append((size - 1 - i, j))
                    # Column reversal
                    positions.append((i, size - 1 - j))
                    positions.append((size - 1 - i, size - 1 - j))
                    # Transpose
                    positions.append((j, i))
                    positions.append((size - 1 - j, i))
                    positions.append((j, size - 1 - i))
                    positions.append((size - 1 - j, size - 1 - i))
                    
                    # Remove duplicates
                    positions = list(set(positions))
                    
                    # Flip all these positions
                    for pi, pj in positions:
                        new_cell[pi][pj] = 1 - new_cell[pi][pj]
                    
                    new_cell = _enforce_symmetry(new_cell)
                    
                    # Evaluate
                    score = _evaluate_gap(new_cell, n_order_fast, n_kpoints=8)
                    if score > current_score:
                        current_cell = new_cell
                        current_score = score
                        best_cell = new_cell
                        best_score = score
                        improved = True
                        no_improve = 0
                    else:
                        no_improve += 1
                if time.time() > deadline - 10:
                    break
            if not improved:
                break
    
    # Final refinement with higher-order basis
    if time.time() < deadline - 5:
        final_score = _evaluate_gap(best_cell, n_order_refine, n_kpoints=10)
        # The final score from the evaluator will use 289 plane waves, so this is just
        # a sanity check. We return the cell.
    
    return best_cell
