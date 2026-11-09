import time
import math
import numpy as np

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the 4-5 TM band gap."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin within 120s budget

    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    # Plane-wave setup: 289 = 17x17
    N = 17
    nmax = (N - 1) // 2  # 8

    # Generate G-vectors
    G = np.array([(mx, my) for mx in range(-nmax, nmax + 1) for my in range(-nmax, nmax + 1)], dtype=float)
    NG = len(G)  # 289

    def compute_fourier(cell):
        """Compute Fourier coefficients of 1/eps for the cell using FFT."""
        eps = np.where(np.array(cell) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps

        # Use FFT to compute the DFT efficiently
        # F[mx,my] = (1/size^2) * sum_{i,j} inv_eps[i,j] * exp(-i*2*pi*(mx*i/size + my*j/size))
        # This is exactly the 2D FFT normalized by size^2
        F = np.fft.fft2(inv_eps) / (size * size)

        # Shift so that F[0,0] corresponds to (0,0)
        # Actually, np.fft.fft2 gives F[kx,ky] where kx,ky are 0..size-1
        # We want F for G-vectors in range [-nmax, nmax]
        # The FFT output index k corresponds to frequency k if k <= size/2, else k-size
        # Since size=32 and nmax=8, we need indices 0..8 and 24..31 (which correspond to -8..-1)
        
        # Let's construct the F array in the order matching our G-vectors
        # G-vectors are (mx, my) for mx, my in [-8, 8]
        # F_fft[i, j] where i, j in [0, 31]
        # i corresponds to mx = i if i <= 16 else i - 32
        # j corresponds to my = j if j <= 16 else j - 32
        
        F_result = np.zeros((N, N), dtype=complex)
        for idx_mx in range(N):
            mx = idx_mx - nmax
            if mx >= 0:
                fft_idx_mx = mx
            else:
                fft_idx_mx = mx + size
            for idx_my in range(N):
                my = idx_my - nmax
                if my >= 0:
                    fft_idx_my = my
                else:
                    fft_idx_my = my + size
                F_result[idx_mx, idx_my] = F[fft_idx_mx, fft_idx_my]

        return F_result

    def solve_bands(F):
        """Solve the eigenvalue problem for TM bands at given k-points."""
        # k-points in irreducible BZ: 30 points
        k_points = []
        # Along Gamma-X: (0,0) to (pi,0)
        for t in np.linspace(0, 1, 8):
            k_points.append((t * np.pi, 0))
        # Along X-M: (pi,0) to (pi,pi)
        for t in np.linspace(0, 1, 8):
            k_points.append((np.pi, t * np.pi))
        # Along M-Gamma: (pi,pi) to (0,0)
        for t in np.linspace(0, 1, 8):
            k_points.append((t * np.pi, t * np.pi))
        # Interior points
        k_points.append((np.pi/2, np.pi/4))
        k_points.append((3*np.pi/4, np.pi/4))
        k_points.append((np.pi/2, np.pi/2))

        max_omega4 = 0
        min_omega5 = np.inf

        for kx, ky in k_points:
            # Build the matrix A[i,j] = (G_i - k)^2 * F[G_i - G_j]
            A = np.zeros((NG, NG), dtype=complex)
            for i in range(NG):
                Gi = G[i]
                Gi_k2 = (Gi[0] - kx)**2 + (Gi[1] - ky)**2
                for j in range(NG):
                    Gj = G[j]
                    dGx = int(Gi[0] - Gj[0])
                    dGy = int(Gi[1] - Gj[1])
                    fi = dGx + nmax
                    fj = dGy + nmax
                    if 0 <= fi < N and 0 <= fj < N:
                        Fval = F[fi, fj]
                    else:
                        Fval = 0
                    A[i, j] = Gi_k2 * Fval

            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(eigvals)
                eigvals = eigvals[eigvals > 1e-10]
                if len(eigvals) >= 5:
                    omega4 = np.sqrt(eigvals[3])
                    omega5 = np.sqrt(eigvals[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass

        if min_omega5 == np.inf:
            return 0.0
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return 0.0
        return gap / midgap

    # Symmetry helpers
    def get_fundamental(cell):
        """Get the 8x8 top-left quadrant."""
        return [cell[i][:8] for i in range(8)]

    def set_fundamental(fund):
        """Set the full cell from the 8x8 fundamental domain."""
        new_cell = [[0]*size for _ in range(size)]
        for i in range(8):
            for j in range(8):
                val = fund[i][j]
                new_cell[i][j] = val
                new_cell[i][size-1-j] = val
                new_cell[size-1-i][j] = val
                new_cell[size-1-i][size-1-j] = val
                new_cell[j][i] = val
                new_cell[j][size-1-i] = val
                new_cell[size-1-j][i] = val
                new_cell[size-1-j][size-1-i] = val
        return new_cell

    def make_square_rod(radius_frac):
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                if x < radius_frac and y < radius_frac:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell

    def make_air_hole(hole_frac):
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                if x < hole_frac and y < hole_frac:
                    row.append(0)
                else:
                    row.append(1)
            cell.append(row)
        return cell

    def make_cross(arm_width):
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                x = abs((i + 0.5) / size - 0.5)
                y = abs((j + 0.5) / size - 0.5)
                if x < arm_width/2 or y < arm_width/2:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell

    # Initial candidates
    candidates = []
    candidates.append(make_square_rod(0.3))
    candidates.append(make_square_rod(0.35))
    candidates.append(make_square_rod(0.4))
    candidates.append(make_air_hole(0.2))
    candidates.append(make_air_hole(0.25))
    candidates.append(make_air_hole(0.3))
    candidates.append(make_cross(0.3))
    candidates.append(make_cross(0.35))

    best_score = -1
    best_fund = None

    for cell in candidates:
        F = compute_fourier(cell)
        score = solve_bands(F)
        if score > best_score:
            best_score = score
            best_fund = get_fundamental(cell)

    # Local search: simulated annealing on the fundamental domain
    fund = [row[:] for row in best_fund]
    current_score = best_score
    best_score_sa = best_score
    best_fund = [row[:] for row in fund]

    temp = 0.1
    cooling = 0.995
    min_temp = 0.001

    while time.time() < deadline and temp > min_temp:
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)

        fund[i][j] = 1 - fund[i][j]
        cell = set_fundamental(fund)
        F = compute_fourier(cell)
        new_score = solve_bands(F)

        if new_score > current_score or np.random.random() < math.exp((new_score - current_score) / temp):
            current_score = new_score
            if new_score > best_score_sa:
                best_score_sa = new_score
                best_fund = [row[:] for row in fund]
        else:
            fund[i][j] = 1 - fund[i][j]

        temp *= cooling

    best_cell_final = set_fundamental(best_fund)

    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert best_cell_final[i][j] == best_cell_final[size-1-i][size-1-j]
            assert best_cell_final[i][j] == best_cell_final[j][i]
            assert best_cell_final[i][j] == best_cell_final[size-1-i][j]

    return best_cell_final
