"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def design(problem):
    """Design a 32x32 symmetric unit cell maximizing the TM band gap between bands 4 and 5."""
    size = problem["pixels"]  # 32
    start_time = time.time()
    deadline = start_time + 170  # Safety margin
    
    N = size
    
    def make_cell(params):
        """Create symmetric 32x32 cell from parameters [c0, c1, c2, c3, thresh]."""
        c0, c1, c2, c3, thresh = params
        cell = np.zeros((N, N))
        for i in range(N):
            x = (i + 0.5) / N - 0.5
            for j in range(N):
                y = (j + 0.5) / N - 0.5
                val = c0
                val += c1 * math.cos(2 * math.pi * x) * math.cos(2 * math.pi * y)
                val += c2 * math.cos(4 * math.pi * x) * math.cos(4 * math.pi * y)
                val += c3 * math.cos(6 * math.pi * x) * math.cos(6 * math.pi * y)
                cell[i, j] = 1 if val > thresh else 0
        # Enforce D4 symmetry
        cell = np.round(cell).astype(int)
        for i in range(N // 2):
            for j in range(N):
                cell[N-1-i, j] = cell[i, j]
                cell[i, N-1-j] = cell[i, j]
        for i in range(N // 2):
            for j in range(i+1, N):
                avg = (cell[i,j] + cell[j,i]) / 2.0
                cell[i,j] = cell[j,i] = 1 if avg > 0.5 else 0
        return cell
    
    def get_mu_coeffs(cell):
        """Get Fourier coefficients of mu = 1/eps for the cell."""
        mu = np.ones((N, N))
        mu[cell == 1] = 1.0 / 12.0
        fft_mu = np.fft.fft2(mu)
        # <mu>_{(nx,ny)} = (1/N^2) * exp(pi*i*(nx+ny)/N) * fft_mu[nx%N, ny%N]
        max_G = 13
        coeffs = {}
        for nx in range(-max_G, max_G + 1):
            for ny in range(-max_G, max_G + 1):
                phase = np.exp(1j * math.pi * (nx + ny) / N)
                coeffs[(nx, ny)] = phase * fft_mu[nx % N, ny % N] / (N * N)
        return coeffs
    
    def evaluate_gap(coeffs):
        """Compute 2*(min_omega5 - max_omega4)/(min_omega5 + max_omega4)."""
        # 17x17 plane waves: G from -8 to 8
        G_list = list(range(-8, 9))
        nG = 17
        
        # Key k-points in irreducible BZ (units of 2pi/a)
        k_points = []
        # Gamma to X: (0, t)
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            k_points.append((0.0, t))
        # Gamma to M: (t, t)
        for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            k_points.append((t, t))
        # X to M: (t, 0.5)
        for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            k_points.append((t, 0.5))
        # Interior
        k_points += [(0.1, 0.3), (0.2, 0.3), (0.1, 0.4), (0.2, 0.4),
                     (0.15, 0.35), (0.25, 0.35), (0.05, 0.45), (0.3, 0.4),
                     (0.15, 0.25), (0.25, 0.45), (0.1, 0.2), (0.2, 0.2)]
        
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for kx, ky in k_points:
            M = np.zeros((nG * nG, nG * nG), dtype=complex)
            
            for ix in range(nG):
                for iy in range(nG):
                    gx, gy = G_list[ix], G_list[iy]
                    kgx, kgy = kx + gx, ky + gy
                    idx = ix * nG + iy
                    for ix2 in range(nG):
                        for iy2 in range(nG):
                            gx2, gy2 = G_list[ix2], G_list[iy2]
                            kgx2, kgy2 = kx + gx2, ky + gy2
                            dGx, dGy = gx - gx2, gy - gy2
                            mu_c = coeffs.get((dGx, dGy), 0)
                            if abs(mu_c) > 1e-14:
                                M[idx, ix2 * nG + iy2] = (kgx * kgx2 + kgy * kgy2) * mu_c
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            if len(eigvals) >= 5:
                w2 = np.sort(eigvals.real)[:5]
                if w2[3] > 1e-10 and w2[4] > 1e-10:
                    w4 = math.sqrt(w2[3])
                    w5 = math.sqrt(w2[4])
                    max_w4 = max(max_w4, w4)
                    min_w5 = min(min_w5, w5)
        
        if min_w5 == float('inf') or max_w4 < 1e-10:
            return -1.0
        mid = (min_w5 + max_w4) / 2.0
        if mid < 1e-10:
            return -1.0
        return 2.0 * (min_w5 - max_w4) / mid
    
    # Initial parameter guesses
    initial_params_list = [
        [0.30, 0.0, 0.0, 0.0, 0.20],
        [0.25, 0.08, 0.0, 0.0, 0.20],
        [0.20, 0.0, 0.08, 0.0, 0.20],
        [0.22, 0.06, 0.04, 0.0, 0.18],
        [0.28, -0.06, 0.03, 0.0, 0.22],
        [0.35, 0.04, -0.04, 0.0, 0.28],
        [0.18, 0.10, -0.08, 0.04, 0.16],
        [0.40, 0.0, 0.0, 0.0, 0.32],
        [0.15, 0.12, 0.08, 0.06, 0.14],
        [0.32, -0.04, 0.06, -0.03, 0.24],
    ]
    
    best_score = -1.0
    best_params = initial_params_list[0][:]
    
    for ip in initial_params_list:
        if time.time() > deadline:
            break
        cell = make_cell(ip)
        coeffs = get_mu_coeffs(cell)
        score = evaluate_gap(coeffs)
        if score > best_score:
            best_score = score
            best_params = ip[:]
    
    # Coordinate descent optimization
    if time.time() < deadline - 30:
        steps = [0.04, 0.04, 0.04, 0.04, 0.015]
        for _ in range(80):
            if time.time() > deadline - 30:
                break
            improved = False
            for d in range(5):
                for sgn in [-1, 1]:
                    np_ = best_params[:]
                    np_[d] += sgn * steps[d]
                    np_[0] = max(0.0, min(0.55, np_[0]))
                    np_[1] = max(-0.25, min(0.25, np_[1]))
                    np_[2] = max(-0.25, min(0.25, np_[2]))
                    np_[3] = max(-0.25, min(0.25, np_[3]))
                    np_[4] = max(0.05, min(0.45, np_[4]))
                    cell = make_cell(np_)
                    coeffs = get_mu_coeffs(cell)
                    score = evaluate_gap(coeffs)
                    if score > best_score:
                        best_score = score
                        best_params = np_[:]
                        improved = True
                        break
            if not improved:
                steps = [s * 0.65 for s in steps]
                if min(steps) < 0.003:
                    break
    
    final_cell = make_cell(best_params)
    return final_cell.tolist()
# EVOLVE-BLOCK-END
