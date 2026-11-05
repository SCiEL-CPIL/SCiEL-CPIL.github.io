"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _make_pwe_solver(size, eps_high, n_pw=289):
    """Build a plane-wave eigenvalue solver for TM polarization on a square lattice."""
    gw = int(np.sqrt(n_pw))  # 17
    half = gw // 2
    G = []
    for ig in range(gw):
        for jg in range(gw):
            Gx = (ig - half)
            Gy = (jg - half)
            G.append((Gx, Gy))
    G = np.array(G, dtype=float)
    nG = len(G)
    
    def compute_epsilon_inv_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon for the pixel cell."""
        eps = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if cell[i][j] == 1:
                    eps[i, j] = eps_high
        
        inv_eps = 1.0 / eps
        F = np.fft.fft2(inv_eps)
        
        max_freq = 2 * half
        freq_range = range(-max_freq, max_freq + 1)
        inv_eps_F = np.zeros((2*max_freq+1, 2*max_freq+1), dtype=complex)
        for mi, m in enumerate(freq_range):
            for ni, n in enumerate(freq_range):
                km = m % size
                kn = n % size
                phase = np.exp(-1j * 2 * np.pi * (m * 0.5 / size + n * 0.5 / size))
                inv_eps_F[mi, ni] = F[km, kn] * phase / (size * size)
        return inv_eps_F, max_freq

    def solve_bands_at_k(inv_eps_F, max_freq, kx, ky, n_bands=5):
        """Solve for the first n_bands TM bands at wavevector (kx, ky)."""
        k = np.array([kx, ky], dtype=float)
        A = np.zeros((nG, nG), dtype=complex)
        # Precompute |G_q + k|^2 for all q
        Gk_sq = np.zeros(nG)
        for q in range(nG):
            Gq = G[q]
            Gqk = Gq + k
            Gk_sq[q] = Gqk[0]**2 + Gqk[1]**2
            
        for p in range(nG):
            Gp = G[p]
            for q in range(nG):
                Gq = G[q]
                diff = Gp - Gq
                mi = int(diff[0]) + max_freq
                ni = int(diff[1]) + max_freq
                if 0 <= mi < 2*max_freq+1 and 0 <= ni < 2*max_freq+1:
                    A[p, q] = Gk_sq[q] * inv_eps_F[mi, ni]
        
        A = (A + A.conj().T) / 2
        try:
            eigvals, _ = np.linalg.eigh(A)
        except:
            return None
        return eigvals[:n_bands]

    def evaluate_cell(cell, k_points, n_bands=5):
        """Evaluate the gap ratio for a given cell at specific k-points."""
        inv_eps_F, max_freq = compute_epsilon_inv_fourier(cell)
        min_ratio = float('inf')
        for kx, ky in k_points:
            eigvals = solve_bands_at_k(inv_eps_F, max_freq, kx, ky, n_bands)
            if eigvals is None:
                return -1.0
            w4 = np.sqrt(max(eigvals[3], 0))
            w5 = np.sqrt(max(eigvals[4], 0))
            if w4 + w5 < 1e-12:
                ratio = -1.0
            else:
                ratio = 2.0 * (w5 - w4) / (w5 + w4)
            if ratio < min_ratio:
                min_ratio = ratio
        return min_ratio

    return evaluate_cell

def design(problem):
    """Optimize a symmetric 32x32 photonic crystal cell for maximum TM band gap."""
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin
    
    # Define k-points for fast evaluation (subset) and final evaluation (full)
    k_points_fast = [
        (0.0, 0.0),   # Gamma
        (1.0, 0.0),   # X
        (0.5, 0.5),   # M
        (0.5, 0.0),   # Mid Gamma-X
        (0.5, 0.25),  # Interior
    ]
    
    k_points_full = []
    for t in np.linspace(0, 1, 6):
        k_points_full.append((t, 0.0))
    for t in np.linspace(0, 0.5, 4):
        k_points_full.append((1.0, t))
    for t in np.linspace(0, 0.5, 4):
        k_points_full.append((t, t))
    k_points_full.append((0.3, 0.2))
    k_points_full.append((0.5, 0.3))
    k_points_full.append((0.7, 0.1))
    k_points_full.append((0.2, 0.1))
    k_points_full.append((0.6, 0.4))
    
    evaluate_cell = _make_pwe_solver(size, eps_high, n_pw=289)
    
    # Initialize with a symmetric pattern: Central dielectric disk
    cell = [[0]*size for _ in range(size)]
    center = (size - 1) / 2.0
    radius = 0.3 * size
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5) / size - 0.5
            dy = (j + 0.5) / size - 0.5
            if dx*dx + dy*dy < radius*radius:
                cell[i][j] = 1

    def enforce_symmetry(cell):
        n = len(cell)
        new_cell = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                equivs = set()
                equivs.add((i, j))
                equivs.add((n-1-i, j))
                equivs.add((i, n-1-j))
                equivs.add((n-1-i, n-1-j))
                equivs.add((j, n-1-i))
                equivs.add((n-1-j, i))
                equivs.add((n-1-j, n-1-i))
                equivs.add((j, i))
                
                ones = sum(1 for (ei, ej) in equivs if cell[ei][ej] == 1)
                total = len(equivs)
                val = 1 if ones > total/2 else 0
                for (ei, ej) in equivs:
                    new_cell[ei][ej] = val
        return new_cell

    cell = enforce_symmetry(cell)
    
    def get_fundamental_domain_indices(n):
        indices = []
        for i in range(n):
            for j in range(n):
                if i <= j and i <= n // 2:
                    indices.append((i, j))
        return indices

    fd_indices = get_fundamental_domain_indices(size)
    
    current_score = evaluate_cell(cell, k_points_fast)
    best_cell = [row[:] for row in cell]
    best_score = current_score
    
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        for (i, j) in fd_indices:
            if time.time() > deadline:
                break
            
            old_val = cell[i][j]
            new_val = 1 - old_val
            
            equivs = set()
            equivs.add((i, j))
            equivs.add((size-1-i, j))
            equivs.add((i, size-1-j))
            equivs.add((size-1-i, size-1-j))
            equivs.add((j, size-1-i))
            equivs.add((size-1-j, i))
            equivs.add((size-1-j, size-1-i))
            equivs.add((j, i))
            
            for (ei, ej) in equivs:
                cell[ei][ej] = new_val
            
            new_score = evaluate_cell(cell, k_points_fast)
            
            if new_score > best_score + 1e-8:
                best_score = new_score
                best_cell = [row[:] for row in cell]
                improved = True
            else:
                for (ei, ej) in equivs:
                    cell[ei][ej] = old_val
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        else:
            no_improve_count = 0
            
    # Final evaluation with full k-points to ensure validity
    # Note: The score returned by the evaluator is based on the full set.
    # We return the best cell found during the fast search.
    return best_cell
# EVOLVE-BLOCK-END
