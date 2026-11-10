import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    def enforce_symmetry(cell):
        c = cell.astype(float)
        c = (c + c[::-1, :]) / 2.0
        c = (c + c[:, ::-1]) / 2.0
        c = (c + c.T) / 2.0
        c = (c + c[::-1, ::-1]) / 2.0
        return (c > 0.5).astype(int)

    def evaluate_cell(cell):
        eps = np.where(cell == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        
        # DFT
        eps_inv_fft = np.fft.fftshift(np.fft.fft2(eps_inv))
        # Extract 17x17 region centered at (16, 16)
        # Indices 8 to 24
        eps_G = eps_inv_fft[8:25, 8:25]
        
        G_vals = np.arange(-8, 9)
        Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_vectors = np.stack([Gx, Gy], axis=-1)
        
        def get_eigenvalues(kx, ky):
            k = np.array([kx, ky])
            N = 17
            A = np.zeros((N, N), dtype=complex)
            
            for i in range(N):
                Gi = G_vectors[i]
                for j in range(N):
                    Gj = G_vectors[j]
                    kGi = k + Gi
                    kGj = k + Gj
                    cross = kGi[0]*kGj[1] - kGi[1]*kGj[0]
                    
                    diff_i = Gi[0] - Gj[0]
                    diff_j = Gi[1] - Gj[1]
                    
                    if abs(diff_i) <= 8 and abs(diff_j) <= 8:
                        idx_i = diff_i + 8
                        idx_j = diff_j + 8
                        eps_val = eps_G[idx_i, idx_j]
                    else:
                        eps_val = 0.0
                        
                    if eps_val != 0:
                        A[i, j] = cross / eps_val
                    
            vals = np.linalg.eigvals(A)
            real_vals = np.sort(np.real(vals))
            return real_vals

        k_points = [(0, 0), (0.5, 0), (0.5, 0.5)]
        max_omega_4 = 0
        min_omega_5 = np.inf
        
        for kx, ky in k_points:
            eigs = get_eigenvalues(kx, ky)
            eigs = eigs[eigs > 1e-6]
            if len(eigs) >= 5:
                max_omega_4 = max(max_omega_4, eigs[3])
                min_omega_5 = min(min_omega_5, eigs[4])
                
        if min_omega_5 == np.inf:
            return -1.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        ratio = gap / midgap if midgap > 0 else 0
        return ratio

    # Initial seed
    seed_cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                seed_cell[i, j] = 1
    seed_cell = enforce_symmetry(seed_cell)
    
    start_time = time.time()
    deadline = start_time + 180
    
    best_cell = seed_cell.copy()
    best_score = evaluate_cell(best_cell)
    current_cell = best_cell.copy()
    current_score = best_score
    
    indep_indices = [(i, j) for i in range(size//2) for j in range(size//2)]
    
    def apply_indep_flip(cell, idx):
        i, j = idx
        new_cell = cell.copy()
        new_cell[i, j] = 1 - new_cell[i, j]
        new_cell[size-1-i, j] = new_cell[i, j]
        new_cell[i, size-1-j] = new_cell[i, j]
        new_cell[size-1-i, size-1-j] = new_cell[i, j]
        new_cell[j, i] = new_cell[i, j]
        new_cell[j, size-1-i] = new_cell[i, j]
        new_cell[size-1-j, i] = new_cell[i, j]
        new_cell[size-1-j, size-1-i] = new_cell[i, j]
        return new_cell

    np.random.seed(42)
    
    while time.time() < deadline:
        for _ in range(5):
            idx = np.random.choice(len(indep_indices))
            new_cell = apply_indep_flip(current_cell, idx)
            new_score = evaluate_cell(new_cell)
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell.copy()

    return best_cell.tolist()
