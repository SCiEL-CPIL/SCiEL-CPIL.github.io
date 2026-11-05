import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Fundamental domain: 136 pixels
    # i in 0..15, j in i..15
    fund_coords = []
    for i in range(16):
        for j in range(i, 16):
            fund_coords.append((i, j))
            
    def expand_cell(bits):
        cell = [[0]*size for _ in range(size)]
        idx = 0
        for i in range(16):
            for j in range(i, 16):
                val = bits[idx]
                idx += 1
                cell[i][j] = val
                cell[j][i] = val
                cell[31-i][31-j] = val
                cell[31-j][31-i] = val
                cell[31-i][j] = val
                cell[i][31-j] = val
                cell[31-j][i] = val
                cell[j][31-i] = val
        return cell

    # PWE Solver
    R_pw = 5 # Plane wave radius
    G_vecs = []
    for n in range(-R_pw, R_pw+1):
        for m in range(-R_pw, R_pw+1):
            if n*n + m*m <= R_pw*R_pw:
                G_vecs.append((n, m))
    G_vecs = np.array(G_vecs, dtype=float)
    N_pw = len(G_vecs)
    
    R_diff = 2 * R_pw
    G_diff_vecs = []
    for n in range(-R_diff, R_diff+1):
        for m in range(-R_diff, R_diff+1):
            if n*n + m*m <= R_diff*R_diff:
                G_diff_vecs.append((n, m))
    G_diff_vecs = np.array(G_diff_vecs, dtype=float)
    N_diff = len(G_diff_vecs)
    diff_to_idx = { (int(n), int(m)): i for i, (n, m) in enumerate(G_diff_vecs) }
    
    # Precompute phase factors for each G_diff
    # phase[k, i, j] = exp(-2pi i (n_k (i+0.5)/size + m_k (j+0.5)/size))
    # This is N_diff x 32 x 32. 200 x 1024 = 200k complex numbers. ~3MB. OK.
    phases = np.zeros((N_diff, size, size), dtype=complex)
    i_idx = np.arange(size)
    r_i = (i_idx + 0.5) / size
    for k in range(N_diff):
        n, m = G_diff_vecs[k]
        phase_i = np.exp(-2j * np.pi * n * r_i)
        phase_j = np.exp(-2j * np.pi * m * r_i)
        # Outer product
        phases[k] = np.outer(phase_i, phase_j)
        
    # K-points
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.5, 0.25)
    ]
    
    def compute_gap(bits):
        # Compute eps_inv_G
        # eps_inv[i,j] = 1/12 if bits==1 else 1
        # We can compute eps_inv_G directly from bits without expanding cell?
        # No, bits only define the fundamental domain.
        # We need the full cell to compute the sum over all i,j.
        # But we can sum over the fundamental domain and multiply by symmetry factors?
        # No, the phase depends on position.
        
        # So we must expand cell or iterate over all 1024 pixels.
        # Iterating 1024 pixels is fast.
        
        eps_inv_G = np.zeros(N_diff, dtype=complex)
        
        # We need eps_inv for all pixels.
        # We can generate the full cell array once.
        cell = expand_cell(bits)
        
        # Convert to numpy array
        cell_np = np.array(cell, dtype=float)
        eps_inv_np = np.where(cell_np == 1, 1.0/eps_high, 1.0/eps_low)
        
        # Compute eps_inv_G
        # eps_inv_G[k] = sum_{i,j} eps_inv[i,j] * phases[k, i, j]
        # This is a dot product.
        # eps_inv_G = np.einsum('ij, kij -> k', eps_inv_np, phases)
        
        eps_inv_G = np.einsum('ij, kij -> k', eps_inv_np, phases)
        
        max_w4 = 0
        min_w5 = np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            k_plus_G = G_vecs + k
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            M = np.zeros((N_pw, N_pw), dtype=complex)
            for i in range(N_pw):
                for j in range(N_pw):
                    G_diff = G_vecs[i] - G_vecs[j]
                    idx = diff_to_idx.get((int(G_diff[0]), int(G_diff[1])), None)
                    if idx is not None:
                        M[i, j] = k_plus_G_sq[i] * eps_inv_G[idx]
                        
            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                w4 = np.sqrt(eigvals[3])
                w5 = np.sqrt(eigvals[4])
                max_w4 = max(max_w4, w4)
                min_w5 = min(min_w5, w5)
                
        if min_w5 == np.inf:
            return 0.0
        gap = min_w5 - max_w4
        mid = (min_w5 + max_w4) / 2.0
        return gap / mid if mid > 0 else 0.0

    # Optimization
    bits = [np.random.randint(0, 2) for _ in range(136)]
    best_bits = bits[:]
    best_score = compute_gap(bits)
    
    while time.time() < deadline:
        idx = np.random.randint(0, 136)
        bits[idx] = 1 - bits[idx]
        score = compute_gap(bits)
        if score > best_score:
            best_score = score
            best_bits = bits[:]
        else:
            bits[idx] = 1 - bits[idx]
            
    return expand_cell(best_bits)
