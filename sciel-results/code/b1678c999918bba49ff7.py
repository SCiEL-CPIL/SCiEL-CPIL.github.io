import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    time_limit = 180.0
    
    # Initialize with baseline rod
    cell = [[0]*size for _ in range(size)]
    radius = 0.2 * size
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < radius*radius:
                cell[i][j] = 1

    # PWE Setup
    N_pw = 3
    G_vectors = []
    for g1 in range(-N_pw, N_pw+1):
        for g2 in range(-N_pw, N_pw+1):
            if g1*g1 + g2*g2 <= N_pw*N_pw:
                G_vectors.append((g1, g2))
    G_vectors = np.array(G_vectors)
    num_pw = len(G_vectors)
    
    # Precompute G vector differences indices?
    # For each pair (i,j), we need the index of G_i - G_j.
    # If G_i - G_j is outside truncation, coeff is 0.
    # Let's precompute a matrix of indices or just the lookup dict every time.
    
    # K-points
    k_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (0.5, 0.0),
        (0.25, 0.25),
        (0.4, 0.0)
    ]
    
    def compute_gap(cell_2d):
        # Convert to array
        eps_arr = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if cell_2d[i][j] == 1:
                    eps_arr[i][j] = eps_high
                else:
                    eps_arr[i][j] = eps_low
        
        inv_eps = 1.0 / eps_arr
        fft_inv_eps = np.fft.fft2(inv_eps)
        
        coeff = np.zeros(num_pw, dtype=complex)
        coeff_lookup = {}
        for idx, (gx, gy) in enumerate(G_vectors):
            u = gx % size
            v = gy % size
            c_val = fft_inv_eps[u, v] / (size * size)
            coeff[idx] = c_val
            coeff_lookup[(gx, gy)] = c_val
            
        max_w4 = 0.0
        min_w5 = float('inf')
        
        for kx, ky in k_points:
            k_vec = np.array([kx, ky])
            kG_vecs = k_vec + G_vectors
            dot_products = kG_vecs @ kG_vecs.T
            
            A = np.zeros((num_pw, num_pw), dtype=complex)
            for i in range(num_pw):
                gi = G_vectors[i]
                row_dots = dot_products[i]
                for j in range(num_pw):
                    gj = G_vectors[j]
                    d_gx = gi[0] - gj[0]
                    d_gy = gi[1] - gj[1]
                    key = (d_gx, d_gy)
                    if key in coeff_lookup:
                        A[i, j] = coeff_lookup[key] * row_dots[j]
                        
            eigvals = np.linalg.eigvalsh(A)
            if len(eigvals) >= 5:
                w4 = np.sqrt(eigvals[3])
                w5 = np.sqrt(eigvals[4])
                if w4 > max_w4: max_w4 = w4
                if w5 < min_w5: min_w5 = w5
                
        if min_w5 == float('inf'):
            return -1.0
        gap = min_w5 - max_w4
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return 0.0
        return 2.0 * gap / midgap

    best_score = compute_gap(cell)
    best_cell = [row[:] for row in cell]
    current_score = best_score
    temp = 1.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        
        orbit = set()
        orbit.add((i, j))
        orbit.add((size-1-i, j))
        orbit.add((i, size-1-j))
        orbit.add((size-1-i, size-1-j))
        orbit.add((j, i))
        orbit.add((j, size-1-i))
        orbit.add((size-1-j, i))
        orbit.add((size-1-j, size-1-i))
        
        for (r, c) in orbit:
            cell[r][c] = 1 - cell[r][c]
            
        new_score = compute_gap(cell)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = [row[:] for row in cell]
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / temp)
                if random.random() < prob:
                    current_score = new_score
                else:
                    for (r, c) in orbit:
                        cell[r][c] = 1 - cell[r][c]
            else:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_cell = [row[:] for row in cell]
                    
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    return best_cell
# EVOLVE-BLOCK-END
