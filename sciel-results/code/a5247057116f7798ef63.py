"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with D4 symmetry to maximize
    the gap between TM bands 4 and 5.
    
    Strategy:
    1. Use a fast plane-wave method to evaluate band gaps.
    2. Search over D4-symmetric designs parameterized by geometry.
    3. Return the design with the largest estimated 4-5 TM gap.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s kill
    
    def compute_tm_bands(cell, num_g=5, num_k=6):
        """
        Compute TM bands using plane-wave expansion.
        Returns: (band4_max, band5_min) for the 4-5 gap
        """
        eps = np.where(np.array(cell) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        N = size
        fft_inv_eps = np.fft.fft2(inv_eps) / (N * N)
        
        G_values = []
        G_indices = []
        for m in range(-num_g, num_g + 1):
            for n in range(-num_g, num_g + 1):
                if m == 0 and n == 0:
                    continue
                G_values.append((2 * math.pi * m, 2 * math.pi * n))
                G_indices.append((m, n))
        
        num_G = len(G_values)
        
        G_coeffs = np.zeros(num_G, dtype=complex)
        for idx, (m, n) in enumerate(G_indices):
            G_coeffs[idx] = fft_inv_eps[m % N, n % N]
        
        k_points = [(0.0, 0.0)]
        for t in np.linspace(0.1, 1.0, num_k - 1):
            k_points.append((math.pi * t, 0.0))
        for t in np.linspace(0.1, 1.0, num_k - 1):
            k_points.append((math.pi, math.pi * t))
        for t in np.linspace(0.1, 0.9, max(1, num_k - 2)):
            k_points.append((math.pi * (1 - t), math.pi * t))
        
        band4_max = 0.0
        band5_min = float('inf')
        
        for kx, ky in k_points:
            A = np.zeros((num_G, num_G), dtype=complex)
            for m_idx in range(num_G):
                kmx = kx + G_values[m_idx][0]
                kmy = ky + G_values[m_idx][1]
                km_sq = kmx * kmx + kmy * kmy
                for n_idx in range(num_G):
                    knx = kx + G_values[n_idx][0]
                    kny = ky + G_values[n_idx][1]
                    kn_sq = knx * knx + kny * kny
                    dGx = G_values[m_idx][0] - G_values[n_idx][0]
                    dGy = G_values[m_idx][1] - G_values[n_idx][1]
                    dm = int(round(dGx / (2 * math.pi)))
                    dn = int(round(dGy / (2 * math.pi)))
                    coeff = 0.0
                    for g_idx, (gm, gn) in enumerate(G_indices):
                        if gm == dm and gn == dn:
                            coeff = G_coeffs[g_idx]
                            break
                    if dm == 0 and dn == 0:
                        coeff = 1.0
                    A[m_idx, n_idx] = km_sq * (1.0 if m_idx == n_idx else 0.0) + kn_sq * coeff
            
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(eigvals.real)
                if len(eigvals) >= 5:
                    band4_val = eigvals[3]
                    band5_val = eigvals[4]
                    if band4_val > band4_max:
                        band4_max = band4_val
                    if band5_val < band5_min:
                        band5_min = band5_val
            except:
                continue
        
        return band4_max, band5_min
    
    def enforce_d4(cell):
        """Enforce D4 symmetry on a 2D array."""
        cell = np.array(cell, dtype=int)
        for i in range(size):
            for j in range(size):
                vals = [
                    cell[i, j],
                    cell[size-1-i, j],
                    cell[i, size-1-j],
                    cell[size-1-i, size-1-j],
                    cell[j, i],
                    cell[size-1-j, size-1-i],
                    cell[j, size-1-i],
                    cell[size-1-j, i],
                ]
                avg = sum(vals) / len(vals)
                cell[i, j] = 1 if avg >= 0.5 else 0
        return cell.tolist()
    
    best_gap = -1
    best_cell = None
    
    # Candidate 1: Central circular rod with various radii
    for radius in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        cell = np.zeros((size, size))
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if dist <= radius * size / 2:
                    cell[i, j] = 1
        cell_list = enforce_d4(cell)
        
        if time.time() - start_time > deadline:
            break
        
        try:
            b4, b5 = compute_tm_bands(cell_list, num_g=5, num_k=6)
            gap = 2 * (b5 - b4) / (b5 + b4) if (b5 + b4) > 0 else -1
            if gap > best_gap:
                best_gap = gap
                best_cell = cell_list
        except:
            continue
    
    # Candidate 2: Ring patterns
    for inner_r, outer_r in [(0.1, 0.3), (0.2, 0.4), (0.3, 0.5), (0.15, 0.35)]:
        cell = np.zeros((size, size))
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2) / (size / 2)
                if inner_r <= dist <= outer_r:
                    cell[i, j] = 1
        cell_list = enforce_d4(cell)
        
        if time.time() - start_time > deadline:
            break
        
        try:
            b4, b5 = compute_tm_bands(cell_list, num_g=5, num_k=6)
            gap = 2 * (b5 - b4) / (b5 + b4) if (b5 + b4) > 0 else -1
            if gap > best_gap:
                best_gap = gap
                best_cell = cell_list
        except:
            continue
    
    # Candidate 3: Four-rod configuration
    for rod_r, offset in [(0.15, 0.2), (0.2, 0.25), (0.1, 0.15), (0.25, 0.3)]:
        cell = np.zeros((size, size))
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        ox = offset * size / 2
        oy = offset * size / 2
        for i in range(size):
            for j in range(size):
                for dx, dy in [(ox, oy), (-ox, oy), (ox, -oy), (-ox, -oy)]:
                    dist = math.sqrt((i - (cx + dx)) ** 2 + (j - (cy + dy)) ** 2)
                    if dist <= rod_r * size / 2:
                        cell[i, j] = 1
                        break
        cell_list = enforce_d4(cell)
        
        if time.time() - start_time > deadline:
            break
        
        try:
            b4, b5 = compute_tm_bands(cell_list, num_g=5, num_k=6)
            gap = 2 * (b5 - b4) / (b5 + b4) if (b5 + b4) > 0 else -1
            if gap > best_gap:
                best_gap = gap
                best_cell = cell_list
        except:
            continue
    
    # Default fallback: central rod
    if best_cell is None:
        cell = np.zeros((size, size))
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if dist <= 0.3 * size / 2:
                    cell[i, j] = 1
        best_cell = enforce_d4(cell)
    
    return best_cell

# EVOLVE-BLOCK-END
