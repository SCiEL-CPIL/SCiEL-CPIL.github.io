import math
import time
import numpy as np

def design(problem):
    """Design a 32x32 D4-symmetric unit cell for maximum TM 4-5 band gap.
    
    Uses a parametric family of square pillars, projected to D4 symmetry,
    with local optimization.
    """
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def project_to_d4(cell_2d):
        """Project a 32x32 cell to D4 symmetry by averaging orbits and thresholding."""
        cell = np.array(cell_2d, dtype=float)
        sym_cell = np.zeros_like(cell)
        for i in range(size):
            for j in range(size):
                # Generate all symmetric coordinates
                coords = [
                    (i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (j, size - 1 - i),
                    (size - 1 - j, i),
                    (size - 1 - j, size - 1 - i)
                ]
                # Ensure uniqueness
                unique_coords = list(set(coords))
                val = np.mean([cell[ui, uj] for ui, uj in unique_coords])
                for ui, uj in unique_coords:
                    sym_cell[ui, uj] = val
        # Threshold to binary
        return (sym_cell > 0.5).astype(int)

    def make_cell_from_params(s, cx, cy):
        """Create a 32x32 cell with 4 square pillars of side s at (cx,cy) etc."""
        cell = np.zeros((size, size), dtype=int)
        half_s = s / 2.0
        
        centers = [
            (cx, cy),
            (cx, 1.0 - cy),
            (1.0 - cx, cy),
            (1.0 - cx, 1.0 - cy)
        ]
        
        for ccx, ccy in centers:
            x_start = int(np.floor(ccx * size - half_s))
            x_end = int(np.ceil(ccx * size + half_s))
            y_start = int(np.floor(ccy * size - half_s))
            y_end = int(np.ceil(ccy * size + half_s))
            
            x_start = max(0, min(size, x_start))
            x_end = max(0, min(size, x_end))
            y_start = max(0, min(size, y_start))
            y_end = max(0, min(size, y_end))
            
            cell[y_start:y_end, x_start:x_end] = 1
        
        return cell

    def compute_gap(cell_2d, npw=8, k_points=None):
        """Compute the gap-midgap ratio between TM bands 4 and 5."""
        if k_points is None:
            k_points = [
                (0.0, 0.0),
                (np.pi, 0.0),
                (np.pi/2, np.pi/2),
                (np.pi/2, 0.0),
                (np.pi/2, np.pi/4),
                (3*np.pi/8, np.pi/8),
                (np.pi/4, np.pi/4),
                (np.pi/3, np.pi/6),
                (2*np.pi/5, np.pi/5),
                (np.pi/2, np.pi/6),
                (np.pi/2, np.pi/3),
                (3*np.pi/4, np.pi/4),
                (np.pi/2, np.pi/8),
                (5*np.pi/8, np.pi/8),
                (np.pi/4, 0.0),
            ]
            
        eps = np.where(cell_2d == 1, eps_high, eps_low)
        
        G = []
        for mx in range(-npw, npw+1):
            for my in range(-npw, npw+1):
                G.append((2*np.pi*mx, 2*np.pi*my))
        G = np.array(G)
        nG = len(G)
        
        inv_eps = 1.0 / eps
        x = (np.arange(size) + 0.5) / size
        y = (np.arange(size) + 0.5) / size
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx, (gx, gy) in enumerate(G):
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (size * size)
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G
            kG_sq = np.sum(k_plus_G**2, axis=1)
            
            G_dict = {}
            for idx, (gx, gy) in enumerate(G):
                G_dict[(gx, gy)] = idx
            
            A = np.zeros((nG, nG), dtype=complex)
            for i in range(nG):
                for j in range(nG):
                    G_diff = G[i] - G[j]
                    key = (G_diff[0], G_diff[1])
                    if key in G_dict:
                        A[i, j] = inv_eps_coeffs[G_dict[key]]
            
            H = A * kG_sq[np.newaxis, :]
            H = (H + H.conj().T) / 2
            
            try:
                eigvals = np.linalg.eigvalsh(H)
                eigvals = np.sort(eigvals.real)
            except:
                continue
            
            if len(eigvals) >= 5:
                omega4 = eigvals[3]
                omega5 = eigvals[4]
                if omega4 > max_omega4:
                    max_omega4 = omega4
                if omega5 < min_omega5:
                    min_omega5 = omega5
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return 0.0
        
        return gap / midgap

    # Optimization
    best_score = -float('inf')
    best_cell = None
    best_s, best_cx, best_cy = 0.2, 0.25, 0.25
    
    # Coarse grid with small basis
    s_vals = np.linspace(0.1, 0.4, 6)
    c_vals = np.linspace(0.2, 0.3, 6)
    
    for s in s_vals:
        if time.time() > deadline - 30:
            break
        for cx in c_vals:
            if time.time() > deadline - 30:
                break
            for cy in c_vals:
                if time.time() > deadline - 30:
                    break
                
                raw_cell = make_cell_from_params(s, cx, cy)
                sym_cell = project_to_d4(raw_cell)
                score = compute_gap(sym_cell, npw=4, k_points=[(0,0), (np.pi,0), (np.pi/2,np.pi/2), (np.pi/2,0), (np.pi/2,np.pi/4)])
                
                if score > best_score:
                    best_score = score
                    best_cell = sym_cell.copy()
                    best_s, best_cx, best_cy = s, cx, cy
    
    if best_cell is None:
        raw_cell = make_cell_from_params(0.2, 0.25, 0.25)
        best_cell = project_to_d4(raw_cell)
    
    # Fine-tune
    if time.time() < deadline - 10:
        s_range = np.linspace(best_s - 0.05, best_s + 0.05, 5)
        c_range = np.linspace(best_cx - 0.02, best_cx + 0.02, 5)
        
        for s in s_range:
            if time.time() > deadline - 10:
                break
            for cx in c_range:
                if time.time() > deadline - 10:
                    break
                for cy in c_range:
                    if time.time() > deadline - 10:
                        break
                    if s < 0.05 or s > 0.45 or cx < 0.1 or cx > 0.4 or cy < 0.1 or cy > 0.4:
                        continue
                    raw_cell = make_cell_from_params(s, cx, cy)
                    sym_cell = project_to_d4(raw_cell)
                    score = compute_gap(sym_cell, npw=4, k_points=[(0,0), (np.pi,0), (np.pi/2,np.pi/2), (np.pi/2,0), (np.pi/2,np.pi/4)])
                    
                    if score > best_score:
                        best_score = score
                        best_cell = sym_cell.copy()
                        best_s, best_cx, best_cy = s, cx, cy
    
    # Final verification with full basis
    if time.time() < deadline - 2:
        final_score = compute_gap(best_cell, npw=8)
        if final_score > best_score:
            best_score = final_score
        # best_cell is already the best candidate
    
    result = best_cell.tolist()
    
    return result
