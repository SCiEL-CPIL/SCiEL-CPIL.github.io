import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 220  # safety margin before 230s wall-clock limit
    size = 32
    half = size // 2  # 16
    qh = half // 2    # 8
    
    def flip_quadrant(cell, i, j):
        """Flip pixel at (i,j) in the 8x8 quadrant, enforcing full C4v symmetry."""
        new_val = 1 - cell[i][j]
        idxs = [
            (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
            (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
        ]
        seen = set()
        for r, c in idxs:
            if (r, c) not in seen:
                seen.add((r, c))
                cell[r][c] = new_val
    
    def make_cell_from_quadrant(q):
        """Build a 32x32 cell from an 8x8 quadrant with C4v symmetry."""
        cell = [[0]*size for _ in range(size)]
        for i in range(qh):
            for j in range(qh):
                val = q[i][j]
                cell[i][j] = val
                cell[i][size-1-j] = val
                cell[size-1-i][j] = val
                cell[size-1-i][size-1-j] = val
        # Enforce transpose symmetry within the quadrant
        for i in range(qh):
            for j in range(qh):
                if i <= j:
                    # cell[i][j] and cell[j][i] should be equal
                    # cell[i][j] = q[i][j], cell[j][i] = q[j][i]
                    # If they differ, set both to the same value
                    if cell[i][j] != cell[j][i]:
                        # Keep the majority or just pick one
                        v = cell[i][j]
                        # Update all symmetric positions for both (i,j) and (j,i)
                        for r, c in [(i,j),(i,size-1-j),(size-1-i,j),(size-1-i,size-1-j),
                                     (j,i),(j,size-1-i),(size-1-j,i),(size-1-j,size-1-i)]:
                            cell[r][c] = v
        return cell
    
    def make_quadrant_from_cell(cell):
        """Extract the 8x8 quadrant from a 32x32 cell."""
        q = [[cell[i][j] for j in range(qh)] for i in range(qh)]
        return q
    
    # Precompute G vectors for plane-wave method
    N_PW = 8
    G_range = np.arange(-N_PW, N_PW + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    Gx_flat = G_vectors[:, 0]
    Gy_flat = G_vectors[:, 1]
    Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
    Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
    idx_x = (Diff_x + 16) % 32
    idx_y = (Diff_y + 16) % 32
    
    # Generate k-points
    def generate_k_points(num_points=30):
        points = []
        for i in range(10):
            kx = (i + 0.5) * (np.pi / 10)
            points.append((kx, 0))
        for i in range(10):
            ky = (i + 0.5) * (np.pi / 10)
            points.append((np.pi, ky))
        for i in range(10):
            t = (i + 0.5) * (np.pi / 10)
            points.append((t, t))
        return np.array(points)
    
    K_POINTS = generate_k_points()
    KG_SQ = []
    for k in K_POINTS:
        kx, ky = k
        kg = k[None, :] + G_vectors * (2 * np.pi)
        kg_sq = np.sum(kg**2, axis=1)
        KG_SQ.append(kg_sq)
    KG_SQ = np.array(KG_SQ)
    
    def evaluate_cell(cell):
        """Compute the gap-midgap ratio for bands 4 and 5."""
        eps = np.array([[12.0 if c == 1 else 1.0 for c in row] for row in cell], dtype=float)
        mu = 1.0 / eps
        mu_fft = np.fft.fft2(mu)
        mu_fft_shifted = np.fft.fftshift(mu_fft)
        mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for ki in range(len(K_POINTS)):
            kg_sq = KG_SQ[ki]
            B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
            S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
            evals = np.linalg.eigvalsh(S)
            w4 = np.sqrt(evals[3])
            w5 = np.sqrt(evals[4])
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap
    
    # Generate initial candidate designs
    candidates = []
    
    # 1. Dielectric background with air holes (cross-shaped)
    cell1 = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Cross-shaped air holes
            if abs(x) < 0.15 or abs(y) < 0.15:
                cell1[i][j] = 0
    candidates.append(cell1)
    
    # 2. Dielectric background with circular air holes
    cell2 = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if r < 0.2:
                cell2[i][j] = 0
    candidates.append(cell2)
    
    # 3. Air background with dielectric cross
    cell3 = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.15 or abs(y) < 0.15:
                cell3[i][j] = 1
    candidates.append(cell3)
    
    # 4. Dielectric background with square air holes
    cell4 = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if max(abs(x), abs(y)) < 0.18:
                cell4[i][j] = 0
    candidates.append(cell4)
    
    # 5. Merged-hole pattern (two overlapping circles)
    cell5 = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Two overlapping circles along diagonal
            r1 = np.sqrt((x - 0.1)**2 + (y - 0.1)**2)
            r2 = np.sqrt((x + 0.1)**2 + (y + 0.1)**2)
            r3 = np.sqrt((x - 0.1)**2 + (y + 0.1)**2)
            r4 = np.sqrt((x + 0.1)**2 + (y - 0.1)**2)
            if min(r1, r2, r3, r4) < 0.15:
                cell5[i][j] = 0
    candidates.append(cell5)
    
    # 6. Checkerboard inverted
    cell6 = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell6[i][j] = 0
    candidates.append(cell6)
    
    # 7. Diamond-shaped air holes in dielectric
    cell7 = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < 0.25:
                cell7[i][j] = 0
    candidates.append(cell7)
    
    # 8. Ring of air in dielectric
    cell8 = [[1]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.1 < r < 0.3:
                cell8[i][j] = 0
    candidates.append(cell8)
    
    # Enforce symmetry on all candidates
    for c in candidates:
        # Make sure it's C4v symmetric
        for i in range(size):
            for j in range(size):
                # Horizontal flip
                if c[i][j] != c[i][size-1-j]:
                    c[i][j] = c[i][size-1-j]
                # Vertical flip
                if c[i][j] != c[size-1-i][j]:
                    c[i][j] = c[size-1-i][j]
                # Transpose
                if c[i][j] != c[j][i]:
                    c[i][j] = c[j][i]
    
    # Evaluate all candidates
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
    
    # Local search with adaptive multi-flip moves
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    stagnation = 0
    
    while time.time() < deadline:
        # Adaptive multi-flip: increase complexity with stagnation
        if stagnation > 10:
            n_flips = min(2 + stagnation // 5, 8)
        else:
            n_flips = 1
        
        new_cell = [row[:] for row in current_cell]
        for _ in range(n_flips):
            i = np.random.randint(0, qh)
            j = np.random.randint(0, qh)
            flip_quadrant(new_cell, i, j)
        
        new_score = evaluate_cell(new_cell)
        
        if new_score > current_score:
            current_cell = new_cell
            current_score = new_score
            best_cell = [row[:] for row in current_cell]
            best_score = current_score
            stagnation = 0
        else:
            stagnation += 1
            # Restart from best if stuck too long
            if stagnation > 40:
                current_cell = [row[:] for row in best_cell]
                current_score = best_score
                stagnation = 0
    
    return best_cell
