"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)
    
    start_time = time.time()
    deadline = 225.0
    margin = 10.0
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline - margin
    
    def enforce_symmetry(cell):
        """Enforce full d4 symmetry by averaging over the group."""
        cell = np.array(cell, dtype=np.float64)
        size = cell.shape[0]
        # d4 group operations: identity, 90° rot, 180° rot, 270° rot,
        # and reflections: vertical, horizontal, main diag, anti-diag
        sym = np.zeros_like(cell)
        for i in range(size):
            for j in range(size):
                # 8 symmetric positions
                positions = [
                    (i, j),
                    (j, i),
                    (size-1-i, size-1-j),
                    (size-1-j, size-1-i),
                    (i, size-1-j),
                    (size-1-j, i),
                    (j, size-1-i),
                    (size-1-i, j),
                ]
                vals = [cell[p[0]][p[1]] for p in positions]
                sym[i][j] = sum(vals) / len(vals)
        # Threshold at 0.5
        cell_sym = (sym >= 0.5).astype(int)
        return cell_sym
    
    def compute_fourier_coefficients(cell, max_order=8):
        """Compute Fourier coefficients of the dielectric function.
        
        eps(x,y) = eps_low + (eps_high - eps_low) * cell(x,y)
        where cell is 0 or 1.
        
        We compute coefficients for the contrast: delta_eps(x,y) = (eps_high - eps_low) * cell(x,y)
        The plane-wave method needs: delta_eps_{mn} = integral delta_eps(x,y) exp(-i 2pi (mx+ny)) dx dy
        
        For a pixelated cell, this is:
        delta_eps_{mn} = (eps_high - eps_low) / N^2 * sum_{i,j} cell[i,j] * exp(-i 2pi (m*i/N + n*j/N))
        
        But we need to be careful about the coordinate system.
        """
        cell_arr = np.array(cell, dtype=np.float64)
        N = size
        delta_eps = np.zeros((2*max_order+1, 2*max_order+1), dtype=np.complex128)
        
        for m in range(-max_order, max_order+1):
            for n in range(-max_order, max_order+1):
                # Sum over all pixels
                s = 0.0
                for i in range(N):
                    for j in range(N):
                        if cell_arr[i][j] == 1:
                            x = (i + 0.5) / N - 0.5
                            y = (j + 0.5) / N - 0.5
                            s += math.exp(-2j * math.pi * (m * x + n * y))
                delta_eps[m+max_order][n+max_order] = (eps_high - eps_low) * s / (N * N)
        
        return delta_eps
    
    def compute_band_structure(cell, num_pw=17, num_kpts=10):
        """Compute TM band structure using plane-wave expansion.
        
        For TM polarization in 2D:
        The eigenvalue equation is:
        sum_{G'} [ (k+G)^2 * delta_{GG'} + (1/eps)_{G-G'} ] E_{G'} = omega^2 E_G
        
        where (1/eps) is the inverse dielectric function in Fourier space.
        For a binary dielectric: 1/eps(x,y) = 1/eps_low for air, 1/eps_high for dielectric.
        
        Actually, for TM: the equation is:
        sum_{G'} [ (k+G)^2 delta_{GG'} + (1/eps)_{G-G'} ] E_{G'} = (omega/c)^2 E_G
        
        Wait, let me be more careful. For TM polarization (E perpendicular to the plane),
        the wave equation is:
        ∇ × (∇ × E) = (omega^2/c^2) eps(x,y) E
        
        In Fourier space:
        sum_{G'} [ (k+G)^2 delta_{GG'} - (k+G)·(k+G') (1/eps)_{G-G'} ] E_{G'} = (omega/c)^2 E_G
        
        Hmm, this is getting complex. Let me use the standard form.
        
        For TE: sum_{G'} [ (k+G)·(k+G') / eps_{G-G'} ] E_{G'} = (omega/c)^2 E_G
        For TM: sum_{G'} [ (k+G)^2 delta_{GG'} + (1/eps)_{G-G'} ] E_{G'} = (omega/c)^2 E_G
        
        Actually, I think the standard form for TM is:
        (k+G)^2 E_G + sum_{G' != G} (1/eps)_{G-G'} E_{G'} = (omega/c)^2 E_G
        
        No wait. Let me recall. The Maxwell equation for TM (E in z-direction):
        ∇_perp × (1/eps ∇_perp E_z) = (omega^2/c^2) E_z
        
        In Fourier:
        sum_{G'} [(k+G)·(k+G') / eps_{G-G'}] E_{G'} = (omega/c)^2 E_G
        
        That's the same form as TE but with 1/eps instead of eps.
        
        Actually no. Let me think again.
        
        For TM polarization, E = (0, 0, E_z). The relevant equation is:
        ∇ × (∇ × E) = (omega^2/c^2) eps E
        
        ∇ × E = (∂E_z/∂y, -∂E_z/∂x, 0)
        ∇ × (∇ × E) = (∂/∂y(-∂E_z/∂x), ∂/∂x(∂E_z/∂y), 0) = (-∂²E_z/∂x∂y, ∂²E_z/∂x∂y, 0)
        
        Hmm, that doesn't look right. Let me use the vector identity:
        ∇ × (∇ × E) = ∇(∇·E) - ∇²E
        
        For TM, ∇·E = ∂E_z/∂z = 0 (2D problem), so:
        ∇ × (∇ × E) = -∇²E = -∇²_perp E_z
        
        So: -∇²_perp E_z = (omega^2/c^2) eps(x,y) E_z
        
        In Fourier: sum_{G'} [(k+G)·(k+G')] / eps_{G-G'} E_{G'} = (omega/c)^2 E_G
        
        Wait, that's not right either. Let me be more careful.
        
        -∇²_perp E_z = (omega^2/c^2) eps(x,y) E_z
        
        E_z(r) = sum_G E_G exp(i G·r)
        eps(r) = sum_G eps_G exp(i G·r)
        
        -∇²_perp E_z = sum_G (k+G)² E_G exp(i(k+G)·r)
        
        (omega^2/c^2) eps(r) E_z(r) = (omega^2/c^2) sum_{G,G'} eps_{G-G'} E_{G'} exp(i(G+G')·r)
        
        Matching coefficients of exp(i(k+G)·r):
        (k+G)² E_G = (omega^2/c^2) sum_{G'} eps_{G-G'} E_{G'}
        
        So: sum_{G'} [ (k+G)² delta_{GG'} - (omega^2/c^2) eps_{G-G'} ] E_{G'} = 0
        
        This is a generalized eigenvalue problem. But it's not symmetric.
        
        To make it symmetric, we can write:
        sum_{G'} (1/eps)_{G-G'} (k+G)² delta_{GG'} E_{G'} = (omega/c)² E_G
        
        Hmm, this is getting complicated. Let me just use a standard approach.
        
        Actually, for the plane-wave method, the standard formulation for TM is:
        
        sum_{G'} [ (k+G)·(k+G') / eps_{G-G'} ] E_{G'} = (omega/c)² E_G
        
        where 1/eps is the Fourier transform of the inverse dielectric function.
        
        Let me just implement this directly.
        """
        # Compute 1/eps in Fourier space
        cell_arr = np.array(cell, dtype=np.float64)
        N = size
        
        # 1/eps(x,y) = 1/eps_low where cell=0, 1/eps_high where cell=1
        inv_eps_values = np.where(cell_arr == 1, 1.0/eps_high, 1.0/eps_low)
        
        # Compute Fourier coefficients of 1/eps
        max_order = num_pw
        inv_eps_ft = np.zeros((2*max_order+1, 2*max_order+1), dtype=np.complex128)
        
        for m in range(-max_order, max_order+1):
            for n in range(-max_order, max_order+1):
                s = 0.0
                for i in range(N):
                    for j in range(N):
                        x = (i + 0.5) / N - 0.5
                        y = (j + 0.5) / N - 0.5
                        s += inv_eps_values[i][j] * math.exp(-2j * math.pi * (m * x + n * y))
                inv_eps_ft[m+max_order][n+max_order] = s / (N * N)
        
        # Generate G vectors
        G_vectors = []
        for m in range(-max_order, max_order+1):
            for n in range(-max_order, max_order+1):
                G_vectors.append((2*math.pi*m, 2*math.pi*n))
        
        num_G = len(G_vectors)
        
        # K-points in the irreducible BZ
        # Irreducible BZ for square lattice: 0 <= kx <= ky, kx >= 0, ky >= 0
        # Boundary: kx = 0 (from 0 to pi), ky = kx (from 0 to pi), ky = pi (from 0 to pi/2)
        kpts = []
        # Along kx axis (ky=0): kx from 0 to pi
        for i in range(num_kpts):
            kx = math.pi * i / (num_kpts - 1)
            kpts.append((kx, 0.0))
        # Along diagonal (kx=ky): kx from 0 to pi
        for i in range(1, num_kpts):
            kx = math.pi * i / (num_kpts - 1)
            kpts.append((kx, kx))
        # Along ky axis (kx=0): ky from 0 to pi
        for i in range(1, num_kpts):
            ky = math.pi * i / (num_kpts - 1)
            kpts.append((0.0, ky))
        
        # Compute bands at each k-point
        bands = []
        for kx, ky in kpts:
            # Build the matrix: A_{GG'} = (k+G)·(k+G') / inv_eps_{G-G'}
            # Wait, I need to be careful. The equation is:
            # sum_{G'} (1/eps)_{G-G'} (k+G)·(k+G') E_{G'} = omega^2 E_G
            # This is not a standard eigenvalue problem because of the (1/eps) factor.
            
            # Let me reformulate. Define F_G = E_G.
            # sum_{G'} (1/eps)_{G-G'} (k+G)·(k+G') F_{G'} = omega^2 F_G
            
            # This is a generalized eigenvalue problem: A F = omega^2 B F
            # where A_{GG'} = (k+G)·(k+G') and B_{GG'} = (1/eps)_{G-G'}
            
            # Actually, I think the correct formulation is:
            # The matrix is M_{GG'} = (1/eps)_{G-G'} * (k+G)·(k+G')
            # and we solve M F = omega^2 F
            
            # But this matrix is not symmetric in general.
            
            # Let me use a different approach. The standard plane-wave method for TM:
            # (k+G)^2 E_G + sum_{G'!=G} (1/eps)_{G-G'} (k+G)·(k+G') E_{G'} = omega^2 E_G
            
            # Hmm, I'm getting confused. Let me just use a simple approach:
            # Compute the matrix and diagonalize it.
            
            M = np.zeros((num_G, num_G), dtype=np.complex128)
            for ig, (gx, gy) in enumerate(G_vectors):
                kgx = kx + gx
                kgy = ky + gy
                for jg, (g2x, g2y) in enumerate(G_vectors):
                    kg2x = kx + g2x
                    kg2y = ky + g2y
                    # (k+G)·(k+G')
                    dot = kgx * kg2x + kgy * kg2y
                    # (1/eps)_{G-G'}
                    dgx = gx - g2x
                    dgy = gy - g2y
                    # Convert to index
                    im = int(dgx / (2*math.pi))
                    in_ = int(dgy / (2*math.pi))
                    if abs(im) <= max_order and abs(in_) <= max_order:
                        inv_eps_val = inv_eps_ft[im+max_order][in_+max_order]
                    else:
                        inv_eps_val = 0.0
                    M[ig][jg] = inv_eps_val * dot
            
            # Diagonalize
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(M)
                # Take the smallest 5 eigenvalues
                idx = np.argsort(eigenvalues.real)[:5]
                bands.append(eigenvalues.real[idx])
            except:
                bands.append(np.array([0.0]*5))
        
        # Find min of band 5 and max of band 4 (0-indexed: bands[4] and bands[3])
        # bands is a list of arrays, each with 5 values
        min_band5 = min(b[4] for b in bands if len(b) > 4)
        max_band4 = max(b[3] for b in bands if len(b) > 3)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2
        
        if midgap > 0:
            ratio = 2 * gap / (min_band5 + max_band4)
        else:
            ratio = gap
        
        return ratio
    
    def fast_score(cell):
        """Fast heuristic score based on Fourier coefficients."""
        cell_arr = np.array(cell, dtype=np.float64)
        N = size
        fill = np.sum(cell_arr) / (N * N)
        
        if fill < 0.1 or fill > 0.7:
            return -1.0
        
        # Compute key Fourier coefficients
        c20 = 0.0
        c02 = 0.0
        c22 = 0.0
        c10 = 0.0
        c01 = 0.0
        c11 = 0.0
        
        for i in range(N):
            for j in range(N):
                if cell_arr[i][j] == 1:
                    x = (i + 0.5) / N - 0.5
                    y = (j + 0.5) / N - 0.5
                    c20 += math.cos(4*math.pi*x)
                    c02 += math.cos(4*math.pi*y)
                    c22 += math.cos(4*math.pi*x)*math.cos(4*math.pi*y)
                    c10 += math.cos(2*math.pi*x)
                    c01 += math.cos(2*math.pi*y)
                    c11 += math.cos(2*math.pi*x)*math.cos(2*math.pi*y)
        
        N2 = N * N
        c20 /= N2
        c02 /= N2
        c22 /= N2
        c10 /= N2
        c01 /= N2
        c11 /= N2
        
        # Score based on second-order components
        second_order = abs(c20) + abs(c02) + abs(c22)
        first_order = abs(c10) + abs(c01) + abs(c11)
        
        fill_score = 1.0 - abs(fill - 0.35) / 0.35
        
        score = second_order * 10.0 * fill_score + first_order * 3.0 * fill_score
        return score
    
    # Generate candidate patterns
    candidates = []
    
    def four_rod_pattern(radius, offset):
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    if dx*dx + dy*dy < radius*radius:
                        return True
            return False
        return f
    
    def make_cell(pattern_func):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                cell[i][j] = 1 if pattern_func(x, y) else 0
        return enforce_symmetry(cell)
    
    # Four-rod patterns
    for r in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]:
        for off in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32]:
            candidates.append(four_rod_pattern(r, off))
    
    # Ring patterns
    def ring_pattern(inner_r, outer_r):
        def f(x, y):
            r2 = x*x + y*y
            return inner_r*inner_r < r2 < outer_r*outer_r
        return f
    
    for ir, orr in [(0.08, 0.18), (0.10, 0.20), (0.12, 0.22), (0.14, 0.25), (0.16, 0.28), (0.18, 0.30)]:
        candidates.append(ring_pattern(ir, orr))
    
    # Cross patterns
    def cross_pattern(arm_width, arm_length):
        def f(x, y):
            if abs(y) < arm_width/2 and abs(x) < arm_length/2:
                return True
            if abs(x) < arm_width/2 and abs(y) < arm_length/2:
                return True
            return False
        return f
    
    for aw, al in [(0.10, 0.35), (0.12, 0.40), (0.15, 0.45), (0.18, 0.50), (0.20, 0.55), (0.22, 0.60)]:
        candidates.append(cross_pattern(aw, al))
    
    # Four small rings
    def four_rings_pattern(inner_r, outer_r, offset):
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    r2 = dx*dx + dy*dy
                    if inner_r*inner_r < r2 < outer_r*outer_r:
                        return True
            return False
        return f
    
    for ir, orr in [(0.04, 0.10), (0.05, 0.12), (0.06, 0.14), (0.07, 0.16), (0.08, 0.18)]:
        for off in [0.20, 0.22, 0.25, 0.28, 0.30]:
            candidates.append(four_rings_pattern(ir, orr, off))
    
    # Fourier-based patterns
    def fourier_cell(modes, base=0.35):
        def f(x, y):
            val = base
            for amp, m, n in modes:
                val += amp * math.cos(2*math.pi*m*x) * math.cos(2*math.pi*n*y)
            return val > 0.5
        return f
    
    fourier_candidates = [
        [(0.3, 1, 0), (0.3, 0, 1)],
        [(0.4, 1, 0), (0.4, 0, 1)],
        [(0.5, 1, 0), (0.5, 0, 1)],
        [(0.3, 1, 0), (0.3, 0, 1), (0.2, 1, 1)],
        [(0.4, 1, 0), (0.4, 0, 1), (0.2, 1, 1)],
        [(0.3, 1, 0), (0.3, 0, 1), (0.3, 1, 1)],
        [(0.3, 2, 0), (0.3, 0, 2)],
        [(0.4, 2, 0), (0.4, 0, 2)],
        [(0.5, 2, 0), (0.5, 0, 2)],
        [(0.2, 1, 0), (0.2, 0, 1), (0.3, 2, 0), (0.3, 0, 2)],
        [(0.3, 1, 0), (0.3, 0, 1), (0.2, 2, 0), (0.2, 0, 2)],
        [(0.3, 2, 2)],
        [(0.4, 2, 2)],
        [(0.3, 1, 0), (0.3, 0, 1), (0.3, 2, 2)],
        [(0.2, 1, 0), (0.2, 0, 1), (0.2, 2, 0), (0.2, 0, 2), (0.2, 2, 2)],
        [(0.2, 3, 0), (0.2, 0, 3)],
        [(0.3, 3, 0), (0.3, 0, 3)],
        [(0.2, 1, 0), (0.2, 0, 1), (0.2, 3, 0), (0.2, 0, 3)],
        [(0.15, 1, 0), (0.15, 0, 1), (0.25, 2, 0), (0.25, 0, 2), (0.15, 2, 2)],
        [(0.2, 1, 0), (0.2, 0, 1), (0.3, 2, 0), (0.3, 0, 2), (0.2, 2, 2)],
        [(0.1, 1, 0), (0.1, 0, 1), (0.35, 2, 0), (0.35, 0, 2), (0.15, 2, 2)],
    ]
    
    for modes in fourier_candidates:
        candidates.append(fourier_cell(modes))
    
    # Evaluate candidates with fast score first
    best_cell = None
    best_fast_score = -1
    best_full_score = -1
    
    evaluated_cells = []
    
    for pattern_func in candidates:
        if not time_ok():
            break
        
        cell = make_cell(pattern_func)
        
        # Check filling
        fill = np.sum(cell) / (size * size)
        if fill < 0.1 or fill > 0.7:
            continue
        
        # Fast score
        fscore = fast_score(cell)
        
        if fscore > best_fast_score:
            best_fast_score = fscore
            best_cell = cell
        
        evaluated_cells.append((fscore, cell))
    
    # Sort by fast score and evaluate top candidates with full band structure
    evaluated_cells.sort(key=lambda x: -x[0])
    
    num_to_evaluate = min(20, len(evaluated_cells))
    
    for fscore, cell in evaluated_cells[:num_to_evaluate]:
        if not time_ok():
            break
        
        try:
            full_score = compute_band_structure(cell, num_pw=12, num_kpts=8)
        except:
            full_score = -1.0
        
        if full_score > best_full_score:
            best_full_score = full_score
            best_cell = cell
    
    # Local optimization around the best cell
    if best_cell is not None and time_ok():
        # Try flipping individual pixels to improve the score
        best_cell_arr = np.array(best_cell)
        improved = True
        max_iterations = 50
        
        for iteration in range(max_iterations):
            if not time_ok():
                break
            if not improved:
                break
            improved = False
            
            # Try flipping some random pixels
            for _ in range(20):
                if not time_ok():
                    break
                
                i = random.randint(0, size-1)
                j = random.randint(0, size-1)
                
                # Create a modified cell
                new_cell = best_cell_arr.copy()
                new_cell[i][j] = 1 - new_cell[i][j]
                
                # Enforce symmetry
                new_cell_sym = enforce_symmetry(new_cell.tolist())
                
                # Check if it's different
                if np.array_equal(new_cell_sym, best_cell_arr):
                    continue
                
                # Evaluate
                try:
                    score = compute_band_structure(new_cell_sym, num_pw=10, num_kpts=6)
                except:
                    score = -1.0
                
                if score > best_full_score:
                    best_full_score = score
                    best_cell_arr = new_cell_sym
                    improved = True
    
    # Convert to nested list
    best_cell_list = best_cell_arr.tolist() if best_cell_arr is not None else [[0]*size for _ in range(size)]
    
    # Ensure it's a proper nested list of ints
    result = []
    for row in best_cell_list:
        result.append([int(x) for x in row])
    
    return result
