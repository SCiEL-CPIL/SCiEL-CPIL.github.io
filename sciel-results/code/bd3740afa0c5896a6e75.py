"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    t0 = time.time()
    deadline = t0 + 215.0  # 215s of 230s wall-clock budget

    half = size // 2  # 16
    N = size  # 32

    # --- Symmetric cell construction from 16x16 upper-triangle quadrant ---
    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from 16x16 upper-triangle quadrant (i<=j)."""
        cell = [[0]*N for _ in range(N)]
        for i in range(half):
            for j in range(i, half):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
                cell[j][i] = val
                cell[j][N-1-i] = val
                cell[N-1-j][i] = val
                cell[N-1-j][N-1-i] = val
        return cell

    # --- PWE solver ---
    # Precompute G vectors for 17x17 = 289 plane waves (mx,my in -8..8)
    max_g = 8
    g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
    num_g = len(g_indices)  # 289
    G_vecs = np.array([(2*math.pi*mx, 2*math.pi*my) for mx, my in g_indices])
    
    # Precompute G difference indices for eps_inv matrix construction
    # eps_inv_mat[p,q] = eps_inv(G_p - G_q)
    # G_p - G_q has components (2*pi*(mx_p-mx_q), 2*pi*(my_p-my_q))
    # We need Fourier coefficients of 1/epsilon at these differences

    # K-points in the irreducible BZ (30 points)
    # Irreducible BZ: 0 <= kx <= ky, kx + ky <= pi
    k_points = []
    n_per_line = 5
    # Gamma to X: (0, t) for t in [0, pi]
    for t in np.linspace(0, math.pi, n_per_line):
        k_points.append((0.0, float(t)))
    # Gamma to M: (t, t) for t in [0, pi/2]
    for t in np.linspace(0, math.pi/2, n_per_line):
        k_points.append((float(t), float(t)))
    # M to X: (pi/2 - t, pi/2 + t) for t in [0, pi/2]
    for t in np.linspace(0, math.pi/2, n_per_line):
        k_points.append((math.pi/2 - float(t), math.pi/2 + float(t)))
    # Interior points
    k_points.append((math.pi/6, math.pi/3))
    k_points.append((math.pi/4, math.pi/2))
    k_points.append((math.pi/8, 3*math.pi/8))
    k_points.append((math.pi/5, 2*math.pi/5))
    k_points.append((math.pi/3, 2*math.pi/3))
    k_points = k_points[:30]

    def compute_eps_inv_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon using FFT."""
        eps_inv = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps_inv[i, j] = 1.0 / eps_high if cell[i][j] == 1 else 1.0
        fft_coeff = np.fft.fft2(eps_inv) / (N * N)
        return fft_coeff

    def evaluate_gap(cell, fft_coeff=None):
        """Evaluate the gap-midgap ratio between TM bands 4 and 5."""
        if fft_coeff is None:
            fft_coeff = compute_eps_inv_fourier(cell)
        
        # Build eps_inv difference matrix
        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                dm = (mx_p - mx_q) % N
                dn = (my_p - my_q) % N
                eps_inv_mat[p, q] = fft_coeff[dn, dm]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            # |k + G_q|^2
            kG_sq = (kx + G_vecs[:, 0])**2 + (ky + G_vecs[:, 1])**2
            M = eps_inv_mat * kG_sq[None, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            eigvals = np.sort(eigvals)
            if len(eigvals) < 5:
                continue
            
            omega4 = math.sqrt(max(eigvals[3], 0))
            omega5 = math.sqrt(max(eigvals[4], 0))
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # --- Initial designs ---
    def make_ring_quadrant(r_inner, r_outer):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / N
                y = (i + 0.5) / N
                r = math.sqrt(x*x + y*y)
                if r_inner <= r <= r_outer:
                    q[i][j] = 1
        return q

    def make_disk_quadrant(radius):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / N
                y = (i + 0.5) / N
                r = math.sqrt(x*x + y*y)
                if r <= radius:
                    q[i][j] = 1
        return q

    def make_cross_quadrant(width=0.15):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / N
                y = (i + 0.5) / N
                if x < width or y < width:
                    q[i][j] = 1
        return q

    # Generate diverse initial designs
    initial_designs = []
    # Rings with various radii
    for ri, ro in [(0.15, 0.35), (0.2, 0.4), (0.1, 0.3), (0.25, 0.45), (0.3, 0.5)]:
        initial_designs.append(make_ring_quadrant(ri, ro))
    # Disks
    for r in [0.2, 0.3, 0.35]:
        initial_designs.append(make_disk_quadrant(r))
    # Crosses
    for w in [0.08, 0.12, 0.15]:
        initial_designs.append(make_cross_quadrant(w))
    # Checkerboard-like patterns
    for period in [4, 6, 8]:
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                if (i // period + j // period) % 2 == 0:
                    q[i][j] = 1
        initial_designs.append(q)
    # Random designs
    rng_init = random.Random(42)
    for _ in range(5):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                q[i][j] = rng_init.randint(0, 1)
        initial_designs.append(q)

    # Evaluate initial designs
    best_score = -1.0
    best_quadrant = initial_designs[0]
    
    for q in initial_designs:
        if time.time() > deadline - 60:
            break
        cell = build_cell_from_quadrant(q)
        score = evaluate_gap(cell)
        if score > best_score:
            best_score = score
            best_quadrant = [row[:] for row in q]

    # --- Topology optimization guided by field intensities ---
    # This is the key structural change: use local field intensities of bands 4 and 5
    # to guide which pixels to flip, rather than random local search.
    
    def get_field_intensities(cell, fft_coeff=None):
        """Get field intensities for bands 4 and 5 at the critical k-point.
        Returns (intensity4, intensity5, critical_k) where intensity is a 32x32 array."""
        if fft_coeff is None:
            fft_coeff = compute_eps_inv_fourier(cell)
        
        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                dm = (mx_p - mx_q) % N
                dn = (my_p - my_q) % N
                eps_inv_mat[p, q] = fft_coeff[dn, dm]
        
        # Find the k-point where the gap is most critical (minimum gap)
        best_gap = float('inf')
        best_k = k_points[0]
        best_eigvecs = None
        
        for kx, ky in k_points:
            kG_sq = (kx + G_vecs[:, 0])**2 + (ky + G_vecs[:, 1])**2
            M = eps_inv_mat * kG_sq[None, :]
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
            except:
                continue
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            if len(eigvals) < 5:
                continue
            w4 = math.sqrt(max(eigvals[3], 0))
            w5 = math.sqrt(max(eigvals[4], 0))
            gap = w5 - w4
            if gap < best_gap:
                best_gap = gap
                best_k = (kx, ky)
                best_eigvecs = eigvecs
        
        if best_eigvecs is None:
            return None, None, None
        
        kx, ky = best_k
        # Field intensity: |sum_q c_q exp(i(k+G_q).r)|^2
        # For each pixel (i,j) at position (j/N, i/N)
        intensity4 = np.zeros((N, N))
        intensity5 = np.zeros((N, N))
        
        c4 = best_eigvecs[:, 3]
        c5 = best_eigvecs[:, 4]
        
        for i in range(N):
            for j in range(N):
                x = j / N
                y = i / N
                field4 = 0.0 + 0.0j
                field5 = 0.0 + 0.0j
                for q in range(num_g):
                    Gx = G_vecs[q, 0]
                    Gy = G_vecs[q, 1]
                    phase4 = np.exp(1j * ((kx + Gx) * x + (ky + Gy) * y))
                    phase5 = np.exp(1j * ((kx + Gx) * x + (ky + Gy) * y))
                    field4 += c4[q] * phase4
                    field5 += c5[q] * phase5
                intensity4[i, j] = abs(field4)**2
                intensity5[i, j] = abs(field5)**2
        
        return intensity4, intensity5, best_k

    # Topology optimization loop
    current_quadrant = [row[:] for row in best_quadrant]
    current_score = best_score
    current_cell = build_cell_from_quadrant(current_quadrant)
    current_fft = compute_eps_inv_fourier(current_cell)
    
    positions = [(i, j) for i in range(half) for j in range(i, half)]
    rng = random.Random(12345)
    
    topo_iterations = 0
    max_topo_iterations = 200
    
    while time.time() < deadline - 30 and topo_iterations < max_topo_iterations:
        topo_iterations += 1
        
        # Get field intensities
        if topo_iterations % 10 == 1 or topo_iterations == 1:
            int4, int5, crit_k = get_field_intensities(current_cell, current_fft)
            if int4 is None:
                break
            # Normalize
            max_i4 = np.max(int4)
            max_i5 = np.max(int5)
            if max_i4 > 0:
                int4 = int4 / max_i4
            if max_i5 > 0:
                int5 = int5 / max_i5
        else:
            # Reuse cached intensities (approximate)
            pass
        
        # Guided pixel flips
        improved = False
        shuffled_pos = positions[:]
        rng.shuffle(shuffled_pos)
        
        for (i, j) in shuffled_pos:
            if time.time() > deadline - 25:
                break
            
            # Decide flip direction based on field intensities
            # Full cell pixel (i,j) and its D4 orbit
            # Get intensities at the quadrant pixel
            # The quadrant pixel (i,j) maps to full cell positions:
            # (i,j), (i,N-1-j), (N-1-i,j), (N-1-i,N-1-j), (j,i), (j,N-1-i), (N-1-j,i), (N-1-j,N-1-i)
            pix_positions = [
                (i, j), (i, N-1-j), (N-1-i, j), (N-1-i, N-1-j),
                (j, i), (j, N-1-i), (N-1-j, i), (N-1-j, N-1-i)
            ]
            
            avg_int4 = 0.0
            avg_int5 = 0.0
            for (pi, pj) in pix_positions:
                if 0 <= pi < N and 0 <= pj < N:
                    avg_int4 += int4[pi, pj]
                    avg_int5 += int5[pi, pj]
            avg_int4 /= len(pix_positions)
            avg_int5 /= len(pix_positions)
            
            # Strategy: add dielectric where band 4 intensity is high (raises band 4 less)
            # and remove dielectric where band 5 intensity is high (lowers band 5 less)
            # Net effect: separate the bands
            # If int4 > int5: prefer to have dielectric here (helps separate)
            # If int5 > int4: prefer to have air here (helps separate)
            
            current_val = current_quadrant[i][j]
            if avg_int4 > avg_int5:
                # Prefer dielectric
                flip_to = 1
            else:
                # Prefer air
                flip_to = 0
            
            # Sometimes flip the other way for exploration
            if rng.random() < 0.3:
                flip_to = 1 - flip_to
            
            if current_val == flip_to:
                continue
            
            # Try the flip
            current_quadrant[i][j] = flip_to
            new_cell = build_cell_from_quadrant(current_quadrant)
            new_fft = compute_eps_inv_fourier(new_cell)
            new_score = evaluate_gap(new_cell, new_fft)
            
            if new_score > current_score:
                current_score = new_score
                current_cell = new_cell
                current_fft = new_fft
                improved = True
                if new_score > best_score:
                    best_score = new_score
                    best_quadrant = [row[:] for row in current_quadrant]
            else:
                current_quadrant[i][j] = current_val
        
        if not improved and topo_iterations > 20:
            # Try random flips as escape mechanism
            for _ in range(10):
                if time.time() > deadline - 25:
                    break
                i, j = rng.choice(positions)
                current_quadrant[i][j] = 1 - current_quadrant[i][j]
                new_cell = build_cell_from_quadrant(current_quadrant)
                new_fft = compute_eps_inv_fourier(new_cell)
                new_score = evaluate_gap(new_cell, new_fft)
                if new_score > current_score:
                    current_score = new_score
                    current_cell = new_cell
                    current_fft = new_fft
                    if new_score > best_score:
                        best_score = new_score
                        best_quadrant = [row[:] for row in current_quadrant]
                else:
                    current_quadrant[i][j] = 1 - current_quadrant[i][j]

    # --- Final local search with high-order PWE ---
    if time.time() < deadline - 15:
        current_quadrant = [row[:] for row in best_quadrant]
        current_score = best_score
        current_cell = build_cell_from_quadrant(current_quadrant)
        current_fft = compute_eps_inv_fourier(current_cell)
        
        rng2 = random.Random(99999)
        shuffled = positions[:]
        rng2.shuffle(shuffled)
        
        for (i, j) in shuffled:
            if time.time() > deadline - 10:
                break
            
            old_val = current_quadrant[i][j]
            current_quadrant[i][j] = 1 - old_val
            new_cell = build_cell_from_quadrant(current_quadrant)
            new_fft = compute_eps_inv_fourier(new_cell)
            new_score = evaluate_gap(new_cell, new_fft)
            
            if new_score > current_score:
                current_score = new_score
                current_cell = new_cell
                current_fft = new_fft
                if new_score > best_score:
                    best_score = new_score
                    best_quadrant = [row[:] for row in current_quadrant]
            else:
                current_quadrant[i][j] = old_val

    # Build final cell
    final_cell = build_cell_from_quadrant(best_quadrant)
    
    # Verify symmetry
    for i in range(N):
        for j in range(N):
            if final_cell[i][j] != final_cell[N-1-i][j]:
                final_cell[N-1-i][j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[i][N-1-j]:
                final_cell[i][N-1-j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[j][i]:
                final_cell[j][i] = final_cell[i][j]
    
    return final_cell
# EVOLVE-BLOCK-END
