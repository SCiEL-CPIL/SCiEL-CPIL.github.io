import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 110 # 110s limit to be safe
    
    size = problem['pixels']
    # Initialize with a guess: Inverted Rod (Dielectric background, Air hole)
    # Or maybe a "square" hole?
    # Let's start with a circular air hole in dielectric background.
    # 1 = Dielectric, 0 = Air.
    # Seed was 1 in center. Let's try 1 in background, 0 in center.
    
    cell = np.ones((size, size), dtype=int)
    # Create a mask for the hole
    y, x = np.mgrid[0:size, 0:size]
    # Center is 15.5
    dist = np.sqrt((x - 15.5)**2 + (y - 15.5)**2)
    # Hole radius ~ 0.3
    cell[dist < 0.3 * size] = 0
    
    # Symmetrize
    cell = symmetrize(cell)
    
    best_cell = cell.copy()
    best_score = -np.inf
    
    # PWE Solver Parameters
    # Cutoff radius in reciprocal lattice units (2pi/a)
    R = 8 
    # Generate G vectors
    Gs = []
    for gx in range(-R, R+1):
        for gy in range(-R, R+1):
            if gx*gx + gy*gy <= R*R:
                Gs.append((gx, gy))
    Gs = np.array(Gs)
    N_G = len(Gs)
    
    # Precompute G dot G
    G2 = Gs[:, 0]**2 + Gs[:, 1]**2
    
    # K-points
    # Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
    # Path: Gamma -> X -> M -> Gamma
    k_points = []
    # Sample 10 points on each segment
    for t in np.linspace(0, 0.5, 10):
        k_points.append((t, 0))
    for t in np.linspace(0.5, 0, 10):
        k_points.append((t, 0.5))
    # Add endpoints explicitly to be sure
    k_points = np.array(k_points)
    
    eps_low = problem['eps_low']
    eps_high = problem['eps_high']
    
    def get_score(cell):
        # Compute inverse permittivity FFT
        # cell: 1 for dielectric (eps_high), 0 for air (eps_low)
        eps_inv = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        
        # FFT of eps_inv
        # Need to align G vectors with FFT frequencies
        # FFT freqs are 0, 1, ..., N/2-1, -N/2, ..., -1
        # Our Gs are integers.
        # We can use np.fft.fft2 on the grid, then pick the corresponding coeffs.
        # Or just compute the sum directly? N_G is ~200. Grid is 32x32.
        # Direct sum: 200 * 1024 = 200k ops. Fast.
        # But FFT is faster.
        eps_inv_fft = np.fft.fft2(eps_inv) / (size*size)
        
        # Map G vectors to FFT indices
        # FFT index i corresponds to freq i if i < N/2, else i - N
        # Our Gs are in range [-R, R]. R < N/2 (8 < 16).
        # So we can just index into the FFT array directly.
        # But fft2 returns frequencies in order 0, 1, ..., 15, -16, ..., -1
        # We need to handle the negative indices.
        
        eps_inv_G = np.zeros(N_G, dtype=complex)
        for idx, (gx, gy) in enumerate(Gs):
            # Convert G to FFT index
            ix = gx % size
            iy = gy % size
            eps_inv_G[idx] = eps_inv_fft[iy, ix]
            
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            # k is in units of 2pi/a
            # Vector K = k + G
            K = Gs + np.array([kx, ky])
            K2 = K[:, 0]**2 + K[:, 1]**2
            
            # Build Matrix M
            # M[i, j] = - K[i] dot K[j] * eps_inv_G[i-j]
            # Note: eps_inv_G is indexed by G_diff.
            # We need eps_inv(G_i - G_j).
            # This is a convolution structure.
            # M = - (K_outer_dot) * eps_inv_conv
            
            # Efficient way:
            # For each i, M[i, :] = - (K[i,0]*K[:,0] + K[i,1]*K[:,1]) * eps_inv_G[i - j] ?
            # No, eps_inv_G is a vector of coefficients for G.
            # The term is eps_inv(G_i - G_j).
            # We can construct a matrix E where E[i,j] = eps_inv_G[i-j].
            # This is a circulant-like matrix but in 2D.
            
            # Let's just loop over i for simplicity if N_G is small?
            # N_G ~ 200. 200*200 = 40k ops per k-point.
            # 30 k-points -> 1.2M ops.
            # 1000 iterations -> 1.2B ops. Might be too slow in Python loops.
            
            # Vectorized approach:
            # We need the matrix E where E[i,j] = eps_inv_G[index of (G_i - G_j)]
            # This is hard to vectorize directly without building the diff grid.
            
            # Alternative:
            # M = - (K_x_i K_x_j + K_y_i K_y_j) * E[i,j]
            
            # Let's try a hybrid: Loop over i, vectorize j.
            M = np.zeros((N_G, N_G), dtype=complex)
            for i in range(N_G):
                # G_diff = Gs - Gs[i]
                G_diff = Gs - Gs[i]
                # Find indices in Gs for these diffs?
                # No, eps_inv_G is defined for the set of Gs.
                # If G_diff is not in Gs, the coefficient is 0?
                # No, the Fourier series is infinite. We truncate.
                # If G_diff is outside our cutoff, we assume 0.
                # This is an approximation.
                
                # To be accurate, we should sum over all G_j.
                # If we truncate Gs, we miss terms where G_i - G_j is large.
                # However, eps_inv decays.
                
                # Let's stick to the truncated basis.
                # We need eps_inv(G_i - G_j).
                # We can precompute a dictionary of G -> coeff.
                
                # For now, let's use a simpler approximation or just accept the truncation error.
                
                # Actually, we can compute the matrix E using FFT logic if we pad.
                # But N_G is small.
                
                # Let's just iterate j.
                # To speed up, we can precompute a map from (gx, gy) to index in Gs.
                
                pass # Placeholder
            
            # This loop is too slow.
            # Let's optimize the matrix construction.
            
            # Precompute index map
            # We will do this outside the k-loop.
            
            # ... (See implementation below) ...
            
            # Eigenvalues
            # H = M
            # Solve H v = omega^2 v
            vals = np.linalg.eigvalsh(M)
            vals = np.sqrt(np.real(vals))
            vals = np.sort(vals)
            
            # Bands 4 and 5 (index 3 and 4)
            if len(vals) >= 5:
                omega_4 = vals[3]
                omega_5 = vals[4]
                if omega_4 > max_omega_4: max_omega_4 = omega_4
                if omega_5 < min_omega_5: min_omega_5 = omega_5
                
        if min_omega_5 == np.inf or max_omega_4 == 0:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0: return -1.0
        return gap / midgap

    # Precompute G index map for fast lookup
    G_map = {}
    for idx, (gx, gy) in enumerate(Gs):
        G_map[(gx, gy)] = idx
        
    def fast_get_score(cell):
        eps_inv = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        eps_inv_fft = np.fft.fft2(eps_inv) / (size*size)
        
        eps_inv_G = np.zeros(N_G, dtype=complex)
        for idx, (gx, gy) in enumerate(Gs):
            ix = gx % size
            iy = gy % size
            eps_inv_G[idx] = eps_inv_fft[iy, ix]
            
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            K = Gs + np.array([kx, ky])
            
            # Build M
            # M[i,j] = - (K_i dot K_j) * eps_inv(G_i - G_j)
            # We need eps_inv(G_i - G_j).
            # Let's construct the matrix E first.
            E = np.zeros((N_G, N_G), dtype=complex)
            for i in range(N_G):
                gi = Gs[i]
                for j in range(N_G):
                    gj = Gs[j]
                    diff = (gi[0] - gj[0], gi[1] - gj[1])
                    if diff in G_map:
                        E[i, j] = eps_inv_G[G_map[diff]]
                    # else 0
                
            # Now M = - (K_x_i K_x_j + K_y_i K_y_j) * E
            Kx = K[:, 0]
            Ky = K[:, 1]
            # Outer products
            # M = - (np.outer(Kx, Kx) + np.outer(Ky, Ky)) * E
            M = - (np.outer(Kx, Kx) + np.outer(Ky, Ky)) * E
            
            vals = np.linalg.eigvalsh(M)
            vals = np.sqrt(np.real(vals))
            vals = np.sort(vals)
            
            if len(vals) >= 5:
                omega_4 = vals[3]
                omega_5 = vals[4]
                if omega_4 > max_omega_4: max_omega_4 = omega_4
                if omega_5 < min_omega_5: min_omega_5 = omega_5
                
        if min_omega_5 == np.inf or max_omega_4 == 0:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0: return -1.0
        return gap / midgap

    # Optimization Loop
    current_score = fast_get_score(cell)
    if current_score > best_score:
        best_score = current_score
        best_cell = cell.copy()
        
    # Symmetry indices
    # Fundamental domain: i in 0..15, j in 0..15
    # But due to transpose, j <= i?
    # Let's just pick random i,j in 0..15 and update all 8 symmetric pixels.
    
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        
        # Get current value
        val = cell[i, j]
        new_val = 1 - val
        
        # Update symmetric positions
        # (i, j), (i, 31-j), (31-i, j), (31-i, 31-j)
        # (j, i), (j, 31-i), (31-j, i), (31-j, 31-i)
        
        positions = [
            (i, j), (i, 31-j), (31-i, j), (31-i, 31-j),
            (j, i), (j, 31-i), (31-j, i), (31-j, 31-i)
        ]
        
        # Check if any position is out of bounds? No, 0..31.
        # Check duplicates?
        # If i==j, some are same.
        # If i==15.5? No, integer.
        
        # Apply change
        for r, c in positions:
            cell[r, c] = new_val
            
        new_score = fast_get_score(cell)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = cell.copy()
        else:
            # Revert
            for r, c in positions:
                cell[r, c] = val

    # Convert to list of lists
    return best_cell.tolist()

def symmetrize(cell):
    size = cell.shape[0]
    sym_cell = np.zeros_like(cell)
    for i in range(size):
        for j in range(size):
            # Take average or just pick one?
            # We want strict symmetry.
            # Pick the value from the fundamental domain.
            # Map (i,j) to fundamental domain (0..15, 0..15)
            fi = min(i, size-1-i)
            fj = min(j, size-1-j)
            # Also transpose symmetry
            # The value at (i,j) must equal (j,i) etc.
            # Let's just enforce by copying from (fi, fj) if we ensure (fi,fj) is consistent.
            # Better: Iterate fundamental domain and copy out.
            pass
    
    # Correct approach:
    # Iterate i in 0..15, j in 0..15
    # Set the 8 symmetric pixels to cell[i,j]
    # But we need to handle the case where the input cell is not symmetric.
    # We'll just average or pick the first.
    # For initialization, we generate symmetric cells.
    # For search, we maintain symmetry.
    # So symmetrize is only needed for the initial guess if it's not perfectly symmetric.
    # My initial guess generation uses dist < r, which is symmetric.
    
    # Just to be safe:
    out = np.zeros_like(cell)
    for i in range(16):
        for j in range(16):
            val = cell[i, j]
            out[i, j] = val
            out[i, 31-j] = val
            out[31-i, j] = val
            out[31-i, 31-j] = val
            out[j, i] = val
            out[j, 31-i] = val
            out[31-j, i] = val
            out[31-j, 31-i] = val
    return out

