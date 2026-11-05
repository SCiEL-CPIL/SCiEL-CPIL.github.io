import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Build symmetry orbits
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            orbit = set()
            x, y = i, j
            for _ in range(4):
                orbit.add((x, y))
                orbit.add((x, size - 1 - y))
                x, y = y, size - 1 - x
            orbits.append(orbit)
            seen.update(orbit)
    n_orbits = len(orbits)
    
    def orbit_to_grid(orbit_vals):
        grid = np.zeros((size, size), dtype=np.int8)
        for k, orbit in enumerate(orbits):
            val = int(orbit_vals[k])
            for (i, j) in orbit:
                grid[i, j] = val
        return grid
    
    # Parametric representation: concentric square rings
    # We use a set of parameters to define rings of dielectric
    def params_to_orbits(params):
        """Convert continuous params to orbit values using square ring representation."""
        n_params = len(params)
        grid = np.zeros((size, size), dtype=np.int8)
        
        xs = (np.arange(size) + 0.5) / size - 0.5
        ys = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(xs, ys)
        
        # Define up to 8 concentric square rings
        # Each ring has: inner_radius, outer_radius, thickness
        # We use a radial distance measure based on max(|x|,|y|)
        radius = np.maximum(np.abs(X), np.abs(Y))
        
        for p_idx in range(min(n_params // 3, 8)):
            base = p_idx * 3
            if base + 2 >= n_params:
                break
            r_inner = params[base]
            r_outer = params[base + 1]
            width = params[base + 2]
            
            if width <= 0.001:
                continue
            
            r_center = (r_inner + r_outer) / 2.0
            mask = np.abs(radius - r_center) < width / 2.0
            grid[mask] = 1
        
        # Convert to orbit values
        orbit_vals = np.zeros(n_orbits)
        for k, orbit in enumerate(orbits):
            val = sum(grid[i, j] for (i, j) in orbit)
            orbit_vals[k] = 1.0 if val > len(orbit) / 2 else 0.0
        
        return orbit_vals
    
    # Fast PWE evaluator
    N_pw = 7  # -3 to 3
    G_x = np.arange(-N_pw // 2, N_pw // 2 + 1)
    G_y = np.arange(-N_pw // 2, N_pw // 2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y])
    N_modes = len(G_vectors)
    
    # Key K-points in irreducible BZ
    k_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (0.5, 0.0),
        (0.25, 0.25),
        (0.25, 0.0),
        (0.5, 0.25),
        (0.125, 0.125),
        (0.375, 0.375),
    ]
    
    def fast_evaluate(orbit_vals):
        grid = orbit_to_grid(orbit_vals)
        eps_grid = eps_low + (eps_high - eps_low) * grid
        
        eps_fft = np.fft.fft2(eps_grid) / (size * size)
        
        global_max_4 = 0.0
        global_min_5 = float('inf')
        
        for kx, ky in k_points:
            M = np.zeros((N_modes, N_modes), dtype=float)
            
            for i in range(N_modes):
                gx, gy = G_vectors[i]
                kx_i = kx + gx
                ky_i = ky + gy
                k_sq_i = kx_i**2 + ky_i**2
                
                for j in range(N_modes):
                    gx_j, gy_j = G_vectors[j]
                    kx_j = kx + gx_j
                    ky_j = ky + gy_j
                    
                    dot = kx_i * kx_j + ky_i * ky_j
                    
                    dgx = int(gx - gx_j)
                    dgy = int(gy - gy_j)
                    idx_i = dgx % size
                    idx_j = dgy % size
                    eps_val = eps_fft[idx_i, idx_j].real
                    if eps_val < 0.1:
                        eps_val = 0.1
                    
                    if i == j:
                        M[i, j] = k_sq_i - dot / eps_val
                    else:
                        M[i, j] = -dot / eps_val
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                return -1.0
            
            if len(eigvals) < 5:
                return -1.0
            
            omega_4 = np.sqrt(max(eigvals[3], 0))
            omega_5 = np.sqrt(max(eigvals[4], 0))
            
            global_max_4 = max(global_max_4, omega_4)
            global_min_5 = min(global_min_5, omega_5)
        
        if global_min_5 == float('inf'):
            return -1.0
        
        gap = global_min_5 - global_max_4
        midgap = (global_min_5 + global_max_4) / 2.0
        if midgap < 1e-10:
            return 0.0
        return gap / midgap
    
    # Optimization
    start_time = time.time()
    deadline = start_time + 180.0
    
    np.random.seed(42)
    
    best_score = -float('inf')
    best_orbit_vals = None
    
    # Phase 1: Coarse search over parametric space
    n_rings = 4
    n_params = n_rings * 3
    param_ranges = []
    for _ in range(n_rings):
        param_ranges.append((0.05, 0.45))  # r_inner
        param_ranges.append((0.05, 0.45))  # r_outer
        param_ranges.append((0.02, 0.12))  # width
    
    # Random sampling
    n_samples = 500
    for _ in range(n_samples):
        if time.time() > deadline - 30:
            break
        params = []
        for lo, hi in param_ranges:
            params.append(np.random.uniform(lo, hi))
        
        # Ensure r_inner < r_outer
        for r in range(n_rings):
            if params[r*3] > params[r*3+1]:
                params[r*3], params[r*3+1] = params[r*3+1], params[r*3]
        
        orbit_vals = params_to_orbits(params)
        score = fast_evaluate(orbit_vals)
        
        if score > best_score:
            best_score = score
            best_orbit_vals = orbit_vals.copy()
    
    # Phase 2: Local optimization from best
    if best_orbit_vals is not None:
        current = best_orbit_vals.copy()
        current_score = best_score
        
        # Bit-flip local search
        step_size = 0.05
        n_local_steps = 2000
        
        for step in range(n_local_steps):
            if time.time() > deadline - 10:
                break
            
            # Pick 1-3 random orbits to flip
            n_flips = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            indices = np.random.choice(n_orbits, size=n_flips, replace=False)
            
            old_vals = current[indices].copy()
            current[indices] = 1.0 - current[indices]
            
            new_score = fast_evaluate(current)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_orbit_vals = current.copy()
            else:
                current[indices] = old_vals
    
    # Phase 3: Refinement with finer PWE
    if best_orbit_vals is not None and time.time() < deadline - 20:
        # Use more plane waves for refinement
        N_pw_ref = 9
        Gx_r = np.arange(-N_pw_ref//2, N_pw_ref//2+1)
        Gy_r = np.arange(-N_pw_ref//2, N_pw_ref//2+1)
        G_r = np.array([(gx, gy) for gx in Gx_r for gy in Gy_r])
        N_r = len(G_r)
        
        def eval_refined(orbit_vals):
            grid = orbit_to_grid(orbit_vals)
            eps_grid = eps_low + (eps_high - eps_low) * grid
            eps_fft = np.fft.fft2(eps_grid) / (size * size)
            
            g_max4 = 0.0
            g_min5 = float('inf')
            
            for kx, ky in k_points:
                M = np.zeros((N_r, N_r), dtype=float)
                for i in range(N_r):
                    gx, gy = G_r[i]
                    kx_i, ky_i = kx + gx, ky + gy
                    ks = kx_i**2 + ky_i**2
                    for j in range(N_r):
                        gx_j, gy_j = G_r[j]
                        kx_j, ky_j = kx + gx_j, ky + gy_j
                        dot = kx_i*kx_j + ky_i*ky_j
                        dgx, dgy = int(gx-gx_j), int(gy-gy_j)
                        ei = eps_fft[dgx % size, dgy % size].real
                        if ei < 0.1: ei = 0.1
                        M[i,j] = ks - dot/ei if i==j else -dot/ei
                try:
                    ev = np.linalg.eigvalsh(M)
                except:
                    return -1.0
                if len(ev) < 5: return -1.0
                g_max4 = max(g_max4, np.sqrt(max(ev[3],0)))
                g_min5 = min(g_min5, np.sqrt(max(ev[4],0)))
            
            if g_min5 == float('inf'): return -1.0
            mid = (g_min5+g_max4)/2
            return (g_min5-g_max4)/mid if mid > 1e-10 else 0.0
        
        # Refine with refined evaluator
        current = best_orbit_vals.copy()
        current_score = eval_refined(current)
        
        for step in range(1500):
            if time.time() > deadline - 5:
                break
            n_flips = 1
            idx = np.random.randint(n_orbits)
            old = current[idx]
            current[idx] = 1.0 - current[idx]
            ns = eval_refined(current)
            if ns > current_score:
                current_score = ns
                if current_score > best_score:
                    best_score = current_score
                    best_orbit_vals = current.copy()
            else:
                current[idx] = old
    
    if best_orbit_vals is None:
        # Fallback: simple central disk
        orbit_vals = np.zeros(n_orbits)
        cx, cy = (size-1)/2, (size-1)/2
        for k, orbit in enumerate(orbits):
            i0, j0 = list(orbit)[0]
            d = np.sqrt((i0-cx)**2 + (j0-cy)**2)
            orbit_vals[k] = 1.0 if d < size*0.2 else 0.0
        best_orbit_vals = orbit_vals
    
    best_grid = orbit_to_grid(best_orbit_vals)
    best_grid = np.round(best_grid).astype(int)
    
    # Enforce symmetry
    for k, orbit in enumerate(orbits):
        val = best_grid[list(orbit)[0][0], list(orbit)[0][1]]
        for (i, j) in orbit:
            best_grid[i, j] = val
    
    return best_grid.tolist()
