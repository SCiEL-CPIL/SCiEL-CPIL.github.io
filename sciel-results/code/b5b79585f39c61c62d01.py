import numpy as np
import scipy.optimize as so
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    deadline = time.time() + 600.0
    
    # Exact volume computation using critical-point integration
    def compute_volume(X, Y):
        # Build typed endpoint lists for X and Y
        # Each endpoint: (A, B, is_right) where value at c is A + B*c
        eps_x = []
        for i in range(N):
            eps_x.append((X[i], i/N, 0))
            eps_x.append((X[i] + delta, i/N, 1))
        
        eps_y = []
        for j in range(N):
            eps_y.append((Y[j], j/N, 0))
            eps_y.append((Y[j] + delta, j/N, 1))
        
        # Find critical points where endpoint order changes
        def find_crit(eps):
            crit = {0.0, 1.0}
            for i in range(len(eps)):
                A1, B1, _ = eps[i]
                for j in range(i+1, len(eps)):
                    A2, B2, _ = eps[j]
                    if abs(B1 - B2) > 1e-15:
                        c = (A2 - A1) / (B1 - B2)
                        if 0.0 < c < 1.0:
                            crit.add(c)
            return crit
        
        crit_x = find_crit(eps_x)
        crit_y = find_crit(eps_y)
        all_crit = sorted(crit_x.union(crit_y))
        
        # Compute union length at a given c
        def union_length(eps, c):
            pts = []
            for A, B, is_r in eps:
                pts.append((A + B*c, is_r))
            pts.sort()
            total = 0.0
            count = 0
            start_pos = 0.0
            for val, is_r in pts:
                if is_r == 0:
                    if count == 0:
                        start_pos = val
                    count += 1
                else:
                    count -= 1
                    if count == 0:
                        total += val - start_pos
            return total
        
        # Integrate Lx(c)*Ly(c) over [0,1] using piecewise quadratic interpolation
        total_vol = 0.0
        for k in range(len(all_crit) - 1):
            c0 = all_crit[k]
            c1 = all_crit[k+1]
            h = c1 - c0
            if h < 1e-12:
                continue
            
            Lx0 = union_length(eps_x, c0)
            Lx1 = union_length(eps_x, c1)
            Ly0 = union_length(eps_y, c0)
            Ly1 = union_length(eps_y, c1)
            
            dx = Lx1 - Lx0
            dy = Ly1 - Ly0
            
            # Integral of (Lx0 + u*dx)(Ly0 + u*dy) du from 0 to 1, times h
            term = Lx0 * Ly0
            term += 0.5 * (Lx0 * dy + Ly0 * dx)
            term += (1.0/3.0) * dx * dy
            
            total_vol += h * term
        
        return total_vol
    
    # Objective: minimize union volume
    def objective(params):
        X = params[:N]
        Y = params[N:]
        return compute_volume(X, Y)
    
    # Initial guess: use the Keich construction as starting point
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    X0 = keich.copy()
    Y0 = keich.copy()
    p0 = np.concatenate([X0, Y0])
    
    best_p = p0.copy()
    best_v = objective(p0)
    
    # Multi-start L-BFGS-B optimization (faster for smooth objectives)
    rng = np.random.default_rng(42)
    
    starts = [p0.copy()]
    # Try several random perturbations
    for attempt in range(15):
        if time.time() > deadline - 120:
            break
        perturbation = rng.normal(0, 0.2, size=2*N)
        p_try = p0 + perturbation
        starts.append(p_try)
    
    # Structured starts
    sym_x = np.linspace(-delta/2, -(N-1)*delta/2, N)
    sym_y = np.linspace(-delta/2, -(N-1)*delta/2, N)
    starts.append(np.concatenate([sym_x, sym_y]))
    
    anti_x = np.linspace(0, (N-1)*delta, N)
    anti_y = np.linspace(0, (N-1)*delta, N)
    starts.append(np.concatenate([anti_x, anti_y]))
    
    for p_start in starts:
        if time.time() > deadline - 120:
            break
        
        try:
            # L-BFGS-B is generally much faster than Powell for this dimensionality
            # if the objective is sufficiently smooth.
            res = so.minimize(objective, p_start, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Final refinement with higher precision
    if time.time() < deadline - 60:
        try:
            res = so.minimize(objective, best_p, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
            
    # Optional: Nelder-Mead polish for any remaining local issues
    if time.time() < deadline - 30:
        try:
            res = so.minimize(objective, best_p, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    X_opt = best_p[:N]
    Y_opt = best_p[N:]
    
    # Construct final positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
