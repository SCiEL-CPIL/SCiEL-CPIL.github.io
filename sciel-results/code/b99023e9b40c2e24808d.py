import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    
    # Keich binary construction for initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    X0 = keich.copy()
    Y0 = keich.copy()
    
    # Pre-compute the linear coefficients for each tube in x and y
    # Tube (i,j): x_start = X[i] + c*i/N, y_start = Y[j] + c*j/N
    i_coeff = np.arange(N, dtype=np.float64) / N  # i/N for i=0..7
    j_coeff = np.arange(N, dtype=np.float64) / N  # j/N for j=0..7
    
    def _union_length_1d(A, B, c):
        """Compute union length of intervals [A[i] + B[i]*c, A[i] + B[i]*c + delta] for i=0..N-1."""
        starts = A + B * c
        ends = starts + delta
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        total = 0.0
        cur_s = s[0]
        cur_e = e[0]
        for k in range(1, N):
            if s[k] > cur_e:
                total += cur_e - cur_s
                cur_s = s[k]
                cur_e = e[k]
            else:
                if e[k] > cur_e:
                    cur_e = e[k]
        total += cur_e - cur_s
        return total
    
    def compute_volume(X, Y):
        """Exact volume via critical c-values and trapezoidal integration of product of 1D union lengths."""
        # Find critical c-values where the ordering of x-interval endpoints changes
        # x-endpoints: X[i] + (i/N)*c for i=0..N-1, and X[i] + (i/N)*c + delta
        # Two endpoints a + b*c and a' + b'*c cross when c = (a'-a)/(b-b') if b != b'
        
        # For x: endpoints are X[i] + (i/N)*c (left) and X[i] + (i/N)*c + delta (right)
        # Critical c values come from crossings of left endpoints and right endpoints
        # Left endpoints: L_i(c) = X[i] + (i/N)*c
        # Right endpoints: R_i(c) = X[i] + (i/N)*c + delta
        
        # Crossings between L_i and L_j: c = (X[j]-X[i]) / ((i/N)-(j/N)) = N*(X[j]-X[i])/(i-j)
        # Crossings between R_i and R_j: same as L_i and L_j (delta cancels)
        # Crossings between L_i and R_j: c = (X[j]+delta - X[i]) / ((i/N)-(j/N)) = N*(X[j]+delta-X[i])/(i-j)
        
        crit = {0.0, 1.0}
        
        # x-critical values
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = (i - j) / N
                if abs(denom) < 1e-15:
                    continue
                # L_i vs L_j crossing
                c1 = (X[j] - X[i]) / denom
                if 0.0 < c1 < 1.0:
                    crit.add(c1)
                # L_i vs R_j crossing
                c2 = (X[j] + delta - X[i]) / denom
                if 0.0 < c2 < 1.0:
                    crit.add(c2)
        
        # y-critical values (same structure with Y and j_coeff)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = (i - j) / N
                if abs(denom) < 1e-15:
                    continue
                c1 = (Y[j] - Y[i]) / denom
                if 0.0 < c1 < 1.0:
                    crit.add(c1)
                c2 = (Y[j] + delta - Y[i]) / denom
                if 0.0 < c2 < 1.0:
                    crit.add(c2)
        
        all_crit = sorted(crit)
        
        total_vol = 0.0
        for k in range(len(all_crit) - 1):
            c_s = all_crit[k]
            c_e = all_crit[k + 1]
            h = c_e - c_s
            if h < 1e-14:
                continue
            
            # At each endpoint, compute 1D union lengths
            Lx_s = _union_length_1d(X, i_coeff, c_s)
            Lx_e = _union_length_1d(X, i_coeff, c_e)
            Ly_s = _union_length_1d(Y, j_coeff, c_s)
            Ly_e = _union_length_1d(Y, j_coeff, c_e)
            
            # Within each interval, Lx and Ly are piecewise linear in c
            # So Lx*Ly is piecewise quadratic; integrate exactly
            dx = Lx_e - Lx_s
            dy = Ly_e - Ly_s
            
            # ∫ Lx*Ly dc ≈ h * [Lx_s*Ly_s + 0.5*(Lx_s*dy + Ly_s*dx) + (1/3)*dx*dy]
            # This is exact if Lx and Ly are linear on [c_s, c_e]
            term = Lx_s * Ly_s
            term += 0.5 * (Lx_s * dy + Ly_s * dx)
            term += (1.0 / 3.0) * dx * dy
            
            total_vol += h * term
        
        return total_vol
    
    def objective(params):
        X = params[:N]
        Y = params[N:]
        return compute_volume(X, Y)
    
    p0 = np.concatenate([X0, Y0])
    
    start_time = time.time()
    deadline = start_time + 570.0  # 30s safety margin
    
    best_p = p0.copy()
    best_v = objective(p0)
    
    # Multi-restart Powell
    rng_opt = np.random.default_rng(42)
    n_restarts = 15
    
    for r in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        
        if r == 0:
            p_init = p0.copy()
        else:
            # Perturb from best found so far with decreasing noise
            noise_scale = 0.15 / (1 + r * 0.3)
            p_init = best_p + rng_opt.normal(0, noise_scale, 2 * N)
        
        try:
            res = minimize(objective, p_init, method='Powell',
                          options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish
    if time.time() < deadline - 30.0:
        try:
            res2 = minimize(objective, best_p, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res2.fun < best_v:
                best_v = res2.fun
                best_p = res2.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with even tighter settings
    if time.time() < deadline - 10.0:
        try:
            res3 = minimize(objective, best_p, method='Powell',
                           options={'maxiter': 2000, 'xtol': 1e-14, 'ftol': 1e-14})
            if res3.fun < best_v:
                best_v = res3.fun
                best_p = res3.x.copy()
        except Exception:
            pass
    
    X_opt = best_p[:N]
    Y_opt = best_p[N:]
    
    # Build the 64 positions: row k has i=k//8, j=k%8
    positions = np.empty((64, 2), dtype=np.float64)
    for k in range(64):
        i = k // N
        j = k % N
        positions[k, 0] = X_opt[i]
        positions[k, 1] = Y_opt[j]
    
    return positions
