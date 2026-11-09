"""Spectral sinusoid fit + safeguarded secant/Newton hybrid for oscillatory roots."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 143.0          # safety margin before 150 s hard kill
    TOL = 1e-12               # aim well below 1e-8
    HARD_TOL = 1e-8           # contract tolerance
    MAX_CALLS = 450           # Hard limit on calls to ensure safety

    call_count = 0
    best_x = x0
    best_val = float('inf')
    eval_cache = {}

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    def safe_f(x):
        nonlocal call_count, best_x, best_val
        if call_count >= MAX_CALLS:
            return float('nan')
        try:
            val = f(x)
            call_count += 1
            if val != val or not math.isfinite(val):
                return float('nan')
            av = abs(val)
            if av < best_val:
                best_val = av
                best_x = x
            return val
        except Exception:
            call_count += 1
            return float('nan')

    def get_f(x):
        if x in eval_cache:
            return eval_cache[x]
        val = safe_f(x)
        eval_cache[x] = val
        return val

    # Evaluate start
    f0 = get_f(x0)
    if f0 != float('nan') and abs(f0) <= HARD_TOL:
        return x0

    # ------------------------------------------------------------------
    # 1) Try spectral approach: fit a sinusoid to local samples
    # ------------------------------------------------------------------
    # Sample at several points around x0 to estimate frequency and phase
    # We'll try a few scales and see if we can fit A*cos(w*(x-x0)+phi)+C
    
    def try_spectral_fit(scale, n_points=9):
        """Fit a sinusoid to n_points samples around x0 with given scale.
        Returns (predicted_root, confidence) or (None, 0)."""
        if not elapsed_ok() or call_count >= MAX_CALLS - 30:
            return None, 0.0
        
        # Sample points: x0 + k * scale / (n_points-1) for k in 0..n_points-1
        step = scale / (n_points - 1)
        xs = []
        fs = []
        for k in range(n_points):
            x = x0 + (k - (n_points - 1) / 2.0) * step
            fx = get_f(x)
            if fx == float('nan'):
                return None, 0.0
            xs.append(x)
            fs.append(fx)
        
        # Fit A*cos(w*x + phi) + C = A*cos(w*x)*cos(phi) - A*sin(w*x)*sin(phi) + C
        # Let u = A*cos(phi), v = -A*sin(phi)
        # f(x) ≈ u*cos(w*x) + v*sin(w*x) + C
        # For a given w, this is linear in u, v, C.
        # We need to search over w.
        
        # Estimate w from zero crossings in the samples
        # Count sign changes
        sign_changes = 0
        for i in range(n_points - 1):
            if fs[i] * fs[i+1] < 0:
                sign_changes += 1
        
        if sign_changes == 0:
            return None, 0.0
            
        # Estimated frequency: if we see s sign changes over range R,
        # the period is roughly R/s, so w ≈ 2*pi*s/R
        R = 2 * scale
        # More accurate: use the fact that each sign change is half a period
        # s sign changes means s/2 periods, so period = R/(s/2) = 2R/s
        # w = 2*pi / period = pi*s/R
        w_est = math.pi * sign_changes / R
        
        # Now do a more careful fit: try a few w values around w_est
        best_w = w_est
        best_residual = float('inf')
        
        for w_try in [w_est * 0.8, w_est * 0.9, w_est, w_est * 1.1, w_est * 1.2]:
            if w_try <= 0:
                continue
            # Build linear system: for each point, u*cos(w*x) + v*sin(w*x) + C = f(x)
            # Use least squares with 3 unknowns (u, v, C)
            # Normal equations: M^T M theta = M^T y
            # M is n_points x 3
            S_cc = S_cs = S_cs2 = S_ss = S_c = S_s = 0.0
            S_cf = S_sf = S_f = 0.0
            for i in range(n_points):
                c = math.cos(w_try * xs[i])
                s = math.sin(w_try * xs[i])
                S_cc += c * c
                S_cs += c * s
                S_ss += s * s
                S_c += c
                S_s += s
                S_cf += c * fs[i]
                S_sf += s * fs[i]
                S_f += fs[i]
            
            # Build 3x3 matrix and 3x1 vector
            # [S_cc, S_cs, S_c]   [u]   [S_cf]
            # [S_cs, S_ss, S_s] * [v] = [S_sf]
            # [S_c,  S_s,  n  ]   [C]   [S_f ]
            
            a11 = S_cc; a12 = S_cs; a13 = S_c
            a21 = S_cs; a22 = S_ss; a23 = S_s
            a31 = S_c;  a32 = S_s;  a33 = float(n_points)
            b1 = S_cf; b2 = S_sf; b3 = S_f
            
            # Solve using Cramer's rule or Gaussian elimination
            # Determinant
            det = (a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31))
            if abs(det) < 1e-15:
                continue
            
            # Solve
            u = ((b1*(a22*a33 - a23*a32) - a12*(b2*a33 - a23*b3) + a13*(b2*a32 - a22*b3)) / det)
            v = ((a11*(b2*a33 - a23*b3) - b1*(a21*a33 - a23*a31) + a13*(a21*b3 - b2*a31)) / det)
            C = ((a11*(a22*b3 - b2*a32) - a12*(a22*b3 - b2*a32) + b1*(a21*a32 - a22*a31)) / det)
            
            # Actually let me just compute residual directly
            residual = 0.0
            for i in range(n_points):
                pred = u * math.cos(w_try * xs[i]) + v * math.sin(w_try * xs[i]) + C
                residual += (pred - fs[i]) ** 2
            
            if residual < best_residual:
                best_residual = residual
                best_w = w_try
                best_u, best_v, best_C = u, v, C
        
        if best_residual > 0.1 * sum(fx*fx for fx in fs):
            return None, 0.0
        
        # Now find the nearest zero of u*cos(w*x) + v*sin(w*x) + C = 0
        # This is A*cos(w*x + phi) + C = 0 where A = sqrt(u^2+v^2), 
        # phi = atan2(-v, u)
        A = math.sqrt(best_u**2 + best_v**2)
        if A < 1e-15:
            return None, 0.0
        phi = math.atan2(-best_v, best_u)
        
        # Solve A*cos(w*x + phi) + C = 0
        # cos(w*x + phi) = -C/A
        ratio = -best_C / A
        if abs(ratio) > 1.0:
            return None, 0.0
        
        # w*x + phi = ±arccos(ratio) + 2*pi*k
        base = math.acos(ratio)
        
        # Find k such that x is closest to x0
        # x = (±base - phi + 2*pi*k) / w
        candidates = []
        for sign in [1, -1]:
            for k in range(-3, 4):
                x_cand = (sign * base - phi + 2 * math.pi * k) / best_w
                candidates.append(x_cand)
        
        # Pick the candidate closest to x0
        x_pred = min(candidates, key=lambda x: abs(x - x0))
        
        # Verify: check f at predicted root
        f_pred = get_f(x_pred)
        if f_pred == float('nan'):
            return None, 0.0
        if abs(f_pred) <= 1e-6:
            return x_pred, 1.0 - min(1.0, abs(f_pred) / 1e-6)
        
        # Even if not great, return it as a starting point for refinement
        return x_pred, 0.5

    # Try spectral fit at several scales
    spectral_candidate = None
    for scale in [0.1, 0.5, 1.0, 5.0]:
        if not elapsed_ok() or call_count >= MAX_CALLS - 20:
            break
        pred, conf = try_spectral_fit(scale)
        if pred is not None and conf > 0.3:
            spectral_candidate = pred
            break

    # ------------------------------------------------------------------
    # 2) If spectral gave a candidate, refine it with safeguarded secant
    #    Otherwise, find a bracket first
    # ------------------------------------------------------------------
    
    if spectral_candidate is not None:
        # Refine around spectral_candidate
        x1 = spectral_candidate
        f1 = get_f(x1)
        if f1 == float('nan'):
            x1 = x0
            f1 = f0
        
        # Find a second point for secant
        h = max(1e-6, 0.01 * max(1.0, abs(x1)))
        x2 = x1 + h
        f2 = get_f(x2)
        
        if f2 == float('nan'):
            x2 = x1 - h
            f2 = get_f(x2)
        
        # Safeguarded secant/Newton hybrid
        if f1 != float('nan') and f2 != float('nan'):
            # Ensure |f1| <= |f2|
            if abs(f1) > abs(f2):
                x1, x2 = x2, x1
                f1, f2 = f2, f1
            
            for _ in range(100):
                if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                    break
                if abs(f1) <= TOL:
                    return x1
                if abs(f2) <= TOL:
                    return x2
                
                # Secant step
                if f2 != f1:
                    x_new = (x1 * f2 - x2 * f1) / (f2 - f1)
                else:
                    x_new = 0.5 * (x1 + x2)
                
                # Safeguard: if x_new is outside [x1, x2] or not finite, bisect
                lo = min(x1, x2)
                hi = max(x1, x2)
                if not math.isfinite(x_new) or x_new <= lo or x_new >= hi:
                    x_new = 0.5 * (x1 + x2)
                
                f_new = get_f(x_new)
                if f_new == float('nan'):
                    break
                
                if abs(f_new) <= TOL:
                    return x_new
                
                # Replace the point with larger |f|
                if abs(f_new) < abs(f1):
                    # New point is better than x1; but we need a bracket
                    # Actually, for secant we don't strictly need a bracket
                    # But for safeguarding, let's keep the bracket
                    if f1 * f_new < 0:
                        # Bracket found: root in [x_new, x1]... wait
                        # If f1 and f_new have opposite signs, root is between them
                        x2, f2 = x1, f1
                    else:
                        x2, f2 = x_new, f_new
                    x1, f1 = x_new, f_new
                else:
                    if f2 * f_new < 0:
                        x1, f1 = x2, f2
                    else:
                        x1, f1 = x_new, f_new
                    x2, f2 = x_new, f_new
                
                # Re-order: keep x1 as the one with smaller |f|
                if abs(f1) > abs(f2):
                    x1, x2 = x2, x1
                    f1, f2 = f2, f1
                
                if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
                    break
    else:
        # No spectral candidate; find a bracket via local scan
        a = b = fa = fb = None
        
        # Local fine scan
        for scale in [0.01, 0.1, 1.0, 10.0]:
            if not elapsed_ok() or call_count >= MAX_CALLS - 20:
                break
            step = scale / 10.0
            prev_x = prev_f = None
            for k in range(-10, 11):
                if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                    break
                x_t = x0 + k * step
                f_t = get_f(x_t)
                if f_t == float('nan'):
                    continue
                if abs(f_t) <= HARD_TOL:
                    return x_t
                if prev_f is not None and prev_f * f_t < 0:
                    a, b = prev_x, x_t
                    fa, fb = prev_f, f_t
                    break
                prev_x, prev_f = x_t, f_t
            if a is not None:
                break
        
        if a is None:
            # Geometric expansion
            h = max(1e-4, abs(x0) * 1e-4)
            last_pos_x, last_pos_f = x0, f0
            last_neg_x, last_neg_f = x0, f0
            for i in range(1, 30):
                if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                    break
                x_p = x0 + i * h
                f_p = get_f(x_p)
                if f_p != float('nan') and last_pos_f != float('nan'):
                    if f_p * last_pos_f < 0:
                        a, b = last_pos_x, x_p
                        fa, fb = last_pos_f, f_p
                        break
                    if abs(f_p) <= HARD_TOL:
                        return x_p
                if f_p != float('nan'):
                    last_pos_x, last_pos_f = x_p, f_p
                
                x_n = x0 - i * h
                f_n = get_f(x_n)
                if f_n != float('nan') and last_neg_f != float('nan'):
                    if f_n * last_neg_f < 0:
                        a, b = x_n, last_neg_x
                        fa, fb = f_n, last_neg_f
                        break
                    if abs(f_n) <= HARD_TOL:
                        return x_n
                if f_n != float('nan'):
                    last_neg_x, last_neg_f = x_n, f_n
                h *= 2.0
        
        # Refine bracket with Illinois
        if a is not None:
            if a > b:
                a, b = b, a
                fa, fb = fb, fa
            x1, f1 = a, fa
            x2, f2 = b, fb
            if abs(f1) > abs(f2):
                x1, x2 = x2, x1
                f1, f2 = f2, f1
            
            for _ in range(200):
                if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                    break
                if abs(f1) <= TOL:
                    return x1
                if abs(f2) <= TOL:
                    return x2
                
                if f2 != f1:
                    x_new = (x1 * f2 - x2 * f1) / (f2 - f1)
                else:
                    x_new = 0.5 * (x1 + x2)
                
                lo = min(x1, x2)
                hi = max(x1, x2)
                if not math.isfinite(x_new) or x_new <= lo or x_new >= hi:
                    x_new = 0.5 * (x1 + x2)
                
                f_new = get_f(x_new)
                if f_new == float('nan'):
                    break
                if abs(f_new) <= TOL:
                    return x_new
                
                if f_new * f1 < 0:
                    if abs(f2) > abs(f1):
                        f2 *= 0.5
                    x2, f2 = x_new, f_new
                else:
                    if abs(f1) > abs(f2):
                        f1 *= 0.5
                    x1, f1 = x_new, f_new
                
                if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
                    break

    # ------------------------------------------------------------------
    # 3) Return best candidate
    # ------------------------------------------------------------------
    if math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END
