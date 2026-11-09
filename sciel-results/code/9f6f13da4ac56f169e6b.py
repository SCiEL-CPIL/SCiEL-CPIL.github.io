"""DFT spectral root-finding with IQI/secant refinement for oscillatory equations."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 143.0
    TOL = 1e-12
    HARD_TOL = 1e-8
    MAX_REFINE_ITERS = 100

    call_count = 0
    MAX_CALLS = 450

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    def safe_f(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return float('nan')
        try:
            val = f(x)
            call_count += 1
            if val != val or not math.isfinite(val):
                return float('nan')
            return val
        except Exception:
            call_count += 1
            return float('nan')

    eval_cache = {}

    def get_f(x):
        # Use a small tolerance for cache key to handle floating point
        key = round(x, 15)
        if key in eval_cache:
            return eval_cache[key]
        val = safe_f(x)
        eval_cache[key] = val
        return val

    def track_best(x, fx):
        nonlocal best_x, best_val
        if fx != float('nan') and math.isfinite(fx):
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    # Evaluate start
    f0 = get_f(x0)
    best_x = x0
    best_val = abs(f0) if f0 != float('nan') else float('inf')
    track_best(x0, f0)

    if f0 != float('nan') and abs(f0) <= HARD_TOL:
        return x0

    # ------------------------------------------------------------------
    # Phase 1: DFT-based frequency estimation
    # Sample N points around x0 to estimate dominant frequency and phase
    # ------------------------------------------------------------------
    N = 16  # Number of sample points for DFT
    # Try multiple scales to find the right frequency range
    candidate_roots = []
    
    for scale in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        if not elapsed_ok() or call_count >= MAX_CALLS - 30:
            break
        
        # Sample N equally spaced points centered around x0
        step = 2.0 * scale / (N - 1)
        samples_x = []
        samples_f = []
        
        for i in range(N):
            if not elapsed_ok() or call_count >= MAX_CALLS - 20:
                break
            xi = x0 - scale + i * step
            fi = get_f(xi)
            if fi != float('nan') and math.isfinite(fi):
                samples_x.append(xi)
                samples_f.append(fi)
                track_best(xi, fi)
        
        if len(samples_f) < 4:
            continue
        
        # Check if any sample is already a root
        for xi, fi in zip(samples_x, samples_f):
            if abs(fi) <= HARD_TOL:
                return xi
        
        # Check for sign changes in samples
        for i in range(len(samples_f) - 1):
            if samples_f[i] * samples_f[i+1] < 0:
                candidate_roots.append((samples_x[i], samples_f[i], samples_x[i+1], samples_f[i+1]))
        
        # DFT to estimate dominant frequency
        # Compute mean
        mean_f = sum(samples_f) / len(samples_f)
        # Remove mean
        centered = [fi - mean_f for fi in samples_f]
        
        # Compute DFT for first few frequency bins
        # We want to find the dominant oscillation frequency
        best_freq = None
        best_mag = 0
        
        for k in range(1, N // 2 + 1):
            # DFT bin k
            re_sum = 0.0
            im_sum = 0.0
            for n in range(len(centered)):
                angle = 2.0 * math.pi * k * n / len(centered)
                re_sum += centered[n] * math.cos(angle)
                im_sum -= centered[n] * math.sin(angle)
            mag = math.sqrt(re_sum * re_sum + im_sum * im_sum)
            if mag > best_mag:
                best_mag = mag
                best_freq = k
        
        if best_freq is not None and best_mag > 0.1 * (sum(abs(c) for c in centered) / len(centered)):
            # Estimated frequency: k cycles over 2*scale range
            # Angular frequency omega = 2*pi*k / (2*scale) = pi*k/scale
            omega = math.pi * best_freq / scale
            
            # Estimate phase from DFT coefficients
            re_sum = 0.0
            im_sum = 0.0
            for n in range(len(centered)):
                angle = 2.0 * math.pi * best_freq * n / len(centered)
                re_sum += centered[n] * math.cos(angle)
                im_sum -= centered[n] * math.sin(angle)
            
            # Phase: f(x) ≈ A * cos(omega * (x - x0) + phi)
            # At x = x0 - scale + n*step, the phase is omega*(x0 - scale + n*step - x0) + phi
            # = omega*(-scale + n*step) + phi
            # The DFT gives us: sum f_n * exp(-i*2*pi*k*n/N) = A*N/2 * exp(-i*phi) * exp(-i*omega*scale)
            # So the phase of the DFT coefficient = -phi - omega*scale (mod 2pi)
            
            dft_phase = math.atan2(im_sum, re_sum)
            # phi = -dft_phase - omega*scale (mod 2pi)
            phi = -dft_phase - omega * scale
            phi = phi % (2.0 * math.pi)
            
            # Amplitude
            A = 2.0 * best_mag / len(centered)
            
            if A > 1e-10:
                # Find zero crossings of A*cos(omega*(x-x0)+phi) + mean_f = 0
                # cos(theta) = -mean_f / A
                if abs(mean_f / A) <= 1.0:
                    theta_offset = math.acos(-mean_f / A)
                    # Zeros when omega*(x-x0) + phi = ±theta_offset + 2*pi*m
                    # x = x0 + (±theta_offset + 2*pi*m - phi) / omega
                    
                    for m in range(-3, 4):
                        for sign in [1, -1]:
                            theta = sign * theta_offset + 2.0 * math.pi * m - phi
                            x_candidate = x0 + theta / omega
                            # Only consider candidates reasonably close to x0
                            if abs(x_candidate - x0) <= 10.0 * scale:
                                candidate_roots.append((x_candidate, None, x_candidate, None))
                    
                    # If we found good candidates, we can stop searching scales
                    if len(candidate_roots) > 0:
                        break
    
    # ------------------------------------------------------------------
    # Phase 2: Refine best candidate using IQI (Illinois-Quadratic-Interpolation)
    # ------------------------------------------------------------------
    
    # Sort candidates by distance to x0, prefer closer ones
    # For sign-change brackets, we have (a, fa, b, fb)
    # For DFT predictions, we have (x, None, x, None) - need to verify
    
    # First, try to use sign-change brackets if available
    bracket_candidates = [(a, fa, b, fb) for (a, fa, b, fb) in candidate_roots if fa is not None and fb is not None]
    dft_candidates = [(x, None, x, None) for (a, fa, b, fb) in candidate_roots if fa is None]
    
    # Sort DFT candidates by distance to x0
    dft_candidates.sort(key=lambda t: abs(t[0] - x0))
    
    # Try each candidate
    for candidate in bracket_candidates + dft_candidates[:5]:
        if not elapsed_ok() or call_count >= MAX_CALLS - 10:
            break
        
        if candidate[1] is not None:
            # Sign-change bracket: use IQI refinement
            a, fa, b, fb = candidate
            if a > b:
                a, b = b, a
                fa, fb = fb, fa
            
            # Ensure |fa| < |fb|
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            
            x1, f1 = a, fa
            x2, f2 = b, fb
            
            for _ in range(MAX_REFINE_ITERS):
                if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                    break
                
                if abs(f1) <= TOL:
                    return x1
                if abs(f2) <= TOL:
                    return x2
                
                # IQI: Use quadratic interpolation if possible, else bisection
                # Check if we have three points for quadratic
                # For now, use secant (two-point) as primary, bisection as fallback
                
                if f2 - f1 != 0:
                    x_new = (x1 * f2 - x2 * f1) / (f2 - f1)
                else:
                    x_new = 0.5 * (x1 + x2)
                
                # Safeguard
                if not math.isfinite(x_new) or x_new <= x1 or x_new >= x2:
                    x_new = 0.5 * (x1 + x2)
                
                f_new = get_f(x_new)
                
                if f_new != float('nan'):
                    if abs(f_new) <= TOL:
                        return x_new
                    track_best(x_new, f_new)
                    
                    if f_new * f1 < 0:
                        # Replace x2
                        if abs(f2) > abs(f1):
                            f2 *= 0.5
                        x2, f2 = x_new, f_new
                    else:
                        # Replace x1
                        if abs(f1) > abs(f2):
                            f1 *= 0.5
                        x1, f1 = x_new, f_new
                    
                    if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
                        return x1 if abs(f1) < abs(f2) else x2
        else:
            # DFT prediction: verify and refine
            x_pred = candidate[0]
            f_pred = get_f(x_pred)
            
            if f_pred != float('nan') and abs(f_pred) <= HARD_TOL:
                return x_pred
            
            if f_pred != float('nan'):
                # Use secant with x0 and x_pred
                if abs(f0 - f_pred) > 1e-15:
                    x_sec = (x0 * f_pred - x_pred * f0) / (f_pred - f0)
                    if math.isfinite(x_sec) and abs(x_sec - x0) < 100.0:
                        # Refine around x_sec
                        x1, f1 = x0, f0
                        x2, f2 = x_sec, get_f(x_sec) if abs(x_sec - x_pred) > 1e-12 else f_pred
                        
                        if f1 != f2 and math.isfinite(f2):
                            for _ in range(50):
                                if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                                    break
                                if abs(f1) <= TOL:
                                    return x1
                                if abs(f2) <= TOL:
                                    return x2
                                
                                if f2 - f1 != 0:
                                    x_new = (x1 * f2 - x2 * f1) / (f2 - f1)
                                else:
                                    x_new = 0.5 * (x1 + x2)
                                
                                if not math.isfinite(x_new):
                                    break
                                
                                f_new = get_f(x_new)
                                if f_new != float('nan'):
                                    if abs(f_new) <= TOL:
                                        return x_new
                                    track_best(x_new, f_new)
                                    if f_new * f1 < 0:
                                        x2, f2 = x_new, f_new
                                    else:
                                        x1, f1 = x_new, f_new
                                    if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
                                        break
                else:
                    # f0 ≈ f_pred, try bisection between them
                    mid = 0.5 * (x0 + x_pred)
                    f_mid = get_f(mid)
                    if f_mid != float('nan') and abs(f_mid) <= TOL:
                        return mid
                    if f_mid != float('nan'):
                        # Determine bracket
                        if f_mid * f0 < 0:
                            x1, f1, x2, f2 = x0, f0, mid, f_mid
                        elif f_mid * f_pred < 0:
                            x1, f1, x2, f2 = mid, f_mid, x_pred, f_pred
                        else:
                            x1, f1, x2, f2 = x0, f0, x_pred, f_pred
                        
                        for _ in range(50):
                            if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                                break
                            if abs(f1) <= TOL:
                                return x1
                            if abs(f2) <= TOL:
                                return x2
                            
                            x_new = 0.5 * (x1 + x2)
                            f_new = get_f(x_new)
                            if f_new != float('nan'):
                                if abs(f_new) <= TOL:
                                    return x_new
                                track_best(x_new, f_new)
                                if f_new * f1 < 0:
                                    x2, f2 = x_new, f_new
                                else:
                                    x1, f1 = x_new, f_new
                                if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
                                    break

    # ------------------------------------------------------------------
    # Phase 3: Fallback - local scan if DFT didn't find anything
    # ------------------------------------------------------------------
    if best_val > HARD_TOL:
        # Dense local scan
        for scale in [0.01, 0.1, 1.0, 10.0]:
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break
            step = scale / 10.0
            prev_x, prev_f = None, None
            for k in range(-10, 11):
                if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                    break
                x_test = x0 + k * step
                f_test = get_f(x_test)
                if f_test != float('nan'):
                    if abs(f_test) <= HARD_TOL:
                        return x_test
                    track_best(x_test, f_test)
                    if prev_x is not None and prev_f is not None and prev_f * f_test < 0:
                        # Found bracket, refine
                        a, b = prev_x, x_test
                        fa, fb = prev_f, f_test
                        if a > b:
                            a, b = b, a
                            fa, fb = fb, fa
                        if abs(fa) > abs(fb):
                            a, b = b, a
                            fa, fb = fb, fa
                        x1, f1, x2, f2 = a, fa, b, fb
                        for _ in range(50):
                            if not elapsed_ok() or call_count >= MAX_CALLS - 3:
                                break
                            if abs(f1) <= TOL:
                                return x1
                            if abs(f2) <= TOL:
                                return x2
                            x_new = 0.5 * (x1 + x2)
                            f_new = get_f(x_new)
                            if f_new != float('nan'):
                                if abs(f_new) <= TOL:
                                    return x_new
                                track_best(x_new, f_new)
                                if f_new * f1 < 0:
                                    x2, f2 = x_new, f_new
                                else:
                                    x1, f1 = x_new, f_new
                                if abs(x2 - x1) < 1e-14 * max(1.0, abs(x1)):
                                    break
                        return x1 if abs(f1) < abs(f2) else x2
                    prev_x, prev_f = x_test, f_test

    # Return best
    if best_x == best_x and math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END
