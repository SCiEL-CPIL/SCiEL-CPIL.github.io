# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Uses a secant-first hybrid with bracket fallback and a derivative-free
    Nelder-Mead polish phase for ill-conditioned cases.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    max_evals = 500
    evals_used = 0

    def check_time():
        return time.time() < deadline

    def safe_eval(x):
        nonlocal evals_used
        if not check_time():
            return None
        if evals_used >= max_evals:
            return None
        if not math.isfinite(x):
            return None
        try:
            val = f(x)
            evals_used += 1
            if not isinstance(val, (int, float)):
                return None
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        except Exception:
            evals_used += 1
            return None

    # Track best point seen so far
    best_x = float(x0)
    best_fx = safe_eval(best_x)

    if best_fx is None:
        for dx in [1e-8, -1e-8, 1e-4, -1e-4, 1.0, -1.0, 10.0, -10.0]:
            if not check_time():
                break
            trial_x = float(x0) + dx
            trial_fx = safe_eval(trial_x)
            if trial_fx is not None:
                if best_fx is None or abs(trial_fx) < abs(best_fx):
                    best_x = trial_x
                    best_fx = trial_fx
                if best_fx is not None and abs(best_fx) <= 1e-8:
                    return best_x
        if best_fx is None:
            return best_x

    if abs(best_fx) <= 1e-8:
        return best_x

    # ============================================================
    # Phase 1: Secant method
    # ============================================================
    x_prev = best_x
    fx_prev = best_fx
    x_curr = best_x + 1e-6 * max(1.0, abs(best_x))
    fx_curr = safe_eval(x_curr)
    if fx_curr is None:
        x_curr = best_x - 1e-6 * max(1.0, abs(best_x))
        fx_curr = safe_eval(x_curr)
    if fx_curr is None:
        for dx in [1e-4, -1e-4, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
            if not check_time():
                break
            x_curr = best_x + dx
            fx_curr = safe_eval(x_curr)
            if fx_curr is not None:
                break

    if fx_curr is not None:
        if abs(fx_curr) < abs(best_fx):
            best_x = x_curr
            best_fx = fx_curr
        if abs(best_fx) <= 1e-8:
            return best_x

        for _ in range(250):
            if not check_time() or evals_used >= max_evals - 2:
                break
            if abs(fx_curr) <= 1e-8:
                best_x = x_curr
                best_fx = fx_curr
                return x_curr
            if abs(fx_prev) <= 1e-8:
                best_x = x_prev
                best_fx = fx_prev
                return x_prev

            denom = fx_curr - fx_prev
            if abs(denom) < 1e-18:
                break

            x_new = x_curr - fx_curr * (x_curr - x_prev) / denom

            if not math.isfinite(x_new):
                break

            max_step = 10.0 * max(1.0, abs(x_curr))
            if abs(x_new - x_curr) > max_step:
                x_new = x_curr - math.copysign(max_step, x_new - x_curr)

            fx_new = safe_eval(x_new)
            if fx_new is None:
                break

            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
                if abs(best_fx) <= 1e-8:
                    return x_new

            x_prev, fx_prev = x_curr, fx_curr
            x_curr, fx_curr = x_new, fx_new

            if abs(x_curr - x_prev) < 1e-14 * max(1.0, abs(x_curr)):
                break

    # ============================================================
    # Phase 2: Bracket search + bisection
    # ============================================================
    if abs(best_fx) > 1e-8 and check_time() and evals_used < max_evals - 10:
        bracket_found = False
        a, b = None, None
        fa, fb = None, None

        base_scale = max(1.0, abs(best_x))
        for scale in [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]:
            if not check_time() or evals_used >= max_evals - 5:
                break
            delta = scale * base_scale
            xa = best_x - delta
            xb = best_x + delta

            fa = safe_eval(xa)
            if fa is None:
                continue
            fb = safe_eval(xb)
            if fb is None:
                continue

            if fa * fb < 0:
                a, b = xa, xb
                bracket_found = True
                break
            elif abs(fa) < abs(best_fx):
                best_x = xa
                best_fx = fa
                if abs(best_fx) <= 1e-8:
                    return best_x
            elif abs(fb) < abs(best_fx):
                best_x = xb
                best_fx = fb
                if abs(best_fx) <= 1e-8:
                    return best_x

        if not bracket_found and check_time() and evals_used < max_evals - 5:
            for scale in [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]:
                if not check_time() or evals_used >= max_evals - 5:
                    break
                delta = scale * base_scale
                xa = best_x
                fa = best_fx
                xb = best_x + delta
                fb = safe_eval(xb)
                if fb is not None and fa * fb < 0:
                    a, b = xa, xb
                    bracket_found = True
                    break
                xa = best_x - delta
                fa = safe_eval(xa)
                if fa is not None and fa * best_fx < 0:
                    a, b = xa, best_x
                    bracket_found = True
                    break

        if bracket_found and check_time() and evals_used < max_evals - 5:
            for _ in range(80):
                if not check_time() or evals_used >= max_evals - 2:
                    break
                mid = 0.5 * (a + b)
                fm = safe_eval(mid)
                if fm is None:
                    break
                if abs(fm) < abs(best_fx):
                    best_x = mid
                    best_fx = fm
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm < 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
                if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                    break

    # ============================================================
    # Phase 3: Nelder-Mead minimization of f(x)^2 as derivative-free polish
    # ============================================================
    if abs(best_fx) > 1e-8 and check_time() and evals_used < max_evals - 30:
        # Initialize simplex around best_x
        x_center = best_x
        h = max(1e-4 * max(1.0, abs(x_center)), 1e-6)
        
        # Simplex points: center, center+h, center-h
        simplex = [
            [x_center, best_fx * best_fx],
            [x_center + h, None],
            [x_center - h, None]
        ]
        
        # Evaluate remaining simplex points
        for i in range(1, 3):
            if not check_time() or evals_used >= max_evals - 2:
                break
            fi = safe_eval(simplex[i][0])
            if fi is None:
                simplex[i][1] = float('inf')
            else:
                simplex[i][1] = fi * fi
        
        # Sort by objective value
        simplex.sort(key=lambda p: p[1])
        
        # Nelder-Mead iterations
        for nm_iter in range(150):
            if not check_time() or evals_used >= max_evals - 2:
                break
            
            # Update best
            if simplex[0][1] < best_fx * best_fx:
                best_x = simplex[0][0]
                best_fx = math.sqrt(simplex[0][1])
            if abs(best_fx) <= 1e-8:
                return best_x
            
            # Check convergence
            if abs(simplex[0][1] - simplex[2][1]) < 1e-16 * max(1.0, abs(simplex[0][1])):
                break
            
            x0_nm, f0 = simplex[0]
            x1_nm, f1 = simplex[1]
            x2_nm, f2 = simplex[2]
            
            # Centroid of two best points
            xc = x0_nm + 0.5 * (x1_nm - x0_nm)
            
            # Reflection
            alpha = 1.0
            xr = xc + alpha * (xc - x2_nm)
            fr = safe_eval(xr)
            if fr is None:
                fr = float('inf')
            else:
                fr = fr * fr
            
            if f0 < fr < f1:
                # Replace worst with reflection
                simplex[2] = [xr, fr]
                simplex.sort(key=lambda p: p[1])
                continue
            
            # Expansion
            gamma = 2.0
            xe = xc + gamma * (xc - x2_nm)
            fe = safe_eval(xe)
            if fe is None:
                fe = float('inf')
            else:
                fe = fe * fe
            
            if fr < fe and fr < f0:
                simplex[2] = [xe, fe]
                simplex.sort(key=lambda p: p[1])
                continue
            
            # Contraction
            rho = 0.5
            if fr < f2:
                xc_r = xc + rho * (xc - x2_nm)
                fc_r = safe_eval(xc_r)
                if fc_r is None:
                    fc_r = float('inf')
                else:
                    fc_r = fc_r * fc_r
                if fc_r < f2:
                    simplex[2] = [xc_r, fc_r]
                    simplex.sort(key=lambda p: p[1])
                    continue
            
            # Shrink
            for i in range(1, 3):
                simplex[i][0] = x0_nm + 0.5 * (simplex[i][0] - x0_nm)
                fi = safe_eval(simplex[i][0])
                if fi is None:
                    simplex[i][1] = float('inf')
                else:
                    simplex[i][1] = fi * fi
            
            simplex.sort(key=lambda p: p[1])
            
            # Update best
            if simplex[0][1] < best_fx * best_fx:
                best_x = simplex[0][0]
                best_fx = math.sqrt(simplex[0][1])
            if abs(best_fx) <= 1e-8:
                return best_x

    # ============================================================
    # Phase 4: Final secant polish if close
    # ============================================================
    if abs(best_fx) <= 1e-4 and check_time() and evals_used < max_evals - 5:
        x = best_x
        fx = best_fx
        for _ in range(30):
            if not check_time() or evals_used >= max_evals - 1:
                break
            if abs(fx) <= 1e-9:
                return x
            h = 1e-8 * max(1.0, abs(x))
            if abs(fx) < 1e-6:
                h = 1e-10 * max(1.0, abs(x))
            x2 = x + h
            fx2 = safe_eval(x2)
            if fx2 is None:
                x2 = x - h
                fx2 = safe_eval(x2)
            if fx2 is None:
                break
            if abs(fx2 - fx) < 1e-18:
                break
            x_new = x - fx * h / (fx2 - fx)
            if not math.isfinite(x_new):
                break
            max_step = 10.0 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x - math.copysign(max_step, x_new - x)
            fx_new = safe_eval(x_new)
            if fx_new is None:
                break
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            if abs(fx_new) <= 1e-8:
                return x_new
            x = x_new
            fx = fx_new

    return best_x
# EVOLVE-BLOCK-END
