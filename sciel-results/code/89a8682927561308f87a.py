import time
import math

def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 148.0  # safety margin before 150s kill

    def time_ok():
        return (time.time() - start_time) < DEADLINE

    x = float(x0)
    if not math.isfinite(x):
        x = 0.0

    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx) if math.isfinite(best_fx) else float('inf')

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 1: Secant method with tighter convergence ---
    x1 = x
    f1 = best_fx
    x2 = x1 + 1e-4 * max(1.0, abs(x1))
    f2 = f(x2)

    if not math.isfinite(f2):
        x2 = x1 - 1e-4 * max(1.0, abs(x1))
        f2 = f(x2)

    if math.isfinite(f1) and math.isfinite(f2) and abs(f2 - f1) > 1e-30:
        for _ in range(25):
            if not time_ok():
                break
            if best_abs <= 1e-8:
                return best_x
            if abs(f2 - f1) < 1e-30:
                break

            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x3) or abs(x3) > 1e15:
                break

            f3 = f(x3)
            if not math.isfinite(f3):
                break

            if abs(f3) < best_abs:
                best_x = x3
                best_fx = f3
                best_abs = abs(f3)

            # Tighter early exit: 1e-10 is well below 1e-8 requirement
            if best_abs <= 1e-10:
                return best_x

            # Check convergence
            if abs(x3 - x2) < 1e-13 * max(1.0, abs(x3)):
                break

            x1, f1 = x2, f2
            x2, f2 = x3, f3

    if best_abs <= 1e-8:
        return best_x

    # --- Phase 2: Bracket the root with smarter step sizes ---
    a = best_x
    fa = best_fx

    found_bracket = False
    # Try expanding from best_x first (most promising), then x0
    for start_pt in [best_x, x0]:
        if not time_ok():
            break
        fs = f(start_pt) if start_pt != best_x else fa
        if not math.isfinite(fs):
            continue

        for direction in [1, -1]:
            if not time_ok():
                break
            step = 0.5  # Start smaller to find bracket faster
            for _ in range(35):
                if not time_ok():
                    break
                b = start_pt + direction * step
                fb = f(b)
                if not math.isfinite(fb):
                    step *= 2
                    if abs(b) > 1e15:
                        break
                    continue

                if fs * fb < 0:
                    # Found bracket
                    a, b = start_pt, b
                    fa, fb = fs, fb
                    found_bracket = True
                    break

                step *= 1.8  # Slightly slower growth, more chances to hit bracket
                if abs(b) > 1e15:
                    break
            if found_bracket:
                break
        if found_bracket:
            break

    if found_bracket and time_ok():
        # --- Phase 3: Simple bisection (fewer calls than Illinois) ---
        lo, hi = a, b
        flo, fhi = fa, fb

        for _ in range(60):
            if not time_ok():
                break

            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if not math.isfinite(fmid):
                break

            if abs(fmid) < best_abs:
                best_x = mid
                best_fx = fmid
                best_abs = abs(fmid)

            if best_abs <= 1e-10:
                return best_x

            # Standard bisection: keep the side with sign change
            if flo * fmid < 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid

            if abs(hi - lo) < 1e-13 * max(1.0, abs(mid)):
                break

        if best_abs <= 1e-8:
            return best_x

    # --- Phase 4: Secant polish (1 call/step vs Newton's 2) ---
    # Use the last two secant points or best_x with a small perturbation
    if not time_ok():
        return best_x

    x_p1 = best_x
    fp1 = best_fx
    x_p2 = best_x + 1e-6 * max(1.0, abs(best_x))
    fp2 = f(x_p2)

    if not math.isfinite(fp2):
        x_p2 = best_x - 1e-6 * max(1.0, abs(best_x))
        fp2 = f(x_p2)

    if math.isfinite(fp1) and math.isfinite(fp2) and abs(fp2 - fp1) > 1e-30:
        for _ in range(15):
            if not time_ok():
                break
            if best_abs <= 1e-8:
                return best_x
            if abs(fp2 - fp1) < 1e-30:
                break

            x_p3 = x_p2 - fp2 * (x_p2 - x_p1) / (fp2 - fp1)
            if not math.isfinite(x_p3) or abs(x_p3) > 1e15:
                break

            fp3 = f(x_p3)
            if not math.isfinite(fp3):
                break

            if abs(fp3) < best_abs:
                best_x = x_p3
                best_fx = fp3
                best_abs = abs(fp3)

            if best_abs <= 1e-10:
                return best_x

            if abs(x_p3 - x_p2) < 1e-13 * max(1.0, abs(x_p3)):
                break

            x_p1, fp1 = x_p2, fp2
            x_p2, fp2 = x_p3, fp3

    return best_x
