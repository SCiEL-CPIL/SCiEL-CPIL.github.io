"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Deadline guard: 120s limit, use 115s as safety margin
    start_time = time.time()
    deadline = start_time + 115.0
    
    x0 = float(x0)
    
    # Strategy: Try polynomial reconstruction (cubic) first for speed,
    # then fall back to robust iterative methods if needed.
    
    # Step 1: Try cubic interpolation with 4 points
    # Choose points adaptively around x0
    # We want to find a root, so we want points that bracket or are close to the root.
    # Since we don't know the root, we sample around x0.
    
    # Let's try a set of points that spans a reasonable range.
    # If the polynomial is smooth and the root is near x0, this should work well.
    
    # Try multiple sets of points if the first one fails or gives a bad result.
    
    def try_cubic_solve(f, x0, deadline):
        """Try to solve using cubic interpolation."""
        # Choose 4 points. Let's use a geometric progression or symmetric points.
        # Symmetric points around x0 with increasing spacing.
        # Let's try: x0 - 2, x0 - 1, x0 + 1, x0 + 2
        # Or: x0, x0+1, x0+2, x0+3
        
        # Actually, a better strategy for root finding is to bracket.
        # But we don't know the bracket.
        # Let's try a few different sets of points.
        
        # Set 1: Close points
        h = 0.5
        pts = [x0 - 1.5*h, x0 - 0.5*h, x0 + 0.5*h, x0 + 1.5*h]
        
        # Set 2: Wider points
        h = 1.0
        pts2 = [x0 - 1.5*h, x0 - 0.5*h, x0 + 0.5*h, x0 + 1.5*h]
        
        # Set 3: Even wider
        h = 2.0
        pts3 = [x0 - 1.5*h, x0 - 0.5*h, x0 + 0.5*h, x0 + 1.5*h]
        
        # Set 4: Geometric
        h = 1.0
        pts4 = [x0, x0 + h, x0 + 2*h, x0 + 4*h]
        
        # Set 5: Mixed
        pts5 = [x0 - 1, x0 - 0.5, x0 + 0.5, x0 + 1]
        
        for pts in [pts, pts5, pts2, pts3, pts4]:
            if time.time() > deadline - 5:
                break
            try:
                vals = [f(p) for p in pts]
                # Check if any val is already small
                for p, v in zip(pts, vals):
                    if abs(v) <= 1e-8:
                        return p
                
                # Solve cubic via numpy.roots
                # Use Lagrange interpolation to get coefficients
                # Or directly use numpy.polyfit
                coeffs = np.polyfit(pts, vals, 3)
                roots = np.roots(coeffs)
                
                # Find the real root closest to x0
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
                if not real_roots:
                    continue
                
                # Choose the root closest to x0
                best_root = min(real_roots, key=lambda r: abs(r - x0))
                
                # Verify the root
                if abs(f(best_root)) <= 1e-8:
                    return best_root
            except Exception:
                continue
        
        return None
    
    # Try cubic first
    result = try_cubic_solve(f, x0, deadline)
    if result is not None:
        return result
    
    # Fallback: Newton's method with finite difference
    x = x0
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
        h = 1e-6 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0:
            break
        x = x - fx / d
    
    # Final fallback: Bisection-like search if Newton fails
    # Try to find a bracket
    lo, hi = x0 - 10, x0 + 10
    flo, fhi = f(lo), f(hi)
    if flo * fhi < 0:
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            mid = (lo + hi) / 2
            fmid = f(mid)
            if abs(fmid) <= 1e-8:
                return mid
            if flo * fmid < 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
        return (lo + hi) / 2
    
    return x
# EVOLVE-BLOCK-END
