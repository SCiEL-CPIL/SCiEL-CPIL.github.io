"""
Root finder using Global Grid Scan + Brent's Method.
Strategy:
1. Perform a coarse-to-fine grid search over a wide range to find a bracket [a, b] 
   where f(a) and f(b) have opposite signs.
2. Use scipy.optimize.brentq to find the root within that bracket.
   Brent's method is superlinearly convergent and robust, typically requiring 
   very few function evaluations (often < 20) once a bracket is established.
3. If no bracket is found, fall back to a local search or return the best point.
"""
# EVOLVE-BLOCK-START
import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    f: callable, takes float, returns float
    x0: float starting point
    returns: float x with |f(x)| <= 1e-8
    """
    x0 = float(x0)
    
    # Deadline guard: 120s limit, use 115s to be safe
    deadline = time.time() + 115.0
    
    # First check if x0 is already a root
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0
    if not math.isfinite(fx0):
        # If f(x0) is non-finite, we might need to move away
        pass

    # Strategy: Global Grid Search to find a bracket
    # We scan from x0 outwards in both directions with increasing step sizes.
    # This is more robust than local methods because it doesn't rely on 
    # derivative estimates or local curvature which can mislead Newton/Secant.
    
    # Define a set of points to check. Start close, then expand.
    # We want to find a sign change.
    # Let's use a logarithmic-like expansion of the search radius.
    
    # Initial radius
    radius = 1e-3
    max_radius = 1e6
    
    # We will store the best point (min |f(x)|) found so far in case no bracket is found
    best_x = x0
    best_fx = fx0
    
    # Grid points to check relative to x0
    # We'll check points at: x0 +/- radius * factor
    # Factors: 1, 2, 5, 10, 20, 50, 100...
    factors = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    # To make the grid more efficient, we can interleave signs and radii.
    # But a simple expanding ring is often sufficient for polynomials.
    
    # Let's collect candidate points
    candidates = []
    for rad in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        candidates.append(x0 + rad)
        candidates.append(x0 - rad)
        
    # Sort candidates by distance to x0 to prioritize local roots?
    # Or just iterate. Since polynomials are smooth, a root is likely nearby.
    # Let's iterate through the list.
    
    # We need to find a pair (a, b) such that f(a)*f(b) < 0.
    # We can do this by walking out from x0.
    
    # Get f at x0 (already have fx0)
    last_x = x0
    last_f = fx0
    
    # Walk outwards
    # We'll check points in order of increasing distance
    # Distances: 0.1, -0.1, 0.5, -0.5, 1, -1, ...
    
    # Let's generate a sorted list of candidate points by distance
    dists = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    
    # We'll maintain a list of (x, f(x)) points we've visited
    # To save calls, we can just check sequentially.
    
    # Optimization: Check x0 +/- small steps first.
    # If we find a sign change between any two consecutive points in our scan, we have a bracket.
    
    # Let's scan: x0, x0+0.1, x0-0.1, x0+0.5, x0-0.5, ...
    # Actually, it's better to scan one direction, then the other?
    # No, a root could be on either side.
    # Let's do a "spiral" or "zig-zag" expansion.
    
    # Points to check:
    # 1. x0
    # 2. x0 + 0.1
    # 3. x0 - 0.1
    # 4. x0 + 0.5
    # 5. x0 - 0.5
    # ...
    
    # We will keep track of the previous point to check for sign change.
    # But since we jump between + and -, we can't just use the immediate previous.
    # We should check sign change between x0 and the new point? 
    # Or better: maintain a list of all points and check adjacent ones in the sorted-by-x list?
    # That's expensive.
    
    # Simpler: 
    # 1. Check if f(x0) and f(x0+d) have opposite signs.
    # 2. Check if f(x0) and f(x0-d) have opposite signs.
    # 3. If not, expand d.
    
    # Let's try expanding d.
    # We have fx0.
    # For each d in dists:
    #   xp = x0 + d
    #   fp = f(xp)
    #   if fx0 * fp < 0: bracket [x0, xp]
    #   xm = x0 - d
    #   fm = f(xm)
    #   if fx0 * fm < 0: bracket [xm, x0]
    #   # Also check if the new points have opposite signs to each other? Unlikely if they are far.
    #   # What if the root is between xp and xm but not straddling x0?
    #   # i.e. f(x0) > 0, f(x0+d) > 0, f(x0-d) < 0? Then [x0-d, x0] is a bracket.
    #   # i.e. f(x0) > 0, f(x0+d) < 0, f(x0-d) > 0? Then [x0, x0+d] is a bracket.
    #   # The only case missed is if f(x0) is 0 (handled) or if the function doesn't change sign at x0 
    #   # but changes sign between two non-x0 points?
    #   # For polynomials, if there is a root, and we expand from x0, we will eventually 
    #   # straddle the root. If the root is between x0-d and x0+d, and f(x0) has the same sign 
    #   # as both, it implies an even number of roots in that interval (e.g. 2). 
    #   # In that case, we might miss it if we only check against x0.
    #   # To be safe, we should also check for sign change between xp and xm?
    #   # If f(xp) and f(xm) have opposite signs, then there is a root in [xm, xp].
    #   # This is a valid bracket.
    
    for d in dists:
        if time.time() > deadline - 5.0:
            break
            
        xp = x0 + d
        fm_xp = f(xp)
        if not math.isfinite(fm_xp):
            continue
        
        # Check if x0 and xp bracket a root
        if fx0 * fm_xp < 0:
            try:
                root = brentq(f, x0, xp, xtol=1e-12, rtol=1e-12, maxiter=100)
                if math.isfinite(root):
                    return root
            except Exception:
                pass
        
        xm = x0 - d
        fm_xm = f(xm)
        if not math.isfinite(fm_xm):
            continue
        
        # Check if xm and x0 bracket a root
        if fx0 * fm_xm < 0:
            try:
                root = brentq(f, xm, x0, xtol=1e-12, rtol=1e-12, maxiter=100)
                if math.isfinite(root):
                    return root
            except Exception:
                pass
        
        # Check if xm and xp bracket a root (covers the case where x0 is not between them 
        # but the root is, or if x0 has same sign as both but they differ)
        if fm_xp * fm_xm < 0:
            try:
                root = brentq(f, xm, xp, xtol=1e-12, rtol=1e-12, maxiter=100)
                if math.isfinite(root):
                    return root
            except Exception:
                pass
        
        # Update best point
        if abs(fm_xp) < abs(best_fx):
            best_x = xp
            best_fx = fm_xp
        if abs(fm_xm) < abs(best_fx):
            best_x = xm
            best_fx = fm_xm

    # If no bracket found, try a local Newton step from best_x just in case, or return best_x
    # But the task requires |f(x)| <= 1e-8. If we didn't find a bracket, we likely failed.
    # Let's try one last Newton step from best_x if it's close.
    if abs(best_fx) < 1e-4:
        x = best_x
        fx = best_fx
        for _ in range(10):
            h = 1e-7 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = f(x_new)
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                break
        if abs(fx) <= 1e-8:
            return x

    return best_x

# EVOLVE-BLOCK-END
