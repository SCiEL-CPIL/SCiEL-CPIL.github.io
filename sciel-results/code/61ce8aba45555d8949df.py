import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 490
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0
    
    calls = 0
    best_x = float(x0)
    best_f = float('inf')
    f_cache = {}
    
    def safe_f(x):
        nonlocal calls, best_x, best_f
        if calls >= MAX_CALLS:
            return float('inf')
        
        if time.time() > DEADLINE:
            return float('inf')
            
        key = round(float(x), 14)
        
        if key in f_cache:
            val = f_cache[key]
            if abs(val) < best_f:
                best_f = abs(val)
                best_x = float(x)
            return val

        calls += 1
        try:
            val = f(float(x))
            if not math.isfinite(val):
                f_cache[key] = float('inf')
                return float('inf')
            f_cache[key] = val
        except Exception:
            f_cache[key] = float('inf')
            return float('inf')
        
        if abs(val) < best_f:
            best_f = abs(val)
            best_x = float(x)
            
        return val

    x_start = float(x0)
    
    # 1. Check starting point
    fx0 = safe_f(x_start)
    if abs(fx0) <= TOL:
        return x_start

    # 2. Global Adaptive Grid Search for Sign Changes
    # Strategy: Scan intervals [x0 - R, x0 + R] with increasing R.
    # Use a fixed number of samples per scan (e.g., 100 points).
    # This ensures we cover the domain with a consistent resolution relative to the range.
    # If a sign change is found, we have a bracket.
    
    scale = max(1.0, abs(x_start))
    
    # Radii to scan. We start small to catch nearby roots, then expand.
    # The number of points is kept constant to maintain relative resolution.
    # 100 points per scan is a good balance.
    # Total calls budget: 490. 
    # If we do 5 scans of 100 points, that's 500 calls. Too many.
    # Let's do adaptive sampling: 
    # Scan 1: R=1, 20 points
    # Scan 2: R=10, 40 points
    # Scan 3: R=100, 60 points
    # Scan 4: R=1000, 80 points
    # Scan 5: R=10000, 100 points
    # Total ~300 calls for scanning, leaving ~190 for brentq refinements.
    
    scan_configs = [
        (1.0, 20),
        (10.0, 40),
        (100.0, 60),
        (1000.0, 80),
        (10000.0, 100),
        (100000.0, 120)
    ]
    
    brackets = []
    
    for R, num_pts in scan_configs:
        if calls > MAX_CALLS - 20:
            break
        if time.time() > DEADLINE - 5.0:
            break
            
        lower = x_start - R * scale
        upper = x_start + R * scale
        
        # Generate points
        # Ensure x_start is included? linspace includes endpoints.
        # We want to catch sign changes between consecutive points.
        points = np.linspace(lower, upper, num_pts)
        
        prev_x = None
        prev_f = None
        
        for p in points:
            if time.time() > DEADLINE - 2.0:
                break
            if calls > MAX_CALLS - 5:
                break
                
            fp = safe_f(p)
            
            if not math.isfinite(fp):
                prev_x = p
                prev_f = fp
                continue
                
            if abs(fp) <= TOL:
                return p
                
            if prev_x is not None and prev_f is not None and math.isfinite(prev_f):
                if prev_f * fp < 0:
                    # Sign change found in [prev_x, p]
                    # Check if we already have a similar bracket to avoid redundant work
                    # Simple check: is this interval much larger than existing ones?
                    # Or just collect all unique brackets.
                    # To save calls, we might only want the tightest bracket.
                    # But we don't know the tightest until we scan all.
                    # Let's store all brackets and pick the best one later.
                    brackets.append((prev_x, p, prev_f, fp))
            
            prev_x = p
            prev_f = fp
        
        # Optimization: If we found a bracket in a small R scan, 
        # and it's very tight, we might not need to scan larger R.
        # However, oscillatory functions might have multiple roots. 
        # The task implies finding A root. 
        # If we found a bracket, we can try to solve it immediately.
        # If it fails (e.g. not a simple root, or numerical issue), we continue scanning.
        
        if brackets:
            # Try to solve the most recent (likely smallest range) bracket first
            # Actually, let's try all brackets found so far, starting from the smallest range.
            # Sort brackets by interval width
            brackets.sort(key=lambda b: abs(b[1] - b[0]))
            
            for a, b, fa, fb in brackets:
                if calls > MAX_CALLS - 10:
                    break
                if time.time() > DEADLINE - 2.0:
                    break
                
                # Try Brent's method
                try:
                    x_root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                    f_root = safe_f(x_root)
                    if math.isfinite(x_root) and abs(f_root) <= TOL:
                        return x_root
                except Exception:
                    pass
            
            # If Brent's failed for all brackets found so far, 
            # continue scanning to see if we find a better bracket?
            # Or stop? 
            # If Brent's failed, it might be due to a double root (no sign change) 
            # but we only add brackets on sign change. 
            # If sign change exists, Brent's should work unless f is discontinuous or non-smooth.
            # Given "oscillatory equations", they are likely smooth.
            # If Brent's failed, maybe the bracket was too wide and there were multiple roots inside?
            # No, sign change in [a,b] implies at least one root. If multiple, Brent's still finds one.
            # Failure is rare. 
            # Let's break out of scanning if we found a bracket but couldn't solve it?
            # No, maybe a smaller bracket in a later scan (impossible, scans get larger) 
            # or a different bracket.
            # Actually, if we found a bracket, it's a good candidate. 
            # If it failed, we might as well try the next scan to see if we find a *different* bracket.
            # But scans get larger. The first bracket found is likely the closest.
            # Let's just return best_x if we can't solve it, to save time?
            # No, let's continue scanning to potentially find a "cleaner" sign change.
            
            # To avoid infinite loops or excessive scanning if the first bracket is "bad",
            # we can limit the number of brackets we try to solve.
            # But we only have ~6 scans.
            
            # Clear brackets? No, keep them. But if we found one, we probably don't need to scan further 
            # unless the first one fails. 
            # Let's break if we found a bracket, to save calls for other potential equations?
            # No, this is per equation.
            # Let's break the scan loop if we found a bracket, because the root is likely there.
            # If brentq failed, we return best_x.
            break

    # 3. Fallback: If no bracket found, or all brackets failed
    # Return the best x found (minimum |f(x)|)
    return best_x
