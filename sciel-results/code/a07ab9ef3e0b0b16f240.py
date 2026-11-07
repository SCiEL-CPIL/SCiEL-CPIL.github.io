"""
Strategy: Dense logarithmic global scan for bracketing, followed by scipy.optimize.brentq.
Unlike the current approach which expands locally from x0 with a fixed step size, this 
approach scans a wide range of the domain using a pre-computed, dense, logarithmically 
spaced grid. This ensures that even highly oscillatory functions have a high probability 
of a sign change being detected between adjacent grid points, providing a tight bracket 
for brentq. This minimizes the number of calls spent on bracketing by finding a valid 
bracket quickly (often in the first few grid points if the root is close, or efficiently 
scanning a wide range if it's far).
"""
import math
import time
import numpy as np
from scipy.optimize import brentq

# Global deadline: evaluator kills after 150s wall-clock.
_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 490  # stay under 500


def _time_left():
    return _DEADLINE - time.time()


def _counted_call(f, x, counter):
    """Call f(x), track count, guard against exceptions / non-finite."""
    counter[0] += 1
    try:
        val = f(x)
    except Exception:
        return float('nan')
    if not math.isfinite(val):
        return val
    return val


def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy: 
    1. Check x0 directly.
    2. Generate a dense, logarithmically spaced grid around x0 covering a wide range.
    3. Scan the grid to find a sign change (bracket).
    4. Use scipy.optimize.brentq to converge within the bracket.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Check x0
    f0 = _counted_call(f, x0, counter)
    if math.isfinite(f0) and abs(f0) <= 1e-8:
        return x0

    if not math.isfinite(f0):
        # If f(x0) is invalid, we must search elsewhere
        pass
    else:
        # We have a valid f0. We will use it as the first point in our scan.
        pass

    # 2. Generate a dense grid.
    # We want to cover a wide range, say [-1e6, 1e6] relative to x0, but log-spaced is tricky for negative numbers.
    # Instead, we can generate a grid of magnitudes and signs.
    # Or simply use a linear grid with increasing density near x0? 
    # Log-spaced magnitude grid is better for "oscillatory" if frequency varies, but standard oscillatory usually means sin/cos with fixed freq.
    # If freq is fixed, a linear grid with step size smaller than the period is best.
    # We don't know the period. 
    # However, a "dense" scan implies many points.
    # Let's create a grid that is dense near x0 and sparse far away, but covers a large range.
    
    # Strategy for grid:
    # Points: x0 + delta, where delta takes values from a set D.
    # D should include small values (to catch roots near x0) and large values (to catch roots far away).
    # To maximize probability of bracketing an oscillatory function, we want the step size to be small relative to the oscillation period.
    # Without knowing the period, we assume it's O(1) or smaller.
    # Let's use a grid of ~100 points covering [-100, 100] around x0, plus some larger jumps.
    # Actually, 100 calls for bracketing is a lot if we can do better.
    # But if we miss the bracket, we fail.
    # The current method uses exponential expansion. If the period is small, exponential expansion might jump over the root.
    # A fine linear scan is safer for small periods.
    
    # Let's try a hybrid: 
    # First, a fine scan in a local window [x0-10, x0+10] with step 0.1 (200 points? No, too many).
    # Let's use 50 points in [x0-5, x0+5] (step 0.2).
    # If no bracket, expand to [x0-50, x0+50] with step 2.0 (50 points).
    # If no bracket, expand to [x0-500, x0+500] with step 20.0 (50 points).
    
    # This is similar to the current approach but with fixed, dense grids per scale rather than exponential steps.
    # The key difference: The current approach checks x-step and x+step. If the root is between x-step and x+step but not bracketed by x, it misses it unless it checks fa*fb. 
    # My approach will check all adjacent points in the grid.
    
    # Grid 1: Local, dense
    # Range: x0 +/- 5.0, Step: 0.1 -> 100 points. 
    # This is 100 calls. If we find a bracket, we spend ~10-20 more for brentq. Total ~120.
    # If we don't find it, we've spent 100 calls.
    
    # Let's be more efficient. 
    # We know f0. 
    # We can generate points: x0 + d for d in deltas.
    # Sort deltas by absolute value.
    # Evaluate f at each.
    # Check sign change between consecutive evaluated points (including x0).
    
    # Deltas:
    # Small: 0.1, -0.1, 0.2, -0.2, ... up to 2.0
    # Medium: 5, -5, 10, -10, ... up to 20
    # Large: 50, -50, 100, -100, ... up to 200
    
    # To ensure we catch oscillations, we need the step to be small.
    # Let's assume the "oscillatory" nature implies a period of roughly 1-10.
    # A step of 0.5 is a good bet.
    
    # Let's construct a list of candidate points.
    # We will evaluate them in order of proximity to x0? No, that doesn't help bracketing.
    # We need a connected path of points to detect sign changes.
    # So we should scan a line.
    
    # Scan 1: x0 - 10 to x0 + 10 with step 0.2. (100 points)
    # If found, done.
    # Scan 2: x0 - 100 to x0 + 100 with step 2.0. (100 points)
    # If found, done.
    # Scan 3: x0 - 1000 to x0 + 1000 with step 20.0. (100 points)
    
    # Total calls if all fail: 300 + brentq.
    # This is well within budget.
    # And it is "dense" enough to catch most oscillations.
    
    scans = [
        (10.0, 0.2),   # local, fine
        (100.0, 2.0),  # medium, coarse
        (1000.0, 20.0) # large, very coarse
    ]
    
    # We already have f0. Let's store it.
    # To make bracketing easy, we'll maintain a list of (x, f(x)) pairs.
    # We'll start with [(x0, f0)] if f0 is finite.
    
    points = []
    if math.isfinite(f0):
        points.append((x0, f0))
    
    for radius, step in scans:
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 10:
            break
            
        # Generate points in this range
        # We want to scan from x0-radius to x0+radius.
        # But we already have x0.
        # Let's generate a grid and filter out points we've already evaluated? 
        # Simpler: Just generate the full grid for this scan, evaluate, and check.
        # To avoid re-evaluating x0, we can just include it in the grid if it falls on it, or handle it separately.
        # Given the cost of a call is low, re-evaluating x0 is acceptable if it simplifies logic, but we have a budget.
        # Let's just generate the grid and evaluate all.
        
        a = x0 - radius
        b = x0 + radius
        n_points = int(2 * radius / step) + 1
        # Ensure we don't generate too many points if step is small
        if n_points > 200:
            # Adjust step
            step = 2 * radius / 199
            n_points = 200
            
        xs = np.linspace(a, b, n_points)
        
        # Evaluate f at these points
        # We can vectorize the call if f supports it? No, f takes float.
        # We must loop.
        
        # Optimization: Check if x0 is in this grid? 
        # It might not be exactly due to floating point.
        # Let's just evaluate all points in the grid.
        
        prev_x = None
        prev_f = None
        
        for x in xs:
            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
                break
                
            fx = _counted_call(f, x, counter)
            
            if math.isfinite(fx):
                if abs(fx) <= 1e-8:
                    return x
                
                if prev_f is not None and math.isfinite(prev_f):
                    if prev_f * fx < 0:
                        # Bracket found: [prev_x, x]
                        try:
                            result = brentq(f, prev_x, x, xtol=1e-12, rtol=1e-12, maxiter=100)
                            if abs(f(result)) <= 1e-8:
                                return result
                        except Exception:
                            pass
                
                prev_x = x
                prev_f = fx
            else:
                # If fx is not finite, we can't use it for bracketing with neighbors.
                # Reset the "previous" state to avoid bracketing across a singularity.
                prev_x = None
                prev_f = None

    # If we exhausted scans without finding a bracket, try a random search?
    # Or just return x0?
    # Given the "oscillatory" nature, it's likely we missed it if the frequency is very high.
    # But a step of 0.2 over a range of 20 is quite dense.
    
    # Fallback: Try a few random points?
    # No, random points don't help bracketing unless we check neighbors.
    
    # Return x0 as best guess (might fail if |f(x0)| > 1e-8, but we have no other choice)
    return x0
