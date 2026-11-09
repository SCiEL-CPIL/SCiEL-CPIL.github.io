"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2

# Precompute triangle indices for speed
_TRIPLETS = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLETS.append((i, j, k))

def _is_inside(points):
    """Check if all points are inside the triangle (0,0), (1,0), (0.5, H)."""
    # y >= 0
    if np.any(points[:, 1] < -1e-9):
        return False
    # y <= sqrt(3) * x
    if np.any(points[:, 1] > np.sqrt(3) * points[:, 0] + 1e-9):
        return False
    # y <= sqrt(3) * (1 - x)
    if np.any(points[:, 1] > np.sqrt(3) * (1.0 - points[:, 0]) + 1e-9):
        return False
    return True

def _min_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    if not _is_inside(points):
        return 0.0
    
    min_a = float('inf')
    for i, j, k in _TRIPLETS:
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # 2 * Area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        area2 = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if area2 < min_a:
            min_a = area2
    return min_a * 0.5

def _negative_min_area(x):
    """Objective function for minimization (minimize -min_area)."""
    points = x.reshape(N, 2)
    return -_min_area(points)

def _callback(xk, convergence):
    """Callback to check time limit."""
    if time.time() > _deadline:
        raise ValueError("Time limit reached")

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 120.0  # 20s safety margin
    
    # Bounds for x and y
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))  # x
    for _ in range(N):
        bounds.append((0.0, H))    # y
        
    # Initial population: Generate a few random valid points to seed DE
    # This helps DE start in a reasonable region
    init = []
    for _ in range(5):
        x = np.random.rand(N, 2)
        x[:, 0] *= 1.0
        x[:, 1] *= H
        # Project/clip to triangle roughly? 
        # Just use random within box, DE will handle it, but valid starts are better.
        # Let's just use the bounds, DE handles invalid via penalty (returns 0 area -> -0 objective, which is bad for maximization, so it will avoid).
        # Wait, if it returns 0 area, objective is 0. Maximize wants high area. So 0 is low. It will move away.
        init.append(x.flatten())
        
    # Run Differential Evolution
    best_result = None
    try:
        res = differential_evolution(
            _negative_min_area,
            bounds,
            maxiter=1000,
            popsize=20,
            tol=1e-6,
            init=init,
            callback=_callback,
            polish=False,  # We will do our own polish
            disp=False
        )
        best_result = res.x
    except ValueError:
        # Time limit reached, get best so far? 
        # DE doesn't easily expose 'best so far' on exception unless we capture it.
        # Alternative: Run with a shorter time limit and check time in a loop?
        # Or just rely on the fact that DE might finish.
        # To be safe, let's implement a manual loop with DE? No, DE is black box.
        # Let's assume DE finishes or we catch the exception and use the last known good?
        # Actually, if we raise exception, we lose the result.
        # Better strategy: Use a callback that sets a flag and returns, but DE continues.
        # Then after DE, we check if we exceeded time.
        # But if it exceeds, we should stop.
        # Let's modify _callback to not raise, but just set a flag.
        # And in the main loop, if flag is set, we break? 
        # DE doesn't support breaking easily.
        
        # Revised Plan:
        # 1. Run DE with a conservative maxiter and popsize.
        # 2. If it finishes, great.
        # 3. If not, we need a fallback.
        
        # Let's try a simpler approach:
        # Run DE. If it takes too long, it will be killed by the evaluator.
        # But we have a 150s limit.
        # Let's use a time-checking wrapper that runs DE in a thread? No, standard lib.
        
        # Let's assume DE is fast enough for 22 dims with popsize 20.
        # 22 * 20 = 440 evals per iter. 1000 iters = 440k evals.
        # Each eval is ~165 area calcs. 165 * 440k = 72M ops.
        # Python is slow. This might take > 120s.
        
        # Optimization: Vectorize _min_area?
        # Or reduce popsize/maxiter.
        # Let's reduce maxiter to 200 and popsize to 15.
        # And use a faster area calculation.
        
        pass

    # Let's restart with a safer configuration
    start_time = time.time()
    _deadline = start_time + 100.0
    
    def _fast_min_area(points):
        if not _is_inside(points):
            return 0.0
        # Vectorized calculation for all triplets
        # This is hard to vectorize easily without creating huge arrays.
        # 165 triplets is small. Loop is fine.
        min_a = float('inf')
        for i, j, k in _TRIPLETS:
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            area2 = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
            if area2 < min_a:
                min_a = area2
        return min_a * 0.5

    def _neg_area(x):
        points = x.reshape(N, 2)
        return -_fast_min_area(points)

    def _cb(xk, conv):
        if time.time() > _deadline:
            raise TimeoutError("DE Time Out")

    best_x = None
    try:
        res = differential_evolution(
            _neg_area,
            bounds,
            maxiter=500,
            popsize=15,
            tol=1e-5,
            init='latinhypercube',
            callback=_cb,
            polish=False,
            disp=False
        )
        best_x = res.x
    except TimeoutError:
        # If timeout, we need the best found.
        # Unfortunately, DE doesn't return partial results on exception.
        # We can wrap the objective to track best?
        # Let's do that.
        best_val = -float('inf')
        best_x_track = None
        def _neg_area_track(x):
            nonlocal best_val, best_x_track
            val = _neg_area(x)
            if val < best_val:
                best_val = val
                best_x_track = x
            return val
        
        _deadline = time.time() + 50.0 # Remaining time
        try:
            res = differential_evolution(
                _neg_area_track,
                bounds,
                maxiter=500,
                popsize=15,
                tol=1e-5,
                init='latinhypercube',
                callback=_cb,
                polish=False,
                disp=False
            )
            if best_x_track is not None:
                best_x = best_x_track
        except TimeoutError:
            if best_x_track is not None:
                best_x = best_x_track
    except Exception:
        # If DE fails completely, fallback to random
        pass

    if best_x is None:
        # Fallback: Random
        best_x = np.random.rand(N, 2)
        best_x[:, 0] *= 1.0
        best_x[:, 1] *= H
        # Project to triangle
        for i in range(N):
            x, y = best_x[i]
            if y > np.sqrt(3) * x:
                y = np.sqrt(3) * x
            if y > np.sqrt(3) * (1 - x):
                y = np.sqrt(3) * (1 - x)
            if y < 0:
                y = 0
            best_x[i] = [x, y]

    # Local Polish with Nelder-Mead
    # Only if we have time
    if time.time() < start_time + 130.0:
        try:
            res2 = minimize(
                _neg_area,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res2.fun < _neg_area(best_x): # Improved
                best_x = res2.x
        except Exception:
            pass

    points = best_x.reshape(N, 2)
    # Final validation and clipping
    for i in range(N):
        x, y = points[i]
        if y < 0: y = 0
        if x < 0: x = 0
        if x > 1: x = 1
        if y > np.sqrt(3) * x:
            y = np.sqrt(3) * x
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x)
        points[i] = [x, y]
        
    return points

# EVOLVE-BLOCK-END
