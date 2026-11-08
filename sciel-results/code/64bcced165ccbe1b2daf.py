"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

DEADLINE = 145.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(pts):
    """
    Compute areas of all triangles formed by triples of points.
    Returns an array of shape (C(n,3),).
    """
    n = len(pts)
    if n < 3:
        return np.array([])
    
    # Get all unique triples
    # Using itertools.combinations is fast enough for N=14 (364 triples)
    # But let's vectorize a bit if possible or just use efficient indexing.
    # For N=14, even a python loop over combinations is ~0.1ms, which is fine.
    # However, we call this inside an objective function, so speed matters.
    
    # Vectorized approach:
    # Indices i, j, k with i < j < k
    # We can precompute indices.
    
    # Let's stick to a very fast numpy implementation for the objective.
    # Precompute index arrays for all triples
    # This is done once in the closure or global.
    
    # For now, a fast loop is acceptable if we minimize overhead.
    # But let's try to be vectorized.
    
    # i: 0..n-3
    # j: i+1..n-2
    # k: j+1..n-1
    
    # It's easier to just generate the triples indices once.
    # Since N is fixed, we can generate them once.
    
    # We'll generate them inside the function but cache them? 
    # No, let's just make a helper that assumes fixed N for speed.
    
    # Actually, for N=14, C(14,3)=364.
    # Creating a 364x3 array of indices is trivial.
    # Let's create them once globally.
    
    return _precomputed_triangles(pts)

# Precompute triangle indices for N=14
def _generate_triangle_indices(n):
    inds = []
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                inds.append((i, j, k))
    return np.array(inds, dtype=int)

_TRI_INDICES = _generate_triangle_indices(N)

def _precomputed_triangles(pts):
    """
    Fast triangle area calculation using precomputed indices.
    """
    # pts is (N, 2)
    # _TRI_INDICES is (M, 3)
    i, j, k = _TRI_INDICES.T
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    
    # Area = 0.5 * | cross(pj-pi, pk-pi) |
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hard_score(pts):
    """
    Compute min triangle area / hull area.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    areas = _precomputed_triangles(pts)
    if len(areas) == 0:
        return 0.0
        
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0
        
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
        
    return min_area / hull_a

def _soft_obj(x, beta=100.0):
    """
    Soft objective for Differential Evolution.
    We want to MAXIMIZE min(area) / hull_area.
    So we MINIMIZE -soft_min(area) / hull_area.
    
    soft_min(a, beta) = -1/beta * log(sum(exp(-beta * a_i)))
    As beta -> inf, soft_min -> min(a_i).
    
    To avoid overflow in exp, we can use the log-sum-exp trick, 
    but for areas ~0.1 and beta=100, exp(-10) is small, exp(0) is 1.
    Actually, if areas are small, -beta * area is large negative? No.
    If area = 0.1, -100 * 0.1 = -10. exp(-10) ~ 4.5e-5.
    If area = 0.01, -100 * 0.01 = -1. exp(-1) ~ 0.36.
    
    Wait, the formula is:
    m = min(a_i)
    soft_min = -1/beta * log(sum(exp(-beta * a_i)))
    
    If a_i are all equal to m, sum = M * exp(-beta*m).
    log(sum) = log(M) - beta*m.
    soft_min = -1/beta * (log(M) - beta*m) = m - log(M)/beta.
    
    This is a lower bound on the min.
    
    We want to maximize this. So we minimize -soft_min / hull_area.
    
    Also, we should penalize if hull_area is small or points are degenerate.
    """
    pts = x.reshape((N, 2))
    
    # Basic sanity check
    if not np.all(np.isfinite(pts)):
        return 1e6
        
    areas = _precomputed_triangles(pts)
    if len(areas) == 0:
        return 1e6
        
    # If any area is extremely small, the soft min will be small.
    # We can add a penalty for very small areas to help DE navigate?
    # The soft min handles this smoothly.
    
    # Calculate soft min
    # To prevent underflow/overflow issues with exp, we can shift by the min.
    # But for N=14, M=364.
    # Let's just compute directly. If beta is large, exp(-beta*area) can underflow to 0.
    # That's fine, log(0) is -inf, soft_min becomes +inf? No.
    # If all terms underflow to 0, sum=0, log(0)=-inf, soft_min=+inf.
    # This would make -soft_min = -inf, which is good for minimization?
    # Wait. If areas are large, exp(-beta*area) is small.
    # If areas are small, exp(-beta*area) is large.
    
    # Let's check the direction.
    # We want to maximize min_area.
    # soft_min approximates min_area from below.
    # We want to maximize soft_min.
    # So we minimize -soft_min.
    
    # If areas are huge, exp(-beta*area) ~ 0. Sum ~ 0. Log ~ -inf.
    # -1/beta * (-inf) = +inf.
    # So -soft_min = -inf.
    # This means huge areas are "good" (low cost).
    
    # If areas are tiny (near 0), exp(0) = 1. Sum ~ M. Log ~ log(M).
    # soft_min ~ -log(M)/beta.
    # -soft_min ~ log(M)/beta.
    
    # So the cost function behaves reasonably.
    
    # However, if some areas are 0, exp(0)=1.
    # If one area is 0 and others are large:
    # Sum = 1 + small. Log ~ 0.
    # soft_min ~ 0.
    # -soft_min ~ 0.
    
    # This seems correct.
    
    # One issue: if pts are identical, areas are 0.
    # We should also normalize by hull area?
    # The task asks for min_area / hull_area.
    # If we just maximize min_area, we might expand the points to infinity.
    # So we MUST divide by hull area.
    
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e6
        
    # Calculate soft min of areas
    # Use log-sum-exp for stability?
    # beta * areas can be large.
    # Let's just use np.exp and hope for the best, or clip.
    # beta=100, area max ~ 10? -1000. exp(-1000) = 0.
    # area min ~ 0. exp(0) = 1.
    
    exp_terms = np.exp(-beta * areas)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        # All areas are very large
        soft_min_val = np.max(areas) # Approximation, or just return 0 cost
        # Actually if sum_exp is 0, it means areas are huge.
        # The ratio min/hull will be small if hull is also huge?
        # If we scale points by k, areas scale by k^2, hull scales by k^2. Ratio is invariant.
        # So we don't need to worry about scaling for the ratio, but DE needs a bounded space or it will drift.
        # We should constrain the space or normalize.
        
        # Better strategy: Normalize the points to unit hull area inside the objective?
        # If we normalize, the objective becomes just max(soft_min(areas_normalized)).
        # This removes the scale degree of freedom.
        
        # Let's try normalizing inside the objective.
        # pts_norm = pts / sqrt(hull_a)
        # Then compute areas of pts_norm.
        # Then soft_min(areas_norm).
        # Maximize this.
        
        # This is safer for DE.
        
        return 1e6 # Should not happen if hull_a > 1e-12 and areas are reasonable
        
    soft_min_val = -1.0 / beta * np.log(sum_exp)
    
    # We want to maximize soft_min_val / hull_a?
    # No, if we normalize pts to unit hull area, hull_a becomes 1.
    # Then we just maximize soft_min_val.
    
    # Let's implement the normalization strategy.
    
    # 1. Compute hull area of x
    # 2. If hull area is too small, return penalty
    # 3. Normalize x to unit hull area
    # 4. Compute soft min of triangle areas of normalized points
    # 5. Return -soft_min_val (to minimize)
    
    # This is cleaner.
    
    # Let's rewrite _soft_obj to do this.
    
    return 0.0 # Placeholder, will rewrite

def _soft_obj_normalized(x, beta=100.0):
    """
    Objective for DE: Minimize -soft_min(areas of unit-hull-normalized points).
    """
    pts = x.reshape((N, 2))
    
    if not np.all(np.isfinite(pts)):
        return 1e6
        
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e6
        
    # Normalize to unit hull area
    scale = np.sqrt(hull_a)
    pts_norm = pts / scale
    
    areas = _precomputed_triangles(pts_norm)
    
    if len(areas) == 0:
        return 1e6
        
    # Soft min
    # To avoid overflow in exp(-beta * area), note that area is ~ 0.01 to 0.1 for unit hull.
    # beta=100 -> exp(-1) to exp(-10). Safe.
    exp_terms = np.exp(-beta * areas)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        # Areas are very large? Unlikely for unit hull.
        return 0.0 
        
    soft_min_val = -1.0 / beta * np.log(sum_exp)
    
    # We want to maximize soft_min_val.
    # So we minimize -soft_min_val.
    return -soft_min_val

def _local_polish(pts, time_limit=5.0):
    """
    Coordinate descent / Nelder-Mead polish on the hard score.
    """
    start_t = time.time()
    
    # Use Nelder-Mead on the negative hard score?
    # Hard score is non-smooth (min).
    # Nelder-Mead works on non-smooth functions but can be slow.
    # Let's try a simple coordinate descent with adaptive step.
    
    current_pts = pts.copy()
    current_score = _hard_score(current_pts)
    
    step = 0.01
    n_dims = current_pts.size
    
    # Directions
    # We'll try moving each coordinate up and down.
    
    while time.time() - start_t < time_limit:
        improved = False
        for d in range(n_dims):
            # +
            trial = current_pts.copy()
            trial[d % N, (d // N) % 2] += step
            s = _hard_score(trial)
            if s > current_score:
                current_pts = trial
                current_score = s
                improved = True
                continue # Move to next direction? Or stay?
                # Standard CD: move to next direction.
                
            # -
            trial = current_pts.copy()
            trial[d % N, (d // N) % 2] -= step
            s = _hard_score(trial)
            if s > current_score:
                current_pts = trial
                current_score = s
                improved = True
        
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        else:
            # Keep step or increase slightly?
            # Keep step to be safe.
            pass
            
    return current_pts, current_score

def construct():
    start_time = time.time()
    
    # Bounds for DE
    # Points should be in a reasonable range.
    # Since we normalize in the objective, the scale doesn't matter much,
    # but the shape does.
    # Let's bound coordinates to [-2, 2] to keep numbers stable.
    bounds = [(-2, 2)] * (2 * N)
    
    # Initial guess: Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Run Differential Evolution
    # maxiter: 100
    # popsize: 15 (255 individuals)
    # tol: 1e-4
    # update_interval: 1 (check time every 1 iteration? No, DE is vectorized internally usually)
    # We need to be careful with time.
    
    # DE in scipy is a black box. We can't easily interrupt it mid-iteration.
    # But we can set maxiter low enough.
    # Each iteration evaluates ~ popsize * N individuals?
    # Actually, DE generates popsize new individuals per generation.
    # With popsize=15, 210 individuals.
    # Each eval is ~ 0.1 ms (hull + triangles).
    # 210 * 0.1 ms = 21 ms per generation.
    # 100 generations = 2.1 seconds.
    # This is very fast. We can afford more iterations.
    
    # Let's try maxiter=200, popsize=20.
    # 20 * 20 = 400 individuals/gen.
    # 400 * 0.1 ms = 40 ms/gen.
    # 200 gens = 8 seconds.
    
    # We have 140 seconds.
    # We can run multiple DE restarts with different betas or initial populations?
    # Or just one big DE run.
    
    # Let's do one DE run with maxiter=100, popsize=20.
    # Then polish.
    
    # To be safe with time, let's use a callback to stop if time is up.
    # scipy.optimize.differential_evolution supports callback.
    
    def de_callback(xk, convergence):
        if time.time() - start_time > DEADLINE - 10:
            return True
        return False

    try:
        result = differential_evolution(
            _soft_obj_normalized, 
            bounds, 
            args=(50.0,), # beta=50 for smoother landscape initially
            maxiter=100, 
            popsize=20, 
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            callback=de_callback,
            polish=False, # We do our own polish
            init='latinhypercube'
        )
        
        best_x = result.x
    except Exception as e:
        # Fallback to initial guess
        best_x = x0
        print(f"DE failed: {e}")
        
    best_pts = best_x.reshape((N, 2))
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_pts
        
    # Polish with local search
    # Use a tighter beta for the polish objective if we were using gradient methods,
    # but we are using hard score for CD.
    
    # Run local polish
    polish_time = min(10.0, DEADLINE - 2 - (time.time() - start_time))
    if polish_time > 0:
        polished_pts, polished_score = _local_polish(best_pts, time_limit=polish_time)
        if polished_score > _hard_score(best_pts):
            best_pts = polished_pts
            
    # Final check
    if time.time() - start_time > DEADLINE - 1:
         return best_pts
         
    # One more quick DE run with higher beta to refine?
    # The soft min with beta=50 approximates min well.
    # Higher beta makes it sharper, harder for DE.
    # Lower beta is smoother but less accurate.
    # Let's try a second short DE run with beta=200 to sharpen the minimum.
    
    if time.time() - start_time < DEADLINE - 15:
        try:
            result2 = differential_evolution(
                _soft_obj_normalized, 
                bounds, 
                args=(200.0,),
                x0=best_pts.flatten(), # Start from previous best
                maxiter=50, 
                popsize=15, 
                tol=1e-5,
                mutation=(0.3, 0.8),
                recombination=0.8,
                seed=43,
                callback=de_callback,
                polish=False,
                init='best' # Use x0 as best? No, init='best' isn't a standard option for DE in scipy, 
                # actually init can be 'latinhypercube', 'halton', 'sobol', or a sequence.
                # If we pass x0, it's ignored if init is not None?
                # Check scipy docs: x0 is ignored if init is not None.
                # So we must set init=None to use x0?
                # Actually, if init is None, x0 is used.
                # Let's set init=None.
            )
            # Wait, if I set init=None, does it use x0?
            # Yes.
            # But I passed init='latinhypercube' in the first call.
            # For the second call, I should pass init=None.
            
            # Let's redo the second call properly.
            result2 = differential_evolution(
                _soft_obj_normalized, 
                bounds, 
                args=(200.0,),
                x0=best_pts.flatten(),
                maxiter=50, 
                popsize=15, 
                tol=1e-5,
                mutation=(0.3, 0.8),
                recombination=0.8,
                seed=43,
                callback=de_callback,
                polish=False,
                init=None
            )
            
            best_x2 = result2.x
            best_pts2 = best_x2.reshape((N, 2))
            
            if _hard_score(best_pts2) > _hard_score(best_pts):
                best_pts = best_pts2
                
        except Exception as e:
            pass
            
    # Final polish if time remains
    if time.time() - start_time < DEADLINE - 2:
        polish_time = min(5.0, DEADLINE - 2 - (time.time() - start_time))
        if polish_time > 0:
            polished_pts, polished_score = _local_polish(best_pts, time_limit=polish_time)
            if polished_score > _hard_score(best_pts):
                best_pts = polished_pts
                
    return best_pts

# EVOLVE-BLOCK-END
