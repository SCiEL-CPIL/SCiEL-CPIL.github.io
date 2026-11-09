import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    n = 8
    _start_time = time.time()
    _DEADLINE = 145.0  # Safety margin
    
    # Regular octagon as initial best
    angles = [2 * math.pi * k / n for k in range(n)]
    reg_pts = [[math.cos(a), math.sin(a)] for a in angles]
    
    def score_points(pts):
        """
        Calculate the score for a set of points.
        pts: list of [x, y]
        Returns score = Delta / n^n after scaling to diameter 2.
        """
        pts = np.array(pts, dtype=float)
        n = len(pts)
        # Pairwise distances
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        diam = d.max()
        if diam <= 1e-15:
            return 0.0
        # Scale to diameter 2
        scaled_d = d / (diam / 2.0)
        # Product over ordered pairs (i != j)
        mask = ~np.eye(n, dtype=bool)
        # Use log to avoid underflow
        # Note: scaled_d[mask] contains distances for all i != j
        # If any distance is 0, log is -inf, score is 0
        if np.any(scaled_d[mask] <= 0):
            return 0.0
        log_prod = np.sum(np.log(scaled_d[mask]))
        # Score = exp(log_prod) / n^n
        log_score = log_prod - n * math.log(n)
        # Avoid overflow/underflow in exp
        if log_score > 700:
            return float('inf')
        if log_score < -700:
            return 0.0
        return math.exp(log_score)
    
    best_score = score_points(reg_pts)
    best_pts = reg_pts
    
    # Symmetric parametrization:
    # 4 points in upper half plane (y >= 0), and their reflections.
    # Variables: x1, y1, x2, y2, x3, y3, x4, y4
    # We restrict y_i >= 0.
    
    def neg_log_score_sym(params):
        # params: 8 floats
        x1, y1, x2, y2, x3, y3, x4, y4 = params
        
        # Ensure y >= 0 for symmetry (optimization might push below, we reflect)
        # Actually, it's better to let the optimizer work on unconstrained space
        # and just use the absolute value or constrain it.
        # Let's use abs(y) to maintain symmetry if it crosses 0.
        y1 = abs(y1)
        y2 = abs(y2)
        y3 = abs(y3)
        y4 = abs(y4)
        
        pts = [
            [x1, y1], [x2, y2], [x3, y3], [x4, y4],
            [x1, -y1], [x2, -y2], [x3, -y3], [x4, -y4]
        ]
        
        s = score_points(pts)
        if s <= 0:
            return 1e10
        return -math.log(s)
    
    # Initial guess: Regular octagon
    # Angles: 0, 45, 90, 135, 180, 225, 270, 315
    # Upper half: 0, 45, 90, 135 (and 180 is on axis, can be represented as y=0)
    # Let's map the 4 upper points of the regular octagon:
    # p0: (1, 0) -> x1=1, y1=0
    # p1: (cos45, sin45) -> x2=0.707, y2=0.707
    # p2: (0, 1) -> x3=0, y3=1
    # p3: (-cos45, sin45) -> x4=-0.707, y4=0.707
    # The lower points are reflections.
    # Note: p4 (180 deg) is (-1, 0). In our symmetric set, do we have it?
    # Our set:
    # (1,0), (0.707, 0.707), (0,1), (-0.707, 0.707)
    # (1,0), (0.707, -0.707), (0,-1), (-0.707, -0.707)
    # This set has duplicates! (1,0) appears twice.
    # The regular octagon has 8 distinct points.
    # My symmetric parametrization creates 8 points, but if y_i=0, the point and its reflection are identical.
    # This leads to zero distance and score 0.
    # So, we must ensure that points with y_i=0 are not duplicated in the set.
    # Or, better, we should parameterize such that we don't force duplicates.
    
    # Alternative Symmetric Parametrization:
    # Assume the set is invariant under reflection across x-axis.
    # This implies that if (x,y) is in the set, (x,-y) is in the set.
    # Points on the axis (y=0) are their own reflections.
    # Let k be the number of points on the axis. k must be even? No, just any number.
    # The remaining 8-k points form (8-k)/2 pairs.
    # For n=8, possible k: 0, 2, 4, 6, 8.
    # Regular octagon has k=2 (points at 0 and 180 deg).
    # Let's try k=2.
    # Points: (x_a, 0), (x_b, 0) and 3 pairs (x1,y1), (x2,y2), (x3,y3) with y_i > 0.
    # Total points: 2 + 6 = 8.
    # Variables: x_a, x_b, x1, y1, x2, y2, x3, y3. (8 variables)
    
    def neg_log_score_sym2(params):
        x_a, x_b, x1, y1, x2, y2, x3, y3 = params
        
        # Enforce y > 0 for pairs to avoid duplicates with axis?
        # If y becomes 0, we have a duplicate with an axis point only if x matches.
        # But generally, if y_i -> 0, the distance between (x_i, y_i) and (x_i, -y_i) goes to 0.
        # The product goes to 0. So the optimizer will avoid y_i=0 unless it's a valid limit.
        # However, if we want to explore configurations where a pair merges into an axis point,
        # this parametrization might struggle.
        # Let's just use abs(y) to be safe, but distinctness is key.
        
        y1 = abs(y1)
        y2 = abs(y2)
        y3 = abs(y3)
        
        pts = [
            [x_a, 0], [x_b, 0],
            [x1, y1], [x1, -y1],
            [x2, y2], [x2, -y2],
            [x3, y3], [x3, -y3]
        ]
        
        s = score_points(pts)
        if s <= 0:
            return 1e10
        return -math.log(s)
    
    # Initial guess for sym2: Regular octagon
    # Axis points: (1,0) and (-1,0)
    # Pairs:
    # (cos45, sin45) and (cos45, -sin45) -> x1=cos45, y1=sin45
    # (cos90, sin90) and (cos90, -sin90) -> x2=0, y2=1
    # (cos135, sin135) and (cos135, -sin135) -> x3=-cos45, y3=sin45
    
    x0_sym2 = [1.0, -1.0, 
               math.cos(math.pi/4), math.sin(math.pi/4),
               0.0, 1.0,
               -math.cos(math.pi/4), math.sin(math.pi/4)]
    
    # Run optimization from multiple starts
    rng = np.random.RandomState(42)
    
    # Strategy 1: Local optimization from regular octagon with small perturbations
    for i in range(20):
        if time.time() > _DEADLINE - 10:
            break
        
        if i == 0:
            x_start = np.array(x0_sym2)
        else:
            # Perturb the regular octagon configuration
            noise = rng.randn(8) * 0.2
            x_start = np.array(x0_sym2) + noise
        
        try:
            res = minimize(neg_log_score_sym2, x_start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.success or res.fun < -math.log(best_score):
                cand_pts = []
                x_a, x_b, x1, y1, x2, y2, x3, y3 = res.x
                y1 = abs(y1); y2 = abs(y2); y3 = abs(y3)
                cand_pts = [
                    [x_a, 0], [x_b, 0],
                    [x1, y1], [x1, -y1],
                    [x2, y2], [x2, -y2],
                    [x3, y3], [x3, -y3]
                ]
                
                # Check distinctness
                # If points are too close, score is low.
                cand_score = score_points(cand_pts)
                
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
                    
        except Exception:
            pass

    # Strategy 2: Try k=0 (no points on axis)
    # 4 pairs: (x1,y1)...(x4,y4) with y_i > 0
    def neg_log_score_sym0(params):
        x1, y1, x2, y2, x3, y3, x4, y4 = params
        y1 = abs(y1); y2 = abs(y2); y3 = abs(y3); y4 = abs(y4)
        
        pts = [
            [x1, y1], [x1, -y1],
            [x2, y2], [x2, -y2],
            [x3, y3], [x3, -y3],
            [x4, y4], [x4, -y4]
        ]
        
        s = score_points(pts)
        if s <= 0:
            return 1e10
        return -math.log(s)
    
    # Initial guess for sym0: Perturbed regular octagon
    # Shift angles slightly so no point is on axis
    # Regular octagon angles: 0, 45, 90, 135, 180, 225, 270, 315
    # Shift by 22.5 deg: 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5
    # Upper half: 22.5, 67.5, 112.5, 157.5
    ang_shift = math.pi / 8 # 22.5 deg
    x0_sym0 = []
    for k in range(4):
        a = ang_shift + k * math.pi / 4
        x0_sym0.append(math.cos(a))
        x0_sym0.append(math.sin(a))
        
    for i in range(20):
        if time.time() > _DEADLINE - 5:
            break
        
        if i == 0:
            x_start = np.array(x0_sym0)
        else:
            noise = rng.randn(8) * 0.2
            x_start = np.array(x0_sym0) + noise
        
        try:
            res = minimize(neg_log_score_sym0, x_start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.success or res.fun < -math.log(best_score):
                cand_pts = []
                x1, y1, x2, y2, x3, y3, x4, y4 = res.x
                y1 = abs(y1); y2 = abs(y2); y3 = abs(y3); y4 = abs(y4)
                cand_pts = [
                    [x1, y1], [x1, -y1],
                    [x2, y2], [x2, -y2],
                    [x3, y3], [x3, -y3],
                    [x4, y4], [x4, -y4]
                ]
                
                cand_score = score_points(cand_pts)
                
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
                    
        except Exception:
            pass
            
    # Final check and return
    # Ensure format is list of lists of floats
    result = [[float(p[0]), float(p[1])] for p in best_pts]
    return result

# EVOLVE-BLOCK-END
