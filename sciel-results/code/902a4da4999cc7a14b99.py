"""
Strategy: Algebraic Parameterization via Conjugate Symmetry.
The problem of maximizing the product of distances for 8 points in the plane
is equivalent to maximizing the discriminant of the polynomial whose roots
are the complex coordinates of the points. For real-coefficient polynomials,
roots come in conjugate pairs. We assume the optimal configuration has 4
conjugate pairs (which includes the regular octagon and many symmetric
deformations). This reduces the parameterization from 16 real numbers (8 points)
to 8 real numbers (coefficients of 4 quadratic factors).
We optimize these 8 coefficients directly using L-BFGS-B with multiple starts.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb

def _score_from_points(points):
    """
    Compute Delta / n^n for 8 points.
    """
    n = len(points)
    pts = np.array(points, dtype=float)
    if pts.shape[1] != 2:
        return 0.0
        
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    
    # Scale to diameter 2
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = dists[mask]
    scaled_dists = ordered_dists / diam * 2.0
    
    # Check for distinct points
    if np.any(scaled_dists < 1e-12):
        return 0.0
        
    # Product of ordered pair distances
    log_delta = np.sum(np.log(scaled_dists))
    log_score = log_delta - n * np.log(n)
    
    return np.exp(log_score)

def _coeffs_to_points(coeffs):
    """
    Convert 8 coefficients (4 pairs of (a, b, c) for quadratics z^2 + a*z + b? 
    No, let's use real quadratic factors: (z - (u_k + i v_k))(z - (u_k - i v_k)) 
    = z^2 - 2u_k z + (u_k^2 + v_k^2).
    So each pair is defined by center (u_k) and squared radius (r_k^2).
    However, to ensure roots are distinct and real-coefficient polynomial is valid,
    we parameterize by 4 centers (u_k) and 4 radii (v_k).
    Actually, simpler: just define 4 complex centers c_k = u_k + i v_k.
    The points are c_k and conj(c_k).
    This is 8 parameters: u_0..u_3, v_0..v_3.
    """
    # coeffs: [u0, v0, u1, v1, u2, v2, u3, v3]
    points = []
    for i in range(4):
        u = coeffs[2*i]
        v = coeffs[2*i+1]
        points.append([u, v])
        points.append([u, -v])
        
    return np.array(points, dtype=float)

def _neg_score_from_coeffs(coeffs):
    """
    Objective: Minimize negative log-score.
    """
    pts = _coeffs_to_points(coeffs)
    # If points are too close, penalize
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 1e-8:
        return 1e6 # Large penalty
    
    score = _score_from_points(pts)
    if score <= 0:
        return 1e6
    return -np.log(score)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial guess: Regular Octagon
    # Regular octagon has vertices at angles 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
    # Conjugate pairs:
    # k=0: (1, 0) and (1, 0) -> Wait, regular octagon on unit circle:
    # z_k = exp(i * k * pi/4).
    # Pairs: 
    # k=0: 1, k=4: -1 (not conjugate unless v=0)
    # k=1: e^{i pi/4}, k=7: e^{-i pi/4} -> Conjugate pair. Center u=cos(pi/4), v=sin(pi/4).
    # k=2: i, k=6: -i -> Conjugate pair. Center u=0, v=1.
    # k=3: e^{i 3pi/4}, k=5: e^{-i 3pi/4} -> Conjugate pair. Center u=cos(3pi/4), v=sin(3pi/4).
    # k=0 and k=4 are on the real axis. They are their own conjugates.
    # My parameterization assumes 4 distinct conjugate pairs (z, z*). 
    # If v=0, the pair is (u, 0) and (u, 0), which are identical points. 
    # The problem requires 8 DISTINCT points.
    # The regular octagon has 2 points on the real axis (v=0) and 3 pairs off-axis?
    # No, regular octagon:
    # 0: (1, 0)
    # 1: (0.707, 0.707)
    # 2: (0, 1)
    # 3: (-0.707, 0.707)
    # 4: (-1, 0)
    # 5: (-0.707, -0.707)
    # 6: (0, -1)
    # 7: (0.707, -0.707)
    
    # Conjugate pairs:
    # (1,0) and (-1,0) are NOT conjugates of each other. (1,0)* = (1,0).
    # So (1,0) is a self-conjugate root.
    # My parameterization `z` and `z*` produces two points at `(u,v)` and `(u,-v)`.
    # If v=0, it produces `(u,0)` and `(u,0)` -> Duplicate points.
    # To handle the regular octagon, I need to allow for real roots.
    # A real root is just a point on the x-axis.
    # Let's modify the parameterization to be more flexible or handle the v=0 case carefully.
    # If v is very small, the two points are very close. The score will drop due to the distance product term becoming small.
    # The regular octagon has v=0 for two points. This is a singular point in my parameterization if I strictly enforce pairs.
    # However, the optimal solution for n=8 is known to be the regular octagon (or close to it).
    # If the true optimum is the regular octagon, my parameterization might struggle near v=0 because the gradient might be ill-defined or the function might be flat/penalized.
    
    # Alternative: Use 8 parameters but constrain them to be conjugate symmetric?
    # Let's try a different parameterization:
    # 4 complex numbers c_k. Points are c_k and conj(c_k).
    # If c_k is real, c_k == conj(c_k). We get duplicate points.
    # We must avoid duplicates.
    # Perhaps the optimal configuration for n=8 is NOT the regular octagon?
    # The prompt says "regular octagon scores exactly 1". "Scores above 1 beat the regular octagon".
    # "The paper's constructions show this is possible".
    # This implies the regular octagon is NOT the global maximum.
    # Therefore, the global maximum likely has NO real roots (all v_k != 0), or at least allows for a better arrangement.
    # If the global max has no real roots, my parameterization is valid and smooth.
    
    # Let's proceed with the assumption that the global max has 4 non-real conjugate pairs.
    # Initial guess: Perturb the regular octagon slightly to move points off the real axis.
    # Regular octagon points:
    # P0: (1, 0) -> Move to (1, eps) and (1, -eps)? No, that changes the set.
    # Let's start with a configuration close to the regular octagon but with v_k > 0.
    # P1, P7 pair: u=cos(pi/4), v=sin(pi/4)
    # P2, P6 pair: u=0, v=1
    # P3, P5 pair: u=cos(3pi/4), v=sin(3pi/4)
    # P0, P4 are on real axis. We can treat them as a pair with u=0, v=eps?
    # If we set u0=0, v0=eps, we get (0, eps) and (0, -eps).
    # This is close to (0,1) and (0,-1) if eps=1.
    # So we can map the 4 pairs to:
    # Pair 1: u=cos(pi/4), v=sin(pi/4)  (approx P1, P7)
    # Pair 2: u=0, v=1                  (approx P2, P6)
    # Pair 3: u=cos(3pi/4), v=sin(3pi/4)(approx P3, P5)
    # Pair 4: u=0, v=1                  (Wait, P0 and P4 are at 1 and -1. They are not a conjugate pair in the sense of u,v. 
    #        But (1,0) and (-1,0) are symmetric about y-axis.
    #        My parameterization generates (u,v) and (u,-v). 
    #        To get (1,0) and (-1,0), I would need two pairs: (1,0) and (-1,0).
    #        But (1,0) is a duplicate if v=0.
    #        So I cannot represent the regular octagon exactly with 4 non-degenerate pairs.
    #        I can represent a configuration CLOSE to it by setting:
    #        Pair 4: u=0, v=1 (This gives (0,1) and (0,-1)).
    #        But I already used Pair 2 for (0,1) and (0,-1).
    #        
    #        Let's look at the 8 points of regular octagon again:
    #        1, e^{i pi/4}, i, e^{i 3pi/4}, -1, e^{i 5pi/4}, -i, e^{i 7pi/4}
    #        Conjugate pairs:
    #        (e^{i pi/4}, e^{-i pi/4}) -> u=sqrt(2)/2, v=sqrt(2)/2
    #        (i, -i) -> u=0, v=1
    #        (e^{i 3pi/4}, e^{-i 3pi/4}) -> u=-sqrt(2)/2, v=sqrt(2)/2
    #        (1, 1) and (-1, -1) -> These are real.
    #        
    #        To approximate this with 4 pairs (u_k, v_k) -> (u_k, v_k), (u_k, -v_k):
    #        We can set:
    #        k=0: u=sqrt(2)/2, v=sqrt(2)/2
    #        k=1: u=0, v=1
    #        k=2: u=-sqrt(2)/2, v=sqrt(2)/2
    #        k=3: u=0, v=1  <-- This duplicates the points from k=1!
    #        
    #        So I cannot represent the regular octagon with 4 DISTINCT pairs of non-real conjugates.
    #        I need 3 non-real pairs and 2 real points.
    #        My parameterization forces 4 pairs.
    #        If I force 4 pairs, I am restricted to configurations with 8 non-real points (4 conjugate pairs).
    #        The regular octagon has 2 real points.
    #        If the global max is the regular octagon, this parameterization is invalid for the global max.
    #        BUT, the prompt says scores > 1 are possible.
    #        Maybe the global max has 0 real points?
    #        Let's try to optimize in this 4-pair subspace.
    #        Initial guess:
    #        u0 = 0.7, v0 = 0.7
    #        u1 = 0.0, v1 = 1.0
    #        u2 = -0.7, v2 = 0.7
    #        u3 = 0.0, v3 = 1.0  (This creates duplicates with k=1. Bad start.)
    #        
    #        Let's spread them out:
    #        u0 = 0.8, v0 = 0.5
    #        u1 = 0.2, v1 = 0.9
    #        u2 = -0.2, v2 = 0.9
    #        u3 = -0.8, v3 = 0.5
    #        
    #        This creates 8 distinct points.
    
    bounds = [(-2.0, 2.0)] * 8
    
    best_score = 0.0
    best_params = None
    
    # Multi-start L-BFGS-B
    n_starts = 50
    for i in range(n_starts):
        if time.time() > deadline - 2.0:
            break
            
        if i == 0:
            # Structured start
            x0 = np.array([0.8, 0.5, 0.2, 0.9, -0.2, 0.9, -0.8, 0.5])
        else:
            # Random start within bounds
            x0 = np.random.uniform(-1.5, 1.5, 8)
            # Ensure v is not too close to 0 to avoid duplicates in the pair
            # v_k is at index 1, 3, 5, 7
            for j in [1, 3, 5, 7]:
                if abs(x0[j]) < 0.1:
                    x0[j] = 0.1 if np.random.rand() > 0.5 else -0.1
        
        try:
            res = minimize(
                _neg_score_from_coeffs, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            final_neg_log_score = res.fun
            if final_neg_log_score < 1e5: # Valid score
                score = np.exp(-final_neg_log_score)
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to regular octagon if optimization failed
        # We can't represent regular octagon exactly, so we return a valid set close to it
        # Or just return the regular octagon directly, which scores 1.0
        octagon = np.array([
            [np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)
        ])
        return [[float(p[0]), float(p[1])] for p in octagon]
    
    points = _coeffs_to_points(best_params)
    
    # Sanity check
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-6:
        octagon = np.array([
            [np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)
        ])
        points = octagon
        
    return [[float(p[0]), float(p[1])] for p in points]
