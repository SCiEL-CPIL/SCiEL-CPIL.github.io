import numpy as np
from scipy.optimize import minimize
import itertools

N = 11

def construct():
    # Vertices of the equilateral triangle
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    # Precompute combinations of indices for area calculation
    idxs = list(itertools.combinations(range(N), 3))
    
    def get_points(z):
        """
        Map unconstrained variables z (shape 2N) to points in the triangle.
        z is flattened [x1, y1, ..., xN, yN].
        Uses softmax-like mapping to ensure u, v > 0 and u+v < 1.
        """
        z = z.reshape(N, 2)
        # Shift to avoid overflow in exp
        # x, y are the unnormalized weights for B and C.
        # Weight for A is 1.
        x = z[:, 0]
        y = z[:, 1]
        
        # For numerical stability, subtract max
        # We want u = exp(x) / (1 + exp(x) + exp(y))
        # v = exp(y) / (1 + exp(x) + exp(y))
        # 1-u-v = 1 / (1 + exp(x) + exp(y))
        
        ex = np.exp(x - np.max(x)) # Not quite right for joint max, but let's just use stable softmax
        # Actually, simpler stable log-sum-exp trick:
        # Let M = max(0, x, y)
        # u = exp(x-M) / (exp(0-M) + exp(x-M) + exp(y-M))
        
        M = np.maximum(0, np.maximum(x, y))
        exp_x = np.exp(x - M)
        exp_y = np.exp(y - M)
        exp_0 = np.exp(0 - M)
        
        denom = exp_0 + exp_x + exp_y
        u = exp_x / denom
        v = exp_y / denom
        
        # P = A + u(B-A) + v(C-A) = u B + v C
        pts = np.zeros((N, 2))
        pts[:, 0] = u * B[0] + v * C[0]
        pts[:, 1] = u * B[1] + v * C[1]
        
        return pts

    def calc_min_area(pts, beta=1.0):
        """
        Calculate the soft-min of all triangle areas.
        """
        areas = np.zeros(len(idxs))
        for idx, (i, j, k) in enumerate(idxs):
            p1 = pts[i]
            p2 = pts[j]
            p3 = pts[k]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            areas[idx] = 0.5 * abs(cross)
            
        # Softmin: -1/beta * log(sum(exp(-beta * areas)))
        # We want to maximize min(areas), so we minimize -softmin(areas)
        # Or we can just return the softmin and maximize it.
        
        # To avoid underflow/overflow in exp:
        min_a = np.min(areas)
        shifted = areas - min_a
        # softmin approx min_a + (1/beta) * log(sum(exp(-beta * shifted)))
        # If beta is large, this is close to min_a.
        
        # For optimization, we want a smooth function.
        # Let's use the standard log-sum-exp form for softmin.
        # softmin(a) = -1/beta * logsumexp(-beta * a)
        # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
        
        x = -beta * areas
        m = np.max(x)
        lse = m + np.log(np.sum(np.exp(x - m)))
        softmin_val = -lse / beta
        
        return softmin_val

    def objective(z, beta=1.0):
        pts = get_points(z)
        val = calc_min_area(pts, beta)
        return -val # Minimize negative softmin

    # Optimization loop with increasing beta (annealing)
    best_z = None
    best_score = -float('inf')
    
    # Random initializations
    np.random.seed(42)
    
    # Start with a few random restarts
    for restart in range(5):
        # Initialize z with random values
        z0 = np.random.randn(2 * N) * 0.1
        
        # Annealing schedule for beta
        betas = [1.0, 5.0, 20.0, 100.0]
        
        current_z = z0.copy()
        
        for beta in betas:
            # Use L-BFGS-B
            # We don't have bounds on z, so L-BFGS-B is fine.
            res = minimize(
                objective, 
                current_z, 
                method='L-BFGS-B', 
                args=(beta,),
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            current_z = res.x
            
        # Evaluate the final result with high beta (close to true min)
        final_pts = get_points(current_z)
        # Calculate true min area for scoring
        true_min_a = float('inf')
        for i, j, k in idxs:
            p1, p2, p3 = final_pts[i], final_pts[j], final_pts[k]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            area = 0.5 * abs(cross)
            if area < true_min_a:
                true_min_a = area
        
        if true_min_a > best_score:
            best_score = true_min_a
            best_z = current_z.copy()
            best_pts = final_pts.copy()

    # If no best found (shouldn't happen), fallback
    if best_pts is None:
         # Random points
         np.random.seed(123)
         best_pts = np.random.rand(N, 2)
         # Project to triangle? Just return random, but ensure they are inside.
         # The get_points function ensures they are inside.
         # Let's just use the last generated points from the last restart
         # Or generate new ones
         z0 = np.random.randn(2 * N)
         best_pts = get_points(z0)

    return best_pts
