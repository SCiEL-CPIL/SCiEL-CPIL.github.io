import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def smooth_ratio_objective(x, alpha=10.0):
    """
    Computes a smooth approximation of the max/min squared distance ratio.
    Uses log-sum-exp trick for numerical stability.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Numerical stability for log-sum-exp
    # max_d approx: (1/alpha) * log(sum(exp(alpha * d^2)))
    # min_d approx: -(1/alpha) * log(sum(exp(-alpha * d^2)))
    
    # To avoid overflow, subtract max before exp
    max_sq = np.max(sq_dists)
    min_sq = np.min(sq_dists)
    
    # If alpha is large, exp(alpha * (d^2 - max_sq)) is safe.
    # sum(exp(alpha * d^2)) = exp(alpha * max_sq) * sum(exp(alpha * (d^2 - max_sq)))
    # log(sum(exp(alpha * d^2))) = alpha * max_sq + log(sum(exp(alpha * (d^2 - max_sq))))
    
    term_max = alpha * max_sq + np.log(np.sum(np.exp(alpha * (sq_dists - max_sq))))
    
    # For min:
    # sum(exp(-alpha * d^2)) = exp(-alpha * min_sq) * sum(exp(-alpha * (d^2 - min_sq)))
    # log(sum(exp(-alpha * d^2))) = -alpha * min_sq + log(sum(exp(-alpha * (d^2 - min_sq))))
    
    term_min = -alpha * min_sq + np.log(np.sum(np.exp(-alpha * (sq_dists - min_sq))))
    
    # Ratio approx = (term_max / alpha) / (-term_min / alpha) = term_max / -term_min
    # Actually, the alpha cancels out in the ratio if we just take the logs?
    # d_max ~ term_max/alpha
    # d_min ~ -term_min/alpha
    # ratio ~ term_max / -term_min
    
    if term_min == 0:
        return 1e10
    return term_max / (-term_min)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin
    
    best_score = np.inf
    best_points = None
    rng = np.random.default_rng(42)
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        if not np.all(np.isfinite(pts)):
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
            
    def local_optimize(x0, max_iter=500):
        """
        Perform local optimization using a sequence of increasing alphas
        to anneal into a sharp minimum.
        """
        current_x = x0.copy()
        alphas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        
        for alpha in alphas:
            if time.time() - start_time > timeout - 10:
                break
                
            def obj(x, a=alpha):
                return smooth_ratio_objective(x, a)
                
            try:
                res = minimize(
                    obj,
                    current_x,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.x is not None:
                    current_x = res.x
            except Exception:
                pass
                
        return current_x

    # Initial Candidates
    candidates = []
    
    # 1. Fibonacci Sphere
    def fibonacci_sphere(n):
        points = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            z = 1.0 - 2.0 * i / (n - 1)
            r = np.sqrt(max(0.0, 1.0 - z * z))
            theta = golden_angle * i
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
        
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 2. Random Initializations
    for _ in range(10):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Icosahedron + Poles
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())

    # Process Initial Candidates
    for x0 in candidates:
        if time.time() - start_time > timeout - 20:
            break
        x_opt = local_optimize(x0)
        update_best(x_opt)
        
    # Basin Hopping Loop
    # We will perform random perturbations and re-optimize.
    # This helps escape local minima.
    
    n_hops = 200 # Max number of hops, limited by time
    
    for i in range(n_hops):
        if time.time() - start_time > timeout - 10:
            break
            
        if best_points is None:
            break
            
        # Perturb the best points
        # Move 1 to 3 random points by a random amount
        n_moves = rng.integers(1, 4)
        indices = rng.choice(N, size=n_moves, replace=False)
        perturbations = rng.normal(scale=0.5, size=(n_moves, D))
        
        x_new = best_points.flatten().copy()
        x_new.reshape(N, D)[indices] += perturbations
        
        # Local optimization on the perturbed point
        x_opt = local_optimize(x_new, max_iter=200)
        update_best(x_opt)
        
    # Final Polish
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            # Use a very high alpha for final polish to sharpen the min/max
            def final_obj(x):
                return smooth_ratio_objective(x, alpha=100.0)
            
            res = minimize(
                final_obj,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None:
                update_best(res.x)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = normalize_points(candidates[0].reshape(N, D))
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
