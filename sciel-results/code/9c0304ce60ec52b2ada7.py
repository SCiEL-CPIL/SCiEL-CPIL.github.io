import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Exploit symmetries to reduce dimensionality from 42 to 24.
       - Translation: Fix center of mass to 0 (removed by normalization).
       - Scaling: Normalize d_max to 1 (handled in objective).
       - Rotation: Represent points via Gram matrix (dot products) or 
         directly optimize pairwise distances and reconstruct? 
         Actually, optimizing pairwise distances is tricky due to Euclidean constraints.
         
    Alternative Structural Change:
    Use a "Soft-Min" objective function that is differentiable (or smooth enough for NM)
    and optimize the raw coordinates directly but with a much more robust initialization 
    and a specific focus on the "bottleneck" distances.
    
    Wait, the directive says "MATERIALLY DIFFERENT".
    Current: DE on 42D.
    New: 
    1. Use `scipy.optimize.differential_evolution` (global) but with `polish=True`? 
       No, DE is already used.
       
    Let's try a **Landscape Smoothing** approach combined with **Multi-Start Nelder-Mead**.
    The objective function $f(x) = \min_{i,j} d_{ij}^2 / \max_{i,j} d_{ij}^2$ is non-smooth (min/max).
    We can approximate $\min$ and $\max$ using soft-min/soft-max (log-sum-exp variants) to make
    the landscape smoother, allowing gradient-based or Nelder-Mead to navigate better.
    
    However, Nelder-Mead doesn't use gradients. But smooth landscapes help NM avoid getting
    stuck on "ridges" created by the sharp kinks of min/max.
    
    Let's implement:
    1. A smooth objective: $S(x) = (\text{softmin}(d^2) / \text{softmax}(d^2))^2$
       where softmin/softmax use a temperature parameter $\tau$.
    2. Multi-start Nelder-Mead from diverse symmetric seeds (Icosahedron variants, Simplex, etc.).
    3. Final polish with the exact non-smooth objective.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620
    
    def get_score(pts):
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        if d_max_sq < 1e-12:
            return 0.0
        return d_min_sq / d_max_sq

    def smooth_score(pts, tau=0.05):
        """
        Smooth approximation of the score.
        softmin(a) = -log(sum(exp(-a/tau))) * tau
        softmax(a) = log(sum(exp(a/tau))) * tau
        """
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        
        # Numerical stability for log-sum-exp
        # softmin
        m_min = np.min(dists)
        softmin_val = m_min - tau * np.log(np.sum(np.exp(-(dists - m_min) / tau)))
        
        # softmax
        m_max = np.max(dists)
        softmax_val = m_max + tau * np.log(np.sum(np.exp((dists - m_max) / tau)))
        
        if softmax_val < 1e-12:
            return 0.0
            
        # We want to maximize (softmin / softmax)^2
        # To use minimize, we return negative
        ratio = softmin_val / softmax_val
        return -ratio * ratio

    def objective_wrapper(x, tau):
        pts = x.reshape(n, d)
        return smooth_score(pts, tau)

    # Generate diverse initial points
    def generate_seed(idx):
        if idx == 0:
            # Icosahedron + 2 points
            phi = (1 + np.sqrt(5)) / 2
            ico = np.array([
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
            ])
            norms = np.linalg.norm(ico, axis=1, keepdims=True)
            ico = ico / norms
            # Add 2 points at opposite poles of a tetrahedron inscribed in icosahedron?
            # Or just random on sphere
            p1 = np.array([0, 0, 1])
            p2 = np.array([0, 0, -1])
            pts = np.vstack([ico, [p1], [p2]])
            return pts.flatten()
        elif idx == 1:
            # Simplex-like: 4 vertices of tetrahedron, then subdivide?
            # Just random uniform in sphere
            pts = np.random.randn(n, d)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            return pts.flatten()
        elif idx == 2:
            # Cubic lattice subset
            pts = np.random.randint(-1, 2, (n, d)).astype(float)
            # Ensure distinct
            for i in range(n):
                while np.sum((pts[:i] == pts[i]).all(axis=1)) > 0:
                    pts[i] = np.random.randint(-1, 2, d).astype(float)
            return pts.flatten()
        else:
            # Random Gaussian
            return np.random.randn(n * d)

    best_pts = None
    best_score = -1.0
    
    num_starts = 50
    tau = 0.05 # Temperature for smoothing
    
    for k in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        x0 = generate_seed(k)
        
        # Normalize initial points to have d_max = 1 to help scaling
        pts0 = x0.reshape(n, d)
        d0 = pdist(pts0, metric='sqeuclidean')
        if np.max(d0) > 1e-6:
            pts0 = pts0 / np.sqrt(np.max(d0))
            x0 = pts0.flatten()
        
        # Nelder-Mead on the smoothed objective
        # Use a tight tolerance to ensure convergence
        res = minimize(
            objective_wrapper, 
            x0, 
            args=(tau,),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if time.time() > deadline - 5:
            break
            
        x_opt = res.x
        pts_opt = x_opt.reshape(n, d)
        
        # Evaluate with the TRUE objective
        score = get_score(pts_opt)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
            
            # If we get a very high score, we can break early
            if best_score > 0.7:
                break
                
        # Optional: Do a second pass with lower temperature (sharper) starting from x_opt
        # to refine the "kinks"
        if k % 2 == 0: # Do this for half the starts to save time
            tau_fine = 0.01
            res_fine = minimize(
                objective_wrapper, 
                x_opt, 
                args=(tau_fine,),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            if time.time() > deadline - 5:
                continue
                
            x_fine = res_fine.x
            pts_fine = x_fine.reshape(n, d)
            score_fine = get_score(pts_fine)
            
            if score_fine > best_score:
                best_score = score_fine
                best_pts = pts_fine.copy()
                if best_score > 0.7:
                    break

    if best_pts is None:
        # Fallback: Return a valid random configuration
        best_pts = np.random.randn(n, d)
        
    # Final sanity check and normalization (though metric is scale invariant, 
    # it's good to return "clean" numbers)
    # Center the points
    best_pts -= np.mean(best_pts, axis=0)
    
    return best_pts
