"""Improved packing of 50 equal circles in a unit circle."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 50

def _score(pts):
    """Compute the packing radius (score) for a set of points."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(dist_boundary)
    # Pairwise distances
    tree = cKDTree(pts)
    # Use query_pairs to get all pairs within a certain distance, but we need min
    # Actually, let's just compute all pairwise distances for N=50 (1225 pairs)
    # This is fast enough
    from scipy.spatial.distance import pdist
    d = pdist(pts, metric='euclidean')
    min_pair = np.min(d) if len(d) > 0 else np.inf
    return min(min_boundary, min_pair / 2.0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_pts = None
    best_score = 0.0
    
    # Strategy: Start with a good initial configuration and use simulated annealing
    # with multiple restarts
    
    # Initial configuration: hexagonal packing
    def hex_init():
        pts = []
        r = 0.12
        dx = r * 2
        dy = r * np.sqrt(3)
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * dx + (j % 2) * dx / 2
                y = j * dy
                if x*x + y*y <= (1 - r)**2:
                    pts.append([x, y])
        pts = np.array(pts)
        # Shuffle and take first N
        idx = np.random.permutation(len(pts))
        return pts[idx[:N]].copy()
    
    def random_init():
        """Random points inside circle of radius 0.85"""
        pts = []
        for _ in range(N):
            r = np.sqrt(np.random.random()) * 0.85
            theta = np.random.random() * 2 * np.pi
            pts.append([r * np.cos(theta), r * np.sin(theta)])
        return np.array(pts)
    
    def anneal(pts, time_budget):
        """Simulated annealing to improve packing."""
        t_end = time.time() + time_budget
        current = pts.copy()
        current_score = _score(current)
        best_local = current.copy()
        best_local_score = current_score
        
        T = 0.01
        T_min = 0.00001
        cooling = 0.995
        step_size = 0.05
        
        while time.time() < t_end:
            # Propose a move: perturb one random point
            i = np.random.randint(N)
            old_x, old_y = current[i]
            dx = (np.random.random() - 0.5) * 2 * step_size
            dy = (np.random.random() - 0.5) * 2 * step_size
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Keep inside container (with margin)
            if new_x*new_x + new_y*new_y > (1 - 0.01)**2:
                continue
            
            current[i] = [new_x, new_y]
            new_score = _score(current)
            
            # Accept if better, or with probability exp((new-old)/T)
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local = current.copy()
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(delta / T):
                    current_score = new_score
                else:
                    current[i] = [old_x, old_y]
            
            T = max(T * cooling, T_min)
            step_size = max(step_size * 0.999, 0.001)
        
        return best_local, best_local_score
    
    # Try multiple strategies
    # 1. Hexagonal init + anneal
    pts = hex_init()
    pts, sc = anneal(pts, 30)
    if sc > best_score:
        best_score = sc
        best_pts = pts.copy()
    
    # 2. Random init + anneal (multiple restarts)
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        pts = random_init()
        pts, sc = anneal(pts, 15)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
    
    # 3. If we have time, do a final refinement on the best
    if time.time() < deadline - 10 and best_pts is not None:
        # Use a more careful approach: optimize using scipy
        from scipy.optimize import minimize
        
        def neg_score(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        # Multi-start local optimization
        x0 = best_pts.flatten()
        try:
            res = minimize(neg_score, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'gtol': 1e-8})
            if -res.fun > best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(N, 2).copy()
        except:
            pass
        
        # Try a few more local optimizations from perturbed best
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            perturbed = best_pts + np.random.randn(N, 2) * 0.01
            # Clip to inside circle
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms > 0.99
            perturbed[mask] *= 0.99 / norms[mask]
            x0 = perturbed.flatten()
            try:
                res = minimize(neg_score, x0, method='L-BFGS-B',
                              options={'maxiter': 300, 'gtol': 1e-8})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_pts = res.x.reshape(N, 2).copy()
            except:
                pass
    
    # Ensure validity: project any points outside the circle back inside
    if best_pts is not None:
        norms = np.linalg.norm(best_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            best_pts[mask] *= (1.0 - 1e-6) / norms[mask]
    
    if best_pts is None:
        # Fallback: hexagonal
        best_pts = hex_init()
    
    return best_pts
