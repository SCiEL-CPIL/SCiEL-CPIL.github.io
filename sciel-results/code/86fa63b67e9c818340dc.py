import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    timeout = 600.0
    
    def compute_score(pts):
        if pts.shape != (16, 2):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_pts(pts):
        pts = pts - np.mean(pts, axis=0)
        dm = np.max(pdist(pts))
        if dm > 0:
            pts = pts / dm
        return pts

    # Fekete objective: minimize -sum(log(dist))
    def obj_fekete(x):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        dists = np.maximum(dists, 1e-10)
        return -np.sum(np.log(dists))

    def grad_fekete(x):
        pts = x.reshape(-1, 2)
        grad = np.zeros_like(x)
        for i in range(n):
            for j in range(i + 1, n):
                vec = pts[i] - pts[j]
                dist_sq = np.dot(vec, vec)
                if dist_sq < 1e-20:
                    vec = np.array([1e-10, 1e-10])
                    dist_sq = np.dot(vec, vec)
                # grad of -log(dist) = - (1/dist) * (1/dist) * vec = -vec / dist^2
                grad[2*i:2*i+2] -= vec / dist_sq
                grad[2*j:2*j+2] += vec / dist_sq
        return grad

    # Soft ratio objective: minimize soft_max / soft_min
    def soft_max(dists, tau=0.01):
        d2 = dists * dists
        m = np.max(d2)
        if m < 1e-15: return 0.0
        log_sum = m / tau + np.log(np.sum(np.exp((d2 - m) / tau)))
        return np.sqrt(tau * log_sum)

    def soft_min(dists, tau=0.01):
        d2 = dists * dists
        m = np.max(d2)
        if m < 1e-15: return 0.0
        log_sum = -m / tau + np.log(np.sum(np.exp((d2 - m) / tau)))
        return -np.sqrt(tau * log_sum)

    def obj_ratio(x, tau=0.01):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        smax = soft_max(dists, tau)
        smin = soft_min(dists, tau)
        if smin < 1e-15 or smax < 1e-15: return 1e10
        return smax / smin

    def grad_ratio(x, tau=0.01):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        smax = soft_max(dists, tau)
        smin = soft_min(dists, tau)
        if smin < 1e-15 or smax < 1e-15: return np.zeros_like(x)
        
        d2 = dists * dists
        m = np.max(d2)
        if m < 1e-15: return np.zeros_like(x)
        
        grad = np.zeros_like(x)
        # dsmax/dx
        exp_d2 = np.exp((d2 - m) / tau)
        sum_exp = np.sum(exp_d2)
        w_max = exp_d2 * dists / (sum_exp * smax)
        
        # dsmin/dx
        d2_neg = -d2
        m_neg = np.max(d2_neg)
        exp_neg = np.exp((d2_neg - m_neg) / tau)
        sum_exp_neg = np.sum(exp_neg)
        Lmin = m_neg / tau + np.log(sum_exp_neg)
        if Lmin > 1e-15:
            sqrt_tau_L = np.sqrt(tau * Lmin)
            w_min = exp_neg * dists / (sum_exp_neg * sqrt_tau_L)
        else:
            w_min = np.zeros_like(dists)
            
        coeff_max = 1.0 / smin
        coeff_min = -smax / (smin * smin)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Need pair index mapping
                pass # This is slow, let's precompute pairs
                
    # Precompute pairs for gradient
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            
    def grad_ratio_fast(x, tau=0.01):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        smax = soft_max(dists, tau)
        smin = soft_min(dists, tau)
        if smin < 1e-15 or smax < 1e-15: return np.zeros_like(x)
        
        d2 = dists * dists
        m = np.max(d2)
        if m < 1e-15: return np.zeros_like(x)
        
        grad = np.zeros_like(x)
        exp_d2 = np.exp((d2 - m) / tau)
        sum_exp = np.sum(exp_d2)
        w_max = exp_d2 * dists / (sum_exp * smax)
        
        d2_neg = -d2
        m_neg = np.max(d2_neg)
        exp_neg = np.exp((d2_neg - m_neg) / tau)
        sum_exp_neg = np.sum(exp_neg)
        Lmin = m_neg / tau + np.log(sum_exp_neg)
        if Lmin > 1e-15:
            sqrt_tau_L = np.sqrt(tau * Lmin)
            w_min = exp_neg * dists / (sum_exp_neg * sqrt_tau_L)
        else:
            w_min = np.zeros_like(dists)
            
        coeff_max = 1.0 / smin
        coeff_min = -smax / (smin * smin)
        
        for idx, (i, j) in enumerate(pairs):
            vec = pts[i] - pts[j]
            w = coeff_max * w_max[idx] + coeff_min * w_min[idx]
            grad[2*i:2*i+2] += w * vec
            grad[2*j:2*j+2] -= w * vec
            
        return grad

    best_score = 0.0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1-5-10 structure
    r1 = 0.5
    r2 = 1.0
    angles1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 10, endpoint=False) + np.pi/10
    p1 = np.column_stack([r1*np.cos(angles1), r1*np.sin(angles1)])
    p2 = np.column_stack([r2*np.cos(angles2), r2*np.sin(angles2)])
    center = np.array([[0,0]])
    init_1510 = np.vstack([center, p1, p2])
    inits.append(init_1510)
    
    # 1-6-9 structure
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 9, endpoint=False) + np.pi/9
    p1 = np.column_stack([0.6*np.cos(angles1), 0.6*np.sin(angles1)])
    p2 = np.column_stack([1.0*np.cos(angles2), 1.0*np.sin(angles2)])
    init_169 = np.vstack([center, p1, p2])
    inits.append(init_169)
    
    # Hexagonal
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5*j
            y = j*np.sqrt(3)/2
            hex_pts.append([x,y])
    hex_pts = np.array(hex_pts)
    dists_c = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dists_c)[:16]
    inits.append(hex_pts[idx])
    
    # Random
    np.random.seed(42)
    for _ in range(10):
        inits.append(np.random.randn(16, 2))
        
    for init in inits:
        if time.time() - start_time > timeout - 30:
            break
            
        init_norm = normalize_pts(init)
        
        # Phase 1: Fekete (maximize product of distances)
        x0 = init_norm.flatten()
        try:
            res1 = minimize(obj_fekete, x0, method='L-BFGS-B', jac=grad_fekete, options={'maxiter': 200, 'ftol': 1e-12})
            pts1 = res1.x.reshape(-1, 2)
            pts1 = normalize_pts(pts1)
        except:
            pts1 = init_norm
            
        score1 = compute_score(pts1)
        if score1 > best_score:
            best_score = score1
            best_points = pts1.copy()
            
        # Phase 2: Ratio polish
        if time.time() - start_time > timeout - 20:
            break
        x1 = pts1.flatten()
        try:
            res2 = minimize(obj_ratio, x1, method='L-BFGS-B', jac=lambda xx: grad_ratio_fast(xx, 0.01), args=(0.01,), options={'maxiter': 200, 'ftol': 1e-12})
            pts2 = res2.x.reshape(-1, 2)
            pts2 = normalize_pts(pts2)
        except:
            pts2 = pts1
            
        score2 = compute_score(pts2)
        if score2 > best_score:
            best_score = score2
            best_points = pts2.copy()
            
    if best_points is None:
        best_points = np.random.randn(16, 2)
        best_points = normalize_pts(best_points)
        
    return best_points
