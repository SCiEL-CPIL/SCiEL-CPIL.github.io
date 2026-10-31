import numpy as np
import time
import scipy.spatial as spat
from itertools import combinations

N = 13

# Pre-compute triangle indices for N=13
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])
_n_triples = len(_triple_idx)

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0

    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    Uses log-sum-exp for numerical stability.
    """
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return -1e10
    
    areas = _triangle_areas(points)
    # Soft min: -log(sum(exp(-beta * a))) / beta
    # For stability, subtract max
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_neg)) + beta * max_a
    soft_min_val = log_sum / beta
    
    return soft_min_val / hull_area

def _soft_min_gradient(points, beta=50.0):
    """
    Gradient of the soft-min score w.r.t. point positions.
    Returns gradient array of shape (N, 2).
    """
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return np.zeros((N, 2))
    
    areas = _triangle_areas(points)
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    weights = exp_neg / np.sum(exp_neg)  # softmax weights (sum to 1)
    
    # Gradient of soft_min w.r.t. areas: -weights
    # Gradient of area_t w.r.t. point positions
    # area_t = 0.5 * |det(v1, v2)| where v1 = p_j - p_i, v2 = p_k - p_i
    # d(area)/dp_i = 0.5 * |det(v2, -v1)| ... let's use the sign-aware version
    
    # For each triangle, compute the signed area gradient
    # signed_area = 0.5 * (v1[0]*v2[1] - v1[1]*v2[0])
    # d(signed_area)/dp_i = 0.5 * (v2[1]*(-1) - v2[0]*(-1)) ... 
    # Actually: d/dp_i [0.5*( (p_j-p_i)[0]*(p_k-p_i)[1] - (p_j-p_i)[1]*(p_k-p_i)[0] )]
    # = 0.5 * [ -1*(p_k-p_i)[1] + (p_j-p_i)[0]*(-1) - (-1)*(p_k-p_i)[0] + (p_j-p_i)[1]*(-1) ]
    # = 0.5 * [ -(p_k-p_i)[1] - (p_j-p_i)[0] + (p_k-p_i)[0] - (p_j-p_i)[1] ]
    # = 0.5 * [ (p_k-p_i)[0] - (p_j-p_i)[0] - (p_k-p_i)[1] - (p_j-p_i)[1] ]
    # = 0.5 * [ (p_k - p_j)[0] - (p_k + p_j - 2*p_i)[1] ]
    # Hmm, let me just do it numerically or use the cross product formula properly.
    
    # Let me use: area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # For the gradient, I'll use the fact that:
    # d(area)/dp_i = -0.5 * cross(p_k - p_j, sign) ... this is getting complex.
    
    # Simpler: use finite differences for the gradient? No, let's do it analytically.
    # cross(a, b) = a[0]*b[1] - a[1]*b[0]
    # area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # Let s = sign(cross(p_j - p_i, p_k - p_i))
    # d(area)/dp_i = -0.5 * s * cross(p_k - p_j, ...) 
    
    # Actually, the cleanest way:
    # area = 0.5 * |det(M)| where M = [[p_jx-p_ix, p_kx-p_ix], [py_j-p_iy, py_k-p_iy]]
    # d(area)/dp_ix = 0.5 * sign(det) * d(det)/dp_ix
    # det = (p_jx-p_ix)(p_ky-p_iy) - (p_jy-p_iy)(p_kx-p_ix)
    # d(det)/dp_ix = -(p_ky-p_iy) + (p_jy-p_iy) = p_jy - p_ky
    # d(det)/dp_iy = -(p_kx-p_ix) + (p_jx-p_ix) = p_jx - p_kx
    # So d(area)/dp_i = 0.5 * sign(det) * (p_jy - p_ky, p_jx - p_kx)
    
    # d(det)/dp_jx = (p_ky-p_iy)
    # d(det)/dp_jy = -(p_kx-p_ix)
    # d(area)/dp_j = 0.5 * sign(det) * (p_ky-p_iy, -(p_kx-p_ix))
    
    # d(det)/dp_kx = -(p_jy-p_iy)
    # d(det)/dp_ky = (p_jx-p_ix)
    # d(area)/dp_k = 0.5 * sign(det) * (-(p_jy-p_iy), (p_jx-p_ix))
    
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det_signed = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    signs = np.sign(det_signed)
    signs[signs == 0] = 1.0
    
    grad = np.zeros((N, 2))
    
    # For point i: d(area)/dp_i = 0.5 * sign * (p_jy - p_ky, p_jx - p_kx)
    for t in range(_n_triples):
        w = weights[t]
        i, j, k = _i_arr[t], _j_arr[t], _k_arr[t]
        s = 0.5 * signs[t]
        
        # Gradient contribution to point i
        grad[i, 0] += w * s * (pts[j, 1] - pts[k, 1])
        grad[i, 1] += w * s * (pts[j, 0] - pts[k, 0])
        
        # Gradient contribution to point j
        grad[j, 0] += w * s * v2[t, 1]
        grad[j, 1] += w * s * (-v2[t, 0])
        
        # Gradient contribution to point k
        grad[k, 0] += w * s * (-v1[t, 1])
        grad[k, 1] += w * s * (v1[t, 0])
    
    # Now we need to account for the division by hull_area
    # score = soft_min / hull_area
    # d(score)/dp = grad_soft_min / hull_area - soft_min * d(hull_area)/dp / hull_area^2
    # Approximation: ignore the hull area gradient (it's often small or we can estimate)
    # For now, just use grad / hull_area as an approximation
    return grad / hull_area

def find_bottleneck(points):
    """
    Returns the indices (i, j, k) of the triangle with the minimum area,
    and the area itself.
    """
    areas = _triangle_areas(points)
    min_idx = np.argmin(areas)
    return _i_arr[min_idx], _j_arr[min_idx], _k_arr[min_idx], areas[min_idx]

def _polish(points, max_iter=2000, tol=1e-12):
    """
    High-precision polish using gradient-based optimization on the soft-min surrogate
    with adaptive beta and learning rate.
    """
    pts = points.copy().astype(np.float64)
    best_score = calculate_score(pts)
    best_pts = pts.copy()
    
    beta = 30.0
    lr = 0.02
    no_improve = 0
    
    for it in range(max_iter):
        score = _soft_min_score(pts, beta)
        if score < -1e9:
            break
        
        grad = _soft_min_gradient(pts, beta)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Gradient ascent (we want to maximize)
        trial = pts + lr * grad / (grad_norm + 1e-10)
        
        trial_score = calculate_score(trial)
        if trial_score > best_score:
            best_score = trial_score
            best_pts = trial.copy()
            pts = trial
            no_improve = 0
            # Decay learning rate slowly
            lr *= 0.9995
            # Gradually increase beta to sharpen
            if beta < 150.0:
                beta = min(beta * 1.02, 150.0)
        else:
            no_improve += 1
            lr *= 0.96
            if no_improve > 100 or lr < 1e-9:
                break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Initialize with multiple structured seeds
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. Perturbed regular 13-gon
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, (N, 2))
    initial_pts2 = initial_pts + noise
    
    # 3. Fibonacci lattice
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    fib_pts = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt((i + 0.5) / N)
        theta = golden_angle * i
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    fib_pts /= np.max(np.linalg.norm(fib_pts, axis=1))
    
    # 4. Another perturbation
    np.random.seed(123)
    noise3 = np.random.normal(0, 0.05, (N, 2))
    initial_pts3 = initial_pts + noise3
    
    # 5. Concentric rings: 1 center + 12 on circle
    ring_pts = np.zeros((N, 2))
    ring_pts[0] = [0.0, 0.0]
    ring_angles = np.linspace(0, 2*np.pi, N-1, endpoint=False)
    ring_pts[1:] = np.column_stack((np.cos(ring_angles), np.sin(ring_angles)))
    
    candidates = [initial_pts, initial_pts2, fib_pts, initial_pts3, ring_pts]
    
    for init in candidates:
        if time.time() > deadline - 10.0:
            break
        
        current_pts = init.copy()
        current_score = calculate_score(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
        
        # Phase 1: Adaptive beta gradient-based optimization on soft-min surrogate
        # Start with moderate beta, increase as we converge
        beta = 20.0
        lr = 0.03
        no_improve = 0
        max_no_improve = 300
        phase1_start = time.time()
        
        while time.time() < deadline - 15.0 and no_improve < max_no_improve:
            score = _soft_min_score(current_pts, beta)
            if score < -1e9:
                break
            
            grad = _soft_min_gradient(current_pts, beta)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-10:
                break
            
            trial = current_pts + lr * grad / (grad_norm + 1e-10)
            trial_score = calculate_score(trial)
            
            if trial_score > current_score:
                current_pts = trial
                current_score = trial_score
                no_improve = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
                
                # Gradually increase beta to sharpen the soft-min
                if no_improve == 0 and beta < 100.0:
                    beta = min(beta * 1.05, 100.0)
            else:
                no_improve += 1
                lr *= 0.97
                
                # Periodic random restart to escape local minima
                if no_improve % 20 == 0:
                    np.random.seed(int(time.time() * 1000) % 100000)
                    perturb = np.random.normal(0, 0.02, (N, 2))
                    trial = current_pts + perturb
                    trial_score = calculate_score(trial)
                    if trial_score > current_score:
                        current_pts = trial
                        current_score = trial_score
                        no_improve = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_pts.copy()
                    
                    if lr < 1e-7:
                        break
        
        # Phase 2: Bottleneck-directed local search with adaptive step
        step_size = 0.05
        no_improve2 = 0
        max_no_improve2 = 150
        
        while time.time() < deadline - 8.0 and no_improve2 < max_no_improve2:
            i, j, k, min_area = find_bottleneck(current_pts)
            
            p_i = current_pts[i]
            p_j = current_pts[j]
            p_k = current_pts[k]
            
            centroid_jk = (p_j + p_k) / 2.0
            centroid_ik = (p_i + p_k) / 2.0
            centroid_ij = (p_i + p_j) / 2.0
            
            dir_i = p_i - centroid_jk
            dir_j = p_j - centroid_ik
            dir_k = p_k - centroid_ij
            
            norm_i = np.linalg.norm(dir_i)
            norm_j = np.linalg.norm(dir_j)
            norm_k = np.linalg.norm(dir_k)
            
            if norm_i > 1e-6: dir_i /= norm_i
            if norm_j > 1e-6: dir_j /= norm_j
            if norm_k > 1e-6: dir_k /= norm_k
            
            trial_pts = current_pts.copy()
            trial_pts[i] += step_size * dir_i
            trial_pts[j] += step_size * dir_j
            trial_pts[k] += step_size * dir_k
            
            trial_score = calculate_score(trial_pts)
            
            if trial_score > current_score:
                current_pts = trial_pts
                current_score = trial_score
                no_improve2 = 0
                step_size = min(step_size * 1.02, 0.1)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
            else:
                no_improve2 += 1
                step_size *= 0.93
                
                if no_improve2 % 10 == 0:
                    np.random.seed(int(time.time() * 1000) % 100000)
                    perturb = np.random.normal(0, step_size / 2, (N, 2))
                    trial_pts = current_pts + perturb
                    trial_score = calculate_score(trial_pts)
                    if trial_score > current_score:
                        current_pts = trial_pts
                        current_score = trial_score
                        no_improve2 = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_pts.copy()
                    
                    if step_size < 1e-6:
                        break
        
        # Phase 3: High-precision polish with adaptive beta
        if time.time() < deadline - 3.0:
            polished, polish_score = _polish(current_pts, max_iter=2000, tol=1e-14)
            if polish_score > best_score:
                best_score = polish_score
                best_points = polished.copy()
    
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
