import numpy as np
import time
import itertools

N = 11
SQRT3 = np.sqrt(3)

# Precompute all triples (i, j, k) with i < j < k
_TRIPLES = list(itertools.combinations(range(N), 3))
_T_IDX = np.array(_TRIPLES)
_NUM_TRIPLES = len(_TRIPLES)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum triangle area (normalized).
    """
    # p is (N, 3)
    # We need det of matrix with rows p[i], p[j], p[k]
    # Vectorized calculation for all triples
    # pi = p[_T_IDX[:, 0]]
    # pj = p[_T_IDX[:, 1]]
    # pk = p[_T_IDX[:, 2]]
    
    # det = pi[0]*(pj[1]*pk[2] - pj[2]*pk[1]) - pi[1]*(pj[0]*pk[2] - pj[2]*pk[0]) + pi[2]*(pj[0]*pk[1] - pj[1]*pk[0])
    
    pi = p[_T_IDX[:, 0]]
    pj = p[_T_IDX[:, 1]]
    pk = p[_T_IDX[:, 2]]
    
    # Calculate determinants
    # Using the formula:
    # det = a1(b2c3 - b3c2) - a2(b1c3 - b3c1) + a3(b1c2 - b2c1)
    dets = (pi[:, 0] * (pj[:, 1] * pk[:, 2] - pj[:, 2] * pk[:, 1])
           - pi[:, 1] * (pj[:, 0] * pk[:, 2] - pj[:, 2] * pk[:, 0])
           + pi[:, 2] * (pj[:, 0] * pk[:, 1] - pj[:, 1] * pk[:, 0]))
           
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y) for equilateral triangle."""
    # u, v, w correspond to vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # x = u*0 + v*1 + w*0.5 = v + 0.5*w
    # y = u*0 + v*0 + w*sqrt(3)/2 = w*sqrt(3)/2
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """Project barycentric coordinates to be valid (non-negative, sum to 1)."""
    b = bary.copy()
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    b = np.clip(b, 0, 1)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def _sqp_step(p, t, max_iter=50, tol=1e-8):
    """
    Perform a few steps of SQP to improve p.
    p: current barycentric coordinates (N, 3)
    t: temperature/step size for noise injection if needed? No, SQP is deterministic.
    """
    current_p = p.copy()
    current_area = _min_area_bary(current_p)
    
    for _ in range(max_iter):
        # 1. Compute areas for all triples
        pi = current_p[_T_IDX[:, 0]]
        pj = current_p[_T_IDX[:, 1]]
        pk = current_p[_T_IDX[:, 2]]
        
        dets = (pi[:, 0] * (pj[:, 1] * pk[:, 2] - pj[:, 2] * pk[:, 1])
               - pi[:, 1] * (pj[:, 0] * pk[:, 2] - pj[:, 2] * pk[:, 0])
               + pi[:, 2] * (pj[:, 0] * pk[:, 1] - pj[:, 1] * pk[:, 0]))
               
        areas = 0.5 * np.abs(dets)
        min_area = np.min(areas)
        
        # 2. Identify active set (triples with area close to min)
        # Use a small tolerance to include near-minimal triangles
        active_mask = areas < min_area + 1e-6
        active_idx = np.where(active_mask)[0]
        
        if len(active_idx) == 0:
            break
            
        # 3. Compute gradients of the active area constraints w.r.t. barycentric coords
        # We want to maximize min_area.
        # For each active triangle (i,j,k), we have area = 0.5 * |det|.
        # Let's assume the sign of det is constant locally.
        # grad_area = 0.5 * sign(det) * grad_det
        # grad_det w.r.t. p[a] (where a is one of i,j,k)
        
        # Let's compute the gradient of the objective function f = min(areas)
        # This is a max-min problem.
        # Linearization: max delta_s.t. areas_active + grad_dot_delta >= current_min
        
        # Actually, let's just use a simple gradient ascent on the average of the active areas?
        # Or better, solve the QP:
        # max t
        # s.t. t <= area_i + grad_i . delta  (for active i)
        #      t <= area_i (for inactive i, but we can ignore if they are far)
        #      p + delta is valid barycentric
        
        # Since we are in barycentric coords, constraints are:
        # sum(delta_row) = 0 for all rows
        # p + delta >= 0
        
        # Let's simplify: Just do a projected gradient ascent on the sum of active areas?
        # Or better, compute the gradient of the min area function.
        # The subgradient of min(f1, ..., fn) is the convex hull of the gradients of the active functions.
        # A good direction is the average of the gradients of the active functions.
        
        # Compute gradients for active triples
        # For a triple (i, j, k), det = p_i x p_j x p_k (scalar triple product)
        # d(det)/dp_i = p_j x p_k (cross product)
        # d(det)/dp_j = p_k x p_i
        # d(det)/dp_k = p_i x p_j
        
        # Area = 0.5 * |det|
        # grad Area = 0.5 * sign(det) * grad_det
        
        grad_sum = np.zeros((N, 3))
        for idx in active_idx:
            i, j, k = _T_IDX[idx]
            det_val = dets[idx]
            if np.abs(det_val) < 1e-12:
                continue
            sign = 1.0 if det_val > 0 else -1.0
            
            # Gradients of det
            # grad_i = p_j x p_k
            grad_i = np.cross(pj[idx], pk[idx])
            # grad_j = p_k x p_i
            grad_j = np.cross(pk[idx], pi[idx])
            # grad_k = p_i x p_j
            grad_k = np.cross(pi[idx], pj[idx])
            
            # Gradient of area = 0.5 * sign * grad_det
            factor = 0.5 * sign
            grad_sum[i] += factor * grad_i
            grad_sum[j] += factor * grad_j
            grad_sum[k] += factor * grad_k
            
        # Normalize gradient
        norm = np.linalg.norm(grad_sum)
        if norm < 1e-12:
            break
        grad_dir = grad_sum / norm
        
        # Project gradient onto feasible space
        # Feasible directions must satisfy sum(delta_row) = 0
        # This is equivalent to removing the mean of each row? No, sum of components in a row is 0.
        # So for each row r, delta[r, 0] + delta[r, 1] + delta[r, 2] = 0
        # The projection of a vector g onto the subspace sum(x)=0 is x - mean(x)
        # Wait, the constraint is on the sum of the 3 components of each point.
        # So we subtract the mean of the 3 components for each point.
        
        # g is (N, 3). For each row i, we want d_i such that sum(d_i) = 0.
        # The projection of g_i onto the plane sum(x)=0 is g_i - mean(g_i) * [1,1,1]
        
        proj_grad = grad_sum.copy()
        for i in range(N):
            mean_val = np.mean(proj_grad[i])
            proj_grad[i] -= mean_val
            
        # Check if we are at a boundary (p[i] >= 0)
        # If p[i, j] == 0 and proj_grad[i, j] < 0, we cannot move in that direction.
        # Simple approach: Clip the step.
        
        # Line search
        alpha = 0.05
        for _ in range(20):
            new_p = current_p + alpha * proj_grad
            new_p = _project_bary(new_p)
            new_area = _min_area_bary(new_p)
            
            if new_area > current_area:
                current_p = new_p
                current_area = new_area
                break
            else:
                alpha *= 0.5
        else:
            break
            
    return current_p, current_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(12345)
    
    best_area = 0.0
    best_bary = None
    
    def _is_time_up():
        return time.time() > deadline - 5.0
        
    # Initializations
    # 1. Random
    # 2. Structured
    # 3. Grid
    
    inits = []
    
    # Random
    inits.append(rng.dirichlet([1, 1, 1], size=N))
    
    # Structured: Vertices, Midpoints, Center
    bary_struct = np.zeros((N, 3))
    bary_struct[0] = [0.9, 0.05, 0.05]
    bary_struct[1] = [0.05, 0.9, 0.05]
    bary_struct[2] = [0.05, 0.05, 0.9]
    bary_struct[3] = [0.45, 0.45, 0.1]
    bary_struct[4] = [0.1, 0.45, 0.45]
    bary_struct[5] = [0.45, 0.1, 0.45]
    bary_struct[6] = [1/3, 1/3, 1/3]
    for i in range(7, N):
        bary_struct[i] = rng.dirichlet([1, 1, 1])
    inits.append(bary_struct)
    
    # Grid-like
    # Place points on a triangular lattice
    inits.append(rng.dirichlet([1, 1, 1], size=N)) # Fallback random for now
    
    for restart in range(50):
        if _is_time_up():
            break
            
        if restart < len(inits):
            current_bary = _project_bary(inits[restart].copy())
        else:
            current_bary = rng.dirichlet([1, 1, 1], size=N)
            current_bary = _project_bary(current_bary)
            
        # Run SQP
        current_bary, current_area = _sqp_step(current_bary, 0.0)
        
        if current_area > best_area:
            best_area = current_area
            best_bary = current_bary.copy()
            
        # Occasionally do a random restart perturbation
        if restart % 5 == 4:
             # Perturb best solution and try again
             perturbed = best_bary + rng.normal(0, 0.05, size=(N, 3))
             perturbed = _project_bary(perturbed)
             perturbed, area_pert = _sqp_step(perturbed, 0.0)
             if area_pert > best_area:
                 best_area = area_pert
                 best_bary = perturbed.copy()

    if best_bary is None:
        # Fallback
        bary = np.zeros((N, 3))
        for i in range(N):
            bary[i] = [1/3, 1/3, 1/3] + rng.normal(0, 0.1, size=3)
        best_bary = _project_bary(bary)
    
    cart = _bary_to_cart(best_bary)
    return cart
