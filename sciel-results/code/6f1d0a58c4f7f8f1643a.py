# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    N = 11
    
    def bary_to_cartesian(bary):
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        x = 0.5 - 0.5 * l1 + 0.5 * l2
        y = (1 - l1 - l2) * np.sqrt(3) / 2
        return np.column_stack([x, y])
    
    # Pre-compute indices for all combinations
    comb_indices = np.array(list(itertools.combinations(range(N), 3)))
    
    def get_areas_and_grads(params):
        bary = params.reshape(N, 2)
        # Clip to valid range
        bary = np.clip(bary, 1e-5, 1 - 1e-5)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 1 - 1e-5
        if np.any(mask):
            bary[mask, 0] *= (1 - 1e-5) / s[mask]
            bary[mask, 1] *= (1 - 1e-5) / s[mask]
        
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        l3 = 1 - l1 - l2
        M_full = np.column_stack([l1, l2, l3])
        
        # M_full is (N, 3)
        # Extract triangles
        # comb_indices shape (165, 3)
        M_tri = M_full[comb_indices] # (165, 3, 3)
        
        # Calculate determinants
        dets = np.linalg.det(M_tri) # (165,)
        areas = np.abs(dets)
        signs = np.sign(dets)
        
        # Calculate gradients of Det w.r.t params
        # Grad of Det w.r.t l_p1 is C_p1 - C_p3
        # Grad of Det w.r.t l_p2 is C_p2 - C_p3
        # Cofactors can be computed from the 3x3 matrices.
        # For a 3x3 matrix, cofactors are:
        # C00 = M11*M22 - M12*M21
        # C01 = -(M10*M22 - M12*M20)
        # C02 = M10*M21 - M11*M20
        # ... and so on.
        
        # Let's compute cofactor matrix for all triangles at once? 
        # It's easier to just loop over triangles if vectorization is complex, 
        # but 165 is small.
        
        grads = np.zeros_like(params)
        
        for idx in range(len(comb_indices)):
            i, j, k = comb_indices[idx]
            M_t = M_tri[idx]
            d = dets[idx]
            a = abs(d)
            sgn = signs[idx]
            
            # Cofactors for row i (point i)
            # C_i0 = M_t[1,1]*M_t[2,2] - M_t[1,2]*M_t[2,1]
            # C_i1 = -(M_t[1,0]*M_t[2,2] - M_t[1,2]*M_t[2,0])
            # C_i2 = M_t[1,0]*M_t[2,1] - M_t[1,1]*M_t[2,0]
            
            # Derivatives of Det w.r.t l_i1, l_i2
            # dDet/dl_i1 = C_i0 - C_i2
            # dDet/dl_i2 = C_i1 - C_i2
            
            Ci0 = M_t[1,1]*M_t[2,2] - M_t[1,2]*M_t[2,1]
            Ci1 = -(M_t[1,0]*M_t[2,2] - M_t[1,2]*M_t[2,0])
            Ci2 = M_t[1,0]*M_t[2,1] - M_t[1,1]*M_t[2,0]
            
            dDi1 = sgn * (Ci0 - Ci2)
            dDi2 = sgn * (Ci1 - Ci2)
            
            # Derivatives for j
            # C_j0 = M_t[0,1]*M_t[2,2] - M_t[0,2]*M_t[2,1]
            # C_j1 = -(M_t[0,0]*M_t[2,2] - M_t[0,2]*M_t[2,0])
            # C_j2 = M_t[0,0]*M_t[2,1] - M_t[0,1]*M_t[2,0]
            
            Cj0 = M_t[0,1]*M_t[2,2] - M_t[0,2]*M_t[2,1]
            Cj1 = -(M_t[0,0]*M_t[2,2] - M_t[0,2]*M_t[2,0])
            Cj2 = M_t[0,0]*M_t[2,1] - M_t[0,1]*M_t[2,0]
            
            dDj1 = sgn * (Cj0 - Cj2)
            dDj2 = sgn * (Cj1 - Cj2)
            
            # Derivatives for k
            # C_k0 = M_t[0,1]*M_t[1,2] - M_t[0,2]*M_t[1,1]
            # C_k1 = -(M_t[0,0]*M_t[1,2] - M_t[0,2]*M_t[1,0])
            # C_k2 = M_t[0,0]*M_t[1,1] - M_t[0,1]*M_t[1,0]
            
            Ck0 = M_t[0,1]*M_t[1,2] - M_t[0,2]*M_t[1,1]
            Ck1 = -(M_t[0,0]*M_t[1,2] - M_t[0,2]*M_t[1,0])
            Ck2 = M_t[0,0]*M_t[1,1] - M_t[0,1]*M_t[1,0]
            
            dDk1 = sgn * (Ck0 - Ck2)
            dDk2 = sgn * (Ck1 - Ck2)
            
            # Accumulate into grads array
            # params is flat (N, 2). index i*2 is l1, i*2+1 is l2
            grads[i*2] += dDi1 * a # Wait, this is for Det, we need for soft-min
            grads[i*2+1] += dDi2 * a
            grads[j*2] += dDj1 * a
            grads[j*2+1] += dDj2 * a
            grads[k*2] += dDk1 * a
            grads[k*2+1] += dDk2 * a
            # Note: This accumulation is for sum(exp(-A/t) * dA). 
            # We will normalize later.
            
        return areas, grads

    def objective_and_grad(params, t):
        areas, grads_raw = get_areas_and_grads(params)
        # Soft min: S = -t log(sum(exp(-A/t)))
        # f = -S = t log(sum(exp(-A/t)))
        # f' = sum(exp(-A/t) * (-1/t * dA)) / sum(exp(-A/t)) = - sum(exp(-A/t) * dA) / sum(exp(-A/t))
        
        exp_terms = np.exp(-areas / t)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 0, np.zeros_like(params)
        
        # grads_raw currently holds sum(exp(-A/t) * dA) ? 
        # No, in the loop above I did `grads[...] += dD * a`. That was wrong.
        # I need to weight by exp(-A/t).
        
        # Let's re-calculate properly in the loop or weight after.
        # Actually, let's just compute the weighted sum in the loop.
        # Re-doing the loop logic inside objective for clarity and correctness.
        
        bary = params.reshape(N, 2)
        bary = np.clip(bary, 1e-5, 1 - 1e-5)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 1 - 1e-5
        if np.any(mask):
            bary[mask, 0] *= (1 - 1e-5) / s[mask]
            bary[mask, 1] *= (1 - 1e-5) / s[mask]
        
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        l3 = 1 - l1 - l2
        M_full = np.column_stack([l1, l2, l3])
        M_tri = M_full[comb_indices]
        dets = np.linalg.det(M_tri)
        areas = np.abs(dets)
        signs = np.sign(dets)
        exp_terms = np.exp(-areas / t)
        
        grads = np.zeros_like(params)
        
        for idx in range(len(comb_indices)):
            i, j, k = comb_indices[idx]
            M_t = M_tri[idx]
            w = exp_terms[idx]
            if w < 1e-10:
                continue
            sgn = signs[idx]
            
            Ci0 = M_t[1,1]*M_t[2,2] - M_t[1,2]*M_t[2,1]
            Ci1 = -(M_t[1,0]*M_t[2,2] - M_t[1,2]*M_t[2,0])
            Ci2 = M_t[1,0]*M_t[2,1] - M_t[1,1]*M_t[2,0]
            dDi1 = sgn * (Ci0 - Ci2)
            dDi2 = sgn * (Ci1 - Ci2)
            
            Cj0 = M_t[0,1]*M_t[2,2] - M_t[0,2]*M_t[2,1]
            Cj1 = -(M_t[0,0]*M_t[2,2] - M_t[0,2]*M_t[2,0])
            Cj2 = M_t[0,0]*M_t[2,1] - M_t[0,1]*M_t[2,0]
            dDj1 = sgn * (Cj0 - Cj2)
            dDj2 = sgn * (Cj1 - Cj2)
            
            Ck0 = M_t[0,1]*M_t[1,2] - M_t[0,2]*M_t[1,1]
            Ck1 = -(M_t[0,0]*M_t[1,2] - M_t[0,2]*M_t[1,0])
            Ck2 = M_t[0,0]*M_t[1,1] - M_t[0,1]*M_t[1,0]
            dDk1 = sgn * (Ck0 - Ck2)
            dDk2 = sgn * (Ck1 - Ck2)
            
            grads[i*2] += w * dDi1
            grads[i*2+1] += w * dDi2
            grads[j*2] += w * dDj1
            grads[j*2+1] += w * dDj2
            grads[k*2] += w * dDk1
            grads[k*2+1] += w * dDk2
            
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 0, np.zeros_like(params)
            
        # f = t log(sum_exp)
        # f_grad = - grads / sum_exp
        val = t * np.log(sum_exp)
        grad = -grads / sum_exp
        return val, grad

    # Initial guess
    np.random.seed(42)
    # Symmetric initialization: 1 center, 3 vertices, 3 edge midpoints, 3 interior
    # Vertices: (0,0), (1,0), (0,1) in barycentric (l1,l2) -> A=(1,0)? No.
    # A=(0,0) cart -> bary (1,0) l1=1, l2=0, l3=0? 
    # My bary definition: x = 0.5 - 0.5*l1 + 0.5*l2, y = (1-l1-l2)*sqrt(3)/2.
    # If l1=1, l2=0: x=0, y=0. Correct (A).
    # If l1=0, l2=1: x=1, y=0. Correct (B).
    # If l1=0, l2=0: x=0.5, y=sqrt(3)/2. Correct (C).
    
    init = np.array([
        [1.0, 0.0], # A
        [0.0, 1.0], # B
        [0.0, 0.0], # C
        [1/3, 1/3], # Center
        [0.5, 0.0], # Mid AB
        [0.0, 0.5], # Mid BC
        [0.5, 0.5], # Mid AC
        [0.2, 0.2],
        [0.2, 0.6],
        [0.6, 0.2],
        [0.4, 0.4]
    ])
    
    # Add small noise to break symmetry
    init += np.random.normal(0, 0.05, init.shape)
    init = np.clip(init, 0.01, 0.99)
    s = init[:, 0] + init[:, 1]
    mask = s > 0.99
    if np.any(mask):
        init[mask, 0] *= 0.99 / s[mask]
        init[mask, 1] *= 0.99 / s[mask]
        
    best_val = np.inf
    best_params = init.flatten()
    
    # Annealing schedule
    temps = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    for t in temps:
        def obj_p(params):
            v, g = objective_and_grad(params, t)
            return v
        
        def grad_p(params):
            v, g = objective_and_grad(params, t)
            return g
            
        res = minimize(
            obj_p,
            best_params,
            jac=grad_p,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        if res.success or res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    final_bary = best_params.reshape(N, 2)
    final_bary = np.clip(final_bary, 0.0, 1.0)
    s = final_bary[:, 0] + final_bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        final_bary[mask, 0] /= s[mask]
        final_bary[mask, 1] /= s[mask]
        
    cart = bary_to_cartesian(final_bary)
    eps = 1e-6
    cart = np.clip(cart, eps, 1 - eps)
    cart[:, 1] = np.clip(cart[:, 1], eps, np.sqrt(3)/2 - eps)
    
    return cart
# EVOLVE-BLOCK-END
