"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    # Precompute all triangle indices
    tri_indices = list(itertools.combinations(range(N), 3))
    n_tris = len(tri_indices)
    
    # Precompute index arrays for vectorized operations
    tri_i = np.array([t[0] for t in tri_indices])
    tri_j = np.array([t[1] for t in tri_indices])
    tri_k = np.array([t[2] for t in tri_indices])
    
    def bary_to_cartesian(bary):
        """Convert barycentric coords (l1, l2) to Cartesian in the unit equilateral triangle."""
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        x = 0.5 - 0.5 * l1 + 0.5 * l2
        y = (1 - l1 - l2) * np.sqrt(3) / 2
        return np.column_stack([x, y])
    
    def get_valid_bary(params):
        """Reshape and clip parameters to valid barycentric coordinates."""
        bary = params.reshape(N, 2)
        bary = np.clip(bary, 0.001, 0.999)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.998
        if np.any(mask):
            bary[mask, 0] *= 0.998 / s[mask]
            bary[mask, 1] *= 0.998 / s[mask]
        return bary
    
    def compute_min_area(bary):
        """Compute the actual minimum triangle area using vectorized operations."""
        l3 = 1.0 - bary[:, 0] - bary[:, 1]
        # For each triangle (i,j,k), area = 0.5 * |det of 3x3 matrix with rows [l1,l2,l3]|
        # det = l1_i*(l2_j*l3_k - l2_k*l3_j) - l2_i*(l1_j*l3_k - l1_k*l3_j) + l3_i*(l1_j*l2_k - l1_k*l2_j)
        l1 = bary[:, 0]
        l2 = bary[:, 1]
        l3_vals = 1.0 - l1 - l2
        
        li = l1[tri_i]; lj = l1[tri_j]; lk = l1[tri_k]
        mi = l2[tri_i]; mj = l2[tri_j]; mk = l2[tri_k]
        ni = l3_vals[tri_i]; nj = l3_vals[tri_j]; nk = l3_vals[tri_k]
        
        dets = li * (mj * nk - mk * nj) - mi * (lj * nk - lk * nj) + ni * (lj * mk - lk * mj)
        areas = np.abs(dets)
        return np.min(areas)
    
    def make_softmin_objective(temperature):
        """Create objective and gradient functions with given temperature, vectorized."""
        t = temperature
        
        def compute_obj_and_grad(params):
            bary = get_valid_bary(params)
            l1 = bary[:, 0]
            l2 = bary[:, 1]
            l3 = 1.0 - l1 - l2
            
            li = l1[tri_i]; lj = l1[tri_j]; lk = l1[tri_k]
            mi = l2[tri_i]; mj = l2[tri_j]; mk = l2[tri_k]
            ni = l3[tri_i]; nj = l3[tri_j]; nk = l3[tri_k]
            
            # Compute determinants
            dets = li * (mj * nk - mk * nj) - mi * (lj * nk - lk * nj) + ni * (lj * mk - lk * mj)
            areas = np.maximum(np.abs(dets), 1e-12)
            signs = np.sign(dets)
            
            # Soft min via log-sum-exp
            exp_terms = -areas / t
            max_exp = np.max(exp_terms)
            exp_shifted = np.exp(exp_terms - max_exp)
            sum_exp = np.sum(exp_shifted)
            weights = exp_shifted / sum_exp
            
            soft_min = -t * (max_exp + np.log(sum_exp))
            obj_val = -soft_min
            
            # Gradient: d(obj)/d(params) = -weights @ d(areas)/d(params) / t
            # d(area)/d(l1_i) for triangle (i,j,k):
            # d/dl1_i [li*(mj*nk-mk*nj) - mi*(lj*nk-lk*nj) + ni*(lj*mk-lk*mj)]
            # = (mj*nk-mk*nj) - mi*(0) + ni*(0) [since l1_i only appears in li and ni=1-l1_i-l2_i]
            # Wait, need to be careful. l3_i = 1 - l1_i - l2_i, so dl3_i/dl1_i = -1
            # d/dl1_i: (mj*nk - mk*nj) + ni*(-1)*(lj*mk - lk*mj) 
            #         = (mj*nk - mk*nj) - ni*(lj*mk - lk*mj)
            
            # Similarly for other variables. Let's compute all partials.
            # For triangle (i,j,k) with barycentric coords:
            # det = li*mj*nk - li*mk*nj - mi*lj*nk + mi*lk*nj + ni*lj*mk - ni*lk*mj
            # d/dli = mj*nk - mk*nj
            # d/dmi = -lj*nk + lk*nj
            # d/dni = lj*mk - lk*mj
            # d/dlj = -mi*nk + ni*mk
            # d/dmj = li*nk - ni*lk
            # d/dnj = -li*mk + mi*lk
            # d/dlk = mi*nj - ni*mj
            # d/dmk = -mi*nj + ni*mj  [wait, let me recompute]
            
            # Actually: det = li*mj*nk - li*mk*nj - mi*lj*nk + mi*lk*nj + ni*lj*mk - ni*lk*mj
            # d/dlk = mi*nj - ni*mj
            # d/dmk = li*nj - ni*lj
            # d/dmj = li*nk - ni*lk
            # d/dmj... let me redo carefully
            
            # det = li*(mj*nk - mk*nj) - mi*(lj*nk - lk*nj) + ni*(lj*mk - lk*mj)
            # d/dli = mj*nk - mk*nj
            # d/dmi = -(lj*nk - lk*nj) = -lj*nk + lk*nj
            # d/dni = lj*mk - lk*mj
            # d/dlj = -mi*nk + ni*mk
            # d/dmj = li*nk - ni*lk
            # d/dnj = -li*mk + mi*lk
            # d/dlk = mi*nj - ni*mj
            # d/dmk = li*nj - ni*lj
            
            # Now, l3_x = 1 - l1_x - l2_x, so:
            # d/dl1_i = d/dli + d/dni * (-1) = (mj*nk - mk*nj) - (lj*mk - lk*mj)
            # d/dl2_i = d/dmi + d/dni * (-1) = (-lj*nk + lk*nj) - (lj*mk - lk*mj)
            
            # Similarly for j and k.
            
            # Compute for each triangle the gradient contributions
            dli = mj * nk - mk * nj
            dmi = -lj * nk + lk * nj
            dni = lj * mk - lk * mj
            dlj = -mi * nk + ni * mk
            dmj = li * nk - ni * lk
            dnj = -li * mk + mi * lk
            dlk = mi * nj - ni * mj
            dmk = li * nj - ni * lj
            
            # Account for l3 = 1 - l1 - l2
            grad_l1_i = dli - dni
            grad_l2_i = dmi - dni
            grad_l1_j = dlj - dnj
            grad_l2_j = dmj - dnj
            grad_l1_k = dlk - dnk
            grad_l2_k = dmk - dnk
            
            # Wait, dnk: d/dnk = lj*mk - lk*mj? No.
            # d/dnk from det = li*mj*nk - li*mk*nj - mi*lj*nk + mi*lk*nj + ni*lj*mk - ni*lk*mj
            # d/dnk = li*mj - mi*lj
            
            dnk = li * mj - mi * lj
            grad_l1_k = dlk - dnk
            grad_l2_k = dmk - dnk
            
            # Multiply by weights and accumulate
            grad_acc = np.zeros((N, 2))
            w = weights
            
            # Use np.add.at for accumulation
            np.add.at(grad_acc, (tri_i, 0), w * grad_l1_i)
            np.add.at(grad_acc, (tri_i, 1), w * grad_l2_i)
            np.add.at(grad_acc, (tri_j, 0), w * grad_l1_j)
            np.add.at(grad_acc, (tri_j, 1), w * grad_l2_j)
            np.add.at(grad_acc, (tri_k, 0), w * grad_l1_k)
            np.add.at(grad_acc, (tri_k, 1), w * grad_l2_k)
            
            grad_obj = -grad_acc / t
            return obj_val, grad_obj.flatten()
        
        def objective(params):
            val, _ = compute_obj_and_grad(params)
            return val
        
        def objective_grad(params):
            _, grad = compute_obj_and_grad(params)
            return grad
        
        return objective, objective_grad
    
    def symmetric_init():
        """Create a symmetric initial configuration for 11 points."""
        bary = np.zeros((N, 2))
        # Center
        bary[0] = [1/3, 1/3]
        # Three points near midpoints of sides
        bary[1] = [0.05, 0.5]
        bary[2] = [0.5, 0.05]
        bary[3] = [0.5, 0.5]
        # Six points in a ring around center
        r = 0.18
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i in range(6):
            angle = angles[i]
            bary[4+i, 0] = 1/3 + r * np.cos(angle)
            bary[4+i, 1] = 1/3 + r * np.sin(angle)
        
        bary = np.clip(bary, 0.01, 0.98)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.98
        if np.any(mask):
            bary[mask, 0] *= 0.98 / s[mask]
            bary[mask, 1] *= 0.98 / s[mask]
        return bary
    
    def hex_init():
        """Hexagonal lattice initialization."""
        bary = np.zeros((N, 2))
        # 1 center + 6 inner ring + 3 outer
        bary[0] = [1/3, 1/3]
        r1 = 0.22
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i in range(6):
            a = angles1[i]
            bary[1+i, 0] = 1/3 + r1 * np.cos(a)
            bary[1+i, 1] = 1/3 + r1 * np.sin(a)
        # 3 more near corners
        bary[7] = [0.15, 0.15]
        bary[8] = [0.7, 0.15]
        bary[9] = [0.15, 0.7]
        # One more
        bary[10] = [0.5, 0.35]
        
        bary = np.clip(bary, 0.01, 0.98)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.98
        if np.any(mask):
            bary[mask, 0] *= 0.98 / s[mask]
            bary[mask, 1] *= 0.98 / s[mask]
        return bary
    
    def random_init(seed):
        """Create a random initial configuration."""
        np.random.seed(seed)
        r = np.random.exponential(1.0, (N, 3))
        r /= r.sum(axis=1, keepdims=True)
        bary = r[:, :2]
        bary += np.random.normal(0, 0.02, (N, 2))
        
        bary = np.clip(bary, 0.01, 0.98)
        s = bary[:, 0] + bary[:, 1]
        mask = s > 0.98
        if np.any(mask):
            bary[mask, 0] *= 0.98 / s[mask]
            bary[mask, 1] *= 0.98 / s[mask]
        return bary
    
    best_result = None
    best_score = -np.inf
    
    initializations = [
        symmetric_init(),
        hex_init(),
        random_init(42),
        random_init(123),
        random_init(456),
        random_init(789),
        random_init(1024),
        random_init(2048),
    ]
    
    for init_bary in initializations:
        for temp in [0.04, 0.015, 0.005, 0.001]:
            objective, objective_grad = make_softmin_objective(temp)
            
            try:
                result = minimize(
                    objective,
                    init_bary.flatten(),
                    method='L-BFGS-B',
                    jac=objective_grad,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                
                final_bary = get_valid_bary(result.x)
                min_area = compute_min_area(final_bary)
                
                if min_area > best_score:
                    best_score = min_area
                    best_result = final_bary.copy()
                    init_bary = final_bary.copy()
                    
            except Exception:
                continue
    
    if best_result is None:
        best_result = symmetric_init()
    
    final_bary = best_result
    cart = bary_to_cartesian(final_bary)
    
    eps = 1e-6
    cart = np.clip(cart, eps, 1 - eps)
    cart[:, 1] = np.clip(cart[:, 1], eps, np.sqrt(3)/2 - eps)
    
    return cart
# EVOLVE-BLOCK-END
