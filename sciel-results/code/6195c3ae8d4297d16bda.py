import numpy as np
from scipy.optimize import minimize

N = 50
w = 1.0 / N
centers = -0.25 + (np.arange(N) + 0.5) * w
sum_centers = (centers[:, None] + centers[None, :]).flatten()
# M[k, p] where p is flattened index of (i,j)
# M[k, p] = max(0, 1 - |sum_centers[k] - sum_centers[p]| / w)
# Note: sum_centers is the same array for peaks and pairs.
# So M is symmetric? No, M[k,p] depends on distance between k-th sum and p-th sum.
# The set of sums is the same.
# So M is a distance matrix of the sum_centers.
diffs = np.abs(sum_centers[:, None] - sum_centers[None, :])
M = np.maximum(0, 1 - diffs / w)

def construct():
    def objective(h_flat):
        h = np.maximum(0, h_flat)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 1e6
        l2_sq = np.sum(h**2)
        
        # Compute autoconvolution peaks
        h_outer = np.outer(h, h)
        h_flat_sq = h_outer.flatten()
        
        # V_k = sum_p M[k, p] * h_i h_j
        V = M @ h_flat_sq
        peak = np.max(V)
        if peak < 1e-12:
            return 1e6
            
        ratio = l2_sq / (l1 * peak)
        return -ratio

    def jac(h_flat):
        h = np.maximum(0, h_flat)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return np.zeros_like(h)
        l2_sq = np.sum(h**2)
        
        h_outer = np.outer(h, h)
        h_flat_sq = h_outer.flatten()
        V = M @ h_flat_sq
        peak = np.max(V)
        if peak < 1e-12:
            return np.zeros_like(h)
            
        ratio = l2_sq / (l1 * peak)
        
        # Gradient of -ratio
        # f = - A / (B C)
        # df/dh = - (B C grad A - A grad (B C)) / (B C)^2
        # grad A = 2h
        # grad B = 1
        # grad C = grad V_k* where k* is argmax V
        
        k_star = np.argmax(V)
        T_k = M[k_star].reshape(N, N)
        # grad V_k = 2 * (T_k h)
        grad_C = 2 * (T_k @ h)
        
        # grad (B C) = B grad C + C grad B = B grad C + C * 1
        grad_BC = l1 * grad_C + peak * np.ones_like(h)
        
        # df/dh = - ( (l1*peak)*(2h) - l2_sq*(grad_BC) ) / (l1*peak)^2
        num = (l1 * peak) * (2 * h) - l2_sq * grad_BC
        den = (l1 * peak) ** 2
        
        grad_f = -num / den
        
        return grad_f

    h0 = np.zeros(N)
    h0[N // 2] = 1.0
    
    bounds = [(0, None)] * N
    
    try:
        res = minimize(objective, h0, method='trust-constr', jac=jac, bounds=bounds,
                       options={'maxiter': 200, 'gtol': 1e-8, 'xtol': 1e-10})
        h_opt = np.maximum(0, res.x)
    except Exception:
        # Fallback to L-BFGS-B if trust-constr fails or is too slow
        try:
            res = minimize(objective, h0, method='L-BFGS-B', jac=jac, bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-8})
            h_opt = np.maximum(0, res.x)
        except Exception:
            h_opt = h0
            
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt / s
    else:
        h_opt = h0
        
    return h_opt
