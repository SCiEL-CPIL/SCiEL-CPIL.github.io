import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # 1. Generate Keich positions as initial guess
    j = np.arange(1, k + 1, dtype=np.float64)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.zeros(k, dtype=np.float64)
        for bit in range(k):
            if i & (1 << bit):
                eps[bit] = 1.0
        positions[i] = np.sum((1.0 - j) * eps * (2.0 ** -j) / k)
    x0 = positions

    # Pre-compute constants
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n

    def compute_area_and_grad(x):
        # x shape (n,)
        L = x[None, :] + a[None, :] * y[:, None]
        
        # Sort L along axis 1
        L_sorted, idx = np.sort(L, axis=1)
        
        # Compute diffs
        diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
        
        # Compute union lengths
        mins = np.minimum(diffs, w[:, None])
        union_lengths = w + np.sum(mins, axis=1)
        area = np.mean(union_lengths)
        
        # Compute gradient
        # IndLeft[m, j] is 1 if L_sorted[m, j+1] - L_sorted[m, j] < w[m]
        IndLeft = (diffs < w[:, None]).astype(np.float64)
        
        # C[m, j] is the derivative of union_length w.r.t L_sorted[m, j]
        # C[:, 0] = -IndLeft[:, 0]
        # C[:, j] = IndLeft[:, j-1] - IndLeft[:, j]
        # C[:, n-1] = IndLeft[:, n-2]
        
        C = np.zeros((M, n), dtype=np.float64)
        C[:, 0] = -IndLeft[:, 0]
        C[:, n-1] = IndLeft[:, n-2]
        if n > 2:
            C[:, 1:n-1] = IndLeft[:, :-1] - IndLeft[:, 1:]
            
        # Scatter back to original indices
        # Grad[m, idx[m, j]] = C[m, j]
        Grad = np.zeros((M, n), dtype=np.float64)
        # Use advanced indexing
        # rows: np.arange(M)[:, None]
        # cols: idx
        Grad[np.arange(M)[:, None], idx] = C
        
        grad_x = Grad.sum(axis=0) / M
        
        return area, grad_x

    start_time = time.time()
    deadline = start_time + 590 # 10s safety margin

    # Use L-BFGS-B with analytic gradient
    # Since the function is piecewise linear, the gradient is discontinuous.
    # L-BFGS-B assumes smoothness. It will still work as a subgradient method essentially.
    # To help it, we can use a small line search step or just let it run.
    
    try:
        res = minimize(
            compute_area_and_grad, 
            x0, 
            method='L-BFGS-B',
            jac=True, # Returns (f, g)
            options={
                'maxiter': 500,
                'ftol': 1e-14,
                'gtol': 1e-8,
                'maxls': 50,
                'maxfun': 2000
            }
        )
        x_opt = res.x
    except Exception:
        x_opt = x0

    # Optional: Coordinate Descent Polishing
    # If we have time, we can do a few iterations of coordinate descent
    # where we pick the best breakpoint for each x_i.
    # This is expensive, so we do it only if time < 500s.
    
    if time.time() < deadline - 50:
        # Simple Coordinate Descent
        # For each i, try all breakpoints
        # Breakpoints for x_i: x_k + (a_k - a_i) * y_m
        # This is 63 * 20000 = 1.26M points.
        # We can sample a subset of m's to speed this up, or just do 1-2 passes.
        
        # Let's do 1 pass of coordinate descent with subsampled m's for speed
        m_sub = np.linspace(0, M-1, 1000, dtype=int)
        y_sub = y[m_sub]
        w_sub = w[m_sub]
        
        for _ in range(2): # 2 passes
            for i in range(n):
                # Get current best x_i
                x_i_current = x_opt[i]
                
                # Generate candidates
                # x_k + (a_k - a_i) * y_m
                # k != i
                k_indices = [k for k in range(n) if k != i]
                a_k = a[k_indices]
                x_k = x_opt[k_indices]
                
                # Candidates shape: (len(k_indices) * len(m_sub))
                # We can vectorize this
                # a_k[:, None] - a_i
                # x_k[:, None] + (a_k[:, None] - a_i) * y_sub[None, :]
                candidates = x_k[:, None] + (a_k[:, None] - a[i]) * y_sub[None, :]
                candidates = candidates.flatten()
                
                # Also include current x_i and maybe some neighbors
                # Evaluate area for each candidate
                # This is slow if we call compute_area for each.
                # We need a fast batched evaluation or just accept the cost.
                # 1.26M candidates * 1000 m's * 64 n's is too slow.
                # Let's skip full coordinate descent and rely on L-BFGS-B.
                pass
                
    return x_opt

# EVOLVE-BLOCK-END
