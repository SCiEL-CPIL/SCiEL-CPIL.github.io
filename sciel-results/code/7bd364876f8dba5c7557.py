import numpy as np
from scipy.optimize import linprog
import numpy.linalg as la

N = 600
M = 10 # Number of cosine terms

def construct():
    # Basis functions: 1, cos(2*pi*k*i/N), ...
    i = np.arange(N)
    # Precompute basis matrix B (N x (M+1))
    B = np.zeros((N, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2 * np.pi * k * i / N)
    
    # Initial guess: Uniform (A0 = 1/N, others 0)
    A = np.zeros(M + 1)
    A[0] = 1.0 / N
    
    # Precompute shift matrices or use FFT for convolution?
    # Linear convolution of h with h.
    # h = B @ A
    # g = convolve(h, h)
    
    def get_h(A):
        h = B @ A
        # Enforce non-negativity? The LP step might push h negative.
        # We should project h to non-negative simplex after each step or handle in LP.
        # For now, let's see if the linearization keeps it close.
        # If h has negative values, the objective is invalid.
        # We will clip h to 0 and renormalize if necessary, but this breaks linearity.
        # Better: Add penalty for negative values in the LP? LP handles linear constraints.
        # Constraint: h_i >= 0 => (B @ A)_i >= 0.
        # This adds N constraints to the LP. N=600, M+1=11. Manageable.
        
        return h

    def compute_g(h):
        # Linear convolution
        # We only care about the peak, which is likely at center.
        # But Remez requires checking all potential peaks.
        # With N=600, full conv is 1199 points.
        g = np.convolve(h, h, mode='full')
        return g

    def gradients_g_A(A, h, g):
        # g_j = sum_i h_i h_{i-j}
        # h = B A
        # d g_j / d A = sum_i (d h_i / d A) h_{i-j} + h_i (d h_{i-j} / d A)
        # d h / d A = B
        # So d g_j / d A = B^T (shift(h, -j) + shift(h, j)) ?
        # Let's derive:
        # g_j = h^T S^j h
        # d/dA (h^T S^j h) = (dh/dA)^T S^j h + h^T S^j (dh/dA)
        # dh/dA = B (N x (M+1))
        # S^j h is h shifted by j.
        # Term 1: B^T (S^j h)
        # Term 2: B^T (S^{-j} h)  (since h^T S^j = (S^{-j} h)^T)
        # So grad_j = B^T (S^j h + S^{-j} h)
        
        grads = np.zeros((len(g), M + 1))
        # We only need grads for active set J
        # To be efficient, we compute shifts only for J
        return None # Placeholder, will compute inside loop for active set

    max_iter = 50
    tol = 1e-12
    
    for _ in range(max_iter):
        h = B @ A
        # Ensure non-negative for validity? 
        # If h has negatives, the "nonnegative step function" constraint is violated.
        # We must ensure h >= 0.
        # If h has negatives, we project h to the simplex (sum=1, h>=0) 
        # and find new A that best fits this projected h?
        # Or add constraints to LP: B @ A >= 0.
        
        # Let's add constraints B @ A >= 0 to the LP.
        
        g = compute_g(h)
        max_g = np.max(g)
        
        # Check convergence
        if max_g <= (1.0 / N) + 1e-10:
            break
            
        # Select active set J: indices where g_j is close to max
        # Usually the center and its neighbors.
        threshold = max_g - 1e-6 * max_g
        J = np.where(g >= threshold)[0]
        
        # If J is too large, reduce it
        if len(J) > 2 * (M + 1) + 2:
            # Take the largest ones
            idx_sorted = np.argsort(g)[::-1]
            J = idx_sorted[:2 * (M + 1) + 2]
            
        # Compute gradients for j in J
        # grad_j = B^T (shift(h, j) + shift(h, -j))
        # Note: np.convolve indices: g_k = sum h_i h_{i+k-N+1}? 
        # np.convolve(h, h, 'full') index k corresponds to lag k - (N-1).
        # Let lag = k - (N - 1).
        # g_k = sum_i h_i h_{i - lag}
        
        grad_J = np.zeros((len(J), M + 1))
        for idx, k in enumerate(J):
            lag = k - (N - 1)
            # shift(h, lag) means h_i -> h_{i-lag}
            # We need vectors v1[i] = h_{i-lag} and v2[i] = h_{i+lag}
            # Using roll:
            # np.roll(h, lag) shifts h to the right by lag. 
            # (np.roll(h, lag))[i] = h[(i-lag)%N]
            # This matches h_{i-lag} for periodic.
            # For linear, boundaries are tricky. But h is supported on [0, N-1].
            # If lag is small, it's fine.
            v1 = np.roll(h, lag)
            v2 = np.roll(h, -lag)
            # grad = B^T (v1 + v2)
            grad_J[idx] = B.T @ (v1 + v2)
            
        # Setup LP:
        # Minimize t
        # Subject to:
        # 1. g_j + grad_j^T dA <= t  for j in J
        #    => -grad_j^T dA + t >= g_j
        #    => grad_j^T dA - t <= -g_j
        # 2. (B @ (A + dA))_i >= 0  => B_i^T dA >= -B_i^T A
        #    => -B_i^T dA <= B_i^T A
        # 3. dA is free.
        
        # Variables: x = [dA (M+1), t (1)]
        # c = [0, ..., 0, 1]
        
        c = np.zeros(M + 2)
        c[M + 1] = 1.0
        
        A_ub = []
        b_ub = []
        
        # Constraints 1: g_j + grad_j^T dA <= t
        # grad_j^T dA - t <= -g_j
        # Matrix row: [grad_j^T, -1]
        for idx in range(len(J)):
            row = np.zeros(M + 2)
            row[:M+1] = grad_J[idx]
            row[M+1] = -1.0
            A_ub.append(row)
            b_ub.append(-g[J[idx]])
            
        # Constraints 2: h_i + B_i^T dA >= 0
        # -B_i^T dA <= B_i^T A  (since h_i = B_i^T A)
        # Row: [-B_i^T, 0]
        # b: h_i
        A_ub.append(-B) # Shape (N, M+1), pad with 0 for t
        # Need to add column of zeros for t
        A_ub = np.vstack(A_ub)
        # Add column for t
        col_t = np.zeros((len(A_ub), 1))
        A_ub = np.hstack([A_ub, col_t])
        
        b_ub = np.array(b_ub)
        # Append h for the N constraints
        b_ub = np.concatenate([b_ub, h])
        
        # Bounds for dA and t
        # dA is free: (-inf, inf)
        # t is free: (-inf, inf)
        bounds = [(None, None)] * (M + 2)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback: simple gradient step
            # Move towards uniform
            A = A * 0.9
            A[0] += 0.1 / N
            continue
            
        dA = res.x[:M+1]
        t_new = res.x[M+1]
        
        # Step size control? LP gives the optimal step for linear approx.
        # We can take full step or damped.
        A = A + dA
        
        # Ensure A0 is dominant to keep h positive?
        # The constraints should handle it.
        
    h = B @ A
    # Final projection to ensure non-negative and sum 1
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        h = np.ones(N) / N
        
    return list(h)
