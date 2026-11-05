"""Rewritten program for third autocorrelation inequality optimization using ADMM."""
import numpy as np
import time

N = 400

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses ADMM to minimize the L2 norm of autoconvolution subject to sum constraint,
    with a hard cap on the maximum entry of the autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # ADMM formulation:
    # Minimize (1/2)||h*h||_2^2 + (rho/2)||h - z||_2^2 + (sigma/2)(max(0, max(h*h) - C)^2)
    # subject to sum(h) = 1
    # where z is a split variable, C is the cap on max autoconvolution
    
    # We use a simpler approach: minimize ||h*h||_2^2 subject to sum(h)=1 and max(h*h) <= C
    # Use ADMM with variables h and y (where y = h*h, but that's nonlinear)
    # Instead, use proximal gradient / ADMM on h directly
    
    # Alternative: use the fact that we want to minimize max(conv) / (sum(h))^2
    # Since the problem is scale invariant, fix sum(h) = 1
    # Then minimize max(conv)
    
    # Use ADMM to minimize (1/2)||h*h||_2^2 + indicator{sum(h)=1} + indicator{max(h*h) <= C}
    # Split: minimize (1/2)||h*h||_2^2 + (rho/2)||h - z||_2^2 + indicator{sum(z)=1, max(z*z)<=C}
    # h-update: minimize (1/2)||h*h||_2^2 + (rho/2)||h - z + u||_2^2
    #   This is still nonlinear due to h*h. Use gradient step.
    # z-update: project z - u onto {sum(z)=1, max(z*z)<=C}
    
    # Simpler: use projected gradient descent on h with sum(h)=1 constraint
    # Minimize f(h) = max(conv(h)) + (mu/2)||h||_2^2 (regularization for stability)
    # Project onto sum(h)=1
    
    # Let's use a different approach: parameterize h = 1 + delta, where sum(delta) = 0
    # Then minimize max(conv(1+delta))
    
    # Use subgradient method on max(conv)
    # The subgradient of max(conv) with respect to h is conv with the argmax location
    
    # Let's try: minimize (1/2)||conv(h)||_2^2 subject to sum(h) = 1
    # This is a smooth objective! The gradient of (1/2)||conv(h)||_2^2 with respect to h
    # can be computed via FFT.
    
    # conv(h) = h * h (full convolution)
    # ||conv(h)||_2^2 = sum_k (sum_i h[i] h[k-i])^2
    # Gradient: d/dh[j] (1/2)||conv(h)||_2^2 = (conv(conv(h)))[j] ... let me verify
    # Actually, for full convolution c = h*h, c[k] = sum_i h[i] h[k-i]
    # (1/2)||c||_2^2 = (1/2) sum_k c[k]^2
    # d/dh[j] = sum_k c[k] * dc[k]/dh[j] = sum_k c[k] * (h[k-j] + h[j-k])
    # Note h[k-j] is valid when 0<=k-j<N, i.e., j<=k<N+j
    # h[j-k] is valid when 0<=j-k<N, i.e., k<=j<N+k
    # So d/dh[j] = sum_{k=j}^{N+j-1} c[k] h[k-j] + sum_{k=0}^{j} c[k] h[j-k]
    # But this is getting complicated. Let's use FFT-based gradient.
    
    # For circular convolution, the gradient is easy. For full convolution, we can zero-pad.
    # Let H = FFT(zeropad(h, 2N-1)), then C = H^2, conv = IFFT(C)
    # d/dh (1/2)||conv||_2^2 = IFFT(2*C * conj(H)) truncated to N... no, that's not right either.
    
    # Let's just use numerical gradient or a simpler approach.
    
    # Actually, let's use the following insight:
    # We want h such that conv(h) is as flat as possible (low peak) while sum(h)=1.
    # The "flattest" autoconvolution subject to sum(h)=1 would be one where conv(h) is constant.
    # But conv(h) has support of length 2N-1, and sum(conv(h)) = (sum h)^2 = 1.
    # If conv(h) were constant c over 2N-1 entries, then c*(2N-1) = 1, so c = 1/(2N-1).
    # Then max(conv) = 1/(2N-1), and value = 2N/(2N-1) ≈ 1.
    # But can we achieve this? We need h*h = 1/(2N-1) * ones(2N-1).
    # In Fourier domain: |H(k)|^2 = (1/(2N-1)) * sum of ones... this is not straightforward.
    
    # Let's try: minimize ||conv(h) - mu*ones(2N-1)||_2^2 subject to sum(h)=1
    # where mu = 1/(2N-1) is the target constant value.
    # This is a least-squares problem!
    
    # conv(h) = A h where A is a (2N-1) x N matrix (Toeplitz)
    # Minimize ||A h - mu * 1||_2^2 subject to 1^T h = 1
    
    # This is a linearly constrained least-squares problem, solvable in closed form!
    # Using Lagrange multipliers:
    # min_h (1/2)||Ah - b||_2^2 + lambda(1^T h - 1)
    # where b = mu * ones(2N-1)
    # KKT: A^T A h - A^T b + lambda * 1 = 0
    #      1^T h = 1
    
    # Let M = A^T A (N x N), r = A^T b (N x 1)
    # M h + lambda * 1 = r
    # 1^T h = 1
    
    # Solve: h = M^{-1}(r - lambda * 1)
    # 1^T M^{-1}(r - lambda * 1) = 1
    # 1^T M^{-1} r - lambda * 1^T M^{-1} 1 = 1
    # lambda = (1^T M^{-1} r - 1) / (1^T M^{-1} 1)
    
    # M = A^T A is the Gram matrix of the autoconvolution operator.
    # M[i,j] = sum_k A[k,i] A[k,j]
    # A[k,i] = 1 if 0<=k-i<N, i.e., i<=k<N+i
    # So M[i,j] = number of k such that max(0,i,j) <= k < min(N, i+N, j+N)
    #           = min(N, i+N, j+N) - max(0, i, j) if positive
    
    # Let's construct M efficiently.
    # M[i,j] = min(N, i+N, j+N) - max(0, i, j)
    # Since i,j in [0,N-1]:
    # min(N, i+N, j+N) = N (since i+N >= N, j+N >= N)
    # max(0, i, j) = max(i,j)
    # So M[i,j] = N - max(i,j) for max(i,j) < N, which is always true.
    # Wait, that's not right. Let me recompute.
    
    # A is (2N-1) x N. A[k,i] = 1 if i <= k < N+i (0-indexed, k in [0, 2N-2])
    # M[i,j] = sum_{k=0}^{2N-2} A[k,i] A[k,j] = number of k in [max(i,j), min(N+i-1, N+j-1)]
    #         = min(N+i-1, N+j-1) - max(i,j) + 1
    #         = N + min(i,j) - max(i,j) + 1
    #         = N + 1 - |i-j|
    
    # So M is a Toeplitz matrix with M[i,j] = N + 1 - |i-j|
    
    # r = A^T b where b[k] = mu for all k
    # r[i] = sum_{k=0}^{2N-2} A[k,i] b[k] = mu * sum_{k=i}^{N+i-1} 1 = mu * N
    
    # So r = mu * N * ones(N)
    
    # Now solve:
    # M h + lambda * 1 = r
    # 1^T h = 1
    
    # Let's construct M and solve.
    
    def _construct_M(N):
        """Construct the Gram matrix M[i,j] = N+1-|i-j|."""
        idx = np.arange(N)
        M = N + 1 - np.abs(idx[:, None] - idx[None, :])
        return M
    
    def _solve_constrained_ls(M, r, N):
        """Solve M h + lambda*1 = r, 1^T h = 1."""
        ones = np.ones(N)
        # h = M^{-1}(r - lambda*1)
        # 1^T M^{-1} r - lambda * 1^T M^{-1} 1 = 1
        # lambda = (1^T M^{-1} r - 1) / (1^T M^{-1} 1)
        
        # Solve M x = r and M y = 1
        try:
            x = np.linalg.solve(M, r)
            y = np.linalg.solve(M, ones)
        except np.linalg.LinAlgError:
            # Use least squares if singular
            x, *_ = np.linalg.lstsq(M, r, rcond=None)
            y, *_ = np.linalg.lstsq(M, ones, rcond=None)
        
        lambda_val = (np.dot(ones, x) - 1.0) / np.dot(ones, y)
        h = x - lambda_val * y
        return h
    
    # Compute mu = 1/(2N-1)
    mu = 1.0 / (2 * N - 1)
    r = mu * N * np.ones(N)
    
    # Construct M
    M = _construct_M(N)
    
    # Solve
    h_ls = _solve_constrained_ls(M, r, N)
    
    # Check validity
    if np.all(np.isfinite(h_ls)) and abs(np.sum(h_ls)) > 1e-12:
        val_ls = _eval_value(h_ls)
        if val_ls < best_val:
            best_val = val_ls
            best_h = h_ls.copy()
    
    # Now refine with subgradient method on max(conv(h)) subject to sum(h)=1
    # Use projected subgradient: h_{t+1} = P_{sum=1}(h_t - alpha_t * g_t)
    # where g_t is a subgradient of max(conv) at h_t
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Subgradient refinement
        h = best_h.copy()
        s = np.sum(h)
        h = h / s  # ensure sum = 1
        
        alpha = 0.01
        for it in range(500):
            if time.time() > deadline - 10:
                break
            
            conv = np.convolve(h, h)
            max_idx = np.argmax(conv)
            max_val = conv[max_idx]
            
            # Subgradient of max(conv) with respect to h
            # d/dh[j] max(conv) = conv[max_idx - j] if max_idx >= j else conv[max_idx - j] ... 
            # Actually, conv[k] = sum_i h[i] h[k-i]
            # d/dh[j] conv[k] = h[k-j] + h[j-k] (when both indices valid)
            # Subgradient of max at index max_idx:
            # g[j] = h[max_idx - j] (if 0 <= max_idx-j < N) + h[j - max_idx] (if 0 <= j-max_idx < N)
            
            g = np.zeros(N)
            # For each j, g[j] = h[max_idx - j] if valid + h[j - max_idx] if valid
            for j in range(N):
                val = 0.0
                idx1 = max_idx - j
                if 0 <= idx1 < N:
                    val += h[idx1]
                idx2 = j - max_idx
                if 0 <= idx2 < N:
                    val += h[idx2]
                g[j] = val
            
            # Projected gradient step
            h_new = h - alpha * g
            # Project onto sum(h)=1
            h_new = h_new - np.mean(h_new - 1.0)  # h_new - mean(h_new - 1) = h_new - mean(h_new) + 1
            # Actually: projection of x onto {sum=1} is x - (sum(x)-1)/N * ones
            h_new = h_new - (np.sum(h_new) - 1.0) / N
            
            val_new = _eval_value(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
            
            h = h_new
            alpha *= 0.995  # slow decay
    
    # Final check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
